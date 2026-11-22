"""한 설정(line / line_syl / sect / sect_syl)의 한 단계를 학습한다 (4.1절, 그림 4, 표 4, 부록 C).

- general 단계는 무작위 초기화에서, 그다음 단계는 앞 단계의 가장 좋은 체크포인트에서 시작한다.
- 손실은 NLL(교차 엔트로피), 최적화는 Adam 계열(HF Trainer 기본 AdamW, weight decay 0.1), warmup 500,
  batch 8. 검증 손실이 3 epoch 연속 줄지 않으면 멈추고 가장 좋은 체크포인트를 남긴다 (부록 C).
- [해석] 학습률 5e-5 (HF Trainer 기본), warmup 500 뒤 고정. 최대 epoch 100 (조기 종료가 먼저 걸림).
- [해석] 토큰 1024 개(표 4 최대 위치)를 넘는 단위는 뺀다.
- --max-train 점검 실행은 runs\_smoke 에 쓰고 done.json 을 남기지 않는다.
- 마디 단위 데이터의 줄 섞기는 꺼낼 때마다 새로 섞는다 (매 epoch 다른 순서). 검증 데이터는 섞지 않는다.

실행: python -m kplt.model.train --config sect_syl --stage general
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time

import torch
from torch.utils.data import Dataset

from ..config import (BATCH_SIZE, CONFIGS, DATA, EARLY_STOP_PATIENCE, LEARNING_RATE, LR_SCHEDULER, MAX_EPOCHS,
                      MAX_LEN, RUNS, SEED, STAGES, WARMUP_STEPS, WEIGHT_DECAY)
from .common import get_tokenizer, new_model, render_unit


class UnitDataset(Dataset):
    """[해석] 토큰이 MAX_LEN 을 넘는 단위는 자르면 원문·정답 줄이 어긋나므로 뺀다 (뺀 수는 done.json 에 기록)."""

    def __init__(self, path, tok, syl: bool, augment: bool, seed: int = SEED, limit: int | None = None):
        units = [json.loads(l) for l in open(path, encoding="utf-8")]
        if limit:
            units = units[:limit]
        self.tok, self.syl, self.augment = tok, syl, augment
        self.rng = random.Random(seed)
        self.units, self.dropped = [], 0
        for u in units:
            s, t = render_unit(u, syl, None)
            if len(tok(s)["input_ids"]) > MAX_LEN or len(tok(text_target=t)["input_ids"]) > MAX_LEN:
                self.dropped += 1
            else:
                self.units.append(u)

    def __len__(self):
        return len(self.units)

    def __getitem__(self, i):
        src, tgt = render_unit(self.units[i], self.syl, self.rng if self.augment else None)
        enc = self.tok(src, text_target=tgt)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": enc["labels"]}


def stage_dir(config: str, stage: str):
    return RUNS / config / stage


def best_dir(config: str, stage: str):
    return stage_dir(config, stage) / "best"


def train(config: str, stage: str, bf16: bool = False, max_epochs: int = MAX_EPOCHS,
          max_train: int | None = None, out_root=None):
    from transformers import (DataCollatorForSeq2Seq, EarlyStoppingCallback, MarianMTModel, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments, set_seed)
    set_seed(SEED)
    form, syl = CONFIGS[config]
    tok = get_tokenizer()
    i = STAGES.index(stage)
    if max_train and out_root is None:
        out_root = RUNS / "_smoke"          # 점검용 실행이 진짜 단계 결과로 남지 않게
    if i == 0:
        model = new_model(tok)
    else:
        prev = (out_root or RUNS) / config / STAGES[i - 1] / "best" if max_train else best_dir(config, STAGES[i - 1])
        if not (prev / "config.json").exists():
            raise SystemExit(f"앞 단계 결과가 없음: {prev}")
        model = MarianMTModel.from_pretrained(prev)
    data_dir = DATA / "stages" / stage / form
    train_ds = UnitDataset(data_dir / "train.jsonl", tok, syl, augment=True, limit=max_train)
    valid_ds = UnitDataset(data_dir / "valid.jsonl", tok, syl, augment=False,
                           limit=max(1, max_train // 9) if max_train else None)
    out = (out_root or RUNS) / config / stage
    args = Seq2SeqTrainingArguments(
        output_dir=str(out / "ckpt"),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 8,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=max_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=EARLY_STOP_PATIENCE + 1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=500,
        report_to="none",
        bf16=bf16,
        dataloader_num_workers=0,
        seed=SEED,
        save_safetensors=True,
        restore_callback_states_from_checkpoint=True,   # 다시 시작해도 조기 종료 셈을 이어감
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model, padding=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )
    last = None
    ck = out / "ckpt"
    if ck.exists():
        cks = sorted(ck.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        last = str(cks[-1]) if cks else None
    t0 = time.time()
    trainer.train(resume_from_checkpoint=last)
    best = out / "best"
    trainer.save_model(str(best))
    tok.save_pretrained(str(best))
    hist = trainer.state.log_history
    (out / "history.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
    info = {"config": config, "stage": stage, "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_eval_loss": trainer.state.best_metric, "epochs_run": trainer.state.epoch,
            "seconds": time.time() - t0, "n_train": len(train_ds), "n_valid": len(valid_ds),
            "dropped_over_max_len": {"train": train_ds.dropped, "valid": valid_ds.dropped},
            "smoke": bool(max_train)}
    (out / ("smoke.json" if max_train else "done.json")).write_text(json.dumps(info, indent=1), encoding="utf-8")
    shutil.rmtree(ck, ignore_errors=True)
    print(info)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--bf16", action="store_true", help="[논문 밖] 속도용 bf16. 기본은 fp32")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--max-train", type=int, default=None, help="점검용: 학습 단위 수 제한")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    train(a.config, a.stage, bf16=a.bf16, max_epochs=a.max_epochs, max_train=a.max_train)
