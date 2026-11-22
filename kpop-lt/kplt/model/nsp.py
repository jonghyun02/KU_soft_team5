"""4.3절 NSP 점수용 모델: bert-base-uncased 의 NSP 머리를 영어 가사 7,103곡으로 미세조정한다.

논문에는 학습 쌍 만드는 법과 학습 값이 없다. [해석] BERT 원래 방식을 따랐다.
- 한 곡의 줄을 순서대로 이어 놓고, 이웃한 두 줄 (A, B) 마다 반은 진짜 다음 줄(라벨 0), 반은 다른 곡의
  무작위 줄(라벨 1)로 바꾼다.
- 곡 단위 9:1 분할, 3 epoch, 학습률 2e-5, batch 32, 최대 128 토큰, 검증 손실이 가장 낮은 epoch 를 남긴다.

실행: python -m kplt.model.nsp
"""
from __future__ import annotations

import json
import random

from torch.utils.data import Dataset

from ..config import DATA, NSP_BASE, NSP_BATCH, NSP_EPOCHS, NSP_LR, RUNS, SEED, VALID_RATIO

OUT_DIR = RUNS / "nsp"


def make_pairs(songs: list[list[str]], rng: random.Random):
    all_lines = [(si, l) for si, s in enumerate(songs) for l in s]
    pairs = []
    for si, lines in enumerate(songs):
        for a, b in zip(lines, lines[1:]):
            if rng.random() < 0.5:
                pairs.append((a, b, 0))
            else:
                while True:
                    sj, rb = rng.choice(all_lines)
                    if sj != si:
                        break
                pairs.append((a, rb, 1))
    return pairs


class PairDS(Dataset):
    def __init__(self, pairs, tok):
        self.pairs, self.tok = pairs, tok

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, b, y = self.pairs[i]
        enc = self.tok(a, b, truncation=True, max_length=128)
        enc["labels"] = y
        return enc


def main():
    from transformers import (BertForNextSentencePrediction, BertTokenizerFast, DataCollatorWithPadding, Trainer,
                              TrainingArguments, set_seed)
    set_seed(SEED)
    rng = random.Random(SEED)
    songs = json.loads((DATA / "nsp_en.json").read_text(encoding="utf-8"))
    songs = [[l for sec in s["sections"] for l in sec] for s in songs]
    rng.shuffle(songs)
    n_val = int(len(songs) * VALID_RATIO)
    train_p, valid_p = make_pairs(songs[n_val:], rng), make_pairs(songs[:n_val], rng)
    tok = BertTokenizerFast.from_pretrained(NSP_BASE)
    model = BertForNextSentencePrediction.from_pretrained(NSP_BASE)
    args = TrainingArguments(
        output_dir=str(OUT_DIR / "ckpt"), per_device_train_batch_size=NSP_BATCH, per_device_eval_batch_size=128,
        learning_rate=NSP_LR, num_train_epochs=NSP_EPOCHS, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, logging_steps=500, report_to="none", seed=SEED)
    trainer = Trainer(model=model, args=args, train_dataset=PairDS(train_p, tok), eval_dataset=PairDS(valid_p, tok),
                      data_collator=DataCollatorWithPadding(tok))
    trainer.train()
    pred = trainer.predict(PairDS(valid_p, tok))
    acc = float((pred.predictions.argmax(-1) == pred.label_ids).mean())
    trainer.save_model(str(OUT_DIR / "best"))
    tok.save_pretrained(str(OUT_DIR / "best"))
    info = {"n_train_pairs": len(train_p), "n_valid_pairs": len(valid_p), "valid_acc": acc,
            "best_eval_loss": trainer.state.best_metric}
    (OUT_DIR / "done.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(info)


if __name__ == "__main__":
    main()
