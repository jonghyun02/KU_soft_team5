"""4.4절 추론: 평가용 K-pop 2,038곡을 각 모델로 번역한다.

- <SYLs> 의 s 는 원문 한국어 줄의 음절 수 (학습 때와 달리, 4.4절).
- 줄 단위 모델과 기준 모델은 줄마다 한 번씩, 마디 단위 모델은 마디마다 한 번 번역한다.
- 빔 서치 빔 4.
- [해석] 원문은 Genius 머리줄만 빼고 그대로 넣는다. 마디 출력 줄 수가 다르면 common.fit_lines.

모델 이름: baseline, {config}_semi (non-singable 단계까지), {config}_ft (singable 단계까지)
실행: python -m kplt.model.infer --model sect_syl_ft
"""
from __future__ import annotations

import argparse
import json

import torch
from tqdm import tqdm

from ..config import BASELINE, CONFIGS, DATA, GEN_MAX_NEW_TOKENS, MAX_LEN, NUM_BEAMS, OUT
from .common import fit_lines, render, source_syllables, split_output
from .train import best_dir

STAGE_OF = {"semi": "nonsingable", "ft": "singable"}


def all_model_names():
    return ["baseline"] + [f"{c}_{k}" for c in CONFIGS for k in ("semi", "ft")]


def load(name: str, device: str):
    from transformers import MarianMTModel, MarianTokenizer
    if name == "baseline":
        tok = MarianTokenizer.from_pretrained(BASELINE)
        model = MarianMTModel.from_pretrained(BASELINE)
        return tok, model.to(device).eval(), None
    config, kind = name.rsplit("_", 1)
    d = best_dir(config, STAGE_OF[kind])
    tok = MarianTokenizer.from_pretrained(d)
    model = MarianMTModel.from_pretrained(d)
    return tok, model.to(device).eval(), CONFIGS[config]


@torch.no_grad()
def generate(tok, model, texts: list[str], device: str, batch_size: int) -> list[str]:
    # 기준 모델(opus-mt-ko-en)은 위치 임베딩이 512 라 그 안에서만 만든다
    limit = model.config.max_position_embeddings
    max_new = min(GEN_MAX_NEW_TOKENS, limit - 1)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out: list[str | None] = [None] * len(texts)
    for b in tqdm(range(0, len(order), batch_size), desc="생성", leave=False):
        idx = order[b:b + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                  max_length=min(MAX_LEN, limit)).to(device)
        gen = model.generate(**enc, num_beams=NUM_BEAMS, max_new_tokens=max_new)
        # <SYLn>, <SEP> 는 special 이 아닌 추가 토큰이라 남고, <pad> </s> <unk> 는 지워진다
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for i, t in zip(idx, dec):
            out[i] = t
    return out  # type: ignore[return-value]


def translate_songs(name: str, songs: list[dict], device: str = "cuda", batch_size: int = 32) -> dict:
    tok, model, cfg = load(name, device)
    if cfg is None or cfg[0] == "line":
        syl = bool(cfg and cfg[1])
        lines = [(si, ki, li, line) for si, s in enumerate(songs) for ki, sec in enumerate(s["sections"])
                 for li, line in enumerate(sec)]
        texts = [render([l], None, syl, source_syllables([l]))[0] if cfg else l for *_, l in lines]
        outs = generate(tok, model, texts, device, batch_size)
        res = {s["id"]: [[""] * len(sec) for sec in s["sections"]] for s in songs}
        for (si, ki, li, _), o in zip(lines, outs):
            res[songs[si]["id"]][ki][li] = " ".join(split_output(o)) if cfg else o.strip()
        return res
    syl = cfg[1]
    secs = [(si, ki, sec) for si, s in enumerate(songs) for ki, sec in enumerate(s["sections"])]
    texts = [render(sec, None, syl, source_syllables(sec))[0] for *_, sec in secs]
    outs = generate(tok, model, texts, device, max(1, batch_size // 4))
    res = {s["id"]: [None] * len(s["sections"]) for s in songs}
    for (si, ki, sec), o in zip(secs, outs):
        res[songs[si]["id"]][ki] = fit_lines(split_output(o), len(sec))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=all_model_names())
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="점검용: 앞 N곡만")
    a = ap.parse_args()
    songs = json.loads((DATA / "eval_kpop.json").read_text(encoding="utf-8"))
    if a.limit:
        songs = songs[:a.limit]
    res = translate_songs(a.model, songs, a.device, a.batch_size)
    d = OUT / "translations"
    d.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{a.limit}" if a.limit else ""
    (d / f"{a.model}{suffix}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print("저장:", d / f"{a.model}{suffix}.json")


if __name__ == "__main__":
    main()
