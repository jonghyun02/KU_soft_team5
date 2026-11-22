"""4.2절 학습 데이터 세 단계(General, Non-singable, Singable)를 만든다.

한 단위(unit) = {"src": [조각...], "tgt": [조각...], "shuffle": bool}. 줄 단위면 조각 1개, 마디 단위면 줄마다 조각.
글자로 펼치는 것(SYL/SEP 토큰 넣기, 줄 섞기)은 학습 때 kplt.model.dataset 이 한다.

- General (4.2절): AI Hub 한영 말뭉치(Park et al., 2022)에서 50만 쌍.
  줄 단위는 문장 그대로. 마디 단위는 두 언어 문장을 무작위로 n 조각으로 나눈다.
  [해석] 원천 파일은 enko-nmt 가 정리해 둔 AI Hub 160만 쌍(clean/*.tsv). n 은 1..min(양쪽 단어 수, 8) 균등,
  자르는 자리는 단어 경계에서 무작위.
- Non-singable (4.2절): 영어 가사 1만 곡을 구글 번역으로 한국어로. 원문 영어가 정답, 번역 한국어가 입력.
  마디 단위는 마디 안 줄 순서를 양쪽 똑같이 섞는다.
- Singable (4.2절): 논문 데이터셋. 마디 단위는 줄 순서를 섞는다.
- 9:1 로 학습·검증 분할 (부록 C). [해석] 가사 두 묶음은 곡 단위로 나눠 같은 곡이 양쪽에 들어가지 않게 했다.

실행: python -m kplt.data.build all
"""
from __future__ import annotations

import argparse
import json
import os
import random

from tqdm import tqdm

from ..config import DATA, N_GENERAL, SEED, SECTION_MAX_SEGMENTS, VALID_RATIO
from ..gtranslate import GoogleTranslator

AIHUB_TSV = [r"D:\transformer-base\enko-nmt\data\clean\train.tsv",
             r"D:\transformer-base\enko-nmt\data\clean\valid.tsv",
             r"D:\transformer-base\enko-nmt\data\clean\test.tsv"]
STAGE_DIR = DATA / "stages"


def _split_words(words: list[str], n: int, rng: random.Random) -> list[str]:
    cuts = sorted(rng.sample(range(1, len(words)), n - 1)) if n > 1 else []
    bounds = [0] + cuts + [len(words)]
    return [" ".join(words[a:b]) for a, b in zip(bounds, bounds[1:])]


def build_general():
    rng = random.Random(SEED)
    pairs = []
    for path in AIHUB_TSV:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                    pairs.append((parts[1].strip(), parts[0].strip()))   # (ko, en)
    rng.shuffle(pairs)
    pairs = pairs[:N_GENERAL]
    n_val = int(len(pairs) * VALID_RATIO)
    splits = {"valid": pairs[:n_val], "train": pairs[n_val:]}
    for split, ps in splits.items():
        line_units = [{"src": [ko], "tgt": [en], "shuffle": False} for ko, en in ps]
        sec_units = []
        for ko, en in ps:
            kw, ew = ko.split(), en.split()
            n = rng.randint(1, max(1, min(len(kw), len(ew), SECTION_MAX_SEGMENTS)))
            sec_units.append({"src": _split_words(kw, n, rng), "tgt": _split_words(ew, n, rng), "shuffle": False})
        _write("general", "line", split, line_units)
        _write("general", "section", split, sec_units)


def _song_split(songs: list, rng: random.Random):
    idx = list(range(len(songs)))
    rng.shuffle(idx)
    n_val = int(len(songs) * VALID_RATIO)
    val = set(idx[:n_val])
    return [s for i, s in enumerate(songs) if i not in val], [s for i, s in enumerate(songs) if i in val]


def _lyric_units(songs: list[list[tuple[list[str], list[str]]]]):
    """songs: 곡마다 [(src_lines, tgt_lines), ...] 마디 목록."""
    line_units, sec_units = [], []
    for song in songs:
        for src, tgt in song:
            for s, t in zip(src, tgt):
                line_units.append({"src": [s], "tgt": [t], "shuffle": False})
            sec_units.append({"src": list(src), "tgt": list(tgt), "shuffle": True})
    return line_units, sec_units


def build_nonsingable():
    rng = random.Random(SEED + 1)
    songs = json.loads((DATA / "nonsingable_en.json").read_text(encoding="utf-8"))
    gt = GoogleTranslator()
    # 모든 곡의 줄을 한 번에 넘겨 여러 곡을 한 요청에 묶는다 (번역은 줄마다 같음)
    all_lines = [l for song in songs for sec in song["sections"] for l in sec]
    all_ko = gt.translate_lines(all_lines, "en", "ko", progress=True)
    out, pos = [], 0
    for song in songs:
        flat = [l for sec in song["sections"] for l in sec]
        ko = all_ko[pos:pos + len(flat)]
        pos += len(flat)
        k, secs = 0, []
        for sec in song["sections"]:
            secs.append((ko[k:k + len(sec)], sec))
            k += len(sec)
        out.append(secs)
    (DATA / "nonsingable_pairs.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    train, valid = _song_split(out, rng)
    for split, ss in [("train", train), ("valid", valid)]:
        lu, su = _lyric_units(ss)
        _write("nonsingable", "line", split, lu)
        _write("nonsingable", "section", split, su)


def build_singable():
    rng = random.Random(SEED + 2)
    lyrics = json.loads((DATA / "lyrics.json").read_text(encoding="utf-8"))
    songs = []
    for lid in sorted(lyrics):
        songs.append([([k for k, _ in sec], [e for _, e in sec]) for sec in lyrics[lid]])
    train, valid = _song_split(songs, rng)
    for split, ss in [("train", train), ("valid", valid)]:
        lu, su = _lyric_units(ss)
        _write("singable", "line", split, lu)
        _write("singable", "section", split, su)


def _write(stage: str, form: str, split: str, units: list):
    d = STAGE_DIR / stage / form
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{split}.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    os.replace(tmp, d / f"{split}.jsonl")   # 다 쓴 뒤 이름을 바꿔 학습 쪽이 반쯤 쓴 파일을 읽지 않게
    print(f"{stage}/{form}/{split}: {len(units)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["general", "nonsingable", "singable", "all"])
    a = ap.parse_args()
    if a.what in ("general", "all"):
        build_general()
    if a.what in ("nonsingable", "all"):
        build_nonsingable()
    if a.what in ("singable", "all"):
        build_singable()
