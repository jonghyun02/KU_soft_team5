"""Sakoe-Chiba 반경에 따라 모음 DTW 가 길이 어긋남을 상 주는지 잰다(Config.sakoe_chiba_radius 의 근거).

앞 실행의 results.jsonl 에서 입 모양 비교를 켠 줄의 통과 후보를 가져와, 반경마다 두 가지를 잰다.
1. 같은 글을 원문 프레임 수 -26~+26(2프레임 간격)으로 늘이고 줄였을 때, 원문과 길이가 같을 때의 DTW 가
   10프레임 이상 어긋난 쪽 평균보다 큰 글의 수. tslearn 은 길이가 다르면 띠를 |n - m| 만큼 넓혀서 반경이 작으면
   이 수가 크다(길이가 맞는 후보가 진다).
2. 그 반경으로 다시 골랐을 때 줄마다 고른 후보의 음절 수(목표 N). COMET 은 results.jsonl 에 적힌 값을 쓴다.
모델을 올리지 않는다(eSpeak, PanPhon 모음표, tslearn 만).

  .venv\\Scripts\\python.exe scripts\\measure_dtw_band.py out\\demo\\results.jsonl
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pstts.config import Config  # noqa: E402
from pstts.dtw import ps_distance  # noqa: E402
from pstts.phonetics import english_syllables, korean_syllables  # noqa: E402
from pstts.records import Candidate, LineItem, Target  # noqa: E402
from pstts.select import ps_comet, source_seconds  # noqa: E402
from pstts.timing import frame_labels  # noqa: E402
from pstts.vowels import load_or_build  # noqa: E402

RADII = (1, 3, 5, 8, 10, 12, 14, 16, 18, 20, 22, 26, 30, 40)
OFFSETS = range(-26, 27, 2)
FAR = 10            # 이만큼(프레임) 이상 어긋난 쪽 평균과 견준다


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sakoe-Chiba 반경별 모음 DTW 길이 쏠림 재기")
    ap.add_argument("results", type=Path, nargs="?", default=ROOT / "out" / "demo" / "results.jsonl",
                    help="앞 실행의 results.jsonl")
    args = ap.parse_args(argv)
    cfg = Config()
    table = load_or_build()
    rows = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = []
    for row in rows:
        if not row["lip"]:
            continue
        t = row["target"]
        target = Target(n=t["n"], tol=t["tol"], en_syllables=t["en_syllables"], basis=t["basis"],
                        seconds_est=t["seconds_est"])
        item = LineItem(id=row["id"], source_en=row["source_en"], seconds=row.get("seconds_input"))
        syls = english_syllables(item.source_en)
        passing = [c for c in row["candidates"] if c["passed"] and c["comet"] is not None]
        if syls and passing:
            src = frame_labels(syls, len(syls) / source_seconds(item, target, cfg), cfg)
            lines.append((row["id"], target, src, passing))
    n_texts = sum(len(p) for *_, p in lines)
    print(f"{args.results}: 입 모양 줄 {len(lines)}개, 통과 후보 {n_texts}개")
    for radius in RADII:
        rcfg = dataclasses.replace(cfg, sakoe_chiba_radius=radius)
        peaks = 0
        picks = []
        for line_id, target, src, passing in lines:
            n = len(src)
            cands = []
            for c in passing:
                ko = korean_syllables(c["text"])
                by_offset: dict[int, float] = {}
                for d in OFFSETS:
                    if n + d < 2:
                        continue
                    tgt = frame_labels(ko, len(ko) / ((n + d) * cfg.frame_sec), cfg)
                    by_offset[len(tgt) - n] = ps_distance(src, tgt, table, rcfg)
                far = [v for off, v in by_offset.items() if abs(off) >= FAR]
                if far and by_offset[min(by_offset, key=abs)] > sum(far) / len(far):
                    peaks += 1
                cand = Candidate(text=c["text"], origin=c["origin"], round=c["round"], syllables=c["syllables"],
                                 comet=c["comet"])
                cand.dtw = ps_distance(src, frame_labels(ko, cfg.ko_rate, cfg), table, rcfg)
                cands.append(cand)
            best = ps_comet(cands, rcfg, True, target)
            picks.append(f"{line_id} {best.syllables}/{target.n}")
        print(f"반경 {radius:>2}({radius * cfg.frame_sec * 1000:4.0f}ms): 원문 길이에서 솟는 글 {peaks:>3}/{n_texts}"
              f" | 고른 음절/목표: {', '.join(picks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
