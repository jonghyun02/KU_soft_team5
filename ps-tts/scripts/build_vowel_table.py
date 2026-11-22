"""모음 거리표 캐시(pstts/data/vowel_table.json)를 다시 만들고 한국어 x 영어 표를 찍는다.

실행: .venv\\Scripts\\python.exe scripts\\build_vowel_table.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pstts.phonetics import EN_VOWELS, KO_VOWEL_JAMO, KO_VOWELS  # noqa: E402
from pstts.vowels import (TABLE_CSV, TABLE_JSON, VowelTable, build_vowel_table,  # noqa: E402
                          nearest_english, save_table, table_symbols)


def render(table: VowelTable) -> str:
    """한국어(행) x 영어(열) 거리표를 글로 만든다."""
    ko = [s for s in table.symbols if s in KO_VOWELS]
    en = [s for s in table.symbols if s in EN_VOWELS]
    head = "ko\\en  " + " ".join(f"{e:>5}" for e in en)
    rows = [head]
    for k in ko:
        cells = " ".join(f"{table.distance(k, e):5.2f}" for e in en)
        rows.append(f"{k:>2}({KO_VOWEL_JAMO[k]}) {cells}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="모음 거리표 캐시를 다시 만든다.")
    ap.add_argument("--json", type=Path, default=TABLE_JSON, help="캐시 JSON 경로")
    ap.add_argument("--csv", type=Path, default=TABLE_CSV, help="한국어 x 영어 CSV 경로")
    args = ap.parse_args(argv)

    table = build_vowel_table(table_symbols())
    save_table(table, args.json, args.csv)
    print(f"기호 {len(table.symbols)}개: {' '.join(table.symbols)}")
    print(f"방법: {table.method}")
    print(render(table))
    print()
    print("한국어 모음마다 가장 가까운 영어 모음(거리):")
    for k, order in nearest_english(table).items():
        best = order[0][1]
        ties = [e for e, d in order if abs(d - best) < 1e-9]
        rest = ", ".join(f"{e} {d:.2f}" for e, d in order[len(ties):len(ties) + 3])
        print(f"  {k}({KO_VOWEL_JAMO[k]}) -> {' = '.join(ties)} {best:.2f} | 다음: {rest}")
    print(f"저장: {args.json}")
    print(f"저장: {args.csv}")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):     # IPA 기호·한글이 cp949 로 깨지지 않게
        stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
