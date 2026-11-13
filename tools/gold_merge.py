"""두 사람이 붙인 정답을 맞춰 보고, 갈린 줄만 판정해 최종 정답 파일을 만든다.

  .venv/Scripts/python -I tools/gold_merge.py <stem>                 # 갈린 줄 보기(stem 예: fm54897_w1)
  .venv/Scripts/python -I tools/gold_merge.py <stem> --resolve r.json   # 판정 반영해 tests/gold/<stem>.jsonl 쓰기

raw 파일: tests/gold/raw/<stem>.a.jsonl, <stem>.b.jsonl  (행: {"n","type","speaker","scene_number","note"})
r.json: 갈린 줄의 판정 목록 [{"n","type","speaker","scene_number","note"}]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from sinkprep import paths, util  # noqa: E402
from eval_parse import norm_name  # noqa: E402

GOLD = ROOT / "tests" / "gold"
TYPES = {"front", "scene", "action", "dialogue", "onscreen", "transition", "dropped", "unknown"}


def same(a: dict, b: dict) -> bool:
    if a.get("type") != b.get("type"):
        return False
    if a["type"] == "dialogue" and norm_name(a.get("speaker")) != norm_name(b.get("speaker")):
        return False
    return True


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    stem = sys.argv[1]
    win = next(w for w in util.read_json(GOLD / "windows.json")["windows"] if w["file"] == f"{stem}.jsonl")
    rows = {r["n"]: r for r in util.read_jsonl(paths.DOCS / win["doc_id"] / "lines.jsonl")[1:]}
    a = {r["n"]: r for r in util.read_jsonl(GOLD / "raw" / f"{stem}.a.jsonl")}
    b = {r["n"]: r for r in util.read_jsonl(GOLD / "raw" / f"{stem}.b.jsonl")}
    ns = list(range(win["start"], win["start"] + win["count"]))
    bad = []
    for n in ns:
        if n not in rows:
            continue
        ra, rb = a.get(n), b.get(n)
        if not ra or not rb or ra.get("type") not in TYPES or rb.get("type") not in TYPES or not same(ra, rb):
            bad.append(n)
    if "--resolve" not in sys.argv:
        agree = sum(1 for n in ns if n in rows) - len(bad)
        print(f"{stem}: 일치 {agree}, 갈림 {len(bad)}")
        for n in bad:
            print(json.dumps({"n": n, "text": rows[n]["text"], "a": a.get(n), "b": b.get(n)}, ensure_ascii=False))
        return 0
    res = {r["n"]: r for r in json.loads(Path(sys.argv[sys.argv.index("--resolve") + 1]).read_text(encoding="utf-8"))}
    missing = [n for n in bad if n not in res]
    if missing:
        print(f"판정이 빠진 줄: {missing}")
        return 1
    out = []
    for n in ns:
        if n not in rows:
            continue
        src = res.get(n) or a[n]
        if src.get("type") not in TYPES:
            print(f"잘못된 type: {n} {src}")
            return 1
        out.append({"doc_id": win["doc_id"], "n": n, "text": rows[n]["text"], "type": src["type"],
                    "speaker": src.get("speaker") if src["type"] == "dialogue" else None,
                    "scene_number": src.get("scene_number") if src["type"] == "scene" else None,
                    "note": src.get("note"), "agreed": n not in bad})
    util.write_jsonl(GOLD / f"{stem}.jsonl", out)
    print(f"썼음: tests/gold/{stem}.jsonl ({len(out)}줄, 판정 {len(bad)}줄)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
