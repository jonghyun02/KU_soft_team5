"""파서 정확도 채점: tests/gold/*.jsonl (정답) vs docs/{doc_id}/script.json (파서 결과).

정답 행: {"n", "text", "type", "speaker", "scene_number", "note"}
  type: front | scene | action | dialogue | onscreen | transition | dropped | unknown
  speaker: 대사 줄의 화자(대본에 적힌 이름에서 (V.O.)·(E)·(계속) 같은 꼬리를 뺀 것, 여럿이면 " / ")
정답 줄은 줄 번호가 밀려도 되게 글 내용으로 지금 lines.jsonl 에 다시 맞춘다.

사용: .venv/Scripts/python -I tools/eval_parse.py [--set dev|test|all] [--fails N] [--json out.json]
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sinkprep import paths, util  # noqa: E402

GOLD = ROOT / "tests" / "gold"
SPLIT = util.read_json(GOLD / "windows.json") if (GOLD / "windows.json").exists() else {"windows": []}


def norm_text(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def norm_name(s: str | None) -> str:
    if not s:
        return ""
    parts = [re.sub(r"[\s.,·'’\"`~!?:;()\[\]{}<>\-_]", "", p).casefold() for p in re.split(r"\s*/\s*", s)]
    return "/".join(sorted(p for p in parts if p))


def line_labels(doc_id: str) -> dict[int, dict]:
    """파서 결과 -> 줄 n: {type, speaker}"""
    sp = paths.DOCS / doc_id / "script.json"
    if not sp.exists():
        return {}
    s = util.read_json(sp)
    people = {p["id"]: p["name"] for p in s.get("people", [])}
    out: dict[int, dict] = {}
    for el in s.get("elements", []):
        spk = None
        if el["type"] == "dialogue":
            spk = " / ".join(people.get(i, i) for i in el.get("speakers", [])) or None
        for n in el.get("lines", []):
            out[n] = {"type": el["type"], "speaker": spk, "el": el["id"]}
    for d in s.get("dropped", []):
        for n in d.get("lines", []):
            out[n] = {"type": "dropped", "speaker": None, "el": None, "why": d.get("why")}
    return out


def realign(doc_id: str, gold: list[dict]) -> dict[int, int]:
    """정답 n -> 지금 lines.jsonl 의 n (글 내용 기준)."""
    rows = util.read_jsonl(paths.DOCS / doc_id / "lines.jsonl")[1:]
    cur = [norm_text(r["text"]) for r in rows]
    g = [norm_text(x["text"]) for x in gold]
    lo = max(0, min(x["n"] for x in gold) - 400)
    hi = min(len(cur), max(x["n"] for x in gold) + 400)
    sm = difflib.SequenceMatcher(a=g, b=cur[lo:hi], autojunk=False)
    m = {}
    for blk in sm.get_matching_blocks():
        for i in range(blk.size):
            m[gold[blk.a + i]["n"]] = lo + blk.b + i
    return m


def evaluate(which: str = "all"):
    wins = SPLIT.get("windows", [])
    sets = {w["file"]: w.get("set", "dev") for w in wins}
    per_doc = {}
    fails = []
    tot = defaultdict(int)
    for gp in sorted(GOLD.glob("*.jsonl")):
        if which != "all" and sets.get(gp.name, "dev") != which:
            continue
        gold = util.read_jsonl(gp)
        if not gold:
            continue
        doc_id = gold[0].get("doc_id") or gp.stem.split("_w")[0]
        lab = line_labels(doc_id)
        mp = realign(doc_id, gold)
        d = defaultdict(int)
        for x in gold:
            if x["type"] == "unknown":
                continue
            d["lines"] += 1
            n = mp.get(x["n"])
            if n is None:
                d["unaligned"] += 1
                fails.append({"doc": doc_id, "n": x["n"], "text": x["text"], "gold": x["type"], "got": "UNALIGNED"})
                continue
            got = lab.get(n, {"type": "MISSING", "speaker": None})
            if got["type"] == x["type"]:
                d["type_ok"] += 1
            else:
                fails.append({"doc": doc_id, "n": n, "text": x["text"], "gold": x["type"], "got": got["type"],
                              "gold_speaker": x.get("speaker"), "got_speaker": got.get("speaker")})
            if x["type"] == "dialogue" and x.get("speaker"):
                d["spk_total"] += 1
                if got["type"] == "dialogue" and norm_name(got.get("speaker")) == norm_name(x["speaker"]):
                    d["spk_ok"] += 1
                elif got["type"] == "dialogue":
                    fails.append({"doc": doc_id, "n": n, "text": x["text"], "gold": "dialogue",
                                  "got": "dialogue", "gold_speaker": x["speaker"], "got_speaker": got.get("speaker"),
                                  "kind": "speaker"})
        key = gp.stem
        per_doc[key] = dict(d)
        for k, v in d.items():
            tot[k] += v
    return per_doc, dict(tot), fails


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="all", choices=["dev", "test", "all"])
    ap.add_argument("--fails", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    per_doc, tot, fails = evaluate(a.set)
    print(f"{'window':<28} {'lines':>5} {'type%':>6} {'spk%':>6} {'unal':>4}")
    for k, d in per_doc.items():
        tl = d.get("lines", 0) or 1
        st = d.get("spk_total", 0)
        print(f"{k:<28} {d.get('lines', 0):>5} {100 * d.get('type_ok', 0) / tl:>6.1f} "
              f"{(100 * d.get('spk_ok', 0) / st) if st else float('nan'):>6.1f} {d.get('unaligned', 0):>4}")
    tl = tot.get("lines", 0) or 1
    st = tot.get("spk_total", 0) or 1
    print(f"{'TOTAL':<28} {tot.get('lines', 0):>5} {100 * tot.get('type_ok', 0) / tl:>6.1f} "
          f"{100 * tot.get('spk_ok', 0) / st:>6.1f} {tot.get('unaligned', 0):>4}")
    for f in fails[: a.fails]:
        print(json.dumps(f, ensure_ascii=False))
    if a.json:
        Path(a.json).write_text(json.dumps({"per_doc": per_doc, "total": tot, "fails": fails},
                                           ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
