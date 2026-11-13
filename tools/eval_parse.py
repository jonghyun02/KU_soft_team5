"""파서 정확도 채점: tests/gold/*.jsonl (정답) vs script.json (파서 결과).

정답 행: {"n", "text", "type", "speaker", "scene_number", "note"}
  type: front | scene | action | dialogue | onscreen | transition | dropped | unknown
  speaker: 대사 줄의 화자(대본에 적힌 이름에서 (V.O.)·(E)·(계속) 같은 꼬리를 뺀 것, 여럿이면 " / ")
정답 줄은 줄 번호가 밀려도 되게 글 내용으로 지금 lines.jsonl 에 다시 맞춘다.

정답 묶음(--set): tests/gold/windows.json 의 set 칸. dev·test(흔한 형식 창 *_w*), hard-dev·hard-test(규칙 파서가
무너지는 문서의 창 *_h*). hard-test 는 마지막 확인용이라 개발 중에 열지 않는다(--set all 에도 넣지 않음).

채점할 결과(--which, 기본 current):
  current  docs/{doc}/script.json (지금 쓰는 판)
  rules    규칙 판: llm 단계와 같은 길(llm_parse.get_rules: 지금 lines·parse 버전의 규칙 판, 낡았으면 다시 파싱해
           cache/parse_rules 에 씀)
  llm      LLM 판: cache/parse_llm/{doc}.json (sinkprep llm 이 만든 것)
  --script-dir DIR 를 주면 DIR/{doc}.json.
--llm-windows: 정답 창(+ 앞뒤 맥락)만 LLM 으로 라벨링하고(문서 전체를 돌리지 않아 싸다) 나머지 줄은 규칙 판으로 채운
  script 를 cache/eval_llm/{모델}[+img]/{doc}.json 에 써서 채점한다. --model 로 모델 ID 를 바꿔 비교(캐시 키에 모델이 들어감).

사용: .venv/Scripts/python -I tools/eval_parse.py [--set dev|test|hard-dev|hard-test|all] [--which current|rules|llm]
      [--script-dir DIR] [--llm-windows [--model ID] [--images] [--workers N]] [--fails N] [--json out.json]
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sinkprep import paths, util  # noqa: E402

GOLD = ROOT / "tests" / "gold"
SPLIT = util.read_json(GOLD / "windows.json") if (GOLD / "windows.json").exists() else {"windows": []}
SETS = ("dev", "test", "hard-dev", "hard-test")
RX_STEM = re.compile(r"^(fm\d+(?:_\d+)?)_([wh])\d+$")


def norm_text(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def norm_name(s: str | None) -> str:
    if not s:
        return ""
    parts = [re.sub(r"[\s.,·'’\"`~!?:;()\[\]{}<>\-_]", "", p).casefold() for p in re.split(r"\s*/\s*", s)]
    return "/".join(sorted(p for p in parts if p))


def gold_set(path: Path) -> str | None:
    """정답 파일의 묶음. windows.json 에 없으면 이름으로: *_w* 는 dev, *_h* 는 모름(어느 묶음에도 넣지 않음)."""
    for w in SPLIT.get("windows", []):
        if w.get("file") == path.name:
            return w.get("set", "dev")
    m = RX_STEM.match(path.stem)
    return "dev" if m and m.group(2) == "w" else None


def gold_files(which: str) -> list[Path]:
    """묶음의 정답 파일(이 함수는 파일 이름만 본다: hard-test 를 고르지 않으면 그 내용을 열지 않음)."""
    out = []
    for gp in sorted(GOLD.glob("*.jsonl")):
        st = gold_set(gp)
        if st is None:
            continue
        if which == "all" and st in ("dev", "test", "hard-dev") or st == which:
            out.append(gp)
    return out


def doc_of(gp: Path, gold: list[dict]) -> str:
    if gold and gold[0].get("doc_id"):
        return gold[0]["doc_id"]
    m = RX_STEM.match(gp.stem)
    return m.group(1) if m else gp.stem.split("_w")[0]


def _rows(doc_id: str) -> list[dict]:
    rows = util.read_jsonl(paths.DOCS / doc_id / "lines.jsonl")
    return rows[1:] if rows and "schema" in rows[0] else rows


def rules_script(doc_id: str) -> dict | None:
    """지금 lines·parse 버전의 규칙 판(llm 단계와 같은 길: 낡은 cache/parse_rules 는 쓰지 않음)."""
    from sinkprep import llm_parse

    lp = paths.DOCS / doc_id / "lines.jsonl"
    if not lp.exists():
        return None
    return llm_parse.get_rules(doc_id, _rows(doc_id), util.sha1_file(lp))[0]


def load_script(doc_id: str, which: str = "current", script_dir: Path | None = None) -> dict | None:
    if script_dir is not None:
        p = Path(script_dir) / f"{doc_id}.json"
    elif which == "rules":
        return rules_script(doc_id)
    elif which == "llm":
        p = paths.CACHE / "parse_llm" / f"{doc_id}.json"
    else:
        p = paths.DOCS / doc_id / "script.json"
    return util.read_json(p) if p.exists() else None


def line_labels(doc_id: str, script: dict | None = None) -> dict[int, dict]:
    """파서 결과 -> 줄 n: {type, speaker}"""
    s = script if script is not None else load_script(doc_id)
    if not s:
        return {}
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
    rows = _rows(doc_id)
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


def evaluate(which: str = "all", script_which: str = "current", script_dir: Path | None = None):
    per_doc = {}
    fails = []
    tot = defaultdict(int)
    cache: dict[str, dict] = {}
    for gp in gold_files(which):
        gold = util.read_jsonl(gp)
        if not gold:
            continue
        doc_id = doc_of(gp, gold)
        if doc_id not in cache:
            cache[doc_id] = line_labels(doc_id, load_script(doc_id, script_which, script_dir) or {})
        lab = cache[doc_id]
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
        per_doc[gp.stem] = dict(d)
        for k, v in d.items():
            tot[k] += v
    return per_doc, dict(tot), fails


def llm_windows(which: str, model: str, images: bool, workers: int, pad: bool = False,
                offline: bool = False) -> tuple[Path, dict]:
    """정답 창만 LLM 으로 라벨링한 script 를 만든다(창 밖은 규칙 판). 반환: (script 폴더, 시간 정보).
    pad 면 창 둘레를 실제 조각 크기(llm_parse.CORE 줄)까지 넓혀 라벨링한다(실제 실행과 같은 부담)."""
    from sinkprep import llm, llm_parse, parse

    llm.set_workers(workers)
    llm.reset_run(offline=offline)
    out_dir = paths.CACHE / "eval_llm" / (model + ("+img" if images else "") + ("+pad" if pad else ""))
    wins: dict[str, list[list[int]]] = defaultdict(list)
    for gp in gold_files(which):
        gold = util.read_jsonl(gp)
        if not gold:
            continue
        doc_id = doc_of(gp, gold)
        mp = realign(doc_id, gold)
        wins[doc_id].append(sorted(mp.values()))

    def one(doc_id: str):
        rows = _rows(doc_id)
        rules = parse.parse_rows(doc_id, rows)
        try:
            s, info = llm_parse.build_script(doc_id, rows, rules, model=model, images=images, workers=workers,
                                             windows=wins[doc_id], pad_to=llm_parse.CORE if pad else None)
        except llm.LLMError as e:
            return doc_id, None, {"error": f"{type(e).__name__}: {e}"}
        util.write_json(out_dir / f"{doc_id}.json", s)
        return doc_id, s, info

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        res = list(ex.map(one, sorted(wins)))
    secs = {"profile": [], "label": []}
    stat = {"docs": len(res), "errors": [], "called": 0, "cached": 0, "wall": 0.0}
    for doc_id, s, info in res:
        if s is None:
            stat["errors"].append((doc_id, info["error"]))
            continue
        stat["called"] += info["called"]
        stat["cached"] += info["cached"]
        for k in info["keys"]:
            rec = util.read_json(llm.cache_path(k))
            secs[rec["task"]].append(float(rec.get("seconds") or 0))
    stat["wall"] = round(time.time() - t0, 1)
    for t, v in secs.items():
        stat[f"{t}_calls"] = len(v)
        stat[f"{t}_sec_mean"] = round(sum(v) / len(v), 1) if v else None
        stat[f"{t}_sec_max"] = round(max(v), 1) if v else None
    return out_dir, stat


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="all", choices=list(SETS) + ["all"])
    ap.add_argument("--which", default="current", choices=["current", "rules", "llm"])
    ap.add_argument("--script-dir", default=None)
    ap.add_argument("--llm-windows", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--pad", action="store_true", help="(--llm-windows) 창 둘레를 실제 조각 크기까지 넓혀 라벨링")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--offline", action="store_true", help="(--llm-windows) 캐시만 쓰고 claude 를 부르지 않음")
    ap.add_argument("--fails", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    script_dir = Path(a.script_dir) if a.script_dir else None
    stat = None
    if a.llm_windows:
        from sinkprep import llm

        script_dir, stat = llm_windows(a.set, a.model or llm.MODEL, a.images, a.workers, a.pad, a.offline)
        print(f"LLM 창 라벨링: {script_dir} {json.dumps(stat, ensure_ascii=False)}")
    per_doc, tot, fails = evaluate(a.set, a.which, script_dir)
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
        util.write_json(Path(a.json), {"per_doc": per_doc, "total": tot, "fails": fails, "llm": stat})
    return 0


if __name__ == "__main__":
    sys.exit(main())
