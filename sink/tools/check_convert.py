"""한컴 변환 PDF 의 글자층이 원본 내용과 맞는지 대조(US-004 증거).

원본을 한컴이 아닌 다른 방법(convert.fallback_text: HWP5 BodyText, HWP3 직접 해석, antiword, striprtf,
python-docx)으로 읽은 글과 한컴 PDF 글자층을 비교한다. 공백을 뺀 글자로
  - 글자 겹침: 두 글의 글자 다중집합 교집합 / 큰 쪽 글자 수
  - 구간%: 대체 글에서 고정 씨앗으로 고른 20자 구간 30개 중 PDF 글에서 그대로 찾은 비율
사용: .venv/Scripts/python -I tools/check_convert.py [--json cache/convert_check.json]
"""
from __future__ import annotations

import argparse
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sinkprep import catalog, convert, paths, util  # noqa: E402

WS = re.compile(r"[\s|]+")   # antiword 는 표를 '|' 로 그린다: 대조에서 뺀다


def squash(t: str) -> str:
    return WS.sub("", util.nfc(t))


def pdf_text(p: Path) -> str:
    import pymupdf
    with pymupdf.open(p) as d:
        return "".join(pg.get_text() for pg in d)


def loss(a: str, b: str) -> float:
    """대체 글에만 있고 PDF 글에는 없는 글자 비율(글자 다중집합 차 / 대체 글 글자 수). PDF 가 내용을 잃었는지 본다."""
    a = [c for c in a if c.isalnum()]          # 구분선(─)·기호는 한컴 글자층이 빈칸 글리프로 잃기도 해 뺀다
    b = [c for c in b if c.isalnum()]
    return sum((Counter(a) - Counter(b)).values()) / max(1, len(a))


def compare(a: str, b: str, seed: str) -> tuple[float, float]:
    """a=대체 글, b=PDF 글(공백 뺀 것)."""
    if not a or not b:
        return 0.0, 0.0
    ca, cb = Counter(a), Counter(b)
    overlap = sum((ca & cb).values()) / max(len(a), len(b))
    rnd = random.Random(seed)
    n = 30
    if len(a) <= 20:
        return overlap, 100.0 if a in b else 0.0
    hits = 0
    for _ in range(n):
        i = rnd.randrange(0, len(a) - 20)
        hits += a[i:i + 20] in b
    return overlap, 100.0 * hits / n


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(paths.CACHE / "convert_check.json"))
    ap.add_argument("--formats", default="hwp3,hwp5,doc,docx,rtf")
    args = ap.parse_args()
    want = set(args.formats.split(","))
    log = {}
    for x in util.read_jsonl(paths.CACHE / "convert_log.jsonl"):
        log[(x["doc_id"], x["k"])] = x
    out = []
    for r in catalog.load():
        if r["status"] == "excluded":
            continue
        for k, (f, fmt) in enumerate(zip(r["files_list"], r["file_formats_list"])):
            if fmt not in want:
                continue
            lg = log.get((r["doc_id"], k))
            pdf = paths.PDF_CACHE / r["doc_id"] / f"{k}.pdf"
            if not lg or not lg.get("ok") or lg.get("method") != "hancom" or not pdf.exists():
                continue
            try:
                got = convert.fallback_text(paths.absp(f), fmt)
                alt = got[0] if isinstance(got, tuple) else got
            except Exception as e:  # noqa: BLE001
                out.append({"doc_id": r["doc_id"], "k": k, "fmt": fmt, "error": f"{type(e).__name__}: {e}"[:200]})
                continue
            a, b = squash(alt or ""), squash(pdf_text(pdf))
            ov, seg = compare(a, b, f"{r['doc_id']}/{k}")
            out.append({"doc_id": r["doc_id"], "k": k, "fmt": fmt, "alt_chars": len(a), "pdf_chars": len(b),
                        "overlap": round(ov, 4), "seg_pct": round(seg, 1), "loss": round(loss(a, b), 4)})
    util.write_json(Path(args.json), {"schema": "sink.convert_check/1", "rows": out})
    by = defaultdict(list)
    for x in out:
        by[x["fmt"]].append(x)
    print(f"{'형식':6} {'파일':>5} {'겹침 중앙값':>10} {'겹침 최저':>9} {'구간% 중앙값':>11} {'겹침<0.95':>9} {'오류':>4}")
    for fmt in sorted(by):
        xs = [x for x in by[fmt] if "error" not in x]
        err = len(by[fmt]) - len(xs)
        ov = [x["overlap"] for x in xs]
        sg = [x["seg_pct"] for x in xs]
        low = [x for x in xs if x["overlap"] < 0.95]
        print(f"{fmt:6} {len(by[fmt]):>5} {statistics.median(ov) if ov else float('nan'):>10.3f} "
              f"{min(ov) if ov else float('nan'):>9.3f} {statistics.median(sg) if sg else float('nan'):>11.1f} "
              f"{len(low):>9} {err:>4}")
        for x in sorted(low, key=lambda x: x["overlap"])[:8]:
            print(f"    낮음 {x['doc_id']}/{x['k']} 겹침 {x['overlap']} 구간 {x['seg_pct']}% 대체 {x['alt_chars']} PDF {x['pdf_chars']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
