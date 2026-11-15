"""정답 표본 작업 도구: 문서의 줄 구간을 보여 주고 그 쪽 그림을 만든다.

사용: .venv/Scripts/python -I tools/gold_window.py <doc_id> <start_n> <count> [--png]
  - 줄 n, 쪽, x, 글을 찍는다(탭은 ⇥ 로 보임).
  - --png 면 그 줄들이 있는 쪽을 cache/gold_png/{doc_id}_{file}_p{page}.png 로 그린다(Read 로 볼 수 있음).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sinkprep import catalog, paths, util  # noqa: E402


def source_pdf(doc: dict, k: int) -> Path | None:
    ocr = paths.OCR_CACHE / doc["doc_id"] / f"{k}.json"
    if ocr.exists():
        return paths.absp(util.read_json(ocr)["source"])
    fmt = doc["file_formats_list"][k]
    if fmt == "pdf_text":
        return paths.absp(doc["files_list"][k])
    pdf = paths.PDF_CACHE / doc["doc_id"] / f"{k}.pdf"
    return pdf if pdf.exists() else None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    doc_id, start, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    png = "--png" in sys.argv
    doc = next(d for d in catalog.load() if d["doc_id"] == doc_id)
    rows = util.read_jsonl(paths.DOCS / doc_id / "lines.jsonl")[1:]
    win = [r for r in rows if start <= r["n"] < start + count]
    for r in win:
        x = "-" if r["x"] is None else f"{r['x']:.3f}"
        print(f"{r['n']:>5} f{r['file']} p{r['page']} x={x} | {r['text'].replace(chr(9), ' ⇥ ')}")
    if png:
        import pymupdf

        out_dir = paths.CACHE / "gold_png"
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, page in sorted({(r["file"], r["page"]) for r in win if r["page"]}):
            pdf = source_pdf(doc, k)
            if not pdf:
                continue
            with pymupdf.open(pdf) as d:
                if page - 1 >= d.page_count:
                    continue
                pix = d[page - 1].get_pixmap(dpi=110)
                o = out_dir / f"{doc_id}_{k}_p{page}.png"
                pix.save(o)
                print(f"PNG {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
