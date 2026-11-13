"""형식 통일 줄: docs/{doc_id}/lines.jsonl

입력 우선순위(파일 k 마다):
  1. cache/ocr/{doc_id}/{k}.json  (스캔 PDF, 글자층 깨진 PDF)      -> src "ocr"
  2. pdf_text 원본                                                   -> src "pdf_text"
  3. cache/pdf/{doc_id}/{k}.pdf   (한컴 변환, hwp/doc/docx/rtf)     -> src 원본 형식(hwp/doc/docx/rtf)
  4. cache/pdf/{doc_id}/{k}.txt   (변환 실패 시 대체 경로 글)       -> src 원본 형식
  5. txt 원본                                                        -> src "txt"

한 줄 = 쪽 위 한 시각적 줄. 같은 높이에 떨어진 조각(이름 칸과 대사 칸 등)은 탭으로 잇는다.
x, y 는 쪽 크기에 대한 0~1 값. 쪽이 없는 글(txt)은 page=null, x=앞 공백 칸/80, y=파일 안 줄 차례/전체 줄 수
(빈 줄은 행을 만들지 않지만 y 간격으로 남는다).
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

from . import __version__, paths, util

STAGE = "lines"
VERSION = "0.3.0"

SRC_OF_FORMAT = {
    "hwp3": "hwp", "hwp5": "hwp", "hwpx": "hwp",
    "doc": "doc", "docx": "docx", "rtf": "rtf",
    "pdf_text": "pdf_text", "txt": "txt", "xls": "xls",
}
WORDPROC = {"hwp3", "hwp5", "hwpx", "doc", "docx", "rtf"}
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


# 글자층이 보통 글자 대신 쓰는 모양 같은 글자(예: 패스트 라이브즈 PDF 는 공백을 U+2029, 쌍점을 U+A789 로 씀)
GLYPH_MAP = str.maketrans({" ": " ", " ": " ", "꞉": ":", "‑": "-", "­": ""})


def clean_text(s: str) -> str:
    return util.nfc(_CTRL.sub("", s.translate(GLYPH_MAP))).strip()


# ---------------- 조각 -> 시각적 줄 ----------------

def merge_segments(segs: list[dict], page_w: float, page_h: float) -> list[dict]:
    """segs: {text, x0, y0, x1, y1, conf?, bold?} (같은 쪽). 같은 기준선 조각을 한 줄로, 큰 틈은 탭."""
    segs = [s for s in segs if s["text"].strip()]
    if not segs:
        return []
    heights = [max(1e-6, s["y1"] - s["y0"]) for s in segs]
    med_h = statistics.median(heights)
    segs.sort(key=lambda s: ((s["y0"] + s["y1"]) / 2, s["x0"]))
    rows: list[list[dict]] = []
    for s in segs:
        cy = (s["y0"] + s["y1"]) / 2
        if rows:
            last = rows[-1]
            lcy = statistics.mean((t["y0"] + t["y1"]) / 2 for t in last)
            if abs(cy - lcy) <= 0.45 * med_h:
                last.append(s)
                continue
        rows.append([s])
    out = []
    for row in rows:
        row.sort(key=lambda s: s["x0"])
        parts: list[str] = []
        prev = None
        for s in row:
            t = s["text"].strip()
            if prev is not None:
                gap = s["x0"] - prev["x1"]
                cw = max(1e-6, (prev["x1"] - prev["x0"]) / max(1, len(prev["text"].strip())))
                if gap > 1.5 * cw:
                    parts.append("\t")
                elif gap > 0.25 * cw:
                    parts.append(" ")
            parts.append(t)
            prev = s
        text = clean_text("".join(parts))
        if not text:
            continue
        confs = [s["conf"] for s in row if s.get("conf") is not None]
        nb = sum(len(s["text"]) for s in row if s.get("bold"))
        nt = sum(len(s["text"]) for s in row) or 1
        out.append({
            "x": round(min(1.0, max(0.0, row[0]["x0"] / page_w)), 4),
            "y": round(min(1.0, max(0.0, min(s["y0"] for s in row) / page_h)), 4),
            "text": text,
            "bold": nb / nt > 0.5,
            "conf": round(min(confs), 4) if confs else None,
        })
    return out


# ---------------- 입력별 줄 만들기 ----------------

_SPACES = "  　\t"
_FILLERS = ""          # 한컴 HWP3 변환 PDF 의 탭 채움 글자 -> 공백
GAP_SPACE = 0.25             # 글자 크기 대비 이만큼 벌어지면 공백으로 본다


def _span_segments(sp: dict) -> list[dict]:
    """rawdict 스팬 하나를 글자 좌표로 조각낸다.

    앞뒤 공백은 좌표에서 빼고(공백으로 들여쓴 고정폭 대본 대응), 공백이 4칸 이상 이어지면 조각을 나눈다.
    """
    chars = [dict(ch, c=" " if ch.get("c", "") in _FILLERS else ch.get("c", "").translate(GLYPH_MAP))
             for ch in sp.get("chars", [])]
    bold = bool(sp.get("flags", 0) & 16)
    size = float(sp.get("size") or 0) or None
    segs, cur, run = [], [], 0
    for ch in chars:
        c = ch.get("c", "")
        if c in _SPACES or c == "":
            run += 1
            if cur:
                cur.append(ch)
            continue
        if cur and run >= 4:
            segs.append(cur)
            cur = []
        run = 0
        cur.append(ch)
    if cur:
        segs.append(cur)
    out = []
    for cs in segs:
        while cs and cs[-1].get("c", "") in _SPACES:
            cs = cs[:-1]
        if not cs:
            continue
        # 글자층에 공백 글자가 없는데 글자 사이가 벌어져 있으면 공백을 되살린다(한컴이 HWP3 를 PDF 로 낼 때).
        buf: list[str] = []
        prev = None
        for ch in cs:
            c = ch.get("c", "")
            if prev is not None and size and c not in _SPACES and prev.get("c", "") not in _SPACES:
                if ch["bbox"][0] - prev["bbox"][2] > GAP_SPACE * size:
                    buf.append(" ")
            buf.append(c)
            prev = ch
        text = "".join(buf)
        x0 = cs[0]["bbox"][0]
        x1 = cs[-1]["bbox"][2]
        y0 = min(ch["bbox"][1] for ch in cs)
        y1 = max(ch["bbox"][3] for ch in cs)
        out.append({"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "bold": bold})
    return out


OVERPRINT = 0.15   # 같은 글자가 글자 크기의 이 배 안쪽에 또 찍혀 있으면 겹쳐 찍은 굵은 글씨로 보고 하나만 남긴다


def _drop_overprint(spans: list[dict]) -> list[dict]:
    """굵게 보이려고 같은 글자를 조금씩 밀어 여러 번 찍은 글자층(예: '5.5.5.5.')에서 겹친 글자를 뺀다."""
    kept: dict[str, list[tuple[float, float]]] = {}
    out = []
    for sp in spans:
        tol = OVERPRINT * float(sp.get("size") or 10)
        chars = []
        for ch in sp.get("chars", []):
            c = ch.get("c", "")
            if c.strip() and c not in _FILLERS:
                x, y = ch["bbox"][0], ch["bbox"][1]
                seen = kept.setdefault(c, [])
                if any(abs(x - a) < tol and abs(y - b) < tol for a, b in seen):
                    continue
                seen.append((x, y))
            chars.append(ch)
        out.append(dict(sp, chars=chars))
    return out


TWO_UP_ASPECT = 1.15    # 가로가 세로의 이 배를 넘는 쪽만 두 쪽 펼침 후보
GUTTER_MIN = 0.015      # 가운데 빈 띠의 최소 폭(쪽 너비 비율)
GUTTER_ZONE = (0.3, 0.7)


def _ink(ch: dict) -> bool:
    c = ch.get("c", "")
    return bool(c.strip()) and c not in _FILLERS


def _gutter(spans: list[dict], w: float, h: float) -> float | None:
    """가로 쪽 한가운데 근처에 글자가 하나도 없는 세로 띠가 있고 양쪽에 글이 고루 있으면 그 띠의 가운데 x.
    한 장에 대본 두 쪽을 나란히 인쇄한 문서(예: 밀정 fm5486140, 쪽 꼬리 '- 2 - ⇥ - 3 -')를 찾는다."""
    if w <= TWO_UP_ASPECT * h:
        return None
    boxes = [(ch["bbox"][0], ch["bbox"][2]) for sp in spans for ch in sp.get("chars", []) if _ink(ch)]
    if len(boxes) < 40:
        return None
    nb = 200
    lo, hi = int(GUTTER_ZONE[0] * nb), int(GUTTER_ZONE[1] * nb)
    used = [False] * nb
    for a, b in boxes:
        for i in range(max(lo, int(a / w * nb)), min(hi, int(b / w * nb) + 1)):
            used[i] = True
    best, run, start = None, 0, 0
    for i in range(lo, hi):
        if not used[i]:
            if run == 0:
                start = i
            run += 1
            if best is None or run > best[1]:
                best = (start, run)
        else:
            run = 0
    if not best or best[1] / nb < GUTTER_MIN:
        return None
    g = (best[0] + best[1] / 2) / nb * w
    left = sum(1 for a, b in boxes if b <= g)
    if not (0.15 <= left / len(boxes) <= 0.85):
        return None
    return g


def _halves(spans: list[dict], g: float) -> tuple[list[dict], list[dict]]:
    """스팬마다 글자 가운데 x 가 g 왼쪽/오른쪽인 글자로 나눈다(공백은 바로 앞 글자 쪽을 따름)."""
    left, right = [], []
    for sp in spans:
        lc, rc, side = [], [], None
        for ch in sp.get("chars", []):
            if _ink(ch):
                side = "L" if (ch["bbox"][0] + ch["bbox"][2]) / 2 < g else "R"
            (lc if (side or ("L" if ch["bbox"][0] < g else "R")) == "L" else rc).append(ch)
        if lc:
            left.append(dict(sp, chars=lc))
        if rc:
            right.append(dict(sp, chars=rc))
    return left, right


def lines_from_pdf(pdf: Path, src: str) -> list[dict]:
    """글자층 PDF -> 줄. 한 장에 두 쪽을 나란히 인쇄한 문서는 가운데 빈 띠에서 나눠 왼쪽 쪽, 오른쪽 쪽 차례의
    논리 쪽으로 만든다(page 는 논리 쪽 번호, x 는 그 쪽 안 위치). 가로 쪽 절반 넘게 띠가 있으면 문서 전체를
    펼침으로 보고, 띠를 못 찾은 쪽(한쪽이 빈 마지막 장 등)은 다른 쪽 띠의 중앙값에서 나눈다."""
    import pymupdf as fitz

    out = []
    with fitz.open(pdf) as doc:
        pages = []
        for page in doc:
            w, h = page.rect.width, page.rect.height
            d = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP)
            spans = _drop_overprint([sp for b in d.get("blocks", []) if b.get("type", 0) == 0
                                     for ln in b.get("lines", []) for sp in ln.get("spans", [])])
            pages.append((w, h, spans, _gutter(spans, w, h)))
        land = [p for p in pages if p[0] > TWO_UP_ASPECT * p[1]]
        found = [p[3] / p[0] for p in land if p[3] is not None]
        two_up = bool(land) and len(found) >= 0.5 * len(land)
        g_med = statistics.median(found) if found else 0.5
        lp = 0
        for w, h, spans, g in pages:
            if two_up and w > TWO_UP_ASPECT * h:
                g = g if g is not None else g_med * w
                left, right = _halves(spans, g)
                parts = [(left, 0.0, g), (right, g, w - g)]
            else:
                parts = [(spans, 0.0, w)]
            for part, off, pw in parts:
                lp += 1
                segs = []
                for sp in part:
                    segs.extend(_span_segments(sp))
                for s in segs:
                    s["x0"] -= off
                    s["x1"] -= off
                for r in merge_segments(segs, pw, h):
                    r.update(page=lp, src=src)
                    out.append(r)
    return out


def lines_from_ocr(js: Path) -> list[dict]:
    data = util.read_json(js)
    out = []
    for pg in data.get("pages", []):
        w, h = float(pg["width"]), float(pg["height"])
        segs = []
        for ln in pg.get("lines", []):
            # OCR 이 한 줄 안에서 떨어진 덩어리(화자 칸·대사 칸)를 segs 로 주면 덩어리마다 넣는다(탭으로 이어짐).
            for s in (ln.get("segs") or [ln]):
                segs.append({"text": s["text"], "x0": s["box"][0], "y0": s["box"][1],
                             "x1": s["box"][2], "y1": s["box"][3], "conf": s.get("conf")})
        for r in merge_segments(segs, w, h):
            r.update(page=pg["page"], src="ocr")
            out.append(r)
    return out


def decode_text(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", "replace")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", "replace")
    for enc in ("utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    from charset_normalizer import from_bytes

    best = from_bytes(raw).best()
    return str(best) if best else raw.decode("latin-1")


def lines_from_text(text: str, src: str) -> list[dict]:
    phys = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    total = max(1, len(phys))
    out = []
    for i, ln in enumerate(phys):
        exp = ln.expandtabs(8) if ln[:1] in (" ", "\t", "　") else ln
        lead = len(exp) - len(exp.lstrip(" \t　"))
        t = clean_text(ln)
        if not t:
            continue
        out.append({"page": None, "x": round(min(1.0, lead / 80), 4), "y": round(i / total, 6),
                    "text": t, "bold": False, "conf": None, "src": src})
    return out


def _xls_cell(v) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return re.sub(r"\s*\n\s*", " ", str(v)).strip()


def lines_from_xls(path: Path) -> list[dict]:
    """엑셀(.xls) 표 대본(녹음 대본 등): 시트의 한 행 = 한 줄, 빈 칸을 뺀 칸들을 탭으로 잇는다.
    x 는 첫 칸의 열 위치(열/열 수), y 는 행/행 수, cell 은 {table: 시트 번호, row, col: 첫 칸}.
    빈 시트와 앞 시트와 내용이 똑같은 시트는 건너뛴다."""
    import xlrd

    out: list[dict] = []
    seen: set = set()
    with xlrd.open_workbook(str(path), on_demand=True) as wb:
        for si in range(wb.nsheets):
            sh = wb.sheet_by_index(si)
            rows = [[_xls_cell(v) for v in sh.row_values(r)] for r in range(sh.nrows)]
            key = tuple(tuple(r) for r in rows)
            if not any(any(r) for r in rows) or key in seen:
                continue
            seen.add(key)
            ncols = max(1, sh.ncols)
            for r, vals in enumerate(rows):
                cells = [(c, v) for c, v in enumerate(vals) if v]
                if not cells:
                    continue
                col = cells[0][0]
                out.append({"page": None, "x": round(col / ncols, 4), "y": round(r / max(1, sh.nrows), 4),
                            "text": clean_text("\t".join(v for _, v in cells)), "bold": False,
                            "cell": {"table": si, "row": r, "col": col}, "src": "xls", "conf": None})
    return out


def _ocr_current(js: Path, pdf: Path) -> bool:
    """OCR 결과가 지금의 입력 PDF 에서 만든 것일 때만 쓴다(원본·변환 PDF 가 바뀌면 낡은 OCR 을 쓰지 않음)."""
    from . import ocr
    try:
        return ocr.result_is_current(util.read_json(js), pdf)
    except Exception:  # noqa: BLE001 - 깨진 결과는 쓰지 않음
        return False


def input_for(doc: dict, k: int, relpath: str, fmt: str) -> tuple[str, Path | None, str]:
    """(종류, 입력 경로, src). 종류: ocr|pdf|text|xls|none"""
    doc_id = doc["doc_id"]
    ocr = paths.OCR_CACHE / doc_id / f"{k}.json"
    if ocr.exists() and _ocr_current(ocr, paths.PDF_CACHE / doc_id / f"{k}.pdf" if fmt in WORDPROC
                                     else paths.absp(relpath)):
        return "ocr", ocr, "ocr"
    if fmt == "pdf_text":
        return "pdf", paths.absp(relpath), "pdf_text"
    if fmt in WORDPROC:
        pdf = paths.PDF_CACHE / doc_id / f"{k}.pdf"
        if pdf.exists():
            return "pdf", pdf, SRC_OF_FORMAT[fmt]
        txt = paths.PDF_CACHE / doc_id / f"{k}.txt"
        if txt.exists():
            return "text", txt, SRC_OF_FORMAT[fmt]
        return "none", None, SRC_OF_FORMAT[fmt]
    if fmt == "txt":
        return "text", paths.absp(relpath), "txt"
    if fmt == "xls":
        return "xls", paths.absp(relpath), "xls"
    return "none", None, fmt


def build_doc(doc: dict, force: bool = False) -> tuple[str, int]:
    doc_id = doc["doc_id"]
    files = doc["files_list"]
    fmts = doc["file_formats_list"]
    plan = [input_for(doc, k, f, fmt) for k, (f, fmt) in enumerate(zip(files, fmts))]
    fp = {"version": VERSION,
          "inputs": [[kind, util.sha1_file(p) if p else None] for kind, p, _ in plan]}
    out = paths.DOCS / doc_id / "lines.jsonl"
    if not force and out.exists() and util.stage_fp(STAGE, doc_id) == fp:
        return "skip", -1
    rows: list[dict] = []
    problems = []
    for k, (kind, p, src) in enumerate(plan):
        try:
            if kind == "ocr":
                got = lines_from_ocr(p)
            elif kind == "pdf":
                got = lines_from_pdf(p, src)
            elif kind == "text":
                got = lines_from_text(decode_text(p.read_bytes()), src)
            elif kind == "xls":
                got = lines_from_xls(p)
            else:
                got = []
                problems.append({"level": "doc", "doc_id": doc_id, "film_id": doc.get("film_id") or None,
                                 "item_id": None, "problem": "extract_broken", "score": 0,
                                 "detail": f"파일 {k}({fmts[k]})에서 글을 얻을 입력이 없음(변환·OCR 결과 없음)",
                                 "path": files[k]})
        except Exception as e:  # noqa: BLE001 - 문서 하나 실패가 전체를 멈추지 않게
            got = []
            problems.append({"level": "doc", "doc_id": doc_id, "film_id": doc.get("film_id") or None,
                             "item_id": None, "problem": "extract_broken", "score": 0,
                             "detail": f"파일 {k} 줄 추출 오류: {type(e).__name__}: {e}"[:300],
                             "path": files[k]})
        for r in got:
            rows.append({"n": len(rows), "file": k, "page": r["page"], "x": r["x"], "y": r["y"],
                         "text": r["text"], "bold": r["bold"], "cell": r.get("cell"), "src": r["src"],
                         "conf": r["conf"]})
    if not rows and not problems:
        problems.append({"level": "doc", "doc_id": doc_id, "film_id": doc.get("film_id") or None,
                         "item_id": None, "problem": "extract_broken", "score": 0,
                         "detail": "글자가 한 줄도 나오지 않음", "path": files[0] if files else ""})
    header = {"schema": "sink.lines/1", "doc_id": doc_id, "files": files,
              "made": {"stage": STAGE, "version": VERSION, "sinkprep": __version__}}
    util.write_jsonl(out, [header] + rows)
    util.set_problems(STAGE, doc_id, problems)
    util.set_stage_fp(STAGE, doc_id, fp)
    return ("ok" if rows else "empty"), len(rows)


def select_docs(args) -> list[dict]:
    from . import catalog

    docs = [d for d in catalog.load() if d.get("status") != "excluded"]
    if getattr(args, "docs", None):
        want = set(args.docs.split(","))
        docs = [d for d in docs if d["doc_id"] in want]
    if getattr(args, "limit", None):
        docs = docs[: args.limit]
    return docs


def run(args) -> int:
    docs = select_docs(args)
    counts = {"ok": 0, "empty": 0, "skip": 0, "error": 0}
    for i, d in enumerate(docs, 1):
        try:
            st, n = build_doc(d, force=getattr(args, "force", False))
        except Exception as e:  # noqa: BLE001
            st, n = "error", 0
            print(f"  ! {d['doc_id']}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        counts[st] += 1
        if st != "skip" and (i % 50 == 0 or st != "ok"):
            print(f"[{i}/{len(docs)}] {d['doc_id']} {st} {n}줄", flush=True)
    print(f"lines: {counts}", flush=True)
    return 1 if counts["error"] else 0
