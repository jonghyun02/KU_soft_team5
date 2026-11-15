"""형식 통일 줄: docs/{doc_id}/lines.jsonl

입력 우선순위(파일 k 마다):
  1. cache/ocr/{doc_id}/{k}.json  (스캔 PDF, 글자층 깨진 PDF)      -> src "ocr"
  2. pdf_text 원본                                                   -> src "pdf_text"
  3. cache/pdf/{doc_id}/{k}.pdf   (한컴 변환, hwp/doc/docx/rtf)     -> src 원본 형식(hwp/doc/docx/rtf)
  4. cache/pdf/{doc_id}/{k}.txt   (변환 실패 시 대체 경로 글)       -> src 원본 형식
  5. txt 원본                                                        -> src "txt"

3번 길(한컴 변환 PDF)에서 줄의 띄어쓰기가 원본보다 뚜렷이 모자란 파일은 줄 글의 공백·탭을 원본 문서 글에 맞춘다
(restore_spacing: 줄 수·차례·좌표·글자는 그대로, text 의 공백·탭만 바뀜). 맞춘 파일과 원본 글과 견주지 못한 파일은
머리 행 made.spacing 에 기록한다. 그 밖의 파일은 줄 글을 건드리지 않는다.

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
VERSION = "0.4.1"

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


# ---------------- 원본 글로 공백·탭 되살리기 ----------------
# 한컴이 낸 PDF 는 글꼴에 공백 글자가 없어 글자 틈으로 공백을 되살린다(_span_segments). 글자 상자가 넓게 찍힌 줄은
# 틈이 문턱을 못 넘어 띄어쓰기가 통째로 사라지고, 이름 칸을 꽉 채운 긴 이름 뒤의 탭은 공백 하나가 된다(추격자 fm55664:
# '씨발년...넌잡히면죽는다.', '모텔남1 빨리왔네.'). 원본 문서의 글(convert 교차 검사가 쓰는 대체 경로 읽기)에는 공백·탭이
# 그대로 있으므로, 공백을 뺀 글자열을 순서대로 맞춰 원본의 낱말 경계를 줄 글에 되살린다. 공백이 뚜렷이 모자란 파일만:
# 그런 파일은 줄 글의 띄어쓰기를 원본에 맞춘다(빠진 공백을 넣고, 이름 뒤 탭을 탭으로 올리고, 원본에 없는 PDF 공백
# '야! ! 우리가' 를 지우고, 원본이 공백 하나인데 넓게 찍혀 탭이 된 자리 '호창,⇥두려워진다.' 를 공백으로 내린다).
# 문턱 아래 파일에도 이름 뒤 탭이 공백이 된 줄(20줄 이상인 문서 48개)과 띄어쓰기가 통째로 사라진 줄(27문서 405줄)이
# 있지만 손대지 않는다: 그 문서들의 대사 요소가 100~400개씩 달라지는 일이라 따로 정한다(notes/parser.md '알려진 문제').

# 공백 손실 = 원본의 낱말 경계(공백·탭) 가운데 PDF 줄에 공백도 탭도 없는 비율(같은 줄 안에서 이웃한 글자로 맞은 경계만).
# 말뭉치의 한컴 PDF 경로 469파일(2026-11-15): 0.001 미만 425, 0.001~0.03 30, 0.04~0.08 4(커밍아웃 둘, 로드 무비,
# 비디오를 보는 남자: 일부 줄만), 0.22~0.52 10(대사나 지문 줄 전체에 띄어쓰기 없음). 그 사이 0.08~0.22 는 비어 있다.
RESPACE_LOSS = 0.15         # 파일의 공백 손실이 이보다 크면 그 파일의 띄어쓰기를 원본에 맞춘다
RESPACE_MIN_BOUNDS = 200    # 견줄 낱말 경계가 이보다 적으면(원본 글을 거의 못 읽음) 판단하지 않는다
_HEAD_TAB_MAX = 20          # 문단 첫 글 덩어리가 이 글자 수(공백 뺌) 이하일 때만 그 뒤의 탭을 이름 뒤 탭 후보로 본다
_HEAD_NAME_MAX = 12         # 그 덩어리에서 꼬리 괄호·쌍점을 뗀 이름 부분의 글자 수(공백 뺌) 한도
_NAME_COL_TOL = 0.003       # 이름 칸: 확인된 이름 뒤 탭 줄이 _NAME_COL_MIN 개 이상 놓인 x 에서 이 안쪽
_NAME_COL_MIN = 3
_ANCHOR = 8                 # 닻으로 쓰는 글자 조각 길이
_GAP_MAX = 2000             # 닻 사이 남은 틈이 이보다 길면 맞추지 않는다(서로 다른 글)
_DIFF_MIN = 6               # 닻 사이 틈을 맞출 때, 두 쪽이 거의 같은 글이 아니면 이 길이 이상 덩어리만 받는다
_DIFF_SAME = 0.5            # 틈의 두 쪽이 거의 같은 글이라고 보는 닮음 비율(difflib ratio)
_DIFF_EDGE = 3              # 한 글의 틈 머리와 다른 글의 틈 끝에 붙은 덩어리는 이 길이부터 받는다

_RX_HEAD_BULLET = re.compile(r"^[*＊•·●○◆◇■□▶\-]\s*")
_RX_HEAD_TAIL = re.compile(r"\s*[(（\[][^()（）\[\]]{0,20}[)）\]]$")
_RX_HEAD_PAREN = re.compile(r"[(（]\s*([^()（）]+?)\s*[)）]")
_RX_HEAD_NUM = re.compile(r"(?i)(?:s\s*#|[#＃]|씬\s*[/#]?\s*\d|sc(?:ene)?\.?\s*\d|\d)")
_RX_HEAD_VOICE = re.compile(r"(?<=[가-힣0-9])(?:[A-Za-z.]{1,4}|목소리|소리)$")


def _name_head(chunk: str) -> tuple[int, str]:
    """원본 문단의 첫 덩어리(첫 탭 앞의 글)가 이름 칸의 글처럼 생겼나. (0 아님 | 1 그렇다 | 2 낱말이 둘 이상이라 약함,
    되풀이를 셀 이름 열쇠).

    이름 칸의 글 = 이름('오형사', '모텔남1'), 괄호에 든 이름('(영민)'), 꼬리표·쌍점이 붙은 이름('견우(나레이션)', '제임스 :'),
    글자를 띄운 이름('진 수', '형 사 2'), 여러 이름('철용, 미수'). 번호·씬 표지('S# 114', '#49.', '씬 /57', '12-13.')와
    문장부호가 든 글('전화를 끊고는 … 봉두.')은 아니다. 낱말이 둘인 덩어리('진태 모', '기자 2')는 낱말 한가운데 잘못 친 탭
    ('전 남자분⇥인 줄 알았는데', '소녀를 다⇥시 바라본다')과 모양이 같아 약하다고 본다(respace 가 PDF 줄의 틈을 함께 본다).
    이름 열쇠는 공백과 끝의 번호·소리 꼬리를 뗀 이름('감식반2'·'(감식반)'·'감식반 (E)' -> '감식반', '사무장E' -> '사무장')."""
    s = _RX_HEAD_BULLET.sub("", chunk.strip())
    s = s.rstrip(" :：")
    m = _RX_HEAD_PAREN.fullmatch(s)
    if m:
        s = m.group(1)
    for _ in range(3):
        t = _RX_HEAD_TAIL.sub("", s).rstrip(" :：")
        if t == s:
            break
        s = t
    core = "".join(s.split())
    if not core or len(core) > _HEAD_NAME_MAX or not any(c.isalpha() for c in core) or core[-1] in "-–—~,":
        return 0, ""
    if re.search(r"[.?!…\"“”‘’()（）\[\]]", s) or _RX_HEAD_NUM.match(s):
        return 0, ""
    key = _RX_HEAD_VOICE.sub("", core.rstrip("0123456789")) or core
    words = s.split()
    if len(words) == 1 or all(len(w) == 1 or w.isdigit() for w in words):
        return 1, key
    parts = [p for p in re.split(r"\s*[,/]\s*", s) if p]
    if len(parts) >= 2 and all(" " not in p and len(p) <= 5 for p in parts):
        return 2, key
    return (2, key) if len(words) == 2 and "," not in s and "/" not in s else (0, "")


def _stream(text: str) -> tuple[str, list[int]]:
    """(공백을 뺀 글자열, 글자마다 원래 위치)."""
    chars: list[str] = []
    pos: list[int] = []
    for p, c in enumerate(text):
        if not c.isspace():
            chars.append(c)
            pos.append(p)
    return "".join(chars), pos


def _source_stream(text: str) -> tuple[str, list[int], dict[int, tuple[int, bool, bool]]]:
    """원본 글 -> (공백을 뺀 글자열, 글자마다 바로 앞 경계, 이름 뒤 탭 후보).

    경계: 0 붙음, 1 공백 하나, 2 이름 뒤 탭 후보(문단 첫 덩어리가 이름 꼴이고 바로 뒤가 그 문단의 첫 탭), 3 줄바꿈(문단
    경계), 4 넓은 틈(공백 여럿, 이름 뒤가 아닌 탭). 후보: 탭 뒤 글자 위치 -> (문단 첫 글자 위치, 약한 덩어리인가, 그 이름이
    이 글에서 이름 뒤 탭 후보로 두 번 이상 나오는가)."""
    chars: list[str] = []
    sep: list[int] = []
    pos: list[int] = []
    nl = tab = False
    run = 0
    for p, c in enumerate(text):
        if c == "\n":
            nl = True
        elif c == "\t":
            tab = True
        elif c.isspace():
            run += 1
        else:
            chars.append(c)
            pos.append(p)
            sep.append(3 if nl else 2 if tab else 4 if run >= 2 else 1 if run else 0)
            nl = tab = False
            run = 0
    found: list[tuple[int, int, bool, str]] = []
    para0, seen = 0, False
    for i, s in enumerate(sep):
        if s == 3:
            para0, seen = i, False
        elif s == 2:
            kind, key = (0, "")
            if not seen and i - para0 <= _HEAD_TAB_MAX:
                kind, key = _name_head(text[pos[para0]:pos[i]])
            seen = True
            if kind:
                found.append((i, para0, kind == 2, key))
            else:
                sep[i] = 4
    freq: dict[str, int] = {}
    for _, _, _, key in found:
        freq[key] = freq.get(key, 0) + 1
    return "".join(chars), sep, {i: (p0, weak, freq[key] >= 2) for i, p0, weak, key in found}


def _align(a: str, b: str) -> list[int]:
    """b 의 글자마다 맞는 a 의 위치(없으면 -1). 두 글 모두에서 차례를 지킨다.

    두 글에 한 번씩만 나오는 _ANCHOR 글자 조각을 닻으로 삼아 차례가 맞는 가장 긴 열을 고르고, 닻 사이는 앞뒤에서
    같은 글자를 따라 넓힌 뒤 남은 짧은 틈만 difflib 로 맞춘다. 틈의 두 쪽이 거의 같은 글(닮음 _DIFF_SAME 이상)이 아니면
    _DIFF_MIN 글자 이상 맞은 덩어리와, 한 글에서는 틈 머리에 다른 글에서는 틈 끝에 붙은 덩어리(한쪽에만 있는 글을 건너
    이어지는 같은 글: 쪽 머리말 '- 13 -' 뒤의 '창밖을 바라', 3글자 이상)만 받는다. 한쪽에만 있는 서로 다른 글끼리 우연히
    맞은 두세 글자에서 원본의 공백을 가져오지 않게 하려는 것이다. 걸리는 시간은 글자 수에 거의 비례한다."""
    import bisect
    import difflib

    n, m, K = len(a), len(b), _ANCHOR
    match = [-1] * m

    def fill(a0: int, a1: int, b0: int, b1: int) -> None:
        while a0 < a1 and b0 < b1 and a[a0] == b[b0]:
            match[b0] = a0
            a0 += 1
            b0 += 1
        while a1 > a0 and b1 > b0 and a[a1 - 1] == b[b1 - 1]:
            a1 -= 1
            b1 -= 1
            match[b1] = a1
        if a0 == a1 or b0 == b1 or a1 - a0 > _GAP_MAX or b1 - b0 > _GAP_MAX:
            return
        sm = difflib.SequenceMatcher(None, a[a0:a1], b[b0:b1], autojunk=False)
        blocks = sm.get_matching_blocks()
        la, lb = a1 - a0, b1 - b0
        same = 2 * sum(size for _, _, size in blocks) >= _DIFF_SAME * (la + lb)
        for i, j, size in blocks:
            edge = size >= _DIFF_EDGE and ((i == 0 and j + size == lb) or (j == 0 and i + size == la))
            if same or size >= _DIFF_MIN or edge:
                for t in range(size):
                    match[b0 + j + t] = a0 + i + t

    chain: list[tuple[int, int]] = []
    if n >= K and m >= K:
        ga: dict[str, int] = {}
        for i in range(n - K + 1):
            g = a[i:i + K]
            ga[g] = -1 if g in ga else i
        gb: dict[str, int] = {}
        for j in range(m - K + 1):
            g = b[j:j + K]
            if g in ga:
                gb[g] = -1 if g in gb else j
        pairs = sorted((j, ga[g]) for g, j in gb.items() if j >= 0 and ga[g] >= 0)
        tails: list[int] = []           # 길이별 가장 작은 끝 a 위치
        tail_at: list[int] = []         # 그 끝의 pairs 차례
        prev = [-1] * len(pairs)
        for k, (_, i) in enumerate(pairs):
            p = bisect.bisect_left(tails, i)
            if p == len(tails):
                tails.append(i)
                tail_at.append(k)
            else:
                tails[p] = i
                tail_at[p] = k
            prev[k] = tail_at[p - 1] if p else -1
        k = tail_at[-1] if tail_at else -1
        while k >= 0:
            chain.append(pairs[k])
            k = prev[k]
        chain.reverse()
    ea = eb = 0
    for j, i in chain:
        sh = max(0, ea - i, eb - j)
        if sh >= K:
            continue
        i0, j0 = i + sh, j + sh
        fill(ea, i0, eb, j0)
        for t in range(K - sh):
            match[j0 + t] = i0 + t
        ea, eb = i + K, j + K
    fill(ea, n, eb, m)
    return match


def respace(texts: list[str], source: str, xs: list[float | None] | None = None) -> tuple[list[str], dict]:
    """PDF 에서 만든 줄 글들의 공백·탭을 원본 글(source)에 맞춘 줄 글들과 통계. 글자와 줄 수·차례는 그대로다.
    xs 는 줄마다 x(이름 칸을 찾는 데 쓴다. 없으면 이름 칸을 보지 않는다).

    줄 안에서 이웃한 두 글자가 원본에서도 이웃한 글자로 맞았을 때만 그 사이를 본다. 그 밖(원본과 맞지 않는 글자: PDF 에만
    있는 자동 번호·쪽 번호, 원본의 문단 경계에 걸친 자리)은 줄 글 그대로다.
      - 이름 뒤 탭: 원본 문단이 '이름⇥대사' 꼴(_name_head)이고 PDF 줄이 그 문단의 첫 글자로 시작하면 탭으로(없던 것은
        넣고 공백은 올림). 그 이름이 원본에서 이름 뒤 탭 꼴로 두 번 이상 나오거나, 줄이 이름 칸(그런 확인된 줄이 여럿 놓인
        x)에 놓였을 때만: 한 번뿐인 '말들을⇥보며 맥주 한 캔을…', '스스로⇥강해질 필요가…' 는 문장 첫 낱말 뒤에 친 탭이다.
        이름 부분이 두 낱말이라 약한 덩어리는 PDF 줄에 틈이 있거나 뒤 글자가 글자가 아닐 때만('진태 모⇥(손잡고)'): 틈 없이
        글자가 이어지면 낱말 한가운데 친 탭이다('전 남자분⇥인 줄'). 자동 번호가 앞에 붙은 줄('＃41. 출판사⇥/ 낮')과 한 줄에
        두 문단이 놓인 줄은 줄 머리가 아니라 받지 않는다. 받지 않은 탭과 문단 중간의 탭(줄바꿈 자리를 맞추려고 친 탭:
        '…시장이냐,⇥슈퍼냐?', 클래식 fm54441), 이름 꼴이 아닌 첫 덩어리 뒤의 탭('S# 114⇥하숙방')은 넓은 공백으로 본다.
      - 원본이 공백(또는 넓은 틈)인데 줄에 아무것도 없으면 공백 하나를 넣는다.
      - 원본에 없는 줄 글의 공백('야! ! 우리가')은 지우고, 원본이 공백 하나인데 줄 글이 탭인 자리(넓게 찍혀 벌어진
        '호창,⇥두려워진다.')는 공백으로 내린다. 원본이 넓은 틈이거나 맞지 않은 자리의 탭은 그대로다.
    통계: bounds(견준 낱말 경계), missing(그중 줄에 공백·탭이 없는 것), loss = missing/bounds, tabs(받은 이름 뒤 탭 경계),
    tab_down(그중 줄에서는 공백이던 것), tab_missing(줄에 아무것도 없던 것), matched(줄 글자 중 원본과 맞은 비율),
    한 일: lines(글이 바뀐 줄), spaces(넣은 공백), raised(탭으로 올린 자리), removed(지운 공백), lowered(공백으로 내린 탭)."""
    import bisect

    # 앞에 줄바꿈을 붙인다: 글 맨 앞의 '⇥이름⇥대사' 문단도 다른 문단처럼 앞머리 탭이 문단 경계에 묻히게
    a, sep_a, head = _source_stream("\n" + util.nfc(_CTRL.sub("", source.translate(GLYPH_MAP))))
    bch: list[str] = []
    spans: list[tuple[int, int]] = []           # 줄마다 b 안 [시작, 끝)
    bpos: list[int] = []
    for t in texts:
        s, pos = _stream(t)
        spans.append((len(bch), len(bch) + len(s)))
        bch.extend(s)
        bpos.extend(pos)
    b = "".join(bch)
    match = _align(a, b)

    def near(j: int) -> bool:
        return match[j - 1] >= 0 and match[j] == match[j - 1] + 1

    # 이름 칸: 원본에서 두 번 이상 나오는 이름의 이름 뒤 탭 줄이 놓인 x
    sure_x: list[float] = []
    if xs is not None:
        for k, (j0, j1) in enumerate(spans):
            if xs[k] is None or j1 - j0 < 2 or match[j0] < 0:
                continue
            for j in range(j0 + 1, j1):
                h = head.get(match[j]) if near(j) else None
                if h:
                    t = texts[k]
                    mid = h[1] and bpos[j - 1] + 1 == bpos[j] and t[bpos[j]].isalpha()      # 낱말 한가운데 친 탭
                    if h[2] and h[0] == match[j0] and not mid:
                        sure_x.append(xs[k])
                    break
        sure_x.sort()

    def name_col(k: int) -> bool:
        if xs is None or xs[k] is None:
            return False
        lo = bisect.bisect_left(sure_x, xs[k] - _NAME_COL_TOL)
        return bisect.bisect_right(sure_x, xs[k] + _NAME_COL_TOL) - lo >= _NAME_COL_MIN

    st = {"bounds": 0, "missing": 0, "tabs": 0, "tab_down": 0, "tab_missing": 0,
          "lines": 0, "spaces": 0, "raised": 0, "removed": 0, "lowered": 0}
    out: list[str] = []
    for k, (t, (j0, j1)) in enumerate(zip(texts, spans)):
        if j1 - j0 < 2:
            out.append(t)
            continue
        parts = [t[: bpos[j0] + 1]]
        for j in range(j0 + 1, j1):
            gap = t[bpos[j - 1] + 1: bpos[j]]
            want = sep_a[match[j]] if near(j) else -1
            if want == 2:
                p0, weak, twice = head[match[j]]
                if match[j0] != p0 or (weak and not gap and t[bpos[j]].isalpha()) or not (twice or name_col(k)):
                    want = 4                     # 줄 머리의 이름 뒤 탭이 아니다: 넓은 공백으로 본다
            if want in (1, 2, 4):
                st["bounds"] += 1
                st["missing"] += not gap
            if want == 2:
                st["tabs"] += 1
                if "\t" not in gap:
                    st["tab_down" if gap else "tab_missing"] += 1
                    st["raised"] += 1
                    gap = "\t"
            elif want in (1, 4) and not gap:
                st["spaces"] += 1
                gap = " "
            elif want == 0 and gap and "\t" not in gap:
                st["removed"] += 1
                gap = ""
            elif want == 1 and "\t" in gap:
                st["lowered"] += 1
                gap = " "
            parts.append(gap)
            parts.append(t[bpos[j]])
        parts.append(t[bpos[j1 - 1] + 1:])
        new = "".join(parts)
        st["lines"] += new != t
        out.append(new)
    st["loss"] = round(st["missing"] / st["bounds"], 4) if st["bounds"] else 0.0
    st["matched"] = round(sum(1 for x in match if x >= 0) / len(match), 4) if match else 0.0
    return out, st


def source_text(path: Path, fmt: str) -> str:
    """원본 문서의 글(공백·탭 그대로). convert 의 대체 경로 읽기를 다시 쓴다. 못 읽으면 그 예외를 그대로 낸다."""
    from . import convert

    return convert.fallback_text(path, fmt)[0]


def _origin(kind: str, relpath: str, fmt: str) -> Path | None:
    """공백·탭 되살리기에 쓸 원본 문서 경로. 한컴 변환 PDF 에서 줄을 만드는 파일(hwp/doc/docx/rtf)만."""
    if kind != "pdf" or fmt not in WORDPROC:
        return None
    return Path(relpath) if Path(relpath).is_absolute() else paths.absp(relpath)


def restore_spacing(rows: list[dict], source: str | None) -> dict | None:
    """한컴 변환 PDF 에서 만든 한 파일의 줄(rows)에서, 공백 손실이 RESPACE_LOSS 를 넘으면 줄 글의 공백·탭을 원본 글에
    맞춘다(respace. rows 의 text 만 바꿈). 머리 행에 남길 기록을 돌려준다: 맞췄으면 {from: "source", loss, matched,
    lines(글이 바뀐 줄), spaces(넣은 공백), tabs(탭으로 올린 이름 뒤 탭), removed(지운 공백), lowered(공백으로 내린 탭)},
    견주지 못했으면 {from: null, skipped: 까닭}. 견줬는데 문턱 아래면 None(줄 글 그대로)."""
    if not rows:
        return None
    if not source or not source.strip():
        return {"from": None, "skipped": "source_empty"}
    texts, st = respace([r["text"] for r in rows], source, xs=[r.get("x") for r in rows])
    if st["bounds"] < RESPACE_MIN_BOUNDS:
        return {"from": None, "skipped": f"few_bounds:{st['bounds']}"}
    if st["loss"] <= RESPACE_LOSS:
        return None
    for r, t in zip(rows, texts):
        r["text"] = t
    return {"from": "source", "loss": st["loss"], "matched": st["matched"], "lines": st["lines"],
            "spaces": st["spaces"], "tabs": st["raised"], "removed": st["removed"], "lowered": st["lowered"]}


def _respace_file(rows: list[dict], origin: Path, fmt: str) -> tuple[dict | None, bool]:
    """한 파일의 줄에 원본 글의 공백·탭을 되살린다. (머리 행에 남길 기록 또는 None, 다음 실행이 다시 해야 하나).

    원본 글을 못 읽었거나 되살리기가 예외로 끝나면 줄은 PDF 에서 만든 그대로 두고 까닭을 기록에 남긴다
    ({from: null, skipped: …}): 기록이 없는 파일은 '견줬고 공백 손실이 문턱 아래'다. 읽기 실패가 한때의 것일 수 있으면
    (시간 초과, 파일 잠김 같은 OSError, 메모리) 둘째 값이 참이다: 그런 문서는 지문이 맞지 않아 실행마다 다시 만든다."""
    import subprocess

    def why(e: Exception) -> str:           # 예외 글의 첫 토막만(뒤에는 파일 경로가 붙는다: 'antiword_rc1:C:\…')
        return util.nfc(" ".join(f"{type(e).__name__}: {str(e).split(':', 1)[0]}".split()))[:80]

    if not origin.exists():
        return {"from": None, "skipped": "source_missing"}, False
    try:
        source = source_text(origin, fmt)
    except (OSError, subprocess.TimeoutExpired, MemoryError) as e:
        return {"from": None, "skipped": "source_unreadable", "error": why(e)}, True
    except Exception as e:  # noqa: BLE001 - 원본 글을 못 읽으면 되살리지 않을 뿐(읽는 코드가 못 읽는 파일: 다시 해도 같다)
        return {"from": None, "skipped": "source_unreadable", "error": why(e)}, False
    try:
        return restore_spacing(rows, source), False
    except Exception as e:  # noqa: BLE001 - 되살리기가 실패하면 PDF 에서 만든 줄 그대로 쓴다(rows 는 끝난 뒤에만 바뀐다)
        return {"from": None, "skipped": "error", "error": why(e)}, False


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
    # 한컴 변환 PDF 에서 줄을 만드는 파일은 원본 문서도 읽는다(공백·탭 되살리기): 그 원본의 sha1 도 지문에 넣는다
    origin = [_origin(kind, f, fmt) for (kind, _, _), f, fmt in zip(plan, files, fmts)]
    fp = {"version": VERSION,
          "inputs": [[kind, util.sha1_file(p) if p else None] for kind, p, _ in plan],
          "sources": [util.sha1_file(o) if o is not None and o.exists() else None for o in origin]}
    out = paths.DOCS / doc_id / "lines.jsonl"
    if not force and out.exists() and util.stage_fp(STAGE, doc_id) == fp:
        return "skip", -1
    rows: list[dict] = []
    problems = []
    spacing: list[dict] = []
    retry = False               # 원본 글 읽기가 한때의 실패(시간 초과·파일 잠김)로 끝났으면 다음 실행이 다시 하게 한다
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
        if got and origin[k] is not None:
            info, again = _respace_file(got, origin[k], fmts[k])
            retry = retry or again
            if info:
                spacing.append({"file": k, **info})
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
    if spacing:
        # 원본 글의 공백·탭을 되살린 파일과 원본 글과 견주지 못한 파일(견줘서 바꿀 것이 없던 파일은 적지 않는다)
        header["made"]["spacing"] = spacing
    util.write_jsonl(out, [header] + rows)
    util.set_problems(STAGE, doc_id, problems)
    # 한때의 실패로 원본 글을 못 읽은 결과에는 맞지 않는 지문을 남겨 다음 실행이 다시 만들게 한다(--force 뒤에도)
    util.set_stage_fp(STAGE, doc_id, dict(fp, retry="source_read") if retry else fp)
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
