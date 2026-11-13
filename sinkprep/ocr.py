"""스캔·깨진 글자층 PDF OCR -> cache/ocr/{doc_id}/{k}.json

엔진: RapidOCR 3.10(onnxruntime CPU)로 PaddleOCR 모델을 돌린다.
  한국어: PP-OCRv5 mobile 검출 + korean_PP-OCRv5 mobile 인식
  영어·로마자: PP-OCRv6 small 검출·인식
고른 이유와 비교표는 notes/ocr_engine.md. 쪽 그림은 300dpi 회색(PyMuPDF, 실패하면 pdftoppm).

결과 형식(DESIGN.md "ocr 결과"):
  {"schema":"sink.ocr/1","engine":…,"engine_version":…,"source":…,"dpi":300,
   "pages":[{"page":1,"width":…,"height":…,
             "lines":[{"text":…,"box":[x0,y0,x1,y1],"conf":0.97}]}]}
줄은 물리적인 줄 하나다. 한 줄 안에서 크게 떨어진 덩어리(화자 칸 | 대사 칸)는 글자 사이를
간격만큼의 공백으로 잇고, 덩어리별 상자를 `segs`에 따로 남긴다(들여쓰기 위치 보존).
"""
from __future__ import annotations

import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator

from . import paths, util

STAGE = "ocr"
VERSION = "0.1.0"
SCHEMA = "sink.ocr/1"
DPI = 300
LOW_CONF = 0.85            # 문서 평균 conf 가 이보다 낮으면 ocr_low
SEG_GAP = 1.5              # 줄 안 덩어리 나누기: 간격 > SEG_GAP × 줄 높이
SAME_LINE = 0.5            # 같은 줄: 세로 겹침 >= SAME_LINE × 낮은 쪽 높이
CONVERTED = {"hwp3", "hwp5", "hwpx", "doc", "docx", "rtf"}

# pdftoppm: 먼저 PATH, 그다음 알려진 설치 위치(Git mingw64 에는 pdftotext 만 있음, 2026-11-13 확인).
_PDFTOPPM_CANDIDATES = [
    Path(r"C:\Program Files\Git\mingw64\bin\pdftoppm.exe"),
    Path(os.path.expandvars(
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe"
        r"\poppler-25.07.0\Library\bin\pdftoppm.exe")),
]


# ---------------------------------------------------------------- 줄 묶기(엔진과 무관한 순수 함수)

def _h(b) -> float:
    return b[3] - b[1]


def _join_texts(items: list[dict], cw: float) -> str:
    """x 순서 상자들을 간격에 맞춰 공백으로 잇는다. cw = 글자 하나 폭 어림."""
    out = ""
    prev = None
    for it in items:
        t = it["text"].strip()
        if not t:
            continue
        if prev is not None:
            gap = it["box"][0] - prev[2]
            if gap > SEG_GAP * it["_lh"]:          # 칸이 바뀜: 간격만큼 공백(최소 2칸)
                out += " " * max(2, round(gap / cw)) if cw > 0 else "  "
            elif gap > 0.15 * it["_lh"]:           # 낱말 사이
                out += " "
            # 그보다 좁으면 한 낱말이 두 상자로 쪼개진 것: 붙인다
        out += t
        prev = it["box"]
    return out


def _conf(items: list[dict]) -> float | None:
    vals = [(it["conf"], max(1, len(it["text"].strip()))) for it in items if it.get("conf") is not None]
    if not vals:
        return None
    return round(sum(c * n for c, n in vals) / sum(n for _, n in vals), 4)


def _union(items: list[dict]) -> list[int]:
    return [round(min(i["box"][0] for i in items)), round(min(i["box"][1] for i in items)),
            round(max(i["box"][2] for i in items)), round(max(i["box"][3] for i in items))]


def group_lines(prims: Iterable[dict]) -> list[dict]:
    """엔진이 준 상자(낱말 또는 글줄 조각) -> 물리적인 줄 목록.

    prims: [{"text", "box":[x0,y0,x1,y1], "conf"(0~1|None)}]
    반환: [{"text", "box", "conf", ("segs": [{"text","box","conf"}] 덩어리가 둘 이상일 때)}], 위에서 아래 순서.
    """
    items = []
    for p in prims:
        t = util.nfc(str(p.get("text") or "")).strip()
        b = [float(v) for v in p["box"]]
        if not t or b[2] <= b[0] or b[3] <= b[1]:
            continue
        items.append({"text": t, "box": b, "conf": p.get("conf")})
    items.sort(key=lambda i: ((i["box"][1] + i["box"][3]) / 2, i["box"][0]))

    lines: list[dict] = []          # {"items": [...], "y0","y1","h"}
    for it in items:
        b = it["box"]
        best, best_ov = None, 0.0
        for ln in lines[-6:]:        # 가까운 줄만 본다(위에서 아래로 정렬돼 있음)
            ov = min(b[3], ln["y1"]) - max(b[1], ln["y0"])
            if ov >= SAME_LINE * min(_h(b), ln["h"]) and ov > best_ov:
                best, best_ov = ln, ov
        if best is None:
            lines.append({"items": [it], "y0": b[1], "y1": b[3], "h": _h(b)})
        else:
            best["items"].append(it)
            best["y0"] = min(best["y0"], b[1])
            best["y1"] = max(best["y1"], b[3])
            best["h"] = statistics.median(_h(i["box"]) for i in best["items"])

    out = []
    for ln in lines:
        its = sorted(ln["items"], key=lambda i: i["box"][0])
        lh = ln["h"]
        nchar = sum(len(i["text"]) for i in its)
        cw = sum(i["box"][2] - i["box"][0] for i in its) / max(1, nchar)
        for i in its:
            i["_lh"] = lh
        segs: list[list[dict]] = [[its[0]]]
        for prev, cur in zip(its, its[1:]):
            if cur["box"][0] - prev["box"][2] > SEG_GAP * lh:
                segs.append([cur])
            else:
                segs[-1].append(cur)
        row = {"text": _join_texts(its, cw), "box": _union(its), "conf": _conf(its)}
        if len(segs) > 1:
            row["segs"] = [{"text": _join_texts(s, cw), "box": _union(s), "conf": _conf(s)} for s in segs]
        out.append(row)
    out.sort(key=lambda r: (r["box"][1], r["box"][0]))
    return out


# ---------------------------------------------------------------- 쪽 그림 만들기

def find_pdftoppm() -> Path | None:
    w = shutil.which("pdftoppm")
    if w:
        return Path(w)
    for c in _PDFTOPPM_CANDIDATES:
        if c.exists():
            return c
    return None


def _np_gray(pix):
    import numpy as np
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return a[:, :, 0].copy()          # (h, w) 2차원


def _blank(img) -> bool:
    """거의 한 색(흰 쪽, 또는 JBIG2 해독 실패로 까맣게 나온 쪽)."""
    import numpy as np
    small = img[::8, ::8]
    return float(np.std(small)) < 2.0


def _bad(img) -> bool:
    """그림이 망가져 보임: 한 색이거나 절반 넘게 까맣다(JBIG2 해독이 반쯤 실패한 쪽)."""
    return _blank(img) or float((img[::8, ::8] < 128).mean()) > 0.5


def _pdftoppm_page(pdf: Path, page: int, dpi: int):
    exe = find_pdftoppm()
    if exe is None:
        return None, "pdftoppm 없음"
    import numpy as np
    import cv2
    with tempfile.TemporaryDirectory(prefix="sinkocr_") as td:
        stem = Path(td) / "p"
        r = subprocess.run([str(exe), "-r", str(dpi), "-gray", "-png", "-f", str(page), "-l", str(page),
                            str(pdf), str(stem)], capture_output=True, timeout=300)
        outs = sorted(Path(td).glob("p*.png"))
        if not outs:
            return None, (r.stderr.decode("utf-8", "replace").strip()[:200] or f"pdftoppm rc={r.returncode}")
        img = cv2.imdecode(np.fromfile(str(outs[0]), np.uint8), cv2.IMREAD_GRAYSCALE)
        warn = r.stderr.decode("utf-8", "replace").strip()[:200]
        return img, warn


def _pdfinfo_pages(pdf: Path) -> int:
    exe = find_pdftoppm()
    if exe is None:
        return 0
    info = exe.with_name("pdfinfo.exe" if exe.suffix else "pdfinfo")
    if not info.exists():
        return 0
    r = subprocess.run([str(info), str(pdf)], capture_output=True, timeout=120)
    m = re.search(rb"Pages:\s+(\d+)", r.stdout)
    return int(m.group(1)) if m else 0


def page_count(pdf: Path) -> int:
    import pymupdf
    try:
        with pymupdf.open(pdf) as d:
            return d.page_count
    except Exception:
        return _pdfinfo_pages(pdf)


def render_pages(pdf: Path, dpi: int = DPI, pages: list[int] | None = None) -> Iterator[tuple[int, object, dict]]:
    """(쪽 번호 1부터, 회색 그림 ndarray|None, 정보{render, warn?}) 를 차례로.

    PyMuPDF 로 그리고, 예외가 나거나 경고(JBIG2 등)와 함께 망가진 그림(한 색·절반 넘게 검정)이 나오면
    pdftoppm 으로 다시 그린다.
    둘 다 안 되면 그림 None(render="failed").
    """
    import pymupdf
    pymupdf.TOOLS.mupdf_display_errors(False)     # JBIG2 오류는 경고 목록으로만 받는다
    pdf = Path(pdf)
    try:
        doc = pymupdf.open(pdf)
    except Exception as e:  # 열리지 않음 -> pdftoppm 만
        doc = None
        open_err = repr(e)[:200]
    n = doc.page_count if doc is not None else _pdfinfo_pages(pdf)
    want = pages or list(range(1, n + 1))
    try:
        for pno in want:
            if pno < 1 or pno > n:
                continue
            info: dict = {}
            img = None
            if doc is not None:
                try:   # 그림을 못 만들어도 쪽 크기(픽셀)는 남긴다
                    r = doc[pno - 1].rect
                    info["_size"] = (round(r.width * dpi / 72), round(r.height * dpi / 72))
                except Exception:
                    pass
                pymupdf.TOOLS.mupdf_warnings()          # 앞 경고 비우기
                try:
                    pix = doc[pno - 1].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
                    img = _np_gray(pix)
                    w = pymupdf.TOOLS.mupdf_warnings()
                    if w:
                        info["warn"] = w.strip()[:200]
                except Exception as e:
                    info["warn"] = repr(e)[:200]
                    img = None
                if img is not None and not (_bad(img) and "warn" in info):
                    info["render"] = "pymupdf"
                    yield pno, img, info
                    continue
            else:
                info["warn"] = open_err
            img2, warn2 = _pdftoppm_page(pdf, pno, dpi)
            if img2 is not None and not _bad(img2):
                info["render"] = "pdftoppm"
                if warn2:
                    info["warn2"] = warn2
                yield pno, img2, info
            elif img is not None and "warn" not in info:
                info["render"] = "pymupdf"            # 정말 빈 쪽
                yield pno, img, info
            else:
                info["render"] = "failed"
                if warn2:
                    info["warn2"] = warn2
                yield pno, None, info
    finally:
        if doc is not None:
            doc.close()


# ---------------------------------------------------------------- 엔진

# 언어 열쇠 -> (검출 모델, 인식 모델). notes/ocr_engine.md 의 비교로 고름.
#  ko: PP-OCRv5 mobile 검출 + korean_PP-OCRv5 mobile 인식(한글 + 로마자)
#  en: PP-OCRv6 small 검출·인식(영어와 로마자 쓰는 유럽어를 한 모델로)
_MODELS = {
    "ko": ("PP-OCRv5_det_mobile", "korean_PP-OCRv5_rec_mobile"),
    "en": ("PP-OCRv6_det_small", "PP-OCRv6_rec_small(en)"),
}


def rec_key(lang_hint: str | None) -> str:
    l = (lang_hint or "").lower()
    if l.startswith("ko"):
        return "ko"
    return "en"     # en, fr, es, it, de … (PP-OCRv6 small 은 로마자 다국어)


class Engine:
    """RapidOCR(onnxruntime CPU) 감싸기. 언어 열쇠마다 엔진 하나."""

    name = "rapidocr"

    def __init__(self, threads: int = -1):
        self.threads = threads
        self._eng: dict[str, object] = {}

    def version(self, key: str) -> str:
        from importlib.metadata import version
        det, rec = _MODELS[key]
        return f"rapidocr {version('rapidocr')}; onnxruntime {version('onnxruntime')}; det {det}; rec {rec}"

    def _get(self, key: str):
        if key not in self._eng:
            from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR
            p = {
                "Global.log_level": "error",
                "Global.use_cls": False,
                "Global.max_side_len": 2000,
                "EngineConfig.onnxruntime.intra_op_num_threads": self.threads,
            }
            if key == "ko":
                p.update({"Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE,
                          "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE,
                          "Rec.lang_type": LangRec.KOREAN})
            else:
                p.update({"Det.ocr_version": OCRVersion.PPOCRV6, "Det.model_type": ModelType.SMALL,
                          "Det.lang_type": LangDet.EN,
                          "Rec.ocr_version": OCRVersion.PPOCRV6, "Rec.model_type": ModelType.SMALL,
                          "Rec.lang_type": LangRec.EN})
            self._eng[key] = RapidOCR(params=p)
        return self._eng[key]

    def prims(self, gray, key: str) -> list[dict]:
        import numpy as np
        import cv2
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        r = self._get(key)(img)
        out = []
        if r.boxes is None:
            return out
        for box, txt, sc in zip(r.boxes, r.txts, r.scores):
            b = np.asarray(box, dtype=float)
            out.append({"text": txt, "box": [float(b[:, 0].min()), float(b[:, 1].min()),
                                             float(b[:, 0].max()), float(b[:, 1].max())],
                        "conf": float(sc)})
        return out


_ENGINE: Engine | None = None


def get_engine(threads: int = -1) -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine(threads)
    return _ENGINE


_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")


def guess_lang(lines: list[dict]) -> str:
    """한국어 모델로 읽은 줄에서 한글/로마자 비율로 언어를 고른다."""
    txt = "".join(l["text"] for l in lines)
    h, a = len(_HANGUL.findall(txt)), len(_LATIN.findall(txt))
    if h + a < 20:
        return "unknown"
    return "ko" if h >= 0.15 * (h + a) else "en"


def ocr_page(gray, key: str, engine: Engine | None = None) -> list[dict]:
    eng = engine or get_engine()
    return group_lines(eng.prims(gray, key))


# ---------------------------------------------------------------- 쪽 방향·두 쪽 펼침

OK_CONF = 0.80        # 쪽 평균 conf 가 이보다 낮으면 방향을 의심한다
ROT_TRIES = 2         # 한 문서에서 돌려 봐도 나아지지 않은 쪽이 이만큼 쌓이면 더 돌려 보지 않는다


def _rotate(img, rot: int):
    import numpy as np
    if rot == 0:
        return img
    return np.ascontiguousarray(np.rot90(img, k={90: 3, 180: 2, 270: 1}[rot]))   # 시계 방향 rot 도


def split_two_up(img) -> list[tuple[int, int]]:
    """가로로 긴 쪽에 두 쪽이 나란히 찍힌 경우(펼침 스캔) 가운데 빈 세로 띠에서 나눈다.

    반환: [(x0, x1), …] 열 범위. 나누지 않으면 [(0, 너비)].
    """
    import numpy as np
    h, w = img.shape[:2]
    if w < 1.15 * h:
        return [(0, w)]
    small = img[::4, ::4] < 128
    col = small.mean(axis=0).astype(float)
    k = max(3, len(col) // 100)
    sm = np.convolve(col, np.ones(k) / k, mode="same")
    lo, hi = int(0.35 * len(sm)), int(0.65 * len(sm))
    band = sm[lo:hi]
    i = int(band.argmin())
    textcols = col[col > 0.01]
    if band[i] > 0.01 or not len(textcols) or band[i] > 0.25 * float(np.median(textcols)):
        return [(0, w)]
    x = (lo + i) * 4
    return [(0, x), (x, w)]


def _valid(lines: list[dict]) -> list[dict]:
    """글줄다운 상자만(높이 0.5인치 이하). 세로로 선 글을 읽으면 여러 줄을 덮는 큰 상자가 나온다."""
    return [l for l in lines if l["box"][3] - l["box"][1] <= 0.5 * DPI]


def _score(lines: list[dict]) -> float:
    return sum((l["conf"] or 0) * len(l["text"].replace(" ", "")) for l in _valid(lines))


def _ink(img) -> float:
    return float((img[::8, ::8] < 128).mean())


def _ok(lines: list[dict], img) -> bool:
    n = sum(len(l["text"].replace(" ", "")) for l in _valid(lines))
    if n < 30:                       # 글자가 거의 없음: 정말 빈 쪽(잉크도 적음)이면 괜찮다
        return _ink(img) < 0.01
    return _score(lines) / n >= OK_CONF


def read_image(img, key: str, eng: Engine, state: dict) -> list[dict]:
    """쪽 그림 하나 -> 부분 목록 [{"rotate","crop","width","height","lines"}].

    state: 문서 안에서 이어지는 방향 기억 {"rot": 0|90|180|270, "fails": n}.
    1) 기억한 방향으로 돌려 읽고 conf 가 낮으면 다른 방향(0·90·270·180)도 읽어 가장 나은 것을 고른다.
    2) 돌린 그림이 가로로 길고 가운데가 비어 있으면 두 쪽으로 나눠 따로 읽는다.
    """
    def parts_of(rot: int) -> list[dict]:
        im = _rotate(img, rot)
        h = im.shape[0]
        out = []
        for x0, x1 in split_two_up(im):
            sub = im[:, x0:x1]
            out.append({"rotate": rot, "crop": [x0, 0, x1, h], "width": x1 - x0, "height": h,
                        "lines": ocr_page(sub, key, eng)})
        return out

    def flat(parts):
        return [l for p in parts for l in p["lines"]]

    rot = state.get("rot", 0)
    best = parts_of(rot)
    if _ok(flat(best), img) or (state.get("fails", 0) >= ROT_TRIES and rot == 0):
        return best
    # 네 방향을 모두 읽고 글줄다운 상자의 (conf × 글자 수) 합이 가장 큰 방향을 고른다.
    best_s = _score(flat(best))
    found = None
    for r in (0, 90, 270, 180):
        if r == rot:
            continue
        cand = parts_of(r)
        sc = _score(flat(cand))
        if sc > best_s:
            best, best_s, found = cand, sc, r
    if found is not None and _ok(flat(best), img):
        state["rot"] = found
    else:
        state["fails"] = state.get("fails", 0) + 1
    return best


# ---------------------------------------------------------------- 문서 하나

def _src_rel(p: Path) -> str:
    try:
        return paths.rel(p)
    except ValueError:
        return Path(p).as_posix()


def doc_conf(result: dict) -> float | None:
    """글자 수로 가중한 평균 conf(그림이 없는 쪽 제외)."""
    num = den = 0.0
    for pg in result.get("pages", []):
        for ln in pg.get("lines", []):
            if ln.get("conf") is None:
                continue
            n = max(1, len(ln["text"].replace(" ", "")))
            num += ln["conf"] * n
            den += n
    return round(num / den, 4) if den else None


def _page_entries(pno: int, info: dict, parts: list[dict]) -> list[dict]:
    out = []
    for i, p in enumerate(parts, 1):
        e = {"page": pno, "width": p["width"], "height": p["height"]}
        if p["rotate"]:
            e["rotate"] = p["rotate"]
        if len(parts) > 1:
            e.update(part=i, parts=len(parts), crop=p["crop"])
        e.update(info)
        e["lines"] = p["lines"]
        out.append(e)
    return out


def ocr_pdf(pdf_path, out_json, lang_hint: str | None = None, *, pages: list[int] | None = None,
            dpi: int = DPI, engine: Engine | None = None, extra: dict | None = None,
            log=None) -> dict:
    """PDF 한 개를 OCR 해 out_json 에 쓴다(원자적). 결과 dict 를 돌려준다.

    lang_hint: ko | en | fr … | None/unknown(앞쪽을 한국어 모델로 읽어 고른다).
    pages: 1부터 쪽 번호 목록(시험·표본용). None 이면 전체.
    한 PDF 쪽이 두 쪽 펼침이면 같은 `page` 번호로 항목 둘(`part` 1·2, `crop`)을 남긴다.
    돌려서 읽은 쪽은 `rotate`(시계 방향 도)를 남기고, 상자는 돌린 그림(나눴으면 그 부분) 기준이다.
    """
    pdf_path = Path(pdf_path)
    eng = engine or get_engine()
    t0 = time.time()
    key = None if (lang_hint or "unknown").lower() in ("", "unknown", "mixed", "none") else rec_key(lang_hint)
    lang_how = "hint" if key else "guess"
    state: dict = {}
    out_pages: list[dict] = []
    pending: list[tuple] = []   # 언어를 정하기 전에 한국어 모델로 읽어 둔 쪽 (pno, img, info, parts)
    failed = []

    def _flush(k: str) -> None:
        # 한국어로 정해지면 읽어 둔 결과를 그대로 쓰고, 아니면 그 쪽들을 다시 읽는다.
        for ppno, pimg, pinfo, pparts in pending:
            parts = pparts if k == "ko" else read_image(pimg, k, eng, state)
            out_pages.extend(_page_entries(ppno, pinfo, parts))
        pending.clear()

    for pno, img, info in render_pages(pdf_path, dpi, pages):
        size = info.pop("_size", None)
        if img is None:
            failed.append(pno)
            w, h = size or (round(8.27 * dpi), round(11.69 * dpi))   # 크기를 모르면 A4
            out_pages.append({"page": pno, "width": w, "height": h, **info, "lines": []})
            continue
        if key is None:
            # 표지만 보고 정하지 않도록 글자 300개 또는 글 있는 쪽 4개가 모일 때까지 모아서 고른다.
            pending.append((pno, img, info, read_image(img, "ko", eng, state)))
            alltxt = [ln for *_, pp in pending for p in pp for ln in p["lines"]]
            letters = sum(len(_HANGUL.findall(l["text"])) + len(_LATIN.findall(l["text"])) for l in alltxt)
            if letters >= 300 or len(pending) >= 4:
                g = guess_lang(alltxt)
                if g != "unknown":
                    key = "ko" if g == "ko" else "en"
                    _flush(key)
            continue
        parts = read_image(img, key, eng, state)
        out_pages.extend(_page_entries(pno, info, parts))
        if log:
            log(f"  p{pno} {sum(len(p['lines']) for p in parts)}줄")
    if pending:   # 쪽이 적어 끝까지 못 정함: 모은 것으로 고른다(글자가 거의 없으면 한국어)
        alltxt = [ln for *_, pp in pending for p in pp for ln in p["lines"]]
        key = "en" if guess_lang(alltxt) == "en" else "ko"
        _flush(key)
    out_pages.sort(key=lambda p: (p["page"], p.get("part", 1)))
    res = {
        "schema": SCHEMA,
        "engine": eng.name,
        "engine_version": eng.version(key or "ko"),
        "source": _src_rel(pdf_path),
        "source_sha1": util.sha1_file(pdf_path),
        "dpi": dpi,
        "lang": key or "ko",
        "lang_how": lang_how,
        "made": {"stage": STAGE, "version": VERSION},
        "seconds": round(time.time() - t0, 1),
        "failed_pages": failed,
        "pages": out_pages,
    }
    if extra:
        res.update(extra)
    res["conf_mean"] = doc_conf(res)
    if out_json is not None:
        util.write_json(Path(out_json), res)
    return res


# ---------------------------------------------------------------- 단계 실행

def _problems_for(doc_id: str, row: dict, k: int, res: dict, out_json: Path) -> list[dict]:
    probs = []
    base = {"level": "doc", "doc_id": doc_id, "film_id": row.get("film_id") or None, "item_id": None,
            "path": _src_rel(out_json)}
    c = res.get("conf_mean")
    if c is not None and c < LOW_CONF:
        probs.append({**base, "problem": "ocr_low", "score": c,
                      "detail": f"OCR 평균 신뢰도 {c:.3f} < {LOW_CONF} (파일 k={k}, {res.get('source')})"})
    elif c is None:
        probs.append({**base, "problem": "ocr_low", "score": 0.0,
                      "detail": f"OCR 로 읽은 글자 없음 (파일 k={k}, {res.get('source')})"})
    if res.get("failed_pages"):
        fp = res["failed_pages"]
        n = res.get("npages") or len({p["page"] for p in res.get("pages", [])}) or 1
        probs.append({**base, "problem": "extract_broken", "score": round(1 - len(fp) / n, 3),
                      "detail": f"쪽 그림을 만들지 못함 {len(fp)}/{n}쪽: {fp[:20]} (파일 k={k}, {res.get('source')})"})
    return probs


def plan(rows: list[dict], force: bool = False) -> list[dict]:
    """catalog 행 -> OCR 할 파일 목록 [{doc_id,k,pdf,why,lang,row,textlayer?}].

    글자층 PDF(원본·변환)는 textlayer.check 로 깨졌을 때만 고른다. 이미 OCR 결과가 있으면
    (--force 가 아니면) 검사를 다시 하지 않고 그대로 대상에 넣는다(_work 가 건너뜀).
    """
    jobs = []
    for row in rows:
        if (row.get("status") or "") == "excluded":
            continue
        doc_id = row["doc_id"]
        files = row.get("files_list") or [f for f in (row.get("files") or "").split("|") if f]
        fmts = row.get("file_formats_list") or []
        for k, f in enumerate(files):
            fmt = (fmts[k] if k < len(fmts) and fmts[k] else row.get("real_format") or "").strip()
            src = paths.absp(f)
            job = None
            done = _cached(paths.OCR_CACHE / doc_id / f"{k}.json") if not force else None
            if fmt in ("pdf_scan", "pdf_broken"):
                job = {"pdf": src, "why": fmt}
            elif fmt in CONVERTED | {"pdf_text"} and done:
                job = done
            elif fmt == "pdf_text":
                bad, chk = needs_ocr(src)
                if bad:
                    job = {"pdf": src, "why": "textlayer_broken", "textlayer": chk}
            elif fmt in CONVERTED:
                cpdf = paths.PDF_CACHE / doc_id / f"{k}.pdf"
                if cpdf.exists():
                    bad, chk = needs_ocr(cpdf)
                    if bad:
                        job = {"pdf": cpdf, "why": "converted_textlayer_broken", "textlayer": chk}
            if job:
                lang = (row.get("script_lang") or "").strip().lower()
                if lang in ("", "mixed"):
                    lang = "unknown"        # 첫 글 쪽을 읽어 고른다(게시판이 한국영화여도 영문 대본이 있음)
                jobs.append({**job, "doc_id": doc_id, "k": k, "lang": lang, "row": row})
    _save_check_cache()
    return jobs


_CHECK_CACHE: dict | None = None


def _check_cache_path() -> Path:
    return paths.STATE / STAGE / "textlayer_check.json"


def needs_ocr(pdf: Path) -> tuple[bool, dict]:
    """글자층 PDF 를 OCR 해야 하는지(textlayer.check 로 깨졌고 lines 로도 되살릴 수 없음)와 검사 결과.
    판정은 파일 크기·수정 시각·검사 코드 버전이 같으면 cache/state/ocr/textlayer_check.json 에서 다시 쓴다
    (다시 돌릴 때 수백 개 PDF 를 또 검사하지 않게)."""
    global _CHECK_CACHE
    from . import lines, textlayer
    if _CHECK_CACHE is None:
        try:
            _CHECK_CACHE = util.read_json(_check_cache_path())
        except Exception:  # noqa: BLE001 - 없거나 깨졌으면 새로
            _CHECK_CACHE = {}
    st = pdf.stat()
    key = _src_rel(pdf)                      # 프로젝트 폴더를 옮겨도 쓰이게 상대 경로로
    ver = f"{VERSION}/{textlayer.VERSION}/{lines.VERSION}"
    hit = _CHECK_CACHE.get(key)
    if hit and hit.get("size") == st.st_size and hit.get("mtime_ns") == st.st_mtime_ns and hit.get("ver") == ver:
        return hit["bad"], hit["chk"]
    chk = textlayer.check(pdf)
    bad = bool(chk["broken"]) and not _restorable_by_lines(pdf, chk)
    _CHECK_CACHE[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "ver": ver, "bad": bad, "chk": chk}
    _CHECK_CACHE["_dirty"] = True
    return bad, chk


def _save_check_cache() -> None:
    if _CHECK_CACHE and _CHECK_CACHE.pop("_dirty", False):
        util.write_json(_check_cache_path(), _CHECK_CACHE)


RESTORABLE = ("ko_spacing_lost", "odd_chars", "single_char_lines", "duplicate_lines", "overlapping_glyphs")
LINES_SINGLE_MAX = 0.35   # 복원한 줄 중 한 글자 줄 비율 상한(한 음절 인물 이름 줄은 정상 대본에도 많음)


def _restorable_by_lines(pdf: Path, chk: dict) -> bool:
    """글자층 검사(get_text 줄)가 깨졌다고 해도, lines 단계가 글자 좌표로 다시 만든 줄
    (띄어쓰기 복원, 겹쳐 찍은 글자 제거, 모양 같은 글자 바꾸기, 같은 높이 조각 잇기)이 정상이면 OCR 하지 않는다.
    다시 만든 줄의 한글 낱말 평균 길이, 한 글자 줄 비율, 앞 줄과 같은 줄 비율, 이상한 문자 비율로 확인한다.
    글자층 자체는 정확한 글자라 OCR 보다 낫다."""
    if not chk.get("reasons") or not all(r.startswith(RESTORABLE) for r in chk["reasons"]):
        return False
    from . import lines, textlayer
    try:
        rows = lines.lines_from_pdf(pdf, "pdf_text")
    except Exception:
        return False
    body = [re.sub(r"\s+", " ", r["text"]).strip() for r in rows]
    body = [t for t in body if t]
    if not body:
        return False
    text = "\n".join(body)
    toks = [t for t in text.split() if re.search(r"[가-힣]", t)]
    hangul = sum(len(re.findall(r"[가-힣]", t)) for t in toks)
    if hangul >= 300 and hangul / len(toks) > textlayer.TH["ko_token_len"]:
        return False
    nonspace = len(re.sub(r"\s", "", text))
    if len(textlayer._ODD.findall(text)) / max(1, nonspace) > textlayer.TH["odd_char"]:
        return False
    single = sum(1 for b in body if len(b.replace(" ", "")) == 1) / len(body)
    dup = sum(1 for a, b in zip(body, body[1:]) if a == b) / len(body)
    return single <= LINES_SINGLE_MAX and dup <= textlayer.TH["dup_line"]


def retire_superseded(rows: list[dict]) -> list[str]:
    """글자층이 깨졌다고 OCR 했던 PDF 중 이제 lines 가 글자층을 되살릴 수 있는 것(_restorable_by_lines)의
    OCR 결과를 cache/ocr_superseded/ 로 옮긴다(lines 는 cache/ocr 에 결과가 있으면 그것을 쓰기 때문).
    plan() 과 같은 판정을 예전 결과에도 적용해, 처음부터 돌린 것과 같은 결과가 되게 한다. 옮긴 '{doc}/{k}' 목록."""
    moved = []
    for row in rows:
        if (row.get("status") or "") == "excluded":
            continue
        doc_id = row["doc_id"]
        for k, f in enumerate(row.get("files_list") or []):
            out = paths.OCR_CACHE / doc_id / f"{k}.json"
            if not out.exists():
                continue
            try:
                res = util.read_json(out)
            except Exception:  # noqa: BLE001 - 깨진 결과는 plan/_work 가 다시 만든다
                continue
            if res.get("why") not in ("textlayer_broken", "converted_textlayer_broken"):
                continue
            src = paths.absp(res["source"])
            if not src.exists():
                continue
            if not needs_ocr(src)[0]:
                dst = paths.OCR_SUPERSEDED / doc_id / f"{k}.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(out, dst)
                moved.append(f"{doc_id}/{k}")
                if not any((paths.OCR_CACHE / doc_id).glob("*.json")):
                    util.set_problems(STAGE, doc_id, [])      # 이 문서에 남은 OCR 결과가 없으면 OCR 문제 기록도 지움
    _save_check_cache()
    return moved


def _cached(out: Path) -> dict | None:
    """이미 만든 OCR 결과(글자층 PDF 쪽) -> 그때의 원본·사유로 된 job. 없거나 깨졌으면 None."""
    if not out.exists():
        return None
    try:
        res = util.read_json(out)
        src = paths.absp(res["source"])
    except Exception:
        return None
    if not src.exists():          # 기록된 입력(예: 대체 글로 바뀐 변환 PDF)이 없어졌으면 낡은 결과: 새로 판정
        return None
    return {"pdf": src, "why": res.get("why", "cached")}


def result_is_current(res: dict, pdf: Path) -> bool:
    """OCR 결과가 지금의 입력 PDF(같은 경로·같은 sha1)에서 만든 것인지. 원본이나 변환 PDF 가 바뀌었거나
    doc_id/k 가 다시 매겨졌으면 False(낡은 OCR 을 조용히 쓰지 않음)."""
    return (res.get("source") == _src_rel(pdf) and Path(pdf).exists()
            and res.get("source_sha1") == util.sha1_file(pdf))


def _work(job: dict, force: bool, threads: int) -> dict:
    out = paths.OCR_CACHE / job["doc_id"] / f"{job['k']}.json"
    if out.exists() and not force:
        try:
            res = util.read_json(out)
            if result_is_current(res, job["pdf"]):
                return {"job": job, "out": out, "res": _summary(res), "skipped": True}
        except Exception:
            pass
    extra = {"why": job["why"]}
    if job.get("textlayer"):
        extra["textlayer"] = job["textlayer"]
    res = ocr_pdf(job["pdf"], out, job["lang"], engine=get_engine(threads), extra=extra)
    return {"job": job, "out": out, "res": _summary(res), "skipped": False}


def _summary(res: dict) -> dict:
    return {"conf_mean": res.get("conf_mean"), "failed_pages": res.get("failed_pages", []),
            "npages": len({p["page"] for p in res.get("pages", [])}), "source": res.get("source"),
            "seconds": res.get("seconds"), "lang": res.get("lang")}


def run(args) -> int:
    try:
        from . import catalog
        rows = catalog.load()
    except Exception as e:  # catalog 모듈·파일이 아직 없을 때
        print(f"[ocr] catalog 를 읽지 못함: {e!r}", file=sys.stderr)
        return 1
    want = set(filter(None, (args.docs or "").split(","))) if getattr(args, "docs", None) else None
    if want:
        rows = [r for r in rows if r["doc_id"] in want]
    if getattr(args, "limit", None):
        rows = rows[: args.limit]
    moved = retire_superseded(rows)
    if moved:
        print(f"[ocr] 글자층 복원으로 충분해 쓰지 않을 예전 OCR 결과 {len(moved)}개 -> {paths.rel(paths.OCR_SUPERSEDED)}: "
              + ", ".join(moved), flush=True)
    jobs = plan(rows, force=bool(getattr(args, "force", False)))
    print(f"[ocr] 대상 파일 {len(jobs)}개 (문서 {len({j['doc_id'] for j in jobs})}개)", flush=True)
    workers = max(1, int(getattr(args, "workers", None) or 1))
    force = bool(getattr(args, "force", False))
    results: dict[str, list] = {}
    t0 = time.time()
    if workers == 1:
        it = (_work(j, force, -1) for j in jobs)
        _collect(it, results)
    else:
        threads = max(1, (os.cpu_count() or 4) // workers)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_work, {k: v for k, v in j.items()}, force, threads) for j in jobs]
            _collect((f.result() for f in futs), results)
    # 문서마다 문제 기록(그 문서의 OCR 파일 전부를 모아서)
    for doc_id, items in results.items():
        probs = []
        for r in items:
            probs += _problems_for(doc_id, r["job"]["row"], r["job"]["k"], r["res"], r["out"])
        util.set_problems(STAGE, doc_id, probs)
    print(f"[ocr] 끝: {len(jobs)}개 파일, {time.time() - t0:.0f}초", flush=True)
    return 0


def _collect(it, results):
    for r in it:
        j = r["job"]
        results.setdefault(j["doc_id"], []).append(r)
        s = r["res"]
        tag = "건너뜀" if r["skipped"] else f"{s.get('seconds')}초"
        print(f"[ocr] {j['doc_id']}/{j['k']} {j['why']} {s['npages']}쪽 lang={s.get('lang')} "
              f"conf={s.get('conf_mean')} 실패쪽={len(s['failed_pages'])} {tag}", flush=True)
