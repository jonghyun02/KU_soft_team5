"""파일 형식 판별: 확장자가 아니라 앞 바이트와 내부 구조로 정한다(수락 기준 A3).

반환값(real_format):
  hwp3, hwp5, hwpx, doc, docx, xls, rtf, pdf_text, pdf_scan, pdf_broken, txt,
  zip, rar, alz(압축), other(실행 파일·그림·알 수 없는 바이너리 등)

catalog.csv 의 real_format 칸에는 압축(zip/rar/alz)을 `archive` 로 적는다(catalog_format).
원본 파일은 읽기만 한다. 어떤 내용도 실행하지 않는다.
"""
from __future__ import annotations

import codecs
import zipfile
from pathlib import Path

ARCHIVE_FORMATS = ("zip", "rar", "alz")
# 대본으로 읽을 수 있는 형식(이 밖의 형식은 catalog 에서 excluded 후보)
SCRIPT_FORMATS = ("hwp3", "hwp5", "hwpx", "doc", "docx", "rtf",
                  "pdf_text", "pdf_scan", "pdf_broken", "txt")

OLE_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")
HWP3_MAGIC = b"HWP Document File V3"
HWP5_SIG = b"HWP Document File"
PDF_SCAN_MIN_CHARS = 50          # 쪽 글자 수가 이보다 적으면 '글자 없는 쪽'
HEAD = 4096


def catalog_format(fmt: str) -> str:
    """catalog.csv 칸 값(DESIGN '형식 값')으로 바꾼다: 압축은 archive."""
    return "archive" if fmt in ARCHIVE_FORMATS else fmt


def _read_head(path: Path, n: int = HEAD) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


# ---------------------------------------------------------------- OLE

def _ole_format(path: Path) -> str:
    try:
        import olefile
    except ImportError:  # pragma: no cover - 설치돼 있음
        return "other"
    try:
        with olefile.OleFileIO(str(path)) as ole:
            streams = {"/".join(s) for s in ole.listdir(streams=True, storages=False)}
            low = {s.lower() for s in streams}
            if "fileheader" in low:
                try:
                    hdr = ole.openstream("FileHeader").read(64)
                except Exception:
                    hdr = b""
                if hdr.startswith(HWP5_SIG):
                    return "hwp5"
            if "worddocument" in low:
                return "doc"
            if "workbook" in low or "book" in low:
                return "xls"
            return "other"
    except Exception:
        return "other"


# ---------------------------------------------------------------- ZIP 계열

def _pk_format(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            low = {n.lower() for n in names}
            if "word/document.xml" in low:
                return "docx"
            if "mimetype" in low:
                try:
                    mt = z.read("mimetype")[:64].decode("ascii", "replace").strip()
                except Exception:
                    mt = ""
                if mt.startswith("application/hwp+zip"):
                    return "hwpx"
            if any(n.startswith("contents/section") and n.endswith(".xml") for n in low):
                return "hwpx"
            if "xl/workbook.xml" in low:
                return "xls"
            return "zip"
    except (zipfile.BadZipFile, OSError, ValueError, NotImplementedError):
        return "zip"  # 분할 zip 조각 등: 머리는 PK 지만 목록을 못 읽음


# ---------------------------------------------------------------- PDF

def pdf_page_chars(path: Path) -> list[int] | None:
    """쪽마다 글자 수(공백 제외). 열기·추출 실패면 None."""
    try:
        import pymupdf as fitz  # PyMuPDF
    except ImportError:  # pragma: no cover
        import fitz  # type: ignore
    try:
        fitz.TOOLS.mupdf_display_errors(False)
    except Exception:
        pass
    try:
        doc = fitz.open(str(path))
    except Exception:
        return None
    try:
        if doc.needs_pass or doc.page_count == 0:
            return None
        out = []
        for page in doc:
            t = page.get_text("text")
            out.append(sum(1 for ch in t if not ch.isspace()))
        return out
    except Exception:
        return None
    finally:
        doc.close()


def _pdf_format(path: Path) -> str:
    counts = pdf_page_chars(path)
    if not counts:
        return "pdf_broken"
    low = sum(1 for c in counts if c < PDF_SCAN_MIN_CHARS)
    return "pdf_scan" if low * 2 >= len(counts) else "pdf_text"


# ---------------------------------------------------------------- 글(txt)

def decode_text(data: bytes, partial: bool = False) -> tuple[str, str] | None:
    """평문이면 (글, 인코딩), 아니면 None. partial=True 면 끝이 글자 중간에서 잘려도 된다."""
    if not data:
        return None
    if data.startswith(b"\xef\xbb\xbf"):
        cands = ["utf-8-sig"]
    elif data.startswith((b"\xff\xfe", b"\xfe\xff")):
        cands = ["utf-16"]
    else:
        if b"\x00" in data:
            return None
        cands = ["utf-8", "cp949"]
    for enc in cands:
        try:
            dec = codecs.getincrementaldecoder(enc)()
            return dec.decode(data, final=not partial), enc
        except UnicodeDecodeError:
            continue
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(data).best()
        if best is not None and best.encoding:
            return str(best), best.encoding
    except Exception:
        pass
    return None


def _looks_like_text(s: str) -> bool:
    if not s.strip():
        return False
    bad = sum(1 for ch in s if (ord(ch) < 32 and ch not in "\t\r\n\f\x1a") or ch == "�")
    return bad <= max(2, len(s) // 200)


def _txt_or_other(path: Path, head: bytes) -> str:
    # 파일 전체(1MB 넘으면 앞 1MB)를 본다. 잘린 끝은 partial 로 허용.
    size = path.stat().st_size
    data = head if size <= len(head) else _read_head(path, min(size, 1 << 20))
    dec = decode_text(data, partial=size > len(data))
    if dec is None:
        return "other"
    return "txt" if _looks_like_text(dec[0]) else "other"


# ---------------------------------------------------------------- 진입점

def sniff(head: bytes) -> str | None:
    """앞 바이트만으로 정해지는 형식(구조 확인 전). 모르면 None."""
    if head.startswith(HWP3_MAGIC):
        return "hwp3"
    if head.startswith(OLE_MAGIC):
        return "ole"
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06") or head.startswith(b"PK\x07\x08"):
        return "pk"
    if head.startswith(b"Rar!\x1a\x07"):
        return "rar"
    if head.startswith(b"ALZ\x01"):
        return "alz"
    if head.lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"{\\rtf"):
        return "rtf"
    if b"%PDF-" in head[:1024]:
        return "pdf"
    if head.startswith(b"MZ"):
        return "other"
    return None


def real_format(path: str | Path) -> str:
    path = Path(path)
    head = _read_head(path)
    kind = sniff(head)
    if kind == "ole":
        return _ole_format(path)
    if kind == "pk":
        return _pk_format(path)
    if kind == "pdf":
        return _pdf_format(path)
    if kind is not None:
        return kind
    return _txt_or_other(path, head)
