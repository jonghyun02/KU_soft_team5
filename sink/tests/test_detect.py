"""detect.real_format: 확장자가 아니라 앞 바이트·내부 구조로 판별하는지(A3)."""
import zipfile

import pytest

from sinkprep import detect, paths


def _ole(path, streams: dict):
    """pywin32 로 작은 OLE(복합 문서) 파일을 만든다."""
    pythoncom = pytest.importorskip("pythoncom")
    from win32com import storagecon as sc

    mode = sc.STGM_CREATE | sc.STGM_READWRITE | sc.STGM_SHARE_EXCLUSIVE
    stg = pythoncom.StgCreateDocfile(str(path), mode, 0)
    for name, data in streams.items():
        st = stg.CreateStream(name, mode, 0, 0)
        st.Write(data)
        del st
    stg.Commit(0)
    del stg
    return path


def _pdf(path, pages: list[str]):
    import pymupdf

    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text, fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------- 위장 형식(.doc 인데 속은 다른 것)

def test_rtf_disguised_as_doc(tmp_path):
    p = tmp_path / "M.doc"
    p.write_bytes(b"{\\rtf1\\ansi\\ansicpg949\\deff0 {\\fonttbl{\\f0 Batang;}} S#1. \\'b9\\'e3 \\par }")
    assert detect.real_format(p) == "rtf"


def test_plain_text_disguised_as_doc_cp949(tmp_path):
    p = tmp_path / "JFK.doc"
    p.write_bytes("S#1. 댈러스 거리 - 낮\r\n\r\n짐   이건 시작일 뿐이야.\r\n".encode("cp949") * 50)
    assert detect.real_format(p) == "txt"


def test_plain_text_utf8_and_utf16(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("INT. DINER - NIGHT\n\nJOHN\nHello there.\n" * 20, encoding="utf-8")
    b = tmp_path / "b.doc"
    b.write_bytes("장면 1. 바닷가\n민수: 안녕\n".encode("utf-16"))
    assert detect.real_format(a) == "txt"
    assert detect.real_format(b) == "txt"


def test_long_text_cut_mid_character_is_still_text(tmp_path):
    # 1MB 넘는 글: 앞 1MB 가 한글 글자 중간에서 잘려도 txt
    p = tmp_path / "long.txt"
    p.write_bytes(("가" * 700_000).encode("utf-8"))
    assert detect.real_format(p) == "txt"


def test_unknown_binary_is_other(tmp_path):
    # 반칙왕[54331] 처럼 02 b3 aa 로 시작하는 짧은 바이너리
    p = tmp_path / "x.doc"
    p.write_bytes(bytes.fromhex("02b3aa") + b"\x00" * 50 + b"\x98\xb0\x00\x1a" + b"\x00" * 100)
    assert detect.real_format(p) == "other"


def test_exe_is_other(tmp_path):
    p = tmp_path / "MovieSmiCh.hwp"
    p.write_bytes(b"MZ\x90\x00" + b"\x00" * 200)
    assert detect.real_format(p) == "other"


# ---------------------------------------------------------------- HWP / OLE

def test_hwp3_magic(tmp_path):
    p = tmp_path / "a.doc"
    p.write_bytes(b"HWP Document File V3.00 \x1a\x01\x02\x03\x04\x05" + b"\x00" * 100)
    assert detect.real_format(p) == "hwp3"


def test_ole_hwp5_doc_xls(tmp_path):
    assert detect.real_format(_ole(tmp_path / "a.doc", {"FileHeader": b"HWP Document File" + b"\0" * 15})) == "hwp5"
    assert detect.real_format(_ole(tmp_path / "b.hwp", {"WordDocument": b"\xec\xa5" + b"\0" * 30})) == "doc"
    assert detect.real_format(_ole(tmp_path / "c.doc", {"Workbook": b"\x09\x08" + b"\0" * 30})) == "xls"
    assert detect.real_format(_ole(tmp_path / "d.doc", {"Other": b"x" * 10})) == "other"


# ---------------------------------------------------------------- ZIP 계열·압축

def test_pk_family(tmp_path):
    docx = tmp_path / "a.zip"
    with zipfile.ZipFile(docx, "w") as z:
        z.writestr("[Content_Types].xml", "<x/>")
        z.writestr("word/document.xml", "<w:document/>")
    hwpx = tmp_path / "b.hwp"
    with zipfile.ZipFile(hwpx, "w") as z:
        z.writestr("mimetype", "application/hwp+zip")
        z.writestr("Contents/section0.xml", "<hs:sec/>")
    plain = tmp_path / "c.docx"
    with zipfile.ZipFile(plain, "w") as z:
        z.writestr("대본.txt", "가나다")
    assert detect.real_format(docx) == "docx"
    assert detect.real_format(hwpx) == "hwpx"
    assert detect.real_format(plain) == "zip"


def test_rar_alz_magic_and_catalog_value(tmp_path):
    r = tmp_path / "a.part2.rar"
    r.write_bytes(b"Rar!\x1a\x07\x00" + b"\x00" * 20)
    a = tmp_path / "a.a00"
    a.write_bytes(b"ALZ\x01\x0a\x00\x01\x00" + b"\x00" * 20)
    assert detect.real_format(r) == "rar"
    assert detect.real_format(a) == "alz"
    assert detect.catalog_format("rar") == "archive"
    assert detect.catalog_format("zip") == "archive"
    assert detect.catalog_format("hwp5") == "hwp5"


# ---------------------------------------------------------------- PDF

def test_pdf_text_scan_broken(tmp_path):
    line = "INT. POLICE STATION - NIGHT  The detective throws the file on the desk. " * 2
    assert detect.real_format(_pdf(tmp_path / "t.pdf", [line, line, line])) == "pdf_text"
    # 글자 없는 쪽이 절반 이상이면 스캔
    assert detect.real_format(_pdf(tmp_path / "s.pdf", ["", "", line, ""])) == "pdf_scan"
    # 정확히 절반이 50자 미만이어도 스캔
    assert detect.real_format(_pdf(tmp_path / "h.pdf", ["", line])) == "pdf_scan"
    b = tmp_path / "b.pdf"
    b.write_bytes(b"%PDF-1.4\n%garbage that is not a pdf\n")
    assert detect.real_format(b) == "pdf_broken"


def test_pdf_header_after_junk(tmp_path):
    p = _pdf(tmp_path / "x.pdf", ["INT. ROOM - DAY  Somebody walks in and sits down at the table." * 2])
    q = tmp_path / "y.doc"
    q.write_bytes(b"\r\n\r\n" + p.read_bytes())
    assert detect.real_format(q) == "pdf_text"


# ---------------------------------------------------------------- 실제 원본(있을 때만)

KNOWN = {
    "_M_ 시나리오 [55542].doc": "rtf",
    "_가을로_ 시나리오 [55354].doc": "rtf",
    "JFK [56704].doc": "txt",
    "드디어 반칙왕 시나라오를 [54331].doc": "other",
    "No Country For Old Men (노인을 위한 나라는 없다) [56897].pdf": "pdf_broken",
    "묻지마패밀리_교회누나_ [54537].xls": "xls",
    "Little Miss Sunshine (미스 리틀 선샤인 part.2) [56874].a00": "alz",
    "황해 시나리오 [773767] 2_황해.part2.rar": "rar",
}


@pytest.mark.skipif(not (paths.RAW / "manifest.csv").exists(), reason="원본 없음")
def test_real_disguised_files():
    found = {p.name: p for p in paths.RAW.rglob("*") if p.name in KNOWN}
    assert set(found) == set(KNOWN)
    for name, want in KNOWN.items():
        assert detect.real_format(found[name]) == want, name
