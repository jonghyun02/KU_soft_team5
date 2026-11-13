"""sinkprep.hwp3text 시험(HWP 3.0 글 뽑기, convert 의 HWP3 대체 경로).

앞부분은 손으로 만든 작은 HWP3 파일로 구조 읽기를 본다(원본 폴더 없이 돈다). 뒷부분은 실제 HWP3 대본을
한컴이 만든 PDF(cache/pdf)와 글자 단위로 견주고, 한컴이 못 여는 fm55645 를 읽는다(없으면 건너뜀).
"""
from __future__ import annotations

import re
import unicodedata
import zlib

import pytest

from sinkprep import convert, hwp3text, paths, util

SIG = b"HWP Document File V3.00 \x1a\x01\x02\x03\x04\x05"


# ------------------------------------------------------------------ 작은 HWP3 만들기

def hc(s: str) -> list[int]:
    """글 -> hchar 목록(ASCII 그대로, 한글은 조합형)."""
    out = []
    for ch in s:
        out.append(ord(ch) if ord(ch) < 0x80 else int.from_bytes(ch.encode("johab"), "big"))
    return out


def u16(*v: int) -> bytes:
    return b"".join(x.to_bytes(2, "little") for x in v)


def para(*items) -> bytes:
    """문단 하나. items: 글(str) 또는 (특수 문자 바이트, 글자 수 몫) 짝."""
    body, nch = b"", 1                      # 1 = 문단 끝(13)
    for it in items:
        if isinstance(it, str):
            body += u16(*hc(it))
            nch += len(it)
        else:
            body += it[0]
            nch += it[1]
    info = bytes([0]) + u16(nch, 1) + bytes([0, 0]) + b"\0" * 4 + bytes([0]) + b"\0" * 31
    return info + b"\0" * 187 + b"\0" * 14 + body + u16(13)


END = b"\0" + u16(0, 0) + b"\0" * 38      # 글자 수 0 문단 = 리스트 끝 (43 B)


def tab() -> tuple[bytes, int]:
    return u16(9, 400, 0, 9), 4


def table(cells: list[tuple[int, int, bytes]]) -> tuple[bytes, int]:
    """표: cells = [(x, y, 문단 리스트 바이트)] 파일 순서."""
    info = bytearray(84)
    info[80:82] = len(cells).to_bytes(2, "little")
    data = u16(10) + b"\0" * 4 + u16(10) + bytes(info)
    for x, y, _ in cells:
        cell = bytearray(27)
        cell[4:6] = x.to_bytes(2, "little")
        cell[6:8] = y.to_bytes(2, "little")
        data += bytes(cell)
    for _, _, plist in cells:
        data += plist
    return data + END, 4                    # 설명 리스트는 빈 리스트


def outline(shape: int, number: int, deco: str = "") -> tuple[bytes, int]:
    """개요 번호(28): 종류 1, 수준 0. deco 는 사용자 모양의 앞뒤 꾸밈 두 글자."""
    d = bytearray(60)
    d[0:2] = u16(1)
    d[2] = shape
    d[4:6] = u16(number)
    if deco:
        d[32:36] = u16(*hc(deco))
    return u16(28) + bytes(d) + u16(28), 32


def header(text: str) -> tuple[bytes, int]:
    return u16(16) + b"\0" * 4 + u16(16) + b"\0" * 10 + para(text) + END, 4


def hwp3_file(tmp_path, body: bytes, compressed: bool = True, password: bool = False, name="a.hwp"):
    info = bytearray(128)
    info[96:98] = (1 if password else 0).to_bytes(2, "little")
    info[124] = 1 if compressed else 0
    fonts = (u16(1) + b"\0" * 40) * 7 + u16(0)          # 언어마다 글꼴 1개, 스타일 0개
    payload = fonts + body + b"\0" * 8                  # 끝 표시 비슷한 꼬리
    if compressed:
        c = zlib.compressobj(9, zlib.DEFLATED, -15)
        payload = c.compress(payload) + c.flush() + b"TRAILER" * 10   # 압축 뒤 남는 자료도 흉내
    p = tmp_path / name
    p.write_bytes(SIG + bytes(info) + b"\0" * 1008 + payload)
    return p


# ------------------------------------------------------------------ 단위

def test_hchar_text_tables():
    assert hwp3text.hchar_text(ord("A")) == "A"
    assert hwp3text.hchar_text(int.from_bytes("형".encode("johab"), "big")) == "형"
    assert hwp3text.hchar_text(int.from_bytes("ㄱ".encode("johab"), "big")) == "ㄱ"
    assert hwp3text.hchar_text(0x4000) == "伽"              # 한자 순번 0 = KS X 1001 0xCAA1
    assert hwp3text.hchar_text(0x3406) == "…"               # 기호 1행 6번째 = 0xA1A6
    assert hwp3text.hchar_text(0x81) + hwp3text.hchar_text(0x82) == "“”"
    assert hwp3text.hchar_text(0x1F02) == "あ"
    assert hwp3text.hchar_text(0x301C) == "─"               # 선 문자(가로)
    assert hwp3text.hchar_text(0x4000 + 4888) == "\ufffd"    # 한컴 사용자 한자: 대응 없음


def test_paragraphs_tab_table_header(tmp_path):
    cells = [(500, 0, para("대사") + END),                  # 파일 순서는 오른쪽 칸 먼저
             (0, 0, para("형사") + END),
             (0, 300, para("둘째 줄") + END)]
    body = (para("S#1. 빵집 - 낮")
            + para("형사", tab(), "너 어제 어디 있었어?")
            + para(header("머리말 글"), "본문")
            + para(table(cells))
            + para("")
            + para(outline(1, 7), " 국방부 (낮)")
            + para("S#.", outline(0, 12, "#."), tab(), "성당 (밤)")     # 번호는 문단 맨 앞에 그려진다
            + END)
    p = hwp3_file(tmp_path, body)
    paras, stop = hwp3text.extract(p)
    assert stop is None
    assert paras == ["S#1. 빵집 - 낮", "형사\t너 어제 어디 있었어?", "본문",
                     "", "형사", "대사", "둘째 줄", "", "7. 국방부 (낮)", "#12. S#.\t성당 (밤)"]
    assert hwp3text.extract_paragraphs(hwp3_file(tmp_path, body, compressed=False, name="b.hwp")) == paras


def test_broken_structure_returns_what_was_read(tmp_path):
    bad = (u16(14) + b"\0" * 4 + u16(99), 4)              # 선(14)의 확인 코드가 틀림
    body = para("첫 문단") + para("둘째", bad, "안 읽힘") + para("셋째") + END
    paras, stop = hwp3text.extract(hwp3_file(tmp_path, body))
    assert paras == ["첫 문단", "둘째"]
    assert stop.startswith("ctrl14_mismatch@")
    # 파일이 문단 도중에 잘려도 죽지 않고, 읽던 문단의 앞부분까지 돌려준다
    long = "가나다" * 50
    for compressed in (False, True):
        p = hwp3_file(tmp_path, para("첫 문단") + para(long) + END, compressed=compressed, name="cut.hwp")
        data = p.read_bytes()
        p.write_bytes(data[:len(data) - (200 if not compressed else 90)])
        paras, stop = hwp3text.extract(p)
        assert stop is not None and stop.startswith("eof@"), (compressed, stop)
        assert paras[0] == "첫 문단" and len(paras) <= 2
        assert all(long.startswith(q) and len(q) < len(long) for q in paras[1:])
        if not compressed:                  # 바이트를 정확히 잘랐으니 둘째 문단 앞부분이 있다
            assert len(paras) == 2 and paras[1]


def test_not_hwp3_and_password_raise(tmp_path):
    p = tmp_path / "x.hwp"
    p.write_bytes(b"\xd0\xcf\x11\xe0" + b"\0" * 2000)        # OLE(HWP5) 머리
    with pytest.raises(ValueError, match="not_hwp3"):
        hwp3text.extract_paragraphs(p)
    q = hwp3_file(tmp_path, para("비밀") + END, password=True, name="pw.hwp")
    with pytest.raises(ValueError, match="hwp3_password"):
        hwp3text.extract_paragraphs(q)


def test_convert_fallback_hwp3(tmp_path, monkeypatch):
    """한컴 없이(pool=None) HWP3 -> {k}.txt, method fallback_hwp3_text, 일부만 읽으면 error 에 hwp3_partial."""
    for name, value in (("PDF_CACHE", tmp_path / "pdf"), ("PROBLEMS", tmp_path / "problems"),
                        ("STATE", tmp_path / "state")):
        monkeypatch.setattr(paths, name, value)
    monkeypatch.setattr(convert, "WORK", tmp_path / "work")
    monkeypatch.setattr(convert, "LOG", tmp_path / "convert_log.jsonl")
    src = tmp_path / "in"
    src.mkdir()
    good = hwp3_file(src, para("S#1. 빵집") + para("형사", tab(), "어디?") + END, name="좋은 [99101].hwp")
    bad = (u16(14) + b"\0" * 4 + u16(99), 4)
    broken = hwp3_file(src, para("앞부분") + para(bad) + END, name="망가짐 [99102].hwp")
    docs = convert.docs_from_paths([str(good), str(broken)])
    assert [d["file_formats_list"] for d in docs] == [["hwp3"], ["hwp3"]]
    log: dict = {}
    assert [convert.convert_doc(d, None, log) for d in docs] == ["partial", "partial"]
    assert (paths.PDF_CACHE / "fm99101" / "0.txt").read_text(encoding="utf-8") == "S#1. 빵집\n형사\t어디?"
    rows = {r["doc_id"]: r for r in util.read_jsonl(convert.LOG)}
    assert (rows["fm99101"]["ok"], rows["fm99101"]["method"], rows["fm99101"]["error"]) == \
        (True, "fallback_hwp3_text", None)
    assert rows["fm99102"]["method"] == "fallback_hwp3_text"
    assert rows["fm99102"]["error"].startswith("hwp3_partial:ctrl14_mismatch@")
    probs = [p for p in util.all_problems() if p["doc_id"] in ("fm99101", "fm99102")]
    # 끝까지 읽은 대체 글은 other(레이아웃 없음), 도중에 멈춘 것은 extract_broken(점수 = 읽은 비율)
    assert [p["problem"] for p in probs] == ["other", "extract_broken"]
    assert ";read=" in rows["fm99102"]["error"] and 0 <= probs[1]["score"] < 1


# ------------------------------------------------------------------ 실제 파일

WS = re.compile("[\\s\uf000]+")            # U+F000: 한컴 PDF 의 빈칸 글리프(lines 도 공백으로 봄)


def _norm(s: str) -> str:
    return WS.sub("", unicodedata.normalize("NFC", s))


def _hwp3_file(doc_id: str):
    from sinkprep import catalog
    if not paths.CATALOG.exists():
        pytest.skip("catalog.csv 없음")
    for r in catalog.load():
        if r["doc_id"] == doc_id:
            k = r["file_formats_list"].index("hwp3")
            p = paths.absp(r["files_list"][k])
            if p.exists():
                return p, k
    pytest.skip(f"원본 표본 없음: {doc_id}")


@pytest.mark.parametrize("doc_id", ["fm54114", "fm54147", "fm54579"])
def test_real_hwp3_matches_hancom_pdf(doc_id):
    """한컴 PDF 글자층과 공백을 빼고 견줘 0.97 이상(fm54147 은 탭 확인 코드에 쓰레기 값이 든 문서)."""
    import pymupdf
    from rapidfuzz import fuzz
    src, k = _hwp3_file(doc_id)
    pdf = paths.PDF_CACHE / doc_id / f"{k}.pdf"
    if not pdf.exists():
        pytest.skip(f"한컴 PDF 없음: {pdf}")
    paras, stop = hwp3text.extract(src)
    assert stop is None
    with pymupdf.open(str(pdf)) as d:
        ref = _norm("".join(page.get_text() for page in d))
    got = _norm("".join(paras))
    assert fuzz.ratio(got, ref) / 100 >= 0.97
    assert got.count("\ufffd") / len(got) < 0.001


def test_real_hwp3_hancom_cannot_open_fm55645():
    """한컴이 못 여는 파일: 앞 40% 뒤로 압축 자료가 망가져 그 앞까지만 읽는다."""
    src, _ = _hwp3_file("fm55645")
    paras, stop = hwp3text.extract(src)
    assert stop is not None
    assert paras[0].startswith("S#1")
    assert len(paras) > 900 and sum(len(p) for p in paras) > 20000
