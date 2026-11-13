"""convert 단계 시험.

앞부분은 한컴 없이 도는 단위 시험이고, 뒷부분(hancom 표시)은 한컴오피스 2024 가 설치된 이 PC 에서
실제 한글을 띄워 변환한다(없으면 건너뜀). 결과는 모두 tmp_path 로 돌린다(실제 cache 를 건드리지 않음).
"""
from __future__ import annotations

import argparse
import threading
import time
import zipfile
from pathlib import Path

import pytest

from sinkprep import convert, paths, util

HANCOM = paths.HWP_EXE.exists()
needs_hancom = pytest.mark.skipif(not HANCOM, reason="한컴오피스 2024 없음")


def raw_file(post: str) -> Path:
    hits = [p for p in paths.RAW.rglob(f"*[[]{post}[]]*") if p.is_file()]
    if len(hits) != 1:
        pytest.skip(f"원본 표본 없음: {post}")
    return hits[0]


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """결과·문제·지문·작업 폴더를 tmp 로."""
    monkeypatch.setattr(paths, "PDF_CACHE", tmp_path / "pdf")
    monkeypatch.setattr(paths, "PROBLEMS", tmp_path / "problems")
    monkeypatch.setattr(paths, "STATE", tmp_path / "state")
    monkeypatch.setattr(convert, "WORK", tmp_path / "work")
    monkeypatch.setattr(convert, "LOG", tmp_path / "convert_log.jsonl")
    return tmp_path


# ------------------------------------------------------------------ 단위

def _u16(s: str) -> bytes:
    return s.encode("utf-16-le")


def test_hwp5_para_text_controls():
    tab = (9).to_bytes(2, "little") + b"\x00" * 14          # 인라인 제어(8글자)
    ext = (11).to_bytes(2, "little") + b"\x01" * 14         # 확장 제어(표 등, 8글자) -> 버림
    body = _u16("형사") + tab + _u16("너 어제") + (10).to_bytes(2, "little") + ext + _u16("어디?") \
        + (13).to_bytes(2, "little")
    assert convert._hwp5_para_text(body) == "형사\t너 어제\n어디?"


def test_hwp5_records_long_size():
    big = b"x" * 5000
    hdr = (convert.HWPTAG_PARA_TEXT | (0xFFF << 20)).to_bytes(4, "little") + len(big).to_bytes(4, "little")
    small = (66 | (3 << 20)).to_bytes(4, "little") + b"abc"
    got = list(convert._hwp5_records(hdr + big + small))
    assert [(t, len(b)) for t, b in got] == [(convert.HWPTAG_PARA_TEXT, 5000), (66, 3)]


def test_prepare_rtf_removes_sectd_only_without_paper_size(tmp_path):
    a = tmp_path / "a.rtf"
    a.write_bytes(b"{\\rtf1\\ansi{\\pard\\margt1985 \\sectd \\sl276 \\'c7\\'d1\\par}}")
    assert convert.prepare_rtf(a) == "rtf_sectd_removed(no_paperw)"
    assert b"\\sectd" not in a.read_bytes()
    assert b"\\margt1985" in a.read_bytes() and b"\\sl276" in a.read_bytes()
    b = tmp_path / "b.rtf"
    orig = b"{\\rtf1\\paperw11906\\paperh16838\\sectd\\pard x\\par}"
    b.write_bytes(orig)
    assert convert.prepare_rtf(b) is None
    assert b.read_bytes() == orig


def test_prepare_rtf_refuses_raw(tmp_path, monkeypatch):
    """손질은 작업 복사본에만: 원본 폴더 안 파일이면 쓰지 않고 멈춘다."""
    monkeypatch.setattr(paths, "RAW", tmp_path)
    p = tmp_path / "a.rtf"
    orig = b"{\\rtf1\\ansi{\\pard\\sectd x\\par}}"
    p.write_bytes(orig)
    with pytest.raises(PermissionError):
        convert.prepare_rtf(p)
    assert p.read_bytes() == orig


def test_rtf_fallback_uses_ansicpg(tmp_path):
    p = tmp_path / "k.rtf"
    p.write_bytes(b"{\\rtf1\\ansi\\ansicpg949{\\pard \\'c7\\'fc\\'bb\\'e7 \\'b3\\'ca\\par}}")
    text, method = convert.fallback_text(p, "rtf")
    assert method == "fallback_rtf"
    assert "형사 너" in text


def test_docx_fallback_keeps_table_cells(tmp_path):
    import docx
    d = docx.Document()
    d.add_paragraph("S#1. 빵집 - 낮")
    t = d.add_table(rows=1, cols=2)
    t.cell(0, 0).text = "형사"
    t.cell(0, 1).text = "너 어제 어디 있었어?"
    d.add_paragraph("끝")
    p = tmp_path / "a.docx"
    d.save(p)
    text, method = convert.fallback_text(p, "docx")
    assert method == "fallback_docx"
    assert text.split("\n") == ["S#1. 빵집 - 낮", "형사\t너 어제 어디 있었어?", "끝"]


def test_hwpx_fallback(tmp_path):
    p = tmp_path / "a.hwpx"
    ns = 'xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph"'
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("mimetype", "application/hwp+zip")
        z.writestr("Contents/section0.xml",
                   f'<hs:sec xmlns:hs="x" {ns}><hp:p><hp:run><hp:t>형사</hp:t></hp:run></hp:p>'
                   f'<hp:p><hp:run><hp:t>너 어제</hp:t><hp:t> 어디?</hp:t></hp:run></hp:p></hs:sec>')
    text, method = convert.fallback_text(p, "hwpx")
    assert (text, method) == ("형사\n너 어제 어디?", "fallback_hwpx")


def test_hwp5_real_file_script_and_bodytext():
    src = raw_file("56162")                     # A.I. (HWP5)
    assert convert.hwp5_script_is_default(src)
    text, method = convert.hwp5_text(src)
    assert method == "fallback_hwp5_bodytext"
    assert len(text) > 10000


@pytest.mark.skipif(not convert.ANTIWORD.exists(), reason="antiword 없음")
def test_doc_fallback_antiword(tmp_path):
    src = raw_file("55041")                     # 친절한 금자씨 (Word97, 한국어)
    cp = tmp_path / "k.doc"
    cp.write_bytes(src.read_bytes())
    text, method = convert.fallback_text(cp, "doc")
    assert method == "fallback_antiword"
    assert "친절한 금자씨" in text


def test_fallback_only_then_skip_then_force(sandbox):
    """한컴 없이(pool=None): 대체 경로 글 + 문제 'other', 같은 지문이면 건너뜀, --force 면 다시."""
    rtf = sandbox / "in" / "샘플 [99001].doc"
    rtf.parent.mkdir()
    rtf.write_bytes(b"{\\rtf1\\ansi\\ansicpg949{\\pard \\'c7\\'fc\\'bb\\'e7\\par}}")
    docs = convert.docs_from_paths([str(rtf)])
    assert docs[0]["doc_id"] == "fm99001" and docs[0]["file_formats_list"] == ["rtf"]
    log: dict = {}
    assert convert.convert_doc(docs[0], None, log) == "partial"
    assert (paths.PDF_CACHE / "fm99001" / "0.txt").read_text(encoding="utf-8").strip() == "형사"
    row = util.read_jsonl(convert.LOG)[0]
    assert set(row) >= {"doc_id", "k", "src", "real_format", "ok", "method", "pages", "error", "seconds"}
    assert (row["ok"], row["method"], row["real_format"]) == (True, "fallback_rtf", "rtf")
    probs = util.read_json(paths.PROBLEMS / "convert" / "fm99001.json")
    assert [p["problem"] for p in probs] == ["other"]
    assert convert.convert_doc(docs[0], None, log) == "skip"
    assert convert.convert_doc(docs[0], None, log, force=True) == "partial"
    assert not list((sandbox / "work").iterdir())          # 작업 복사본이 남지 않음


def test_total_failure_is_extract_broken(sandbox):
    """한컴 없이 돌리고 대체 경로도 실패하면(암호 걸린 HWP3) extract_broken."""
    src = sandbox / "in" / "암호 [99002].hwp"
    src.parent.mkdir()
    info = bytearray(128)
    info[96] = 1                                # 문서 정보 +96: 암호 표시
    src.write_bytes(b"HWP Document File V3.00 \x1a\x01\x02\x03\x04\x05" + bytes(info) + b"\0" * 1100)
    doc = {"doc_id": "fmT", "film_id": "", "files_list": [str(src)], "file_formats_list": ["hwp3"]}
    log: dict = {}
    assert convert.convert_doc(doc, None, log) == "failed"
    probs = util.read_json(paths.PROBLEMS / "convert" / "fmT.json")
    assert probs[0]["problem"] == "extract_broken" and probs[0]["level"] == "doc"
    assert log[("fmT", 0)]["ok"] is False
    assert "fallback:hwp3_password" in log[("fmT", 0)]["error"]
    assert not (paths.PDF_CACHE / "fmT").exists() or not list((paths.PDF_CACHE / "fmT").iterdir())


# ------------------------------------------------------------------ 한컴(실제 한글)

class WindowWatch:
    """한글·한워드가 화면 안에 보이는 창을 띄우는지 30ms 마다 감시."""

    def __init__(self):
        self.seen: list = []
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        import psutil
        import win32gui
        import win32process
        while not self._stop:
            pids = {p.pid: p.info["name"] for p in psutil.process_iter(["name"])
                    if (p.info["name"] or "").lower() in ("hwp.exe", "hword.exe")}

            def cb(h, _):
                try:
                    _, pid = win32process.GetWindowThreadProcessId(h)
                    if pid in pids and win32gui.IsWindowVisible(h):
                        r = win32gui.GetWindowRect(h)
                        if r[2] > -10000 and r[3] > -10000 and r[2] > r[0] and r[3] > r[1]:
                            self.seen.append((pids[pid], win32gui.GetWindowText(h), r))
                except Exception:
                    pass
                return True
            try:
                win32gui.EnumWindows(cb, None)
            except Exception:
                pass
            time.sleep(0.03)

    def stop(self):
        self._stop = True
        self._t.join(2)
        return self.seen


@needs_hancom
def test_hancom_run_hwp5_doc_docx(sandbox):
    before = convert.automation_hwp_pids()
    files = [raw_file("56162"), raw_file("56107"), raw_file("5691744")]   # hwp5, doc, docx
    watch = WindowWatch()
    args = argparse.Namespace(docs=None, limit=None, force=False, workers=None, paths=[str(f) for f in files])
    assert convert.run(args) == 0
    seen = watch.stop()
    assert seen == [], f"한컴 창이 화면에 보였다: {seen}"
    log = {(r["doc_id"], r["k"]): r for r in util.read_jsonl(convert.LOG)}
    for doc_id, word in [("fm56162", "A.I."), ("fm56107", "Alien"), ("fm5691744", "간신")]:
        r = log[(doc_id, 0)]
        assert (r["ok"], r["method"]) == (True, "hancom"), r
        pdf = paths.PDF_CACHE / doc_id / "0.pdf"
        pages, chars = convert.pdf_stats(pdf)
        assert pages == r["pages"] and pages > 40 and chars > 20000
        assert not (paths.PDF_CACHE / doc_id / "0.txt").exists()
    import pymupdf
    with pymupdf.open(paths.PDF_CACHE / "fm56107" / "0.pdf") as d:
        assert "Alien" in d[0].get_text()
    # 끝나면 이번에 띄운 한글이 남지 않는다
    time.sleep(1)
    assert convert.automation_hwp_pids() - before == set()


@needs_hancom
def test_hancom_rtf_without_paper_size(sandbox):
    """가을로 [55354]: \\paperw 없는 RTF. 손질 없이는 28985쪽으로 멈추던 문서."""
    doc = {"doc_id": "fm55354", "film_id": "", "files_list": [paths.rel(raw_file("55354"))],
           "file_formats_list": ["rtf"]}
    pool = convert.WorkerPool()
    try:
        assert convert.convert_doc(doc, pool, {}) == "ok"
    finally:
        pool.close()
    r = util.read_jsonl(convert.LOG)[0]
    assert r["method"] == "hancom" and r["note"] == "rtf_sectd_removed(no_paperw)"
    assert 40 < r["pages"] < 100
    import pymupdf
    with pymupdf.open(paths.PDF_CACHE / "fm55354" / "0.pdf") as d:
        assert "opening" in d[0].get_text()


@needs_hancom
def test_hancom_open_yes_retry(sandbox):
    """마린보이 [55927]: 첫 Open 이 '예/아니오/취소' 상자에서 막힘 -> open_yes 로 열림(빈 PDF 아님)."""
    doc = {"doc_id": "fm55927", "film_id": "", "files_list": [paths.rel(raw_file("55927"))],
           "file_formats_list": ["hwp5"]}
    pool = convert.WorkerPool()
    try:
        assert convert.convert_doc(doc, pool, {}) == "ok"
    finally:
        pool.close()
    r = util.read_jsonl(convert.LOG)[0]
    assert r["method"] == "hancom" and r["note"] == "hancom_open_yes" and r["pages"] > 50


@needs_hancom
def test_timeout_kills_worker_and_hwp_then_recovers(sandbox):
    before = convert.automation_hwp_pids()
    src = raw_file("56107")                     # Alien doc (Insert + PDF 수 초)
    work = sandbox / "work"
    work.mkdir()
    cp = work / "t.doc"
    cp.write_bytes(src.read_bytes())
    pool = convert.WorkerPool(timeout=0.2)
    try:
        res = pool.convert(cp, "doc", work / "t.pdf", "t")
        assert res["ok"] is False and res["error"] == "hancom_timeout"
        assert pool.w is None and pool.restarts == 1
        time.sleep(1)
        assert convert.automation_hwp_pids() - before == set(), "멈춘 한글이 남았다"
        pool.timeout = convert.TIMEOUT
        res = pool.convert(cp, "doc", work / "t.pdf", "t2")        # 새 일꾼으로 다음 문서
        assert res["ok"] is True and res["how"] == "insert"
    finally:
        pool.close()
    time.sleep(1)
    assert convert.automation_hwp_pids() - before == set()


# ---------------------------------------------------------------- 교차 검사(한컴 PDF 가 글을 잃었는지)

def test_letters_loss_counts_only_letters():
    assert convert.letters_loss("가나다라마", "가나다") == pytest.approx(0.4)
    assert convert.letters_loss("가나 ── 다", "가나다") == 0.0          # 구분선·공백은 세지 않음
    assert convert.letters_loss("abc", "cba xyz") == 0.0


def _pdf(path: Path, text: str) -> None:
    import pymupdf
    path.parent.mkdir(parents=True, exist_ok=True)
    d = pymupdf.open()
    d.new_page().insert_text((72, 72), text)
    d.save(path)


def test_crosscheck_warn_and_switch(sandbox, monkeypatch):
    full = "형사가 서류를 던진다 " * 60                      # 문자 540개
    monkeypatch.setattr(convert, "fallback_text", lambda src, fmt, notes=None: (full, "fallback_rtf"))
    base = {"src": "x.rtf", "real_format": "rtf", "ok": True, "method": "hancom", "pages": 1,
            "error": None, "note": None}
    # 1) 거의 같은 PDF: 그대로, loss 기록
    _pdf(paths.PDF_CACHE / "fmA" / "0.pdf", "a")
    monkeypatch.setattr(convert, "letters_loss", lambda a, b: 0.002)
    r = convert.crosscheck({**base, "doc_id": "fmA", "k": 0}, Path("x.rtf"))
    assert r["method"] == "hancom" and r["loss"] == 0.002 and convert.file_problems({"doc_id": "fmA"}, r) == []
    # 2) 조금 잃음(1~3%): PDF 그대로 쓰고 needs_review 에 올림
    monkeypatch.setattr(convert, "letters_loss", lambda a, b: 0.015)
    r = convert.crosscheck({**base, "doc_id": "fmA", "k": 0}, Path("x.rtf"))
    assert r["method"] == "hancom" and (paths.PDF_CACHE / "fmA" / "0.pdf").exists()
    assert [p["problem"] for p in convert.file_problems({"doc_id": "fmA"}, r)] == ["other"]
    # 3) 많이 잃음(>3%): PDF 를 incomplete 로 옮기고 대체 글을 씀
    monkeypatch.setattr(convert, "letters_loss", lambda a, b: 0.16)
    _pdf(paths.PDF_CACHE / "fmB" / "0.pdf", "a")
    r = convert.crosscheck({**base, "doc_id": "fmB", "k": 0}, Path("x.rtf"))
    assert r["method"] == "fallback_rtf" and "hancom_pdf_incomplete" in r["note"]
    assert not (paths.PDF_CACHE / "fmB" / "0.pdf").exists()
    assert (paths.PDF_CACHE / "fmB" / "0.incomplete.pdf").exists()
    assert (paths.PDF_CACHE / "fmB" / "0.txt").read_text(encoding="utf-8") == full
    probs = convert.file_problems({"doc_id": "fmB"}, r)
    assert probs[0]["problem"] == "other" and probs[0]["score"] == pytest.approx(0.84)
