"""ocr: 줄 묶기(순수 함수), 쪽 그림(JBIG2 대체 렌더러), 실제 스캔 쪽 OCR, run() 흐름."""
from __future__ import annotations

import functools
import json
import shutil
import types

import pytest

from sinkprep import ocr, paths, util

KO = paths.RAW / "한국영화 시나리오"
FO = paths.RAW / "외국영화 시나리오"
BYEONHOIN = KO / "변호인 [3710018].pdf"
DARKKNIGHT = FO / "The Dark Knight (다크 나이트) [56924].pdf"
NOCOUNTRY = FO / "No Country For Old Men (노인을 위한 나라는 없다) [56897].pdf"
BERLIN = KO / "베를린 [3659631].pdf"
TUNNEL = KO / "터널 [5691168].pdf"
GUTA = KO / "구타유발자들 [54792].pdf"   # 띄어쓰기·구두점 줄이 깨져 줄 복원으로도 안 됨


def _need(*ps):
    for p in ps:
        if not p.exists():
            pytest.skip(f"원본 없음: {p.name}")


def W(text, x0, y0, x1, y1, conf=None):
    return {"text": text, "box": [x0, y0, x1, y1], "conf": conf}


# ---------------------------------------------------------------- group_lines

def test_group_words_keep_columns():
    """낱말 상자(윈도 OCR 식) -> 한 줄, 화자 칸과 대사 칸은 segs 로 따로."""
    prims = [
        W("부동산", 750, 440, 880, 495, 0.9), W("김광삼", 490, 442, 609, 497, 1.0),
        W("등기를?", 900, 441, 1050, 494, 0.8), W("변호사가?", 1070, 443, 1240, 496, 0.9),
    ]
    lines = ocr.group_lines(prims)
    assert len(lines) == 1
    ln = lines[0]
    assert ln["box"] == [490, 440, 1240, 497]
    assert [s["box"][0] for s in ln["segs"]] == [490, 750]
    assert ln["segs"][1]["text"] == "부동산 등기를? 변호사가?"
    assert ln["text"].startswith("김광삼  ") and ln["text"].endswith("부동산 등기를? 변호사가?")
    # 글자 수로 가중한 평균
    assert 0.85 < ln["conf"] < 0.95


def test_group_lines_order_and_split():
    prims = [W("둘째 줄", 400, 600, 700, 650, 0.9), W("첫째 줄", 400, 500, 700, 550, 0.95),
             W("셋째", 400, 700, 520, 750, 0.9)]
    lines = ocr.group_lines(prims)
    assert [l["text"] for l in lines] == ["첫째 줄", "둘째 줄", "셋째"]
    assert all("segs" not in l for l in lines)


def test_group_det_boxes_small_gap_one_segment():
    """검출기가 한 줄을 두 상자로 쪼갠 경우(간격이 좁음) -> 한 덩어리."""
    prims = [W("Two black SUV's pull", 462, 1420, 1100, 1465, 0.99),
             W("onto the top floor.", 1130, 1421, 1700, 1466, 0.98)]
    lines = ocr.group_lines(prims)
    assert len(lines) == 1 and "segs" not in lines[0]
    assert lines[0]["text"] == "Two black SUV's pull onto the top floor."


def test_group_empty_and_bad_boxes():
    assert ocr.group_lines([]) == []
    assert ocr.group_lines([W("  ", 0, 0, 10, 10), W("x", 5, 5, 5, 9)]) == []


def test_guess_lang():
    assert ocr.guess_lang([{"text": "부동산 등기를? 변호사가? 그래도 되나?"} for _ in range(3)]) == "ko"
    assert ocr.guess_lang([{"text": "Why wouldn't he come? Hopefully because he's busy"}]) == "en"
    assert ocr.guess_lang([{"text": "- 5 -"}]) == "unknown"


# ---------------------------------------------------------------- 쪽 그림

def test_render_jbig2_fallback():
    """No Country: PyMuPDF·pdftoppm 모두 JBIG2 해독 실패하는 쪽은 failed, 나머지는 그림."""
    _need(NOCOUNTRY)
    got = {p: (img is not None, info["render"]) for p, img, info in ocr.render_pages(NOCOUNTRY, 100, [1, 2, 41])}
    assert got[1] == (True, "pymupdf")
    assert got[41] == (True, "pymupdf")
    assert got[2] == (False, "failed")


def test_find_pdftoppm():
    assert ocr.find_pdftoppm() is not None


# ---------------------------------------------------------------- 실제 OCR (쪽 하나씩)

@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("rapidocr")
    return ocr.Engine()


def _find(res, needle):
    for pg in res["pages"]:
        for ln in pg["lines"]:
            for s in ln.get("segs", [ln]):
                if needle in s["text"]:
                    return s
    return None


def test_ocr_korean_scan(tmp_path, engine):
    _need(BYEONHOIN)
    out = tmp_path / "o" / "0.json"
    res = ocr.ocr_pdf(BYEONHOIN, out, None, pages=[5], engine=engine)
    disk = json.loads(out.read_text(encoding="utf-8"))
    assert disk["schema"] == "sink.ocr/1" and disk["dpi"] == 300
    assert disk["source"] == "filmmakers_scripts/한국영화 시나리오/변호인 [3710018].pdf"
    assert res["lang"] == "ko" and res["lang_how"] == "guess"
    pg = disk["pages"][0]
    assert pg["page"] == 5 and pg["width"] == 2480 and pg["height"] in (3508, 3509)
    for ln in pg["lines"]:
        assert set(ln) >= {"text", "box", "conf"} and len(ln["box"]) == 4 and 0 <= ln["conf"] <= 1
    # 대사 칸 들여쓰기 위치가 살아 있다(화자 x≈490, 대사 x≈748, 지문 x≈414, 씬 머리 x≈331)
    s = _find(disk, "부동산 등기를")
    assert s is not None and abs(s["box"][0] - 748) <= 30
    s = _find(disk, "김광삼")
    assert s is not None and abs(s["box"][0] - 490) <= 30
    s = _find(disk, "흥청거리는")
    assert s is not None and abs(s["box"][0] - 414) <= 30
    s = _find(disk, "시내 유흥가")
    assert s is not None and abs(s["box"][0] - 331) <= 30
    assert res["conf_mean"] > 0.85


def test_ocr_english_scan(tmp_path, engine):
    _need(DARKKNIGHT)
    res = ocr.ocr_pdf(DARKKNIGHT, tmp_path / "1.json", "unknown", pages=[10], engine=engine)
    assert res["lang"] == "en"
    s = _find(res, "Why wouldn't he come?")
    assert s is not None and abs(s["box"][0] - 705) <= 30            # 대사 칸
    s = _find(res, "RAMIREZ")
    assert s is not None and abs(s["box"][0] - 990) <= 30            # 화자 칸
    s = _find(res, "PARKING GARAGE")
    assert s is not None and abs(s["box"][0] - 459) <= 30            # 씬 머리
    assert res["conf_mean"] > 0.95


def test_ocr_failed_page_kept(tmp_path, engine):
    _need(NOCOUNTRY)
    res = ocr.ocr_pdf(NOCOUNTRY, tmp_path / "nc.json", "en", pages=[1, 2], engine=engine)
    assert res["failed_pages"] == [2]
    p2 = [p for p in res["pages"] if p["page"] == 2][0]
    assert p2["lines"] == [] and p2["render"] == "failed" and p2["width"] > 0 and p2["height"] > 0
    assert _find(res, "No Country") is not None


# ---------------------------------------------------------------- 쪽 방향·두 쪽 펼침

GBU = KO / "_좋은 놈 나쁜 놈 이상한 놈_ 시나리오 [55777].pdf"


def test_split_two_up_synthetic():
    import numpy as np
    img = np.full((1000, 1600), 255, np.uint8)
    for y in range(100, 900, 40):          # 왼쪽 쪽 글줄, 오른쪽 쪽 글줄
        img[y:y + 15, 100:700] = 0
        img[y:y + 15, 900:1500] = 0
    (a0, a1), (b0, b1) = ocr.split_two_up(img)
    assert a0 == 0 and b1 == 1600 and 700 <= a1 == b0 <= 900
    # 세로로 긴 쪽, 가운데에도 글이 있는 쪽은 나누지 않는다
    assert ocr.split_two_up(img[:, :800].copy()) == [(0, 800)]
    full = np.full((1000, 1600), 255, np.uint8)
    for y in range(100, 900, 40):
        full[y:y + 15, 100:1500] = 0
    assert ocr.split_two_up(full) == [(0, 1600)]


def test_rotated_two_up_scan(engine):
    """좋은 놈 나쁜 놈 이상한 놈: 두 쪽 펼침을 옆으로 눕혀 스캔 -> 90도 돌려 두 쪽으로 나눠 읽는다."""
    _need(GBU)
    res = ocr.ocr_pdf(GBU, None, "ko", pages=[4], engine=engine)
    pg = res["pages"]
    assert [(p["page"], p.get("part"), p.get("rotate")) for p in pg] == [(4, 1, 90), (4, 2, 90)]
    assert pg[0]["crop"][2] == pg[1]["crop"][0] and pg[0]["height"] == 2480
    assert all(len(p["lines"]) >= 25 for p in pg)
    assert res["conf_mean"] > 0.85
    assert _find(res, "도원") is not None


def test_rotation_detect_on_rotated_page(engine):
    """바로 선 쪽을 일부러 눕힌 그림 -> 방향을 찾아 바로 읽는다."""
    _need(BYEONHOIN)
    import numpy as np
    (_, img, _), = list(ocr.render_pages(BYEONHOIN, 300, [5]))
    lying = np.ascontiguousarray(np.rot90(img, k=1))      # 반시계 90도로 눕힘 -> 시계 90도로 돌려야 함
    state: dict = {}
    parts = ocr.read_image(lying, "ko", engine, state)
    assert state.get("rot") == 90 and parts[0]["rotate"] == 90 and len(parts) == 1
    txt = " ".join(l["text"] for l in parts[0]["lines"])
    assert "부동산" in txt and "변호사" in txt
    # 기억한 방향으로 다음 쪽은 바로 읽는다
    parts2 = ocr.read_image(lying, "ko", engine, state)
    assert parts2[0]["rotate"] == 90


def test_blank_page_not_rotated(engine):
    import numpy as np
    state: dict = {}
    parts = ocr.read_image(np.full((3508, 2480), 255, np.uint8), "ko", engine, state)
    assert parts[0]["lines"] == [] and parts[0]["rotate"] == 0 and "rot" not in state


# ---------------------------------------------------------------- run(): catalog -> 대상 고르기, 건너뛰기, 문제 기록

def _row(doc_id, files, fmts, lang="unknown"):
    return {"doc_id": doc_id, "film_id": "local_9999", "category": "ko_film", "status": "pending",
            "files_list": [paths.rel(f) for f in files], "file_formats_list": fmts,
            "real_format": fmts[0] if len(set(fmts)) == 1 else "mixed", "script_lang": lang}


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OCR_CACHE", tmp_path / "cache" / "ocr")
    monkeypatch.setattr(paths, "PROBLEMS", tmp_path / "cache" / "problems")
    monkeypatch.setattr(paths, "PDF_CACHE", tmp_path / "cache" / "pdf")
    monkeypatch.setattr(paths, "OCR_SUPERSEDED", tmp_path / "cache" / "ocr_superseded")
    monkeypatch.setattr(paths, "STATE", tmp_path / "cache" / "state")
    monkeypatch.setattr(ocr, "_CHECK_CACHE", None)      # 글자층 판정 캐시도 tmp 에서 새로
    return tmp_path


def test_plan_targets(sandbox):
    _need(BYEONHOIN, BERLIN, TUNNEL, GUTA)
    # 변환 PDF 자리(cache/pdf/{doc}/0.pdf)에 글자층 깨진 PDF 를 둔다
    conv = paths.PDF_CACHE / "fmconv" / "0.pdf"
    conv.parent.mkdir(parents=True)
    shutil.copyfile(GUTA, conv)
    rows = [
        _row("fmscan", [BYEONHOIN], ["pdf_scan"]),
        _row("fmbroken", [GUTA], ["pdf_text"]),
        _row("fmgood", [TUNNEL], ["pdf_text"]),
        # 글자층 검사는 깨졌지만(띄어쓰기 빠짐) lines 가 글자 간격으로 되살리므로 OCR 하지 않음
        _row("fmrestore", [BERLIN], ["pdf_text"]),
        _row("fmconv", [paths.RAW / "x.hwp"], ["hwp5"]),
        {**_row("fmexcl", [BYEONHOIN], ["pdf_scan"]), "status": "excluded"},
    ]
    jobs = ocr.plan(rows)
    got = {(j["doc_id"], j["k"]): j["why"] for j in jobs}
    assert got == {("fmscan", 0): "pdf_scan", ("fmbroken", 0): "textlayer_broken",
                   ("fmconv", 0): "converted_textlayer_broken"}
    j = [j for j in jobs if j["doc_id"] == "fmbroken"][0]
    assert j["textlayer"]["broken"] and j["pdf"] == GUTA

    # 이미 OCR 결과가 있는 글자층 PDF 는 다시 검사하지 않고 그 결과의 원본·사유로 대상에 넣는다
    util.write_json(paths.OCR_CACHE / "fmbroken" / "0.json",
                    {"source": paths.rel(GUTA), "why": "textlayer_broken", "pages": []})
    jobs2 = {(j["doc_id"], j["k"]): j for j in ocr.plan(rows)}
    assert jobs2[("fmbroken", 0)]["pdf"] == GUTA and "textlayer" not in jobs2[("fmbroken", 0)]
    # --force 면 다시 검사
    assert "textlayer" in {(j["doc_id"], j["k"]): j for j in ocr.plan(rows, force=True)}[("fmbroken", 0)]


def test_run_skip_force_problems(sandbox, monkeypatch, engine):
    _need(BYEONHOIN)
    import sinkprep.catalog as catalog_mod
    rows = [_row("fmscan", [BYEONHOIN], ["pdf_scan"])]
    monkeypatch.setattr(catalog_mod, "load", lambda *a, **k: rows)
    # 시험은 한 쪽만
    monkeypatch.setattr(ocr, "ocr_pdf", functools.partial(ocr.ocr_pdf, pages=[5], engine=engine))
    args = types.SimpleNamespace(docs=None, limit=None, force=False, workers=None, paths=[])

    assert ocr.run(args) == 0
    out = paths.OCR_CACHE / "fmscan" / "0.json"
    assert out.exists()
    first = out.stat().st_mtime_ns
    res = util.read_json(out)
    assert res["why"] == "pdf_scan" and res["pages"][0]["page"] == 5
    assert not (paths.PROBLEMS / "ocr" / "fmscan.json").exists()     # conf 0.89 > 0.85

    # 이미 있으면 건너뜀
    assert ocr.run(args) == 0
    assert out.stat().st_mtime_ns == first

    # --force 면 다시, 기준을 올리면 ocr_low 기록
    monkeypatch.setattr(ocr, "LOW_CONF", 0.999)
    args.force = True
    assert ocr.run(args) == 0
    assert out.stat().st_mtime_ns != first
    probs = util.read_json(paths.PROBLEMS / "ocr" / "fmscan.json")
    assert [p["problem"] for p in probs] == ["ocr_low"]
    p = probs[0]
    assert p["level"] == "doc" and p["doc_id"] == "fmscan" and p["film_id"] == "local_9999"
    assert 0 < p["score"] < 0.999 and set(p) >= {"item_id", "detail", "path"}

    # --docs 로 다른 문서만 고르면 아무것도 안 함
    args.docs = "fmnone"
    assert ocr.run(args) == 0


def test_retire_superseded(sandbox):
    """예전에 글자층이 깨졌다고 OCR 한 결과 중, 이제 lines 가 글자층을 되살릴 수 있는 것은 쓰지 않도록 옮긴다."""
    _need(BERLIN, GUTA, BYEONHOIN)
    rows = [_row("fmrestore", [BERLIN], ["pdf_text"]), _row("fmbroken", [GUTA], ["pdf_text"]),
            _row("fmscan", [BYEONHOIN], ["pdf_scan"])]
    for d, src, why in (("fmrestore", BERLIN, "textlayer_broken"), ("fmbroken", GUTA, "textlayer_broken"),
                        ("fmscan", BYEONHOIN, "pdf_scan")):
        util.write_json(paths.OCR_CACHE / d / "0.json", {"source": paths.rel(src), "why": why, "pages": []})
    util.set_problems("ocr", "fmrestore", [{"level": "doc", "doc_id": "fmrestore", "problem": "ocr_low"}])
    assert ocr.retire_superseded(rows) == ["fmrestore/0"]
    assert not (paths.OCR_CACHE / "fmrestore" / "0.json").exists()
    assert (paths.OCR_SUPERSEDED / "fmrestore" / "0.json").exists()
    assert (paths.OCR_CACHE / "fmbroken" / "0.json").exists() and (paths.OCR_CACHE / "fmscan" / "0.json").exists()
    assert not util.all_problems() or all(p["doc_id"] != "fmrestore" for p in util.all_problems())
    assert ocr.retire_superseded(rows) == []        # 두 번째는 할 일 없음


def test_needs_ocr_cache(sandbox, monkeypatch):
    """글자층 판정은 크기·수정 시각·버전이 같으면 캐시에서(다시 돌릴 때 수백 개 PDF 를 또 검사하지 않음)."""
    _need(TUNNEL)
    from sinkprep import textlayer
    calls = []
    real = textlayer.check
    monkeypatch.setattr(textlayer, "check", lambda p: calls.append(p) or real(p))
    a = ocr.needs_ocr(TUNNEL)
    ocr._save_check_cache()
    monkeypatch.setattr(ocr, "_CHECK_CACHE", None)          # 새 실행처럼 파일에서 다시 읽음
    b = ocr.needs_ocr(TUNNEL)
    assert a == b and a[0] is False and len(calls) == 1
    assert (paths.STATE / "ocr" / "textlayer_check.json").exists()
