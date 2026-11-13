import csv

from sinkprep import paths, review, util


def test_detect_lang():
    ko = ["형사가 서류를 책상에 던진다. 너 어제 어디 있었어?"] * 20
    en = ["The man walks into the room and he looks at the window. What is it?"] * 20
    fr = ["Elle est dans la maison et je ne sais pas qui est le chef des enfants."] * 20
    assert review.detect_lang(ko) == "ko"
    assert review.detect_lang(en) == "en"
    assert review.detect_lang(fr) == "fr"
    assert review.detect_lang(ko[:10] + en[:10]) == "mixed"
    assert review.detect_lang(["짧다"]) == "unknown"


def test_collect_problems_filters_and_sorts(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PROBLEMS", tmp_path / "problems")
    util.set_problems("catalog", "local_0001", [{"level": "film", "doc_id": "", "film_id": "local_0001",
                                                 "item_id": "", "problem": "other", "score": None,
                                                 "detail": "KMDb/TMDB 조회 불가(키 없음)", "path": ""}])
    util.set_problems("parse", "fm2", [{"level": "doc", "doc_id": "fm2", "film_id": None, "item_id": None,
                                        "problem": "speaker_missing", "score": 0.2, "detail": "화자 없음", "path": "p"}])
    util.set_problems("lines", "fm1", [{"level": "doc", "doc_id": "fm1", "film_id": None, "item_id": None,
                                        "problem": "extract_broken", "score": 0, "detail": "없음", "path": "q"}])
    util.set_problems("lines", "fm9", [{"level": "doc", "doc_id": "fm9", "film_id": None, "item_id": None,
                                        "problem": "extract_broken", "score": 0, "detail": "제외 문서", "path": ""}])
    rows = [{"doc_id": "fm1", "film_id": "local_0001", "status": "parsed"},
            {"doc_id": "fm2", "film_id": "local_0002", "status": "parsed"},
            {"doc_id": "fm9", "film_id": "", "status": "excluded"}]
    out = review.collect_problems(rows)
    assert [r["problem"] for r in out] == ["extract_broken", "speaker_missing"]
    assert out[0]["film_id"] == "local_0001" and out[1]["film_id"] == "local_0002"


def test_write_needs_review_header(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "NEEDS_REVIEW", tmp_path / "qa" / "needs_review.csv")
    review.write_needs_review([{"level": "doc", "doc_id": "fm1", "film_id": "local_0001", "item_id": "",
                                "problem": "ocr_low", "score": "0.7", "detail": "낮음", "path": "x"}])
    with open(paths.NEEDS_REVIEW, encoding="utf-8-sig") as f:
        r = list(csv.DictReader(f))
    assert list(r[0].keys()) == list(review.COLUMNS)
    assert r[0]["problem"] == "ocr_low"


def test_catalog_unreadable_flag_dropped_when_ocr_exists(tmp_path, monkeypatch):
    """pdf_broken 이라 catalog 가 '글자를 꺼낼 수 없음'으로 적었어도 OCR 결과가 있으면 그 판정은 목록에서 뺀다."""
    monkeypatch.setattr(paths, "PROBLEMS", tmp_path / "problems")
    monkeypatch.setattr(paths, "OCR_CACHE", tmp_path / "ocr")
    for d in ("fm1", "fm2"):
        util.set_problems("catalog", d, [{"level": "doc", "doc_id": d, "film_id": "", "item_id": "",
                                          "problem": "extract_broken", "score": 0.0,
                                          "detail": "PDF 를 열거나 글자를 꺼낼 수 없음(catalog 판별)", "path": f"x/{d}.pdf"}])
    util.write_json(paths.OCR_CACHE / "fm1" / "0.json", {"pages": []})
    rows = [{"doc_id": "fm1", "film_id": "", "status": "parsed", "files_list": ["x/fm1.pdf"]},
            {"doc_id": "fm2", "film_id": "", "status": "parsed", "files_list": ["x/fm2.pdf"]}]
    assert [r["doc_id"] for r in review.collect_problems(rows)] == ["fm2"]
