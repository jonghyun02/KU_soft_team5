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


def _catalog_with_docs(tmp_path, monkeypatch, rows):
    """임시 catalog 와 문서(줄 하나, script.json)를 만든다. rows: [(doc_id, film_id, status, match, is_primary)]."""
    from sinkprep import catalog

    monkeypatch.setattr(paths, "CATALOG", tmp_path / "catalog.csv")
    monkeypatch.setattr(paths, "DOCS", tmp_path / "docs")
    out = []
    for doc_id, film_id, status, match, primary in rows:
        r = {k: "" for k in catalog.FIELDS}
        r.update(doc_id=doc_id, post_id=doc_id[2:], category="ko_film", files=f"raw/{doc_id}.hwp", sha1="1" * 40,
                 real_format="hwp5", title_raw="시험", title_norm="시험", film_id=film_id, film_title="시험",
                 version_label="unknown", is_primary="true" if primary else "false", status=status, match=match)
        out.append(r)
        util.write_jsonl(paths.DOCS / doc_id / "lines.jsonl", [{"schema": "sink.lines/1"}, {"n": 0, "text": "형사가 묻는다.",
                                                                                           "src": "hwp", "conf": None}])
        util.write_json(paths.DOCS / doc_id / "script.json", {"parser": {"name": "sink-parse"}, "quality": {"score": 0.9}})
    catalog.write_catalog(out)
    return catalog.load()


def test_finalize_catalog_keeps_aligned_and_done(tmp_path, monkeypatch):
    """맞추기·최종본 단계가 올린 status(aligned, done)는 review 가 parsed 로 되돌리지 않는다."""
    from sinkprep import catalog

    rows = _catalog_with_docs(tmp_path, monkeypatch, [("fm1", "local_0001", "aligned", "0.812", True),
                                                      ("fm2", "local_0002", "done", "0.700", True),
                                                      ("fm3", "local_0003", "pending", "", True)])
    review.finalize_catalog(rows)
    got = {r["doc_id"]: r for r in catalog.load()}
    assert [got[d]["status"] for d in ("fm1", "fm2", "fm3")] == ["aligned", "done", "parsed"]
    assert got["fm1"]["match"] == "0.812" and got["fm1"]["is_primary"] == "true"
    (paths.DOCS / "fm1" / "script.json").unlink()                    # 대본이 사라지면 pending
    review.finalize_catalog(catalog.load())
    assert {r["doc_id"]: r for r in catalog.load()}["fm1"]["status"] == "pending"


def test_primary_follows_match_when_docs_were_aligned(tmp_path, monkeypatch):
    """같은 영화의 문서 가운데 영상과 맞춰 본 것이 있으면 일치율이 가장 높은 문서가 대표다."""
    from sinkprep import catalog

    rows = _catalog_with_docs(tmp_path, monkeypatch, [("fm1", "local_0001", "aligned", "0.412", True),
                                                      ("fm2", "local_0001", "aligned", "0.803", False),
                                                      ("fm3", "local_0001", "parsed", "", False)])
    review.finalize_catalog(rows)
    got = {r["doc_id"]: r for r in catalog.load()}
    assert [got[d]["is_primary"] for d in ("fm1", "fm2", "fm3")] == ["false", "true", "false"]
