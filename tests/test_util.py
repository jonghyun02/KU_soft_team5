import json
import unicodedata

import pytest

from sinkprep import paths, util


def test_nfc_joins_decomposed_hangul():
    nfd = unicodedata.normalize("NFD", "한글 대본")
    assert nfd != "한글 대본"
    assert util.nfc(nfd) == "한글 대본"
    assert util.nfc(None) is None


def test_write_json_atomic_roundtrip(tmp_path):
    p = tmp_path / "a" / "b.json"
    util.write_json(p, {"글": "값", "n": 1})
    assert util.read_json(p) == {"글": "값", "n": 1}
    # 임시 파일이 남지 않음
    assert [x.name for x in p.parent.iterdir()] == ["b.json"]


def test_write_jsonl_roundtrip(tmp_path):
    p = tmp_path / "x.jsonl"
    rows = [{"n": 0, "text": "가"}, {"n": 1, "text": "나"}]
    util.write_jsonl(p, rows)
    assert util.read_jsonl(p) == rows
    assert p.read_text(encoding="utf-8").count("\n") == 2


def test_refuses_to_write_into_raw():
    with pytest.raises(PermissionError):
        util.write_json(paths.RAW / "should_not_exist.json", {})
    assert not (paths.RAW / "should_not_exist.json").exists()


def test_failed_write_keeps_old_file(tmp_path):
    p = tmp_path / "keep.json"
    util.write_json(p, {"v": 1})

    class Bad:
        pass

    with pytest.raises(TypeError):
        util.write_json(p, {"v": Bad()})
    assert json.loads(p.read_text(encoding="utf-8")) == {"v": 1}
