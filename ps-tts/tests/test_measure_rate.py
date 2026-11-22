"""scripts/measure_rate.py: 매니페스트 음절 속도와 dub-rhythm 임시 coef."""
from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path
from types import ModuleType

import pytest

from pstts.phonetics import english_syllable_count

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_rate.py"


@pytest.fixture(scope="module")
def mr() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_rate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_rate"] = module
    spec.loader.exec_module(module)
    return module


HELLO, TICKETS = "Hello world.", "I didn't order two tickets."


def test_known_english_counts() -> None:
    # 아래 기대값의 바탕: hə.loʊ.wɜld = 3, aɪ.dɪdnt.ɔɹ.də.tu.tɪ.kɪts = 7
    assert english_syllable_count(HELLO) == 3
    assert english_syllable_count(TICKETS) == 7


def test_measure_manifest(mr: ModuleType) -> None:
    rows = [
        {"lang": "en", "text": HELLO, "seconds": "0.5"},       # 6 음절/초
        {"lang": "en", "text": TICKETS, "seconds": "1.0"},     # 7 음절/초
        {"lang": "ko", "text": "안녕하세요", "seconds": "0.5"},  # 10 음절/초
        {"lang": "ko", "text": "고마워", "seconds": "0.5"},      # 6 음절/초
        {"lang": "ko", "text": "쉼만 있는 줄", "seconds": "0"},  # 길이 0 -> 뺌
        {"lang": "EN", "text": "?!", "seconds": "1.0"},         # 음절 0 -> 뺌
    ]
    res = mr.measure_manifest(rows)
    assert res["en"]["n"] == 2 and res["ko"]["n"] == 2
    assert res["en"]["rate_total"] == pytest.approx(10 / 1.5)
    assert res["en"]["rate_median"] == pytest.approx(6.5)
    assert res["ko"]["rate_total"] == pytest.approx(8.0)
    assert res["ko"]["rate_median"] == pytest.approx(8.0)
    assert res["coef_total"] == pytest.approx(8.0 / (10 / 1.5))
    assert res["coef_median"] == pytest.approx(8.0 / 6.5)
    assert res["skipped"] == ["6행", "7행"]


def test_measure_manifest_errors(mr: ModuleType) -> None:
    with pytest.raises(ValueError):
        mr.measure_manifest([{"lang": "fr", "text": "bonjour", "seconds": "1"}])
    with pytest.raises(ValueError):                         # 한국어 줄이 없다
        mr.measure_manifest([{"lang": "en", "text": HELLO, "seconds": "1"}])


@pytest.mark.parametrize("suffix, delim", [(".csv", ","), (".tsv", "\t")])
def test_read_manifest(mr: ModuleType, tmp_path: Path, suffix: str, delim: str) -> None:
    path = tmp_path / f"m{suffix}"
    lines = [delim.join(["lang", "text", "seconds"]), delim.join(["ko", "안녕하세요", "0.5"]),
             delim.join(["en", "Hello", "0.3"])]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    rows = mr.read_manifest(path)
    assert rows == [{"lang": "ko", "text": "안녕하세요", "seconds": "0.5"},
                    {"lang": "en", "text": "Hello", "seconds": "0.3"}]


def test_read_manifest_missing_column(mr: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("lang,text\nko,안녕\n", encoding="utf-8")
    with pytest.raises(ValueError):
        mr.read_manifest(path)


LINES = [
    {"id": "L0", "text_en": HELLO, "n_target": 4, "usable": True, "dur": 0.5},
    {"id": "L1", "text_en": TICKETS, "n_target": 9, "usable": True, "dur": 1.0},
    {"id": "L2", "text_en": "Oh.", "n_target": 2, "usable": False, "dur": 0.3},
    {"id": "L3", "text_en": "...", "n_target": 2, "usable": True, "dur": 0.4},   # 영어 음절 0 -> 뺌
]


def test_dub_rhythm_coef(mr: ModuleType) -> None:
    res = mr.dub_rhythm_coef(LINES)
    assert res["n_lines"] == 2
    assert res["ko_syllables"] == 13 and res["en_syllables"] == 10
    assert res["coef_total"] == pytest.approx(1.3)
    assert res["coef_median"] == pytest.approx(statistics.median([4 / 3, 9 / 7]))
    assert res["en_rate_total"] == pytest.approx(10 / 1.5)
    assert res["skipped"] == ["L3"]
    every = mr.dub_rhythm_coef(LINES, usable_only=False)
    assert every["n_lines"] == 3
    assert every["en_syllables"] == 10 + english_syllable_count("Oh.")


def test_dub_rhythm_coef_no_lines(mr: ModuleType) -> None:
    with pytest.raises(ValueError):
        mr.dub_rhythm_coef([{"id": "x", "text_en": HELLO, "n_target": 3, "usable": False}])


def test_cli_manifest(mr: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "m.csv"
    path.write_text(f"lang,text,seconds\nen,{HELLO},0.5\nko,안녕하세요,0.5\n", encoding="utf-8")
    out_json = tmp_path / "r.json"
    assert mr.main([str(path), "--json", str(out_json)]) == 0
    text = capsys.readouterr().out
    assert "coef(전체 합 비율) = 1.667" in text           # (5/0.5) / (3/0.5)
    assert json.loads(out_json.read_text(encoding="utf-8"))["coef_total"] == pytest.approx(5 / 3)


def test_cli_dub_rhythm(mr: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "lines.json"
    path.write_text(json.dumps(LINES), encoding="utf-8")
    assert mr.main(["--from-dub-rhythm", str(path)]) == 0
    text = capsys.readouterr().out
    assert "쓴 줄 2개" in text and "coef(전체 합 비율) = 1.300" in text


def test_cli_needs_exactly_one_source(mr: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        mr.main([])
    with pytest.raises(SystemExit):
        mr.main([str(tmp_path / "a.csv"), "--from-dub-rhythm", str(tmp_path / "b.json")])
