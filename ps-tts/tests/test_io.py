"""io.py: 입력 읽기(csv·tsv·txt, 행 번호 오류)와 결과 파일 쓰기."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pstts.config import ROOT, Config
from pstts.io import (RESULT_COLUMNS, TEMPLATE_HEADER, candidates_rows, csv_rows, read_lines, report_markdown,
                      write_outputs)
from pstts.llm import FileBackend
from pstts.pipeline import RunInfo
from pstts.records import Candidate, LineItem, LineResult, Target

HEADER = ",".join(TEMPLATE_HEADER)
TEMPLATE_ROW = "practice_001,train,practice_dinner,dinner_01,I'll pay half.,,친구끼리 식사,반말,짧게,6,on,,,,화자가 절반을 부담한다"


def write(path: Path, text: str, encoding: str = "utf-8-sig") -> Path:
    path.write_text(text, encoding=encoding)
    return path


# ---------------------------------------------------------------- 읽기

def test_template_row_is_read(tmp_path: Path) -> None:
    # 'C:/Users/jongh/Desktop/Study/메모/더빙 실습 코드/대사 양식.csv' 와 같은 머리글·예시 행
    items, columns = read_lines(write(tmp_path / "t.csv", f"{HEADER}\n{TEMPLATE_ROW}\n"))
    assert columns == list(TEMPLATE_HEADER)
    (it,) = items
    assert (it.id, it.source_en, it.relationship, it.register, it.target_syllables, it.lip) == (
        "practice_001", "I'll pay half.", "친구끼리 식사", "반말", 6, True)
    assert (it.prev_en, it.draft_ko, it.meaning_notes, it.seconds) == ("", None, "화자가 절반을 부담한다", None)
    assert it.extra == {"split": "train", "work_id": "practice_dinner", "scene_id": "dinner_01",
                        "length_request": "짧게", "draft_model": "", "approved_ko": ""}


def test_minimal_csv_defaults_and_auto_ids(tmp_path: Path) -> None:
    text = "source_en,seconds,draft_ko\nHello there.,1.25,\n\n,,\nGo.,,가!\n"
    items, columns = read_lines(write(tmp_path / "m.csv", text))
    assert columns == ["source_en", "seconds", "draft_ko"]
    assert [it.id for it in items] == ["L0001", "L0002"]          # 빈 행은 건너뛰고 번호를 매긴다
    assert (items[0].register, items[0].lip, items[0].seconds, items[0].draft_ko) == ("미지정", True, 1.25, None)
    assert items[1].draft_ko == "가!"


def test_tsv_and_txt(tmp_path: Path) -> None:
    tsv = write(tmp_path / "l.tsv", "id\tsource_en\tregister\tlip\nx1\tHi, you.\t해요체\tOFF\n")
    (it,), columns = read_lines(tsv)
    assert (it.id, it.source_en, it.register, it.lip) == ("x1", "Hi, you.", "해요체", False)
    txt = write(tmp_path / "l.txt", "First line.\n\n   Second   line.  \n")
    items, columns = read_lines(txt)
    assert [(it.id, it.source_en) for it in items] == [("L0001", "First line."), ("L0002", "Second line.")]
    assert columns == ["source_en"]


@pytest.mark.parametrize("row, words", [
    ("a,Hi.,존댓말,,on", ["3행", "register", "존댓말"]),
    ("a,Hi.,,0,on", ["3행", "target_syllables"]),
    ("a,Hi.,,2.5,on", ["3행", "target_syllables"]),
    ("a,Hi.,,abc,on", ["3행", "target_syllables"]),
    ("a,Hi.,,,maybe", ["3행", "lip"]),
    ("a,,,,on", ["3행", "source_en"]),
    ("a,Hi.,,,on,extra", ["3행", "칸 수"]),
])
def test_bad_values_name_the_row(tmp_path: Path, row: str, words: list[str]) -> None:
    text = "id,source_en,register,target_syllables,lip\nok,Fine.,,,on\n" + row + "\n"
    with pytest.raises(ValueError) as e:
        read_lines(write(tmp_path / "bad.csv", text))
    for w in words:
        assert w in str(e.value)


def test_bad_seconds_and_duplicates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2행: seconds"):
        read_lines(write(tmp_path / "s.csv", "source_en,seconds\nHi.,-1\n"))
    with pytest.raises(ValueError, match="같은 id"):
        read_lines(write(tmp_path / "d.csv", "id,source_en\na,Hi.\na,Bye.\n"))
    with pytest.raises(ValueError, match="source_en 칸이 없습니다"):
        read_lines(write(tmp_path / "h.csv", "id,text\na,Hi.\n"))
    with pytest.raises(ValueError, match="겹치거나"):
        read_lines(write(tmp_path / "h2.csv", "source_en,source_en\nHi.,Hi.\n"))
    with pytest.raises(ValueError, match="대사가 하나도"):
        read_lines(write(tmp_path / "e.csv", "source_en\n"))
    with pytest.raises(ValueError, match=".csv"):
        read_lines(write(tmp_path / "x.json", "[]"))
    with pytest.raises(FileNotFoundError):
        read_lines(tmp_path / "없음.csv")


def test_tsv_keeps_quotes_as_text(tmp_path: Path) -> None:
    # TSV 에는 따옴표 규칙이 없다. 따옴표로 시작하는 대사가 따옴표를 잃거나 다음 행을 삼키면 안 된다(2026-11-22 검토)
    (it,), _ = read_lines(write(tmp_path / "q1.tsv", 'id\tsource_en\na\t"Run," she said.\n'))
    assert it.source_en == '"Run," she said.'
    items, _ = read_lines(write(tmp_path / "q2.tsv", 'id\tsource_en\na\t"Run!\nb\tStop right there.\nc\tWho are you?\n'))
    assert [(it.id, it.source_en) for it in items] == [("a", '"Run!'), ("b", "Stop right there."), ("c", "Who are you?")]


def test_csv_with_broken_quotes_names_the_row(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2행: 따옴표가 맞지 않습니다"):
        read_lines(write(tmp_path / "q.csv", 'id,source_en\na,"Run,\nb,Stop right there.\n'))
    with pytest.raises(ValueError, match="따옴표"):
        read_lines(write(tmp_path / "q2.csv", 'id,source_en\na,"Run" she said.\n'))
    # 제대로 감싼 칸(쉼표·큰따옴표 두 번)은 그대로 읽는다
    (it,), _ = read_lines(write(tmp_path / "ok.csv", 'id,source_en\na,"""Run,"" she said."\n'))
    assert it.source_en == '"Run," she said.'


def test_cp949_file_gets_a_korean_hint(tmp_path: Path) -> None:
    for name in ("e.csv", "e.tsv", "e.txt"):
        sep = "\t" if name.endswith(".tsv") else ","
        path = write(tmp_path / name, f"id{sep}source_en{sep}relationship\na{sep}Hi.{sep}친구끼리\n", encoding="cp949")
        with pytest.raises(ValueError, match="UTF-8 로 읽을 수 없습니다.*CSV UTF-8"):
            read_lines(path)


def test_auto_ids_skip_ids_written_by_hand(tmp_path: Path) -> None:
    items, _ = read_lines(write(tmp_path / "m.csv", "id,source_en\nL0002,Hello.\n,Bye.\n,Again.\nL0004,End.\n"))
    assert [it.id for it in items] == ["L0002", "L0003", "L0005", "L0004"]
    with pytest.raises(ValueError, match="같은 id 가 두 번 있습니다: x"):
        read_lines(write(tmp_path / "d.csv", "id,source_en\nx,Hi.\n,Bye.\nx,Hey.\n"))


def test_example_file_covers_the_cases() -> None:
    items, columns = read_lines(ROOT / "examples" / "lines.csv")
    assert columns == list(TEMPLATE_HEADER) and len(items) == 8
    regs = [it.register for it in items]
    assert {"반말", "해요체", "하십시오체", "미지정"} <= set(regs)
    assert any(it.register == "미지정" and it.prev_en for it in items)
    assert any(not it.lip for it in items)
    assert any(it.target_syllables and it.meaning_notes for it in items)
    assert any(len(it.source_en.split()) <= 2 for it in items)                     # 아주 짧은 대사
    assert any(it.source_en.count(". ") >= 1 and len(it.source_en) > 60 for it in items)   # 긴 두 문장
    assert any(it.register == "하십시오체" and any(ch.isdigit() for ch in it.source_en) for it in items)


# ---------------------------------------------------------------- 쓰기

def _result() -> tuple[list[LineResult], RunInfo]:
    item = LineItem("A", "I'll pay half.", register="반말", relationship="친구", target_syllables=5,
                    extra={"split": "dev", "scene_id": "s1"})
    target = Target(n=5, tol=2, en_syllables=3, basis="explicit", seconds_est=5 / Config().ko_rate)
    draft = Candidate("제가 절반을 내겠습니다.", "model1", 0, syllables=10, labse=0.9, ok_length=False,
                      ok_meaning=True, ok_register=False, register_found=["하십시오체"],
                      reasons=["길이 10음절(목표 5±2)", "말투: 반말 요청인데 하십시오체 문장"], dtw=40.0)
    best = Candidate("내가 반 낼게.", "llm", 1, syllables=5, labse=0.91, ok_length=True, ok_meaning=True,
                     ok_register=True, register_found=["반말"], dtw=20.0, dtw_inv=1.0, comet=0.3, score=1.72)
    other = Candidate("반은 | 내가 낼게.", "llm", 2, syllables=6, labse=0.88, ok_length=True, ok_meaning=True,
                      ok_register=True, register_found=["반말"], dtw=30.0, dtw_inv=0.0, comet=0.6, score=0.24)
    res = LineResult(item=item, target=target, draft_ko=draft.text, candidates=[draft, best, other],
                     selected=best, status="ok", rounds_used=2, message="")
    info = RunInfo(started_at="2026-11-22T00:00:00", finished_at="2026-11-22T00:00:10", device="cpu",
                   seconds={"model1": 1.0, "total": 10.0}, llm_backend="claude-cli", llm_calls=2,
                   llm_cache_hits=1, status_counts={"ok": 1},
                   rounds=[{"round": 1, "lines": 1, "requested": 20, "new": 1, "new_pass": 1, "calls": 1,
                            "cache_hits": 0, "llm_seconds": 3.0, "labse_seconds": 0.1}])
    return [res], info


def test_write_outputs(tmp_path: Path) -> None:
    results, info = _result()
    columns = list(TEMPLATE_HEADER)
    paths = write_outputs(tmp_path / "o", results, info, Config(), columns, "시험.csv")

    raw = paths["results.csv"].read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")                       # utf-8-sig(엑셀에서 한글이 안 깨짐)
    with paths["results.csv"].open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header, row = rows[0], rows[1]
    rest = [c for c in TEMPLATE_HEADER if c not in RESULT_COLUMNS]
    assert header == list(RESULT_COLUMNS) + rest
    got = dict(zip(header, row))
    assert (got["final_ko"], got["final_syllables"], got["final_register"], got["status"], got["n_candidates"],
            got["n_pass"]) == ("내가 반 낼게.", "5", "반말", "ok", "3", "2")
    assert (got["labse"], got["comet"], got["dtw"], got["score"]) == ("0.9100", "0.3000", "20.000", "1.7200")
    assert (got["target"], got["tol"], got["target_syllables"], got["lip"]) == ("5", "2", "5", "on")
    assert (got["split"], got["scene_id"], got["relationship"]) == ("dev", "s1", "친구")

    line = json.loads(paths["results.jsonl"].read_text(encoding="utf-8").splitlines()[0])
    assert line["final_ko"] == "내가 반 낼게." and line["selected_index"] == 1
    assert line["target"]["low"] == 3 and line["target"]["high"] == 7
    assert [c["passed"] for c in line["candidates"]] == [False, True, True]
    assert line["candidates"][0]["reasons"][0] == "길이 10음절(목표 5±2)"

    run = json.loads(paths["run.json"].read_text(encoding="utf-8"))
    assert run["seconds_per_line"] == 10.0 and run["llm"]["calls"] == 2
    assert run["config"]["coef"] == Config().coef and run["versions"]["pstts"]
    assert set(run["outputs"]) == {"results.jsonl", "results.csv", "report.md", "candidates.jsonl", "run.json"}

    rows = [json.loads(x) for x in paths["candidates.jsonl"].read_text(encoding="utf-8").splitlines()]
    assert rows == [{"id": "A", "round": 1, "candidates": ["내가 반 낼게."]},
                    {"id": "A", "round": 2, "candidates": ["반은 | 내가 낼게."]}]
    back = FileBackend(paths["candidates.jsonl"])          # FileBackend 가 그대로 읽는다
    assert back.table[("A", 2)] == ["반은 | 내가 낼게."]


def test_report_markdown() -> None:
    results, info = _result()
    md = report_markdown(results, info, Config(), "시험.csv")
    assert "# PS-TTS 텍스트판 결과" in md and "## A (고름)" in md
    assert "고른 대사: **내가 반 낼게.**" in md
    assert "목표 5음절(허용 3~7, 사람이 정함)" in md
    draft_line = next(ln for ln in md.splitlines() if ln.startswith("- 초벌"))
    # 출처는 괄호 앞에 한 번만 적는다
    assert draft_line.startswith("- 초벌(모델 1 초벌): 제가 절반을 내겠습니다. (10음절, 하십시오체, LaBSE 0.900, DTW 40.0, 탈락")
    assert draft_line.count("모델 1 초벌") == 1
    assert "DTW 반경 26프레임" in md
    assert "반은 \\| 내가 낼게." in md                          # 표·목록이 깨지지 않게 | 를 바꾼다
    assert "탈락 까닭: 길이 1개, 말투 1개" in md
    assert "1차: 1줄 요청 20개" in md
    for mark in ("—", "→", "->"):
        assert mark not in md


def test_locked_output_changes_nothing(tmp_path: Path) -> None:
    # 엑셀이 results.csv 를 열고 있으면 바꿀 수 없다. 다섯 파일 모두 예전 그대로 두고 한국어로 알린다(2026-11-22 검토)
    results, info = _result()
    out = tmp_path / "o"
    paths = write_outputs(out, results, info, Config(), ["source_en"], "처음")
    before = {p.name: p.read_bytes() for p in paths.values()}
    results[0].selected = results[0].candidates[2]              # 결과를 바꿔 다시 쓰려 한다
    with paths["results.csv"].open("rb"):                       # 열어 둔 파일은 Windows 에서 이름을 바꿀 수 없다
        with pytest.raises(OSError, match="results.csv 를 바꿀 수 없습니다.*닫고 다시 돌리세요"):
            write_outputs(out, results, info, Config(), ["source_en"], "다시")
    assert {p.name: p.read_bytes() for p in paths.values()} == before
    assert sorted(p.name for p in out.iterdir()) == sorted(before)       # .tmp, .old 가 남지 않는다
    write_outputs(out, results, info, Config(), ["source_en"], "다시")   # 닫은 뒤에는 바뀐다
    assert "반은 \\| 내가 낼게." in paths["report.md"].read_text(encoding="utf-8").split("## A")[0]


def test_csv_rows_without_selection() -> None:
    results, _ = _result()
    r = results[0]
    r.selected, r.status, r.message = None, "error", "후보가 하나도 없음"
    header, rows = csv_rows([r], ["source_en"])
    got = dict(zip(header, rows[0]))
    assert (got["final_ko"], got["labse"], got["status"], got["message"]) == ("", "", "error", "후보가 하나도 없음")
    assert candidates_rows([r])[0]["round"] == 1
