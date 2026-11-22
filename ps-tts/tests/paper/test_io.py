"""논문 판 io: results.csv 칸과 값, results.jsonl, report.md, candidates.jsonl, run.json, 한꺼번에 바꾸기, 문장 하나 출력.

결과는 손으로 만든 PaperLine 으로 시험한다(파이프라인 시험과 따로).
"""
from __future__ import annotations

import csv
import io as stdio
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from pstts.records import LineItem
from pstts.paper.config import PaperConfig
from pstts.paper.io import (OUTPUT_NAMES, RESULT_COLUMNS, csv_rows, flat_values, print_choice, report_markdown,
                            result_dict, run_dict, summary_line, write_outputs)
from pstts.paper.iso import IsoCandidate, IsoLine
from pstts.paper.pipeline import PaperLine, PaperRunInfo
from pstts.paper.prompts import PROMPT_VERSION_PAPER
from pstts.paper.ps import PsCandidate, PsChoice

CFG = PaperConfig(device="cpu")


def _iso_cand(text: str, origin: str, rnd: int, it: int | None, k: float, labse: float, ok_len: bool,
              ok_sem: bool) -> IsoCandidate:
    return IsoCandidate(text=text, origin=origin, round=rnd, iteration=it, checked=it is not None, dur_sec=k * 0.0116,
                        k_frames=k, ok_len=ok_len, labse=labse, ok_sem=ok_sem, passed=ok_len and ok_sem)


def _ok_line() -> PaperLine:
    item = LineItem(id="ex_1", source_en="We have to leave before the storm hits.", register="해요체",
                    extra={"split": "dev"})
    iso = IsoLine(id="ex_1", source_en=item.source_en, draft_ko="우리는 폭풍이 치기 전에 떠나야 합니다.", status="paraphrase",
                  target="폭풍 오기 전에 떠나야 해요.", selected=1, iterations=1, k_prime=125.23, k_prime_basis="tts",
                  r=1.0, src_dur_sec=1.454, length_ratio=0.72, rounds=1,
                  candidates=[_iso_cand("우리는 폭풍이 치기 전에 떠나야 합니다.", "draft", 0, 0, 173.9, 0.91, False, True),
                              _iso_cand("폭풍 오기 전에 떠나야 해요.", "llm", 1, 1, 119.6, 0.88, True, True),
                              _iso_cand("폭풍 전에 가야 해요.", "llm", 1, None, 86.9, 0.8, True, True)])
    cands = [PsCandidate("폭풍이 오기 전에 떠나야 해요.", 1.51, 130, 61.5, 1.0, 0.20, 1.68),
             PsCandidate("태풍 오기 전에 떠나요.", 1.14, 98, 70.0, 0.0, 0.90, 0.36),
             PsCandidate("폭풍 전에 출발해요.", 1.01, 87, 65.75, 0.5, 0.95, 1.18),
             PsCandidate("PVC.", 0.0, 0, None, None, 0.1, None)]
    return PaperLine(item=item, iso=iso, ps_candidates=cands, ps_rounds=[1, 1, 1, 2], n_src_frames=125,
                     choice=PsChoice(ps=0, ps_comet=0, excluded=[3]), status="ok",
                     message="프레임이 0개라 PS 고르기에서 뺀 후보 1개")


def _error_line() -> PaperLine:
    item = LineItem(id="ex_2", source_en="Hello there.", lip=False)
    iso = IsoLine(id="ex_2", source_en=item.source_en, draft_ko="", status="error", k_prime=40.0,
                  k_prime_basis="tts", src_dur_sec=0.46, message="초벌(T0)이 비어 ISO 를 할 수 없음")
    return PaperLine(item=item, iso=iso, status="error", message=iso.message)


def _info() -> PaperRunInfo:
    return PaperRunInfo(started_at="2026-11-22T05:00:00", finished_at="2026-11-22T05:01:00", device="cpu",
                        seconds={"model1": 2.0, "iso_llm": 3.0, "ps_llm": 5.0, "comet": 9.0, "total": 20.0},
                        llm_backend="claude-cli", llm={"iso": {"calls": 1, "cache_hits": 0},
                                                       "ps": {"calls": 2, "cache_hits": 1}},
                        iso_rounds=[{"round": 1, "lines": 1, "requested": 10, "new": 10, "checked": 1, "passed": 1,
                                     "calls": 1, "cache_hits": 0, "llm_seconds": 3.0, "labse_seconds": 0.1,
                                     "tts_seconds": 0.2}],
                        ps_rounds=[{"round": 1, "lines": 1, "requested": 60, "new": 3, "calls": 1, "cache_hits": 0,
                                    "llm_seconds": 4.0}],
                        frontend={"name": "approx", "en_rate": 6.19, "ko_rate": 7.9232, "null_cost": 1.0,
                                  "weights": {"vowel": 2.0, "ko_glide_vowel": 2.5, "consonant": 1.0, "glide": 0.5},
                                  "vowel_table": {"symbols": list("aeiou"), "sha1": "abc123"}},
                        unused_columns={"register": 1, "lip": 1},
                        rows=[{"id": "ex_1", "stage": "iso", "round": 1, "candidates": ["폭풍 오기 전에 떠나야 해요."]},
                              {"id": "ex_1", "stage": "ps", "round": 1, "candidates": ["폭풍이 오기 전에 떠나야 해요."]}],
                        status_counts={"ok": 1, "error": 1},
                        iso_counts={"draft": 0, "paraphrase": 1, "fallback": 0, "error": 1})


def test_csv_columns_order_values_and_rest_of_input() -> None:
    header, rows = csv_rows([_ok_line(), _error_line()], ["id", "split", "source_en", "register", "draft_ko"])
    assert header == [*RESULT_COLUMNS, "split", "register"]
    ok = dict(zip(header, rows[0]))
    assert ok["k_prime"] == "125.23" and ok["k_prime_basis"] == "tts" and ok["iso_status"] == "paraphrase"
    assert ok["iso_ko"] == "폭풍 오기 전에 떠나야 해요." and ok["iso_iterations"] == 1 and ok["iso_k"] == "119.60"
    assert ok["iso_labse"] == "0.8800" and ok["ps_ko"] == "폭풍이 오기 전에 떠나야 해요." and ok["ps_dtw"] == "61.500"
    assert ok["final_ko"] == "폭풍이 오기 전에 떠나야 해요." and ok["ps_comet_score"] == "1.6800"
    assert ok["ps_comet_dtw"] == "61.500" and ok["ps_comet_comet"] == "0.2000" and ok["n_ps_candidates"] == 4
    assert ok["split"] == "dev" and ok["register"] == "해요체"
    err = dict(zip(header, rows[1]))
    assert err["iso_ko"] == "" and err["final_ko"] == "" and err["iso_k"] == "" and err["n_ps_candidates"] == 0
    assert "초벌(T0)이 비어" in err["message"]


def test_results_jsonl_has_all_candidates_and_indices() -> None:
    d = result_dict(_ok_line())
    assert {k: d[k] for k in RESULT_COLUMNS} == flat_values(_ok_line())
    assert d["status"] == "ok" and d["draft_origin"] == "model1" and d["length_ratio"] == 0.72
    assert [c["text"] for c in d["iso"]["candidates"]][1] == "폭풍 오기 전에 떠나야 해요."
    assert d["iso"]["selected_index"] == 1 and d["iso"]["candidates"][2]["iteration"] is None
    assert set(d["ps"]["candidates"][0]) == {"text", "round", "dur_sec", "n_frames", "dtw", "dtw_inv", "comet", "score"}
    assert [c["round"] for c in d["ps"]["candidates"]] == [1, 1, 1, 2]
    assert d["ps"]["index"] == 0 and d["ps"]["excluded"] == [3] and d["ps_comet"]["index"] == 0
    assert d["ps"]["candidates"][3]["dtw"] is None and d["input_unused"]["register"] == "해요체"
    assert d["extra"] == {"split": "dev"}
    json.dumps(d, ensure_ascii=False)
    e = result_dict(_error_line())
    assert e["ps"]["candidates"] == [] and e["ps"]["index"] is None and e["final_ko"] == ""


def test_report_has_settings_and_each_line() -> None:
    md = report_markdown([_ok_line(), _error_line()], _info(), CFG, "examples\\lines.csv")
    assert md.startswith("# PS-TTS 논문 그대로 판 결과")
    for words in ("PS-Comet 고름 1줄, 실패 1줄", "ISO 호출 1번(캐시 0번), PS 호출 2번(캐시 1번)", "근사(approx)",
                  "식 6 K' ±26프레임", "LaBSE 0.75 이상", "최대 60번", "PS 후보 60개", "반경 없음(tslearn 기본, 반경 1)",
                  "alpha 1.6", "beta 0.4", "논문 판이 쓰지 않은 입력 칸: register 1줄, lip 1줄", "## ex_1 (고름)",
                  "## ex_2 (실패)", "ISO(바꿔 쓰기, 바꿔 쓴 후보 1개 검사", "PS(식 11, DTW 최소): **폭풍이 오기 전에 떠나야 해요.**",
                  "PS-Comet(식 20, 최종)", "점수 1.680 = 1.6 x 1.000 + 0.4 x 0.200", "PS-Comet 상위 3개:",
                  "PS 후보 4개(1차 3, 2차 1)", "탈락(길이 식 6)", "초벌(T0)이 비어"):
        assert words in md, words
    top = md.split("PS-Comet 상위 3개:")[1].splitlines()[1:4]
    assert [t.split(". ", 1)[1].split(" (")[0] for t in top] == ["폭풍이 오기 전에 떠나야 해요.", "폭풍 전에 출발해요.",
                                                                 "태풍 오기 전에 떠나요."]


def test_run_json_records_versions_stages_and_config() -> None:
    run = run_dict([_ok_line(), _error_line()], _info(), CFG, "x.csv")
    assert run["version"] == "paper" and run["versions"]["paper_prompt"] == PROMPT_VERSION_PAPER
    assert run["llm"]["calls"] == 3 and run["llm"]["cache_hits"] == 1
    assert run["llm"]["by_stage"] == {"iso": {"calls": 1, "cache_hits": 0}, "ps": {"calls": 2, "cache_hits": 1}}
    assert run["seconds_per_line"] == 10.0 and run["config"] == asdict(CFG)
    assert run["frontend"]["name"] == "approx" and run["iso_counts"]["paraphrase"] == 1
    for name in ("phonemizer", "g2pK", "tslearn"):
        assert run["versions"][name]


def test_write_outputs_writes_five_files(tmp_path: Path) -> None:
    lines, info = [_ok_line(), _error_line()], _info()
    paths = write_outputs(tmp_path / "o", lines, info, CFG, ["id", "source_en", "split"], "x.csv")
    assert sorted(p.name for p in paths.values()) == sorted(OUTPUT_NAMES)
    raw = paths["results.csv"].read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(stdio.StringIO(raw.decode("utf-8-sig"))))
    assert rows[0] == [*RESULT_COLUMNS, "split"] and len(rows) == 3
    jl = [json.loads(x) for x in paths["results.jsonl"].read_text(encoding="utf-8").splitlines()]
    assert [x["id"] for x in jl] == ["ex_1", "ex_2"]
    cands = [json.loads(x) for x in paths["candidates.jsonl"].read_text(encoding="utf-8").splitlines()]
    assert cands == info.rows
    run = json.loads(paths["run.json"].read_text(encoding="utf-8"))
    assert run["outputs"]["report.md"].endswith("report.md")
    assert "## ex_1" in paths["report.md"].read_text(encoding="utf-8")


def test_locked_file_keeps_all_old_outputs(tmp_path: Path) -> None:
    """엑셀처럼 results.csv 를 열고 있으면 다섯 파일 모두 예전 그대로 두고 한국어로 알린다(기존 판 _publish)."""
    out = tmp_path / "o"
    paths = write_outputs(out, [_ok_line()], _info(), CFG, ["source_en"], "first")
    before = {n: (out / n).read_bytes() for n in OUTPUT_NAMES}
    with paths["results.csv"].open("rb"):                       # 열어 둔 파일은 Windows 에서 이름을 바꿀 수 없다
        with pytest.raises(OSError, match="results.csv 를 바꿀 수 없습니다"):
            write_outputs(out, [_error_line()], _info(), CFG, ["source_en"], "second")
    assert {n: (out / n).read_bytes() for n in OUTPUT_NAMES} == before
    assert sorted(x.name for x in out.iterdir()) == sorted(OUTPUT_NAMES)      # .tmp, .old 가 남지 않는다


# ---------------------------------------------------------------- 입력 읽기

def test_read_lines_same_as_main_version_for_valid_files(tmp_path: Path) -> None:
    from pstts.config import ROOT
    from pstts.io import read_lines as main_read_lines
    from pstts.paper.io import read_lines

    lines_csv = ROOT / "examples" / "lines.csv"
    assert read_lines(lines_csv) == main_read_lines(lines_csv)
    tsv = tmp_path / "a.tsv"
    tsv.write_text('id\tsource_en\tregister\tseconds\tsplit\n\t"Hi."\t해요체\t1.5\tdev\nx\tGo.\t\t\t\n', encoding="utf-8")
    assert read_lines(tsv) == main_read_lines(tsv)
    txt = tmp_path / "a.txt"
    txt.write_text("Hi.\n\nGo.\n", encoding="utf-8")
    assert read_lines(txt) == main_read_lines(txt)


@pytest.mark.parametrize("content, words", [
    ("id,source_en,seconds\na,Hi.,-1\n", "seconds"),
    ("id,source_en,register\na,,반말\n", "source_en"),
    ("id,text\na,Hi.\n", "source_en 칸이 없습니다"),
    ("id,source_en\na,\"Hi.\n", "따옴표"),
    ("id,source_en\na,Hi.\na,Go.\n", "같은 id"),
    ("id,source_en,source_en\na,Hi.,Go.\n", "겹치거나"),
    ("id,source_en\n", "대사가 하나도"),
])
def test_read_lines_still_checks_used_columns_like_main(tmp_path: Path, content: str, words: str) -> None:
    from pstts.io import read_lines as main_read_lines
    from pstts.paper.io import read_lines

    p = tmp_path / "x.csv"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=words) as ours:
        read_lines(p)
    with pytest.raises(ValueError) as main:
        main_read_lines(p)
    assert str(ours.value) == str(main.value)


def test_read_lines_does_not_stop_on_unused_columns(tmp_path: Path) -> None:
    from pstts.paper.io import input_value, read_lines

    p = tmp_path / "x.csv"
    p.write_text("id,source_en,register,lip,target_syllables,split\na,Hi.,존댓말,yes,1.5,dev\nb,Go.,반말,OFF,9,\n",
                 encoding="utf-8")
    (a, b), columns = read_lines(p)
    assert columns == ["id", "source_en", "register", "lip", "target_syllables", "split"]
    assert (a.register, a.lip, a.target_syllables) == ("미지정", True, None)          # LineItem 칸은 기본값
    assert a.extra == {"split": "dev", "register": "존댓말", "lip": "yes", "target_syllables": "1.5"}
    assert [input_value(a, c) for c in ("register", "lip", "target_syllables", "split")] == ["존댓말", "yes", "1.5", "dev"]
    assert (b.register, b.lip, b.target_syllables, b.extra) == ("반말", False, 9, {"split": ""})
    assert [input_value(b, c) for c in ("register", "lip", "target_syllables")] == ["반말", "off", "9"]
    cp = tmp_path / "cp949.csv"
    cp.write_bytes("id,source_en,register\na,Hi.,반말\n".encode("cp949"))
    with pytest.raises(ValueError, match="UTF-8"):
        read_lines(cp)


def test_print_choice_and_summary() -> None:
    buf = stdio.StringIO()
    print_choice(_ok_line(), CFG, buf)
    text = buf.getvalue()
    for words in ("영어: We have to leave", "K': 125.2프레임(seconds 칸이 없어 원문 TTS 길이 1.454초로 정함, R 1.000)", "초벌 T0:", "ISO(바꿔 쓰기, 검사 1번)", "PS(식 11):",
                  "PS-Comet(식 20, 최종): 폭풍이 오기 전에 떠나야 해요.", "PS-Comet 상위 3개(후보 4개 중):", "메모:"):
        assert words in text, words
    buf = stdio.StringIO()
    print_choice(_error_line(), CFG, buf)
    assert "고른 대사 없음: 초벌(T0)이 비어" in buf.getvalue()
    s = summary_line([_ok_line(), _error_line()], _info())
    assert "PS-Comet 고름 1줄, 실패 1줄" in s and "ISO 1번(캐시 0번), PS 2번(캐시 1번)" in s
