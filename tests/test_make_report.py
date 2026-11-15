"""tools/make_report.py: 영상 쪽 절(6b)과 수락 기준 C1~C3·D1~D9 줄, 파일이 없을 때 '없음', 경로 바꾸기."""
import argparse
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import align_synth as S  # noqa: E402
import make_report as R  # noqa: E402
from sinkprep import align, catalog, final, paths, util, validate  # noqa: E402

DOC, FID, SHA = "fm1", "local_0001", "0" * 40
FILM = [                                  # tests/test_final.py 의 작은 영화와 같은 꼴
    ("1", "scene", "S#1 공원"),
    ("1", "dialogue", "오늘 날씨가 정말 좋네요."),
    ("1", "dialogue", "그러게요. 산책이나 갈까요?"),
    ("1", "dialogue", "예."),
    ("1", "dialogue", "저는 집에서 책을 읽고 싶어요."),
    ("2", "scene", "S#2 극장 앞"),
    ("2", "onscreen", "3년 뒤"),
    ("2", "dialogue", "어제 본 영화는 어땠습니까?"),
    ("2", "dialogue", "생각보다 많이 지루했습니다."),
    ("2", "transition", "CUT TO"),
    ("3", "scene", "S#3 역"),
    ("3", "action", "두 사람이 걸어온다."),
    ("3", "dialogue", "내일 아침 일찍 출발합시다."),
    ("3", "dialogue", "예."),
    ("3", "dialogue", "기차표는 미리 사 두었어요."),
    ("3", "dialogue", "......."),
]
SAID = ["오늘 날씨가 정말 좋네요.", "그러게요. 산책이나 갈까요?", "예.", "저는 집에서 그냥 책을 읽고 싶습니다.",
        ("누구세요?", {"gap": 2.0}), ("내일 아침 일찍 출발합시다.", {"gap": 2.0}), "예.", "기차표는 미리 사 두었어요."]
ARGS = argparse.Namespace(cmd="final", docs=None, limit=None, force=False, workers=None, paths=[])


def speak(utts, t0=10.0, gap=1.0, p=0.9):
    words, t = [], t0
    for u in utts:
        text, kw = (u, {}) if isinstance(u, str) else u
        t += kw.get("gap", gap)
        for w in text.split():
            words.append((w, round(t, 3), round(t + 0.3, 3), p))
            t += 0.3
    return S.say(words, "plain")


@pytest.fixture
def proj(tmp_path, monkeypatch):
    """임시 저장소: 영상이 있는 영화 local_0001(fm1), 영상이 없는 fm2."""
    for name, sub in (("ROOT", ""), ("RAW", "raw"), ("EXTRACTED", "extracted"), ("DOCS", "docs"), ("FILMS", "films"),
                      ("CACHE", "cache"), ("STATE", "cache/state"), ("PROBLEMS", "cache/problems"),
                      ("AUDIO", "cache/audio"), ("OCR_CACHE", "cache/ocr"), ("OCR_SUPERSEDED", "cache/ocr_superseded"),
                      ("CATALOG", "catalog.csv"), ("QA", "qa"), ("NEEDS_REVIEW", "qa/needs_review.csv"),
                      ("EXAMPLES", "examples")):
        monkeypatch.setattr(paths, name, tmp_path / sub if sub else tmp_path)
    rows = []
    for doc, fid, video in ((DOC, FID, True), ("fm2", "local_0002", False)):
        row = {k: "" for k in catalog.FIELDS}
        row.update(doc_id=doc, post_id=doc[2:], category="ko_film", files=f"raw/{doc}.hwp", sha1="1" * 40,
                   real_format="hwp5", file_formats="hwp5", title_raw="시험", title_norm="시험", script_lang="ko",
                   is_translation="false", version_label="shooting", film_id=fid, film_title="시험 영화",
                   is_primary="true", extract_quality="0.990", status="parsed")
        if video:
            row.update(video_path=f"videos/{fid}.mp4", video_sha1=SHA, video_duration="600.000", video_lang="ko")
        rows.append(row)
    catalog.write_catalog(rows)
    return tmp_path


def put_timemap():
    util.write_json(paths.AUDIO / f"{SHA}.timemap.json",
                    {"video_sha1": SHA, "mode": "map", "frame": 0.021333333, "segments": [[0.0, 300.0], [300.0, 599.9]],
                     "audio_duration": 599.9, "video_duration": 600.0, "gaps": 1, "gap_total": 0.1,
                     "made": "sinkprep.media 0.2.0"})


def put_needs_review():
    paths.QA.mkdir(parents=True, exist_ok=True)
    with open(paths.NEEDS_REVIEW, "w", encoding="utf-8-sig", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(validate.NEEDS_REVIEW_COLUMNS)
        cw.writerow(["film", DOC, FID, "", "absent_many", "0.6", "x", f"films/{FID}/final.json"])
        cw.writerow(["item", DOC, FID, "I0003", "asr_low_conf", "0.1", "x", f"films/{FID}/final.json"])
        cw.writerow(["item", DOC, FID, "I0005", "asr_low_conf", "0.1", "x", f"films/{FID}/final.json"])
        cw.writerow(["doc", "fm2", "local_0002", "", "parse_low_dialogue", "0.1", "x", "docs/fm2/script.json"])


def make_film():
    """대본·받아쓰기를 차리고 align → final 을 돌린다(임시 저장소 안)."""
    script = S.tiny_script(FILM)
    script["doc_id"] = DOC
    util.write_json(paths.DOCS / DOC / "script.json", script)
    util.write_json(paths.FILMS / FID / "work" / "asr.json", S.asr_of(speak(SAID), vad=[[10.0, 30.0]]))
    assert align.run(ARGS) == 0
    assert final.run(ARGS) == 0
    put_timemap()
    put_needs_review()


def section(vres=None):
    vres = vres or R.run_validate(workers=1)
    rows = catalog.load()
    return "\n".join(R.video_section(rows, vres)), "\n".join(R.criteria_cd(rows, vres))


def no_script_text(text: str) -> None:
    """보고에 대사 글이 들어가지 않는다(대사·들린 글 가운데 짧지 않은 것으로 확인)."""
    for t in [x for _, typ, x in FILM if typ == "dialogue"] + [s if isinstance(s, str) else s[0] for s in SAID]:
        if len(t) >= 6:
            assert t not in text, t


def test_video_section_reads_each_stage_from_disk(proj):
    make_film()
    v, _ = section()
    al = util.read_json(paths.FILMS / FID / "work" / "align.json")
    fi = util.read_json(paths.FILMS / FID / "final.json")
    asr = util.read_json(paths.FILMS / FID / "work" / "asr.json")
    st, qd = al["method"]["stats"], fi["quality"]
    assert "### local_0001 (대표 문서 fm1, catalog status done" in v and "local_0002" not in v
    assert "| local_0001 | fm1 | 600.000 |" in v
    # 영상·대응표
    assert f"mode map, 틈 1곳 합 0.100초, 소리 599.900초 → 영상 600.000초, sinkprep.media 0.2.0" in v
    # 받아쓰기
    n = len(asr["passes"][0]["words"])
    assert f"말 구간 1개 합 20.0초, 묶음 0개, pass 1개, 검사 통과" in v
    assert f"`plain`: 낱말 {n}개, 평균 확률 0.900, context -, 모델 synth" in v and "rescue 통계: rescue pass 없음" in v
    # 맞추기: align.json 의 method.stats 그대로
    assert (f"same {st['same']}, changed {st['changed']}, absent {st['absent']}(말 있는 대사 가운데 {st['absent_spoken']}), "
            f"일치율 {st['match']}") in v
    plain = al["method"]["passes"]["plain"]
    assert f"`plain` " in v and f"chosen_lines {plain['chosen_lines']}" in v
    assert f"decided_by: rule {len(al['ops'])}" in v and "다시 짝지은 줄:" in v
    # 최종본: quality 전부, 항목 종류, 씬, 화자, 시각 보정
    for k in qd:
        assert f"{k} " in v.split("quality: ", 1)[1].split("\n", 1)[0], k
    types = {t: sum(i["type"] == t for i in fi["items"]) for t in {i["type"] for i in fi["items"]}}
    assert f"dialogue {types['dialogue']}" in v and f"scene {types['scene']}" in v
    with_span = sum(s["start"] is not None for s in fi["scenes"])
    assert f"씬 {len(fi['scenes'])}개, 구간(start)이 있는 씬 {with_span}개" in v
    sd = [i for i in fi["items"] if i["type"] == "dialogue" and i["status"] != "added"]
    assert f"대본 대사 {sum(bool(i['speakers']) for i in sd)}/{len(sd)}(" in v
    assert f"time_fixed {qd['time_fixed']}줄" in v
    assert "final.json`, sinkprep.final" in v and v.count("검사 통과") >= 3          # asr·align·final
    # 확인 필요: 그 영화의 film·item 수준만
    assert "film 수준 1건(absent_many 1), item 수준 2건(asr_low_conf 2)" in v
    assert "단계 문제 기록" in v and "asr 없음" in v
    no_script_text(v)


def test_missing_files_are_reported_as_none(proj):
    v, c = section()
    assert "시각 대응표: 없음" in v and "받아쓰기: 없음" in v and "맞추기: 없음" in v and "최종본: 없음" in v
    assert "확인 필요(`qa/needs_review.csv`): 없음(파일 없음)" in v
    assert "| local_0001 | fm1 | 600.000 | 없음 | - | 없음 | 없음 |" in v
    assert "D1 영상이 있는 영화 1편 가운데 final.json 이 있는 영화 0편; [D1] 오류 1개" in c
    assert "영상이 있는 영화 1편 중 final.json 이 검사를 통과한 영화 0편" in c
    # asr.json 만 있으면 나머지만 없음
    util.write_json(paths.FILMS / FID / "work" / "asr.json", S.asr_of(speak(SAID[:2]), vad=[[10.0, 20.0]]))
    v, _ = section()
    assert "받아쓰기(`films/local_0001/work/asr.json`, 언어 ko): 말 구간 1개 합 10.0초" in v
    assert "맞추기: 없음" in v and "최종본: 없음" in v


def test_no_video_film_means_no_film_blocks(proj):
    catalog.write_catalog([dict(r, video_path="", video_sha1="", video_duration="", video_lang="")
                           for r in catalog.load()])
    v, c = section()
    assert "영상(video_path)이 있는 영화가 없음" in v
    assert "영상이 있는 영화 0편 중 final.json 이 검사를 통과한 영화 0편" in c


def test_criteria_lines_count_check_errors_by_code(proj):
    make_film()
    _, c = section()
    for k in ("C1", "C2", "D2", "D3", "D4", "D5", "D6", "D7", "D8"):
        assert f"- {k} 오류 films 0 · examples 0. 검사: " in c, k
    assert "- C3 asr_sha1 어긋남 0개, asr.json 스키마 오류 0개" in c
    assert "D1 영상이 있는 영화 1편 가운데 final.json 이 있는 영화 1편; [D1] 오류 0개" in c
    assert "D9 오류 films 0 · examples 0, 대본(script.json)과 견준 final.json 1/1개" in c
    assert "**영상이 있는 영화 1편 중 final.json 이 검사를 통과한 영화 1편**(local_0001)" in c
    # 대본에서 복사한 칸 하나를 바꾸면 [D9] 로 잡히고 통과한 영화가 0편이 된다
    fp = paths.FILMS / FID / "final.json"
    data = util.read_json(fp)
    it = next(i for i in data["items"] if i["type"] == "dialogue" and i["status"] == "same")
    it["how"] = "off" if it["how"] != "off" else "on"
    util.write_json(fp, data)
    v, c = section()
    assert "D9 오류 films 1 · examples 0" in c
    assert "영상이 있는 영화 1편 중 final.json 이 검사를 통과한 영화 0편" in c
    assert "검사 오류 1개([D9] 1)" in v
    # asr.json 이 바뀌면 align.json 의 asr_sha1 이 어긋난다(C3)
    ap = paths.FILMS / FID / "work" / "asr.json"
    asr = util.read_json(ap)
    asr["lang"] = "ko"
    asr["chunks"] = [{"start": 10.0, "end": 30.0, "lang": "ko"}]
    util.write_json(ap, asr)
    _, c = section()
    assert "- C3 asr_sha1 어긋남 1개" in c


def test_main_writes_the_whole_report_where_asked(proj, capsys):
    make_film()
    out = proj / "out" / "R.md"
    assert R.main(["--no-eval", "--workers", "1", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# sink 결과 보고")
    assert "## 6b. 영상 쪽" in text and "## 7. 수락 기준 확인(A1~A3, B1~B5, C1~C3, D1~D9, E1~E2)" in text
    assert "- 건너뜀(`--no-eval`)" in text and "- D9 오류 films 0" in text
    assert "- lines 판(머리 행 made.version):" in text and "- 파서 판(parser.name·version): sink-parse 0 1" in text
    assert not (proj / "REPORT.md").exists()
    no_script_text(text)


def test_path_overrides_move_everything_under_root_but_rules_and_examples(tmp_path):
    ov = R.path_overrides(root=tmp_path / "other")
    base = (tmp_path / "other").resolve()
    assert Path(ov["ROOT"]) == base and Path(ov["CATALOG"]) == base / "catalog.csv"
    assert Path(ov["FILMS"]) == base / "films" and Path(ov["AUDIO"]) == base / "cache" / "audio"
    assert "SCHEMA" not in ov and "EXAMPLES" not in ov and "HWP_EXE" not in ov
    ov2 = R.path_overrides(films=tmp_path / "f")
    assert ov2 == {"FILMS": str((tmp_path / "f").resolve())}
    before = paths.FILMS
    old = R.apply_paths(ov2)
    try:
        assert paths.FILMS == (tmp_path / "f").resolve()
    finally:
        R.apply_paths({k: str(v) for k, v in old.items()})
    assert paths.FILMS == before
