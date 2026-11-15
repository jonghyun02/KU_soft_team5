"""sinkprep.final: 최종본 조립, 스스로 검사, catalog·문제 기록, 결정성. tools/final_to_srt.py."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import align_synth as S  # noqa: E402
import final_to_srt as SRT  # noqa: E402
from sinkprep import align, catalog, final, paths, textnorm, timeline, util, validate  # noqa: E402

REAL_DOC = paths.DOCS / "fm55890" / "script.json"
DOC, FILM_ID = "fm1", "local_0001"

FILM = [
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
E = {"날씨": "E0002", "산책": "E0003", "예1": "E0004", "책": "E0005", "뒤": "E0007", "영화": "E0008", "지루": "E0009",
     "출발": "E0013", "예2": "E0014", "기차표": "E0015", "말없음": "E0016"}
# 씬 2 의 대사는 말하지 않고(absent), 한 줄은 바꿔 말하고(changed), 대본에 없는 말을 한다(added)
SAID = ["오늘 날씨가 정말 좋네요.", "그러게요. 산책이나 갈까요?", "예.", "저는 집에서 그냥 책을 읽고 싶습니다.",
        ("누구세요?", {"gap": 2.0}), ("내일 아침 일찍 출발합시다.", {"gap": 2.0}), "예.", "기차표는 미리 사 두었어요."]
ARGS = argparse.Namespace(cmd="final", docs=None, limit=None, force=False, workers=None, paths=[])
FORCE = argparse.Namespace(**dict(vars(ARGS), force=True))


def speak(utts, t0=10.0, gap=1.0, p=0.9, pass_id="plain", context=None):
    """말 차례 목록 -> pass. 차례는 글(공백으로 나눈 낱말) 또는 (글, {"gap": 초, "p": 확률})."""
    words, t = [], t0
    for u in utts:
        text, kw = (u, {}) if isinstance(u, str) else u
        t += kw.get("gap", gap)
        for w in text.split():
            words.append((w, round(t, 3), round(t + 0.3, 3), kw.get("p", p)))
            t += 0.3
    return S.say(words, pass_id, context)


@pytest.fixture
def proj(tmp_path, monkeypatch):
    """임시 프로젝트: 문서 fm1 = 영화 local_0001, 영상 칸이 찬 catalog 한 행."""
    for name, sub in (("ROOT", ""), ("DOCS", "docs"), ("FILMS", "films"), ("CACHE", "cache"), ("STATE", "cache/state"),
                      ("PROBLEMS", "cache/problems"), ("CATALOG", "catalog.csv"), ("QA", "qa"),
                      ("NEEDS_REVIEW", "qa/needs_review.csv")):
        monkeypatch.setattr(paths, name, tmp_path / sub if sub else tmp_path)
    put_catalog()
    return tmp_path


def put_catalog(**kw):
    row = {k: "" for k in catalog.FIELDS}
    row.update(doc_id=DOC, post_id="1", category="ko_film", files="raw/a.hwp", sha1="1" * 40, real_format="hwp5",
               title_raw="시험", title_norm="시험", script_lang="ko", is_translation="false", version_label="shooting",
               film_id=FILM_ID, film_title="시험 영화", video_path=f"videos/{FILM_ID}.mp4", video_sha1="0" * 40,
               video_duration="600.000", video_lang="ko", is_primary="true", extract_quality="0.990", status="parsed")
    row.update(kw)
    catalog.write_catalog([row])


def put_film(turns, *passes, vad=None, edit=None):
    script = S.tiny_script(turns)
    script["doc_id"] = DOC
    if edit:
        edit(script)
    util.write_json(paths.DOCS / DOC / "script.json", script)
    util.write_json(work() / "asr.json", S.asr_of(*passes, vad=vad))
    return script


def work():
    return paths.FILMS / FILM_ID / "work"


def out():
    return paths.FILMS / FILM_ID / "final.json"


def made(turns=FILM, said=SAID, **kw):
    """대본과 받아쓰기를 차리고 align → final 을 돌린 뒤 (script, final.json 내용)."""
    script = put_film(turns, speak(said), **kw)
    assert align.run(ARGS) == 0
    assert final.run(ARGS) == 0
    return script, util.read_json(out())


def dialogue(data):
    return [it for it in data["items"] if it["type"] == "dialogue"]


def by_el(data):
    return {it["src_el"]: it for it in data["items"] if it["src_el"]}


def stage_problems(stage="final"):
    p = paths.PROBLEMS / stage / f"{FILM_ID}.json"
    return util.read_json(p) if p.exists() else []


# ------------------------------------------------------------------ 끝에서 끝까지
def test_run_writes_final_json_that_passes_every_check(proj, capsys):
    """align → final: final.json·align.json·asr.json 이 검사를 모두 통과하고 catalog 가 done 이 된다."""
    script, data = made(vad=[[10.0, 30.0]])
    sp = paths.DOCS / DOC / "script.json"
    assert validate.check_file(out()) == ("final", [])
    assert validate.check_file(work() / "align.json") == ("align", [])       # C1·C2: final 과 1:1
    assert validate.check_file(work() / "asr.json") == ("asr", [])
    assert validate.validate_final(out(), sp) == []
    capsys.readouterr()
    assert validate.run(SimpleNamespace(paths=[str(paths.FILMS)], docs=None, limit=None, force=False, workers=1)) == 0
    assert "실패 0개" in capsys.readouterr().out
    assert not list(out().parent.glob("*.check")) and sorted(p.name for p in work().iterdir()) == ["align.json", "asr.json"]

    assert list(data) == ["schema", "film", "made", "people", "scenes", "items", "quality"]
    al = util.read_json(work() / "align.json")
    match = al["method"]["stats"]["match"]
    assert data["film"] == {"id": FILM_ID, "title": "시험 영화", "title_orig": None, "year": None,
                            "video": {"sha1": "0" * 40, "duration": 600.0, "lang": "ko"},
                            "script": {"doc_id": DOC, "version": "shooting", "lang": "ko", "translation": False,
                                       "match": match}}
    m = data["made"]
    assert m["pipeline"] == "0.1.0" and m["translator"] is None and m["asr"] == "plain: synth"
    assert m["aligner"].startswith(f"sinkprep.align {align.VERSION}") and m["conf"] == align.CONF_TEXT
    assert "forced alignment" in m["forced"] and "일치율" in m["match"]
    assert data["people"] == [{"id": "P01", "name": "가", "aliases": []}]              # count 는 뗀다
    row = catalog.load()[0]
    assert row["status"] == "done" and row["match"] == f"{match:.3f}"
    assert stage_problems() == []


def test_items_cover_every_script_element_once_and_counts_agree(proj):
    """대본의 scene·action·dialogue·onscreen·transition 요소가 빠짐없이 한 번씩, quality 수는 항목 수와 같다."""
    script, data = made()
    assert sorted(it["src_el"] for it in data["items"] if it["src_el"]) == [e["id"] for e in script["elements"]]
    assert [it["id"] for it in data["items"]] == [f"I{n:04d}" for n in range(1, len(data["items"]) + 1)]
    els = {e["id"]: e for e in script["elements"]}
    for it in data["items"]:
        if it["type"] != "dialogue":
            assert it["text"] == els[it["src_el"]]["text"] and it["timing"] == "estimated" and it["end"] is None
            assert ("kind" in it) == (it["type"] == "onscreen")
    assert by_el(data)[E["뒤"]]["kind"] == "insert"
    d = dialogue(data)
    q = data["quality"]
    c = Counter(it["status"] for it in d)
    assert (q["dialogue"], q["same"], q["changed"], q["added"], q["absent"]) == \
        (len(d), c["same"], c["changed"], c["added"], c["absent"]) == (11, 6, 1, 1, 3)
    assert q["speaker_unknown"] == sum(it["speaker_from"] == "unknown" for it in d) == 1 and q["speaker_inferred"] == 0
    assert q["silent_turns"] == 1 and q["spoken"] == 9 and q["absent_spoken"] == 2
    assert q["line_coverage"] == round(7 / 9, 3) and q["match"] == data["film"]["script"]["match"]
    assert q["mean_word_p"] == 0.9 and q["words"] == sum(len(it["words"]) for it in d) and q["by_pass"] == {"plain": 8}
    assert q["diff_kinds"] == dict(Counter(x["kind"] for it in d for x in it["diffs"]))
    assert (q["time_fixed"], q["time_fixed_words"], q["time_suspect"]) == (0, 0, [])
    # 씬 구간: 대사가 있으면 그 범위(그 씬으로 분류한 대본에 없는 말까지), 대사가 모두 absent 면 지문 등의 추정 시각 범위
    sc = {s["id"]: s for s in data["scenes"]}
    assert list(sc["SC001"]) == ["id", "number", "heading", "place", "time", "flags", "start", "end", "timing"]
    added = next(it for it in d if it["status"] == "added")
    assert added["scene"] == "SC001" and added["start"] > by_el(data)[E["책"]]["end"]
    assert (sc["SC001"]["start"], sc["SC001"]["end"]) == (by_el(data)[E["날씨"]]["start"], added["end"])
    assert (sc["SC003"]["start"], sc["SC003"]["end"]) == (by_el(data)[E["출발"]]["start"], by_el(data)[E["기차표"]]["end"])
    assert sc["SC002"]["timing"] == "estimated" and added["end"] <= sc["SC002"]["start"] <= sc["SC003"]["start"]
    # 대사 없는 씬 2 의 머리·화면 글자·전환은 씬 1 에 붙은 말 뒤, 씬 3 머리 앞에 한 덩어리로 놓인다
    order = [it["src_el"] for it in data["items"]]
    assert order.index(None) < order.index("E0006") < order.index(E["뒤"]) < order.index("E0010") < order.index("E0011")


def test_dialogue_items_carry_heard_words_and_script_fields(proj):
    """대사 23칸: 들린 글은 받아쓰기 낱말 그대로, 화자·how·paren 등은 대본에서, 한국어 줄은 ko = heard."""
    def edit(s):
        e = next(x for x in s["elements"] if x["id"] == E["책"])
        e.update(how="phone", paren=[{"text": "웃으며", "at": 0}], overlap=2, sung=True)
    script, data = made(edit=edit)
    items = by_el(data)
    same, changed = items[E["날씨"]], items[E["책"]]
    assert list(same) == ["id", "type", "scene", "speakers", "speaker_from", "how", "lang", "heard", "ko", "ko_src",
                          "start", "end", "timing", "words", "status", "match", "script_ref", "diffs", "paren", "sung",
                          "overlap", "conf", "src_el"]
    assert same["status"] == "same" and same["diffs"] == [] and same["speakers"] == ["P01"]
    assert same["speaker_from"] == "script" and same["match"] == "text" and same["timing"] == "aligned"
    assert same["heard"] == same["ko"] == "오늘 날씨가 정말 좋네요." and same["ko_src"] == "same" and same["lang"] == "ko"
    assert same["script_ref"] == {"el": E["날씨"], "text": "오늘 날씨가 정말 좋네요."}
    assert [list(w) for w in same["words"]] == [["w", "s", "e", "p"]] * 4            # 낱말 번호 i 는 뗀다
    assert same["heard"] == " ".join(w["w"] for w in same["words"])
    assert same["start"] == same["words"][0]["s"] and same["end"] == same["words"][-1]["e"] and same["start"] < same["end"]
    assert changed["status"] == "changed" and changed["heard"] == "저는 집에서 그냥 책을 읽고 싶습니다."
    assert changed["script_ref"]["text"] == "저는 집에서 책을 읽고 싶어요."              # 대본 글은 그대로 남는다
    assert changed["diffs"] == textnorm.diffs(changed["script_ref"]["text"], changed["heard"]) and changed["diffs"]
    assert (changed["how"], changed["paren"], changed["overlap"], changed["sung"]) == \
        ("phone", [{"text": "웃으며", "at": 0}], 2, True)
    added = next(it for it in dialogue(data) if it["status"] == "added")
    assert added["heard"] == added["ko"] == "누구세요?" and added["ko_src"] == "same" and added["lang"] == "ko"
    assert (added["speakers"], added["speaker_from"], added["how"]) == ([], "unknown", "unknown")
    assert added["script_ref"] is None and added["match"] is None and added["src_el"] is None and added["diffs"] == []
    # 앞 대사(씬 1)와 뒤 대사(씬 3) 사이, 쉼이 같으면 앞 대사의 씬. 대사 없는 씬 2 의 머리 추정 시각은 보지 않는다
    assert added["scene"] == "SC001" and added["paren"] == [] and added["sung"] is False and added["overlap"] is None
    # conf 는 align 의 계산식 그대로
    al = util.read_json(work() / "align.json")
    conf = dict(zip((o["item"] for o in al["ops"]),
                    align.op_conf(script, util.read_json(work() / "asr.json"), al["ops"])))
    assert {it["id"]: it["conf"] for it in dialogue(data)} == conf
    assert same["conf"] == 0.95 and items[E["말없음"]]["conf"] is None


def test_absent_items_have_no_time_and_sit_in_script_order(proj):
    """absent 는 시각·들린 글·낱말이 없고 대본 글만 남으며, 대본 순서로 앞뒤 항목 사이에 놓인다(D7)."""
    script, data = made()
    items = by_el(data)
    for el in (E["영화"], E["지루"], E["말없음"]):
        it = items[el]
        assert it["status"] == "absent" and (it["start"], it["end"], it["timing"]) == (None, None, "none")
        assert (it["heard"], it["ko"], it["ko_src"], it["words"], it["diffs"]) == (None, None, None, [], [])
        assert it["match"] == "text" and it["speakers"] == ["P01"] and it["speaker_from"] == "script"
    assert items[E["말없음"]]["script_ref"] == {"el": E["말없음"], "text": "......."}  # 말 없는 대사: 요소 글 그대로
    order = [it["src_el"] for it in data["items"]]
    assert order.index(E["뒤"]) < order.index(E["영화"]) < order.index(E["지루"]) < order.index("E0010")
    assert order[-1] == E["말없음"]
    errs = validate._Errs()
    validate._check_untimed_order(data["items"], errs)
    assert errs.done() == []
    timed = [it["start"] for it in data["items"] if it["start"] is not None]
    assert timed == sorted(timed) and max(timed) <= 600.0


def test_inferred_speaker_is_copied_but_not_counted_as_script(proj):
    """speaker_inferred 요소: 화자는 복사하되 speaker_from 은 unknown(확정값으로 세지 않음), quality 에 따로 센다."""
    def edit(s):
        next(x for x in s["elements"] if x["id"] == E["산책"])["speaker_inferred"] = True
        next(x for x in s["elements"] if x["id"] == E["예1"])["speakers"] = []
    script, data = made(edit=edit)
    items = by_el(data)
    assert (items[E["산책"]]["speakers"], items[E["산책"]]["speaker_from"]) == (["P01"], "unknown")
    assert (items[E["예1"]]["speakers"], items[E["예1"]]["speaker_from"]) == ([], "unknown")
    q = data["quality"]
    assert q["speaker_inferred"] == 1 and q["speaker_unknown"] == 3                   # 추정 1 + 화자 없음 1 + added 1
    assert validate.check_file(out()) == ("final", [])


# ------------------------------------------------------------------ 다시 돌리기
def test_run_is_byte_identical_and_skips_unchanged(proj, capsys):
    """같은 입력이면 final.json 바이트가 같고, 입력이 그대로면 건너뛴다. catalog 는 늘 done 과 같은 match 로 맞춘다."""
    made()
    first = out().read_bytes()
    capsys.readouterr()
    assert final.run(ARGS) == 0 and "'skip': 1" in capsys.readouterr().out
    assert final.run(FORCE) == 0 and out().read_bytes() == first
    assert align.run(FORCE) == 0 and catalog.load()[0]["status"] == "aligned"      # 맞추기를 다시 돌리면 aligned 로
    capsys.readouterr()
    assert final.run(ARGS) == 0 and "'skip': 1" in capsys.readouterr().out          # align.json 바이트가 같아 건너뜀
    assert catalog.load()[0]["status"] == "done" and out().read_bytes() == first
    # 받아쓰기가 바뀌면 align 부터 다시 해야 한다: align 을 돌리기 전에는 final 이 멈춘다
    util.write_json(work() / "asr.json", S.asr_of(speak([u for u in SAID if isinstance(u, str)])))
    assert final.run(ARGS) == 1 and out().read_bytes() == first
    assert align.run(ARGS) == 0 and final.run(ARGS) == 0 and out().read_bytes() != first
    # catalog 행이 바뀌어도 다시 만든다
    second = out().read_bytes()
    catalog.finalize(updates={DOC: {"year": "2008", "film_title_orig": "The Test"}})
    assert final.run(ARGS) == 0 and out().read_bytes() != second
    film = util.read_json(out())["film"]
    assert (film["year"], film["title_orig"]) == (2008, "The Test")


def test_select_docs_needs_video_primary_asr_and_align(proj):
    put_film(FILM, speak(SAID))
    assert final.select_docs(ARGS) == []                                         # align.json 없음
    assert final.run(ARGS) == 0 and not out().exists()                           # 대상이 없으면 아무것도 하지 않는다
    assert align.run(ARGS) == 0
    assert [d["doc_id"] for d in final.select_docs(ARGS)] == [DOC]
    assert final.select_docs(argparse.Namespace(**dict(vars(ARGS), docs="fm2"))) == []
    catalog.finalize(updates={DOC: {"is_primary": False}})
    assert final.select_docs(ARGS) == []


# ------------------------------------------------------------------ 문제 기록, 실패
def test_film_and_item_problems_are_recorded_without_script_text(proj):
    """일치율이 낮으면 version_mismatch, absent 가 많으면 absent_many, conf 가 아주 낮은 줄은 상한까지만 asr_low_conf."""
    junk = [(f"잡음{k} 소리{k} 말{k}.", {"gap": 2.0, "p": 0.05}) for k in range(align.ITEM_PROBLEMS_MAX + 5)]
    put_catalog(video_duration="2000.000")
    made(said=["전혀 다른 영화의 대사입니다.", "여기는 어디인가요?", SAID[0]] + junk)
    probs = stage_problems()
    kinds = Counter(p["problem"] for p in probs)
    assert kinds == {"version_mismatch": 1, "absent_many": 1, "asr_low_conf": align.ITEM_PROBLEMS_MAX}
    assert all(p["film_id"] == FILM_ID and p["doc_id"] == DOC and p["path"] == f"films/{FILM_ID}/final.json" for p in probs)
    items = {it["id"]: it for it in util.read_json(out())["items"]}
    for p in probs:
        if p["level"] == "item":
            assert items[p["item_id"]]["conf"] == p["score"] < align.ITEM_LOW_CONF
        else:
            assert p["level"] == "film" and p["item_id"] == ""
    said_words = [w for t in FILM for w in t[2].split() if len(w) > 2] + ["잡음", "소리"]
    assert not any(w in p["detail"] for p in probs for w in said_words)
    # align 단계와 같은 글이라 review 가 한 줄로 합친다
    assert {(p["level"], p["item_id"], p["problem"], p["detail"]) for p in probs} == \
        {(p["level"], p["item_id"], p["problem"], p["detail"]) for p in stage_problems("align")}
    assert catalog.load()[0]["status"] == "done"                                # 문제가 있어도 최종본은 만든다


def _relabel(el_key, lang, text=None):
    def edit(s):
        e = next(x for x in s["elements"] if x["id"] == E[el_key])
        e["lang"] = lang
        if text is not None:
            e["text"] = e["raw"] = text
    return edit


def test_line_marked_foreign_but_heard_in_hangul_is_kept_as_the_video_language(proj):
    """글자의 기준은 영상: 대본 lang 이 ja 로 잘못 붙은 줄('(일어서며)' 오판)이 한글로 들리면 그 줄은 ko 로 두고 영화를 만든다.
    그런 줄 수는 quality.lang_from_heard, 줄마다 항목 수준 other(코드만). 말하지 않은(absent) 줄은 대본 lang 그대로."""
    def edit(s):
        _relabel("날씨", "ja")(s)
        _relabel("영화", "ja")(s)                                         # 씬 2 는 말하지 않는다(absent)
    put_film(FILM, speak(SAID), edit=edit)
    sp = paths.DOCS / DOC / "script.json"
    assert align.run(ARGS) == 0 and final.run(ARGS) == 0
    data = util.read_json(out())
    it = by_el(data)[E["날씨"]]
    assert (it["status"], it["lang"], it["ko"], it["ko_src"]) == ("same", "ko", it["heard"], "same")
    cut = by_el(data)[E["영화"]]
    assert (cut["status"], cut["lang"], cut["ko_src"]) == ("absent", "ja", None)
    assert data["quality"]["lang_from_heard"] == 1
    assert validate.check_file(out()) == ("final", []) and validate.validate_final(out(), sp) == []
    (p,) = stage_problems()
    assert (p["level"], p["item_id"], p["problem"], p["score"]) == ("item", it["id"], "other", None)
    assert p["detail"] == "대본 lang ja, 들린 글은 영상 언어 ko 글자: lang 을 ko 로 둠" and "날씨" not in p["detail"]
    assert catalog.load()[0]["status"] == "done"

    # 검사도 같은 규칙(textnorm.spoken_lang): 한글로 들린 줄을 대본 lang 그대로 두면 D5·D9, 집계가 어긋나면 Q
    bad = json.loads(out().read_text(encoding="utf-8"))
    k = next(i for i, x in enumerate(bad["items"]) if x["id"] == it["id"])
    bad["items"][k].update(lang="ja", ko_src=None)
    tmp = out().with_name("bad.json")
    tmp.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    codes = {e.split()[0] for e in validate.validate_final(tmp, sp)}
    assert {"[D5]", "[D9]", "[Q]"} <= codes
    assert any("영상 언어 ko 의 글자인데 lang='ja'" in e for e in validate.validate_final(tmp, sp))


def test_timed_line_heard_in_another_script_fails_with_translation_unmatched(proj):
    """번역은 범위 밖: 들린 글이 한글이 아닌(영문으로 들린) 외국어 줄이 시각을 얻으면 그 영화는 실패,
    translation_unmatched 를 기록한다. 대본 lang 이 무엇이든(여기서는 ja 로 잘못 붙음) 들린 글자로 정한다."""
    said = ["Nice weather today."] + SAID[1:]
    put_film(FILM, speak(said), edit=_relabel("날씨", "ja", "Nice weather today."))
    assert align.run(ARGS) == 0
    assert final.run(ARGS) == 1 and not out().exists()
    (p,) = stage_problems()
    assert (p["level"], p["problem"], p["score"], p["film_id"]) == ("film", "translation_unmatched", 1, FILM_ID)
    assert "ja 1" in p["detail"] and "weather" not in p["detail"]
    assert catalog.load()[0]["status"] == "aligned"
    # 영상 언어가 한국어가 아니면 대본에 없는 말(added)부터 번역이 필요하다
    put_film(FILM, speak(SAID))
    assert align.run(ARGS) == 0 and final.run(ARGS) == 0
    put_catalog(video_lang="en", status="aligned")
    assert final.run(ARGS) == 1
    assert stage_problems()[0]["problem"] == "translation_unmatched"


def test_stale_align_json_is_refused(proj):
    """align.json 이 지금 입력과 맞지 않으면(item id, 대본 글, 영상 길이) 멈추고 align 을 다시 돌리라고 알린다."""
    made()
    first = out().read_bytes()
    al_p = work() / "align.json"
    good = al_p.read_bytes()

    def broken(fn):
        al = json.loads(good)
        fn(al)
        util.write_json(al_p, al)
        st, msg, _ = final.process(catalog.load()[0])
        al_p.write_bytes(good)
        return st, msg

    def swap(al):
        a, b = al["ops"][0], al["ops"][1]
        a["item"], b["item"] = b["item"], a["item"]
    st, msg = broken(swap)
    assert st == "fail" and "align" in msg and "item" in msg
    st, msg = broken(lambda al: al["ops"].pop())                                  # 대사 요소에 op 이 없음
    assert st == "fail" and "sinkprep align" in msg
    st, msg = broken(lambda al: al.update(script_doc="fm999"))
    assert st == "fail" and "script_doc" in msg
    assert out().read_bytes() == first and stage_problems()[0]["problem"] == "other"
    # 대본 글이 바뀌어 same 이던 줄이 changed 가 됨
    s = util.read_json(paths.DOCS / DOC / "script.json")
    next(x for x in s["elements"] if x["id"] == E["날씨"])["text"] = "오늘 날씨가 참 좋네요."
    util.write_json(paths.DOCS / DOC / "script.json", s)
    st, msg, _ = final.process(catalog.load()[0])
    assert st == "fail" and "changed" in msg and "날씨" not in msg
    assert final.run(ARGS) == 1 and catalog.load()[0]["status"] == "aligned"      # done 으로 남기지 않는다
    assert align.run(ARGS) == 0 and final.run(ARGS) == 0 and catalog.load()[0]["status"] == "done"
    assert stage_problems() == []


def test_final_json_is_not_written_when_a_check_fails(proj, monkeypatch, capsys):
    """스스로 검사에서 오류가 나오면 final.json 을 쓰지 않고, 문제 기록에는 오류 코드별 수만 남긴다."""
    put_film(FILM, speak(SAID))
    assert align.run(ARGS) == 0
    real = validate.validate_final
    monkeypatch.setattr(validate, "validate_final",
                        lambda p, s=None: real(p, s) + ["[D2] items/3(I0004): 대사 글이 섞인 오류", "[D7] 차례"])
    assert final.run(ARGS) == 1
    assert not out().exists() and not list(out().parent.glob("*.check"))
    err = capsys.readouterr().err
    assert "[D2] items/3(I0004)" in err
    (p,) = stage_problems()
    assert p["problem"] == "other" and p["score"] == 2 and "[D2] 1, [D7] 1" in p["detail"] and "대사 글" not in p["detail"]
    assert "; 항목 I0004)" in p["detail"]                                       # 어느 줄인지는 id 로만 남긴다
    assert final._where(["[D2] items/0(I0001): x"] * 2 + [f"[D2] items/{k}(I{k + 1:04d}): x" for k in range(1, 9)]) == \
        "; 항목 I0001, I0002, I0003, I0004, I0005 …" and final._where(["[Q] quality.same=3"]) == ""
    assert catalog.load()[0]["status"] == "aligned" and util.stage_fp("final", FILM_ID) is None
    monkeypatch.setattr(validate, "validate_final", real)
    assert final.run(ARGS) == 0 and out().exists() and stage_problems() == []


def test_build_refuses_missing_duration_and_other_video(proj):
    script = put_film(FILM, speak(SAID))
    assert align.run(ARGS) == 0
    row = catalog.load()[0]
    asr, al = util.read_json(work() / "asr.json"), util.read_json(work() / "align.json")
    with pytest.raises(final.FinalError, match="video_duration"):
        final.build(dict(row, video_duration=""), script, asr, al)
    with pytest.raises(final.FinalError, match="video_sha1"):
        final.build(dict(row, video_sha1="a" * 40), script, asr, al)
    assert final.build(row, script, asr, al)["film"]["script"]["match"] == al["method"]["stats"]["match"]


# ------------------------------------------------------------------ 낱말 시각 바로잡기
def test_word_times_that_go_back_or_run_past_the_video_are_repaired_and_the_film_survives(proj):
    """줄 안에서 낱말 시작 시각이 거꾸로 가거나 낱말이 영상 길이 밖이어도 영화 전체를 실패시키지 않는다. 시각을 바로잡고
    (quality.time_fixed), 바로잡아도 믿기 어려운 줄만 항목 수준 문제로 남긴다."""
    plain = speak(SAID)
    ws = plain["words"]
    assert (ws[1]["s"], ws[2]["s"]) == (11.3, 11.6) and [w["s"] for w in ws[-4:]] == [26.0, 26.3, 26.6, 26.9]
    ws[2]["s"] = 11.2                                   # 첫 줄 셋째 낱말이 둘째 낱말보다 먼저 시작한다고 적힘
    put_catalog(video_duration="25.200")                # 마지막 줄(26.0~27.2초)은 영상 길이 밖
    put_film(FILM, plain)
    assert align.run(ARGS) == 0 and final.run(ARGS) == 0
    data = util.read_json(out())
    first, last = by_el(data)[E["날씨"]], by_el(data)[E["기차표"]]
    assert [(w["s"], w["e"]) for w in first["words"]] == [(11.0, 11.3), (11.3, 11.6), (11.3, 11.9), (11.9, 12.2)]
    assert first["status"] == "same" and first["heard"] == "오늘 날씨가 정말 좋네요."        # 글자와 차례는 그대로
    assert (last["start"], last["end"], last["status"]) == (25.2, 25.22, "same")
    assert [(w["s"], w["e"]) for w in last["words"]] == [(25.2, 25.2)] * 3 + [(25.2, 25.22)]
    q = data["quality"]
    assert (q["time_fixed"], q["time_fixed_words"], q["time_suspect"]) == (2, 5, [last["id"]])
    assert "time_fixed" in data["made"]["quality"] and "time_fixed" in data["made"]["timing"]
    assert max(t for it in data["items"] for t in (it["start"], it["end"]) if t is not None) <= 25.2 + validate.DURATION_TOL
    # 검사 통과: final 의 낱말은 같은 규칙으로 바로잡은 asr 낱말과 같다(C2)
    assert validate.check_file(out()) == ("final", []) and validate.check_file(work() / "align.json") == ("align", [])
    assert catalog.load()[0]["status"] == "done"
    (p,) = stage_problems()
    assert (p["level"], p["item_id"], p["problem"], p["detail"]) == ("item", last["id"], "other", final.TIME_SUSPECT_TEXT)
    assert p["path"] == f"films/{FILM_ID}/final.json" and "기차표" not in p["detail"]
    assert final.run(FORCE) == 0 and util.read_json(out()) == data


# ------------------------------------------------------------------ validate 가 잡는 어긋남(C2)
def test_validate_catches_ops_pointing_at_other_words_and_a_stale_final(proj, tmp_path, capsys):
    """align.json 의 item·asr 범위가 어긋난 사본과, 받아쓰기가 바뀐 뒤 그대로 남은 낡은 final.json 을 validate 가 잡는다."""
    said = SAID[:5] + [("여기 좀 보세요, 여기요.", {"gap": 2.0})] + SAID[5:]
    made(said=said)
    sp, ap, lp = paths.DOCS / DOC / "script.json", work() / "asr.json", work() / "align.json"
    good = util.read_json(lp)
    adds = [k for k, o in enumerate(good["ops"]) if o["op"] == "added"]
    same = next(k for k, o in enumerate(good["ops"]) if o["op"] == "same" and o["asr"][0] != o["asr"][1])
    assert len(adds) >= 2 and validate.validate_align(lp, out(), ap, sp) == []

    def errs_of(fn):
        al = json.loads(json.dumps(good))
        fn(al["ops"])
        p = tmp_path / "broken_align.json"
        util.write_json(p, al)
        return validate.validate_align(p, out(), ap, sp)

    def swap(key):
        def fn(ops):
            a, b = ops[adds[0]], ops[adds[1]]
            a[key], b[key] = b[key], a[key]
        return fn

    def shift(ops):
        pid, i0 = ops[same]["asr"][0].split(":")
        i1 = ops[same]["asr"][1].split(":")[1]
        ops[same]["asr"] = [f"{pid}:{int(i0) + 1}", f"{pid}:{int(i1) + 1}"]

    for fn in (swap("item"), swap("asr"), shift):
        errs = errs_of(fn)
        assert errs and all(e.startswith("[C2]") for e in errs) and any("words" in e for e in errs), errs
    assert len(errs_of(swap("item"))) == 2                                       # 맞바꾼 두 줄만 알린다

    # 받아쓰기가 바뀌고(글은 같고 확률만) align 만 다시 돌림: final.json 은 예전 낱말을 가진 채 남는다
    asr = util.read_json(ap)
    for w in asr["passes"][0]["words"][:4]:
        w["p"] = 0.5
    util.write_json(ap, asr)
    assert align.run(ARGS) == 0
    kind, errs = validate.check_file(lp)
    assert kind == "align" and errs and all(e.startswith("[C2]") and "words" in e for e in errs)
    capsys.readouterr()
    assert validate.run(SimpleNamespace(paths=[str(paths.FILMS)], docs=None, limit=None, force=False, workers=1)) == 1
    assert "[C2]" in capsys.readouterr().out
    assert final.run(ARGS) == 0 and validate.check_file(lp) == ("align", [])


# ------------------------------------------------------------------ 다른 pass 에서 온 줄
def _hand_aligned(proj):
    """align 을 돌리지 않고 손으로 적은 align.json: 한 줄은 plain, 한 줄은 lv3, 한 줄과 대본에 없는 말은 rescue 에서."""
    turns = [("1", "scene", "S#1"), ("1", "dialogue", "오늘 날씨가 정말 좋네요."), ("1", "dialogue", "그러게요. 산책이나 갈까요?"),
             ("1", "dialogue", "저는 집에서 책을 읽고 싶어요."), ("1", "dialogue", "내일 아침 일찍 출발합시다.")]
    line1 = [("오늘", 11.0, 11.3), ("날씨가", 11.3, 11.6), ("정말", 11.6, 11.9), ("좋네요.", 11.9, 12.2)]
    plain = S.say(line1 + [("그러게요", 13.2, 13.5), ("산책이나", 13.5, 13.8), ("갈까", 13.8, 14.1)])
    lv3 = S.say(line1 + [("그러게요.", 13.2, 13.5, 0.8), ("산책이나", 13.5, 13.8, 0.8), ("갈까요?", 13.8, 14.1, 0.8)], "lv3")
    rescue = S.say([("누구세요?", 20.0, 20.6, 0.7), ("내일", 30.0, 30.3), ("아침", 30.3, 30.6), ("일찍", 30.6, 30.9),
                    ("출발합시다.", 30.9, 31.5)], "rescue", "full_window_stable")
    script = put_film(turns, plain, lv3, rescue)
    asr = util.read_json(work() / "asr.json")

    def line(el, status, pid, a, b, text):
        return {"item": None, "op": status, "script_el": el, "asr": [f"{pid}:{a}", f"{pid}:{b}"], "match": "text",
                "text_score": 1.0, "candidates": [{"text": text, "src": f"asr:{pid}", "score": -0.5},
                                                  {"text": text, "src": "script", "score": None}],
                "chosen": 0, "decided_by": "rule"}
    ops = [line("E0002", "same", "plain", 0, 3, "오늘 날씨가 정말 좋네요."),
           line("E0003", "same", "lv3", 4, 6, "그러게요. 산책이나 갈까요?"),
           {"item": None, "op": "absent", "script_el": "E0004", "asr": None, "match": "text", "decided_by": "rule"},
           line("E0005", "same", "rescue", 1, 4, "내일 아침 일찍 출발합시다."),
           {"item": None, "op": "added", "script_el": None, "asr": ["rescue:0", "rescue:0"], "match": None,
            "decided_by": "rule"}]
    timeline.attach_items(ops, timeline.build_timeline(script, ops, asr, 600.0))
    ops.sort(key=lambda o: int(o["item"][1:]))
    util.write_json(work() / "align.json", {
        "schema": "sink.align/1", "script_doc": DOC, "asr_sha1": util.sha1_file(work() / "asr.json"),
        "method": {"anchor": "hand", "acoustic": None, "forced": None, "translate": None}, "ops": ops, "scenes": []})
    return script, asr


def test_lines_from_lv3_and_rescue_passes_are_copied_like_plain_lines(proj):
    """op 의 asr 참조가 plain 이 아닌 pass(lv3, rescue)를 가리켜도, added 가 rescue 낱말이어도 같은 꼴의 항목이 된다."""
    script, asr = _hand_aligned(proj)
    assert final.run(ARGS) == 0
    data = util.read_json(out())
    assert validate.check_file(out()) == ("final", []) and validate.check_file(work() / "align.json") == ("align", [])
    assert catalog.load()[0]["status"] == "done"
    items = by_el(data)
    index = timeline.asr_index(asr)
    for el, pid, a, b in (("E0002", "plain", 0, 3), ("E0003", "lv3", 4, 6), ("E0005", "rescue", 1, 4), (None, "rescue", 0, 0)):
        it = items[el] if el else next(x for x in dialogue(data) if x["status"] == "added")
        assert it["words"] == [{"w": w["w"], "s": w["s"], "e": w["e"], "p": w["p"]} for w in index[pid][a:b + 1]]
        assert it["heard"] == it["ko"] == " ".join(w["w"] for w in it["words"]) and it["timing"] == "aligned"
        assert list(it) == list(items["E0002"])                                  # 어느 pass 에서 왔든 같은 칸
    assert items["E0003"]["heard"] == "그러게요. 산책이나 갈까요?" and items["E0003"]["status"] == "same"
    added = next(x for x in dialogue(data) if x["status"] == "added")
    # rescue 낱말로 만든 added 는 낱말 확률 평균(0.7)에 rescue 계수를 곱한다
    assert (added["heard"], added["scene"], added["speakers"], added["conf"]) == (
        "누구세요?", "SC001", [], round(0.7 * align.CONF_RESCUE, 3))
    q = data["quality"]
    assert q["by_pass"] == {"lv3": 1, "plain": 1, "rescue": 2} and q["words"] == 12
    assert q["pass_overlaps"] == [] and stage_problems() == []
    assert data["made"]["asr"] == "plain: synth; lv3: synth; rescue: synth [full_window_stable]"
    # 자막도 같은 꼴: 화자 이름, 대본에 없는 말 표시, 영상에 없는 줄
    rows = SRT.cues(data, absent=True)
    assert [t for _, _, t in rows] == ["가: 오늘 날씨가 정말 좋네요.", "가: 그러게요. 산책이나 갈까요?",
                                       "[없음] 가: 저는 집에서 책을 읽고 싶어요.", "[+] 누구세요?", "가: 내일 아침 일찍 출발합시다."]
    assert [(s, e) for s, e, _ in rows] == [(11.0, 12.2), (13.2, 14.1), (14.1, 16.1), (20.0, 20.7), (30.0, 31.5)]


def test_lines_from_two_passes_over_the_same_time_are_noted_not_refused(proj):
    """서로 다른 pass 에서 온 두 줄의 시간 구간이 크게 겹치면(같은 소리가 두 줄에 들어갔을 수 있다) 최종본은 만들되
    quality.pass_overlaps 와 뒤 줄의 항목 수준 문제로 남긴다. 시각만으로는 잘못이라 단정할 수 없어 막지 않는다."""
    _hand_aligned(proj)
    al = util.read_json(work() / "align.json")
    op = next(o for o in al["ops"] if o["script_el"] == "E0003")
    op.update(op="changed", asr=["lv3:0", "lv3:6"])          # lv3 줄이 첫 줄(plain:0~3)의 소리까지 가져감: 1.2초 겹침
    util.write_json(work() / "align.json", al)
    assert final.run(ARGS) == 0
    data = util.read_json(out())
    one, two = by_el(data)["E0002"], by_el(data)["E0003"]
    assert two["heard"] == "오늘 날씨가 정말 좋네요. 그러게요. 산책이나 갈까요?" and two["status"] == "changed"
    assert data["quality"]["pass_overlaps"] == [[one["id"], two["id"], 1.2]]
    assert "pass_overlaps" in data["made"]["quality"]
    (p,) = stage_problems()
    assert (p["level"], p["item_id"], p["problem"], p["score"]) == ("item", two["id"], "other", 1.2)
    assert one["id"] in p["detail"] and "1.200초" in p["detail"] and "날씨" not in p["detail"]
    assert validate.check_file(out()) == ("final", []) and validate.check_file(work() / "align.json") == ("align", [])


def test_rescue_word_claimed_by_two_lines_fails_the_self_check(proj, capsys):
    """같은 rescue 낱말을 두 줄이 가진 align.json 으로는 final.json 을 만들지 않는다(검사 C2)."""
    _hand_aligned(proj)
    al = util.read_json(work() / "align.json")
    added = next(o for o in al["ops"] if o["op"] == "added")
    added["asr"] = ["rescue:0", "rescue:1"]                                        # 넷째 줄의 첫 낱말까지 가져감
    util.write_json(work() / "align.json", al)
    assert final.run(ARGS) == 1 and not out().exists()
    (p,) = stage_problems()
    assert p["problem"] == "other" and "[C2] 1" in p["detail"] and "내일" not in p["detail"]
    ids = sorted(o["item"] for o in al["ops"] if o["asr"] and o["asr"][0].startswith("rescue"))
    assert f"; 항목 {ids[0]}, {ids[1]})" in p["detail"]                            # 낱말을 함께 가진 두 줄의 id
    assert "같은 낱말" in capsys.readouterr().err


# ------------------------------------------------------------------ pass 여럿, 옮긴 씬
def test_lines_from_ctx_and_moved_scenes_keep_ids_in_step_with_align(proj):
    """ctx pass 에서 고른 줄도 그 pass 의 낱말을 그대로 복사하고, 자리를 옮긴 씬이 있어도 align 과 같은 id 를 얻는다."""
    plain = speak(SAID[5:] + ["오늘 날시가 정말 좋네요."] + SAID[1:4])           # 씬 3 을 앞으로, 한 글자 틀리게 받아씀
    ctx = speak(SAID[5:] + SAID[:4], pass_id="ctx", context="script_hotwords")
    put_film(FILM, plain, ctx)
    assert align.run(ARGS) == 0 and final.run(ARGS) == 0
    data, al = util.read_json(out()), util.read_json(work() / "align.json")
    assert validate.check_file(work() / "align.json") == ("align", []) and validate.check_file(out()) == ("final", [])
    items = {it["id"]: it for it in data["items"]}
    index = timeline.asr_index(util.read_json(work() / "asr.json"))
    for op in al["ops"]:
        it = items[op["item"]]
        assert it["status"] == op["op"] and it["src_el"] == op["script_el"]
        if op["asr"]:
            assert it["words"] == timeline.op_words(index, op)
    line = by_el(data)[E["날씨"]]
    assert line["status"] == "same" and line["heard"] == "오늘 날씨가 정말 좋네요."          # ctx 의 낱말
    assert data["quality"]["by_pass"] == {"ctx": 1, "plain": 6} and data["made"]["asr"].endswith("[script_hotwords]")
    assert data["quality"]["moved"] >= 1


# ------------------------------------------------------------------ 가짜 영화(진짜 대본)
@pytest.mark.skipif(not REAL_DOC.exists(), reason="docs/fm55890/script.json 이 없음")
def test_synthetic_film_from_a_real_script(proj, capsys):
    """진짜 대본으로 만든 가짜 영화를 끝에서 끝까지: align → final → validate 가 통과하고 수가 맞는다."""
    script = json.loads(REAL_DOC.read_text(encoding="utf-8"))
    for e in script["elements"]:             # 가짜 영화는 모든 줄을 한국어로 '말한다'(외국어 표기 줄은 번역 단계가 있어야 한다)
        if e["type"] == "dialogue":
            e["lang"] = "ko"
    others = [" ".join(reversed(e["text"].split())) for e in script["elements"]
              if e["type"] == "dialogue" and len(e["text"].split()) >= 3][::5]
    mv = S.make_movie(script, others, seed=1)
    doc = script["doc_id"]
    put_catalog(doc_id=doc, post_id=doc[2:], video_duration=f"{mv['truth']['duration']:.3f}")
    util.write_json(paths.DOCS / doc / "script.json", script)
    util.write_json(work() / "asr.json", mv["asr"])
    assert align.run(ARGS) == 0 and final.run(ARGS) == 0
    capsys.readouterr()
    assert validate.run(SimpleNamespace(paths=[str(paths.FILMS)], docs=None, limit=None, force=False, workers=1)) == 0
    assert "align: 1개 검사, 0개 실패" in capsys.readouterr().out
    data = util.read_json(out())
    need = [e["id"] for e in script["elements"] if e["type"] in timeline.ITEM_TYPES]
    assert sorted(it["src_el"] for it in data["items"] if it["src_el"]) == sorted(need)
    q = data["quality"]
    d = dialogue(data)
    assert q["dialogue"] == len(d) == q["same"] + q["changed"] + q["added"] + q["absent"]
    assert q["spoken"] + q["silent_turns"] + q["added"] == q["dialogue"]
    assert set(q["by_pass"]) == {"plain", "ctx", "rescue"} and sum(q["by_pass"].values()) == q["same"] + q["changed"] + q["added"]
    assert all(it["ko"] == it["heard"] and it["start"] < it["end"] for it in d if it["status"] != "absent")
    assert all(textnorm.key(it["heard"]) == textnorm.key(it["script_ref"]["text"]) for it in d if it["status"] == "same")
    first = out().read_bytes()
    assert final.run(FORCE) == 0 and out().read_bytes() == first
    row = catalog.load()[0]
    assert row["status"] == "done" and float(row["match"]) == data["film"]["script"]["match"] > 0.5


# ------------------------------------------------------------------ tools/final_to_srt.py
def test_final_to_srt_marks_speakers_added_and_absent_and_splits_long_lines(proj):
    long = "이 줄은 아주 길어서 자막 한 칸에 다 들어가지 않으니까 낱말 시각을 써서 여러 칸으로 나누어야 하는 긴 대사입니다 " \
           "그리고 아직도 끝나지 않고 계속 이어지는 말이 한참 더 남아 있습니다."
    turns = [("1", "scene", "S#1"), ("1", "dialogue", "오늘 날씨가 정말 좋네요."), ("1", "dialogue", "어제 본 영화는 어땠습니까?"),
             ("1", "dialogue", long)]
    script, data = made(turns, ["오늘 날씨가 정말 좋네요.", ("누구세요?", {"gap": 2.0}), (long, {"gap": 2.0})])
    rows = SRT.cues(data)
    texts = [t for _, _, t in rows]
    assert texts[0] == "가: 오늘 날씨가 정말 좋네요." and texts[1] == "[+] 누구세요?"
    assert not any("[없음]" in t for t in texts)
    line = by_el(data)["E0004"]
    parts = texts[2:]
    assert len(parts) >= 2 and parts[0].startswith("가: 이 줄은")
    assert all(len(ln) <= SRT.WIDTH for t in parts for ln in t.split("\n")) and all(t.count("\n") <= 1 for t in parts)
    assert " ".join(" ".join(t.split("\n")) for t in parts) == "가: " + line["heard"]
    assert rows[2][0] == line["start"] and rows[-1][1] == line["end"]
    assert all(a[0] <= b[0] for a, b in zip(rows, rows[1:])) and all(s < e for s, e, _ in rows)
    with_absent = SRT.cues(data, absent=True)
    cue = next(c for c in with_absent if c[2].startswith("[없음]"))
    assert cue[2] == "[없음] 가: 어제 본 영화는 어땠습니까?" and cue[0] == by_el(data)["E0002"]["end"]
    # 파일로: 기본은 final.json 옆 preview.srt
    assert SRT.main([FILM_ID, "--absent"]) == 0
    text = (out().parent / "preview.srt").read_text(encoding="utf-8-sig")
    assert text.startswith("1\n00:00:11,000 --> 00:00:12,200\n가: 오늘 날씨가 정말 좋네요.\n\n2\n")
    assert SRT.stamp(3725.0415) == "01:02:05,042" and SRT.main(["local_9999"]) == 1
    assert validate.run(SimpleNamespace(paths=[str(paths.FILMS)], docs=None, limit=None, force=False, workers=1)) == 0


def _line(start, text, speakers=(), status="same", frm="script", ref=None, dur=0.4):
    """자막 도구가 읽는 칸만 가진 대사 항목(final.json items 의 일부)."""
    words = [{"w": w, "s": round(start + k * dur, 3), "e": round(start + (k + 1) * dur, 3), "p": 0.9}
             for k, w in enumerate(text.split())] if status != "absent" else []
    return {"type": "dialogue", "speakers": list(speakers), "speaker_from": frm if speakers else "unknown",
            "status": status, "heard": text if words else None, "words": words,
            "start": words[0]["s"] if words else None, "end": words[-1]["e"] if words else None,
            "script_ref": None if status == "added" else {"el": "E0001", "text": ref or text}}


def test_final_to_srt_speaker_names_and_status_marks():
    """화자 이름(여럿이면 /, 추정 화자는 ?, 모르는 id 는 id 그대로), added 의 [+], absent 의 [없음] 과 여러 줄 묶기."""
    people = [{"id": "P01", "name": "형사"}, {"id": "P02", "name": "창혁"}]
    items = [
        _line(None, "차를 저렇게 대면 어떡해요?", ["P02"], "absent"),             # 첫 대사보다 앞: 0초에 놓인다
        {"type": "scene", "text": "S#1", "start": 0.0, "end": None},
        _line(10.0, "아리랑 아리랑", ["P01", "P02"]),
        _line(12.0, "네.", ["P02"], frm="unknown"),                               # 앞 대사에서 이어받은 추정 화자
        _line(13.0, "누구세요?", status="added"),
        _line(14.0, "저기요.", ["P09"], "changed", ref="여보세요."),              # people 에 없는 id, changed 는 표시 없음
        _line(None, "하나.", ["P01"], "absent"), _line(None, ".......", ["P01"], "absent"),   # 말 없는 대사는 넣지 않는다
        _line(None, "둘.", ["P02"], "absent"), _line(None, "셋.", status="absent"), _line(None, "넷.", status="absent"),
        _line(20.0, "끝났습니다.", ["P01"]),
        _line(None, "마지막 줄.", ["P01"], "absent"),
    ]
    data = {"people": people, "items": items}
    assert [t for _, _, t in SRT.cues(data)] == ["형사/창혁: 아리랑 아리랑", "창혁?: 네.", "[+] 누구세요?", "P09: 저기요.",
                                                 "형사: 끝났습니다."]
    rows = SRT.cues(data, absent=True)
    assert [t for _, _, t in rows] == [
        "[없음] 창혁: 차를 저렇게 대면 어떡해요?", "형사/창혁: 아리랑 아리랑", "창혁?: 네.", "[+] 누구세요?", "P09: 저기요.",
        "[없음] 형사: 하나.\n[없음] … 외 3줄", "형사: 끝났습니다.", "[없음] 형사: 마지막 줄."]
    assert rows[0][:2] == (0.0, 2.0)                                              # 다음 자막까지 여유가 있으면 2초
    assert rows[5][:2] == (14.4, 16.4) and rows[7][:2] == (20.4, 22.4)            # 앞 대사가 끝난 자리
    assert all(a[0] <= b[0] for a, b in zip(rows, rows[1:])) and all(s < e for s, e, _ in rows)
    assert rows[2][:2] == (12.0, 12.7) and rows[3][:2] == (13.0, 13.7)            # 짧은 자막은 0.7초까지 늘린다
    # 다음 자막이 바로 오면 그 시작을 넘지 않는다
    tight = SRT.cues({"people": people, "items": [_line(1.0, "네.", dur=0.1), _line(1.3, "예.", dur=0.1)]})
    assert [(s, e) for s, e, _ in tight] == [(1.0, 1.3), (1.3, 2.0)]
    # 한 줄보다 긴 낱말은 잘라 넣고, 두 줄을 넘는 글은 낱말 시각으로 자막을 나눈다
    long = SRT.cues({"people": people, "items": [_line(5.0, "가" * 30 + " 나나 다다 라라 마마 바바", ["P01"])]}, width=10)
    assert [t for _, _, t in long] == ["형사: " + "가" * 6 + "\n" + "가" * 10, "가" * 10 + "\n" + "가" * 4 + " 나나 다다",
                                       "라라 마마 바바"]
    # 잘린 낱말(5.0~5.4초, 화자 이름까지 34자)의 시간은 글자 수에 비례해 조각에 나눈다: 자막 시간이 겹치지 않는다
    assert [(round(s, 3), round(e, 3)) for s, e, _ in long] == [(5.0, 5.235), (5.235, 6.2), (6.2, 7.4)]
    assert SRT.to_srt(long[2:]) == "1\n00:00:06,200 --> 00:00:07,400\n라라 마마 바바\n\n"
