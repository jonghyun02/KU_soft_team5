"""sinkprep.align: 닻, 줄 맞추기, 남은 낱말, pass 고르기, align.json."""
import argparse
import json
import math
import re
import sys
import time
import unicodedata
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import align_synth as S  # noqa: E402
from sinkprep import align, catalog, paths, textnorm, timeline, util, validate  # noqa: E402

REAL_DOC = paths.DOCS / "fm55890" / "script.json"

FILM = [
    ("1", "scene", "S#1 공원"),
    ("1", "dialogue", "오늘 날씨가 정말 좋네요."),
    ("1", "dialogue", "그러게요. 산책이나 갈까요?"),
    ("1", "dialogue", "예."),
    ("1", "dialogue", "저는 집에서 책을 읽고 싶어요."),
    ("2", "scene", "S#2 극장 앞"),
    ("2", "dialogue", "어제 본 영화는 어땠습니까?"),
    ("2", "dialogue", "생각보다 많이 지루했습니다."),
    ("3", "scene", "S#3 역"),
    ("3", "action", "두 사람이 걸어온다."),
    ("3", "dialogue", "내일 아침 일찍 출발합시다."),
    ("3", "dialogue", "예."),
    ("3", "dialogue", "기차표는 미리 사 두었어요."),
]
L = {"날씨": "E0002", "산책": "E0003", "예1": "E0004", "책": "E0005", "영화": "E0007", "지루": "E0008",
     "출발": "E0011", "예2": "E0012", "기차표": "E0013"}


def speak(utts, t0=10.0, gap=1.0, p=0.9, pass_id="plain", context=None):
    """말 차례 목록 -> pass. 차례는 글(공백으로 나눈 낱말) 또는 (글, {"gap": 초, "p": 확률, "at": 시작 시각})."""
    words, t = [], t0
    for u in utts:
        text, kw = (u, {}) if isinstance(u, str) else u
        t = kw["at"] if "at" in kw else t + kw.get("gap", gap)
        for w in text.split():
            words.append((w, round(t, 3), round(t + 0.3, 3), kw.get("p", p)))
            t += 0.3
    return S.say(words, pass_id, context)


def run(script_turns, *passes, duration=600.0, vad=None):
    script = S.tiny_script(script_turns)
    return script, align.build_ops(script, S.asr_of(*passes, vad=vad), duration)


def by_el(res):
    return {o["script_el"]: o for o in res.ops if o["script_el"]}


def heard_of(res, op):
    return next(it["heard"] for it in res.items if it["id"] == op["item"])


SAID = ["오늘 날씨가 정말 좋네요.", "그러게요. 산책이나 갈까요?", "예.", "저는 집에서 책을 읽고 싶어요.",
        "어제 본 영화는 어땠습니까?", "생각보다 많이 지루했습니다.", "내일 아침 일찍 출발합시다.", "예.",
        "기차표는 미리 사 두었어요."]


# ------------------------------------------------------------------ 닻
def test_unique_blocks_need_grams_that_occur_once_on_both_sides():
    """닻 씨앗은 양쪽에서 한 번씩만 나오는 k-gram 이다. 되풀이되는 조각은 씨앗이 못 된다."""
    assert align._unique_blocks("가나다라마xyz가나다라마", "가나다라마", 5) == []        # 대본에 두 번
    assert align._unique_blocks("가나다라마", "가나다라마xyz가나다라마", 5) == []        # 받아쓰기에 두 번
    assert align._unique_blocks("ab가나다라마cd", "xy가나다라마zw", 5) == [(2, 2, 5)]
    # 씨앗에서 양쪽으로 같은 글자만큼 늘린 최대 블록
    assert align._unique_blocks("예예예알겠습니다끝", "예예예알겠습니다", 5) == [(0, 0, 8)]
    assert align._unique_blocks("가나", "가나", 5) == []


def test_chain_takes_heaviest_monotone_chain():
    """대본 순서와 영상 순서가 함께 늘어나는 사슬 가운데 일치 글자 합이 가장 큰 것."""
    blocks = [(0, 0, 5), (10, 30, 6), (20, 10, 7), (30, 20, 8)]
    assert align._chain(blocks) == [0, 2, 3]
    assert align._chain([]) == []
    assert align._chain([(0, 0, 5), (3, 3, 5)]) in ([0], [1])       # 겹치는 블록은 함께 못 든다


def test_film_said_as_written_is_all_same():
    script, res = run(FILM, speak(SAID))
    assert [o["op"] for o in by_el(res).values()] == ["same"] * 9
    assert res.match == 1.0 and res.stats["added"] == 0 and res.stats["absent"] == 0
    assert all(o["decided_by"] == "rule" and o["match"] == "text" and o["text_score"] == 1.0 for o in res.ops)


def test_cut_scene_is_absent():
    """씬이 통째로 잘리면 그 대사는 absent 이고 나머지는 제자리에 맞는다."""
    script, res = run(FILM, speak(SAID[:4] + SAID[6:]))
    ops = by_el(res)
    assert ops[L["영화"]]["op"] == ops[L["지루"]]["op"] == "absent"
    assert ops[L["영화"]]["asr"] is None
    assert [ops[L[k]]["op"] for k in ("날씨", "산책", "예1", "책", "출발", "예2", "기차표")] == ["same"] * 7
    sc = {s["scene"]: s for s in res.scenes}
    assert sc["SC002"] == {"scene": "SC002", "start": None, "end": None, "how": "none"}
    assert sc["SC001"]["how"] == "anchors" and sc["SC001"]["start"] < sc["SC001"]["end"] < sc["SC003"]["start"]


def test_moved_scene_is_found_where_it_is():
    """자리를 옮긴 씬은 뼈대 밖에서 따로 찾는다."""
    script, res = run(FILM, speak(SAID[6:] + SAID[:6]))
    ops = by_el(res)
    assert all(o["op"] == "same" for o in ops.values())
    item = {it["id"]: it for it in res.items}
    assert item[ops[L["출발"]]["item"]]["start"] < item[ops[L["날씨"]]["item"]]["start"]
    hows = {s["scene"]: s["how"] for s in res.scenes}
    assert sorted(hows.values()) == ["anchors", "anchors", "moved"]
    assert sum(it.get("moved", False) for it in res.items) in (3, 6)   # 옮겨진 쪽 줄만 흐름 밖
    assert [it["id"] for it in res.items] == sorted(it["id"] for it in res.items)


def test_short_repeated_line_goes_to_its_place():
    """'예' 처럼 되풀이되는 짧은 대사는 닻이 아니라 앞뒤 줄 사이 자리로 맞춘다. 말하지 않은 '예' 는 absent."""
    script, res = run(FILM, speak(SAID[:7] + SAID[8:]))                # 둘째 '예' 를 말하지 않음
    ops = by_el(res)
    assert ops[L["예1"]]["op"] == "same" and ops[L["예2"]]["op"] == "absent"
    a, b, c = (timeline.op_range(ops[L[k]])[1] for k in ("산책", "예1", "책"))
    assert a < b < c
    # 첫째 '예' 만 말하지 않으면 뒤의 '예' 가 첫째 자리로 끌려오지 않는다
    script, res = run(FILM, speak(SAID[:2] + SAID[3:]))
    ops = by_el(res)
    assert ops[L["예1"]]["op"] == "absent" and ops[L["예2"]]["op"] == "same"


def test_lines_too_short_for_anchors_are_matched_by_place():
    """닻이 하나도 없는 글(짧은 줄뿐)도 줄 단위 맞추기로 순서대로 맞춘다."""
    turns = [("1", "dialogue", t) for t in ("예.", "왜요?", "아니요.", "가자.", "예.")]
    script, res = run(turns, speak(["예", "왜요", "가자", "예"]))
    assert [o["op"] for o in sorted(res.ops, key=lambda o: o["script_el"])] == ["same", "same", "absent", "same", "same"]
    assert [timeline.op_range(o)[1] for o in sorted(res.ops, key=lambda o: o["script_el"]) if o["asr"]] == [0, 1, 2, 3]


def test_short_line_must_match_whole_words():
    """짧은 줄은 낱말 범위가 글자 그대로 같아야 한다: '예전에' 속의 '예' 에 붙지 않는다."""
    said = SAID[:2] + ["예전에 그랬죠."] + SAID[3:]
    script, res = run(FILM, speak(said))
    ops = by_el(res)
    assert ops[L["예1"]]["op"] == "absent"
    assert [heard_of(res, o) for o in res.ops if o["op"] == "added"] == ["예전에 그랬죠."]


def test_misheard_short_line_is_paired_by_place_when_it_sounds_alike():
    """자리로 맞추기: 맞춘 두 줄 사이에 못 맞춘 줄 하나와 남은 낱말이 마주 있고 소리가 닮았으면 그 줄로 본다."""
    turns = FILM[:3] + [("1", "dialogue", "괜찮니?")] + FILM[3:]
    el = "E0004"
    script, res = run(turns, speak(SAID[:2] + ["괜창니?"] + SAID[2:]))
    op = by_el(res)[el]
    assert op["op"] == "changed" and heard_of(res, op) == "괜창니?" and res.stats["added"] == 0
    assert textnorm.verdict("괜찮니?", "괜창니?")[0] == "changed"
    # 소리가 닮지 않은 말은 짝짓지 않는다: 줄은 absent, 말은 added
    script, res = run(turns, speak(SAID[:2] + ["뭐라고요?"] + SAID[2:]))
    assert by_el(res)[el]["op"] == "absent"
    assert [heard_of(res, o) for o in res.ops if o["op"] == "added"] == ["뭐라고요?"]


def test_short_line_far_from_matched_neighbours_is_not_taken():
    """짧은 줄은 대본에서 바로 앞뒤 줄이 가까이 맞춰져 있을 때만 남긴다. 잘린 대목 한가운데의 '예' 는 붙이지 않는다."""
    turns = [FILM[1], ("1", "dialogue", "그 얘기는 나중에 다시 하도록 하죠."), ("1", "dialogue", "예."),
             ("1", "dialogue", "지금은 시간이 너무 늦었습니다."), FILM[4]]
    said = [SAID[0], ("예", {"gap": 60.0}), (SAID[3], {"gap": 60.0})]
    script, res = run(turns, speak(said))
    ops = sorted(by_el(res).values(), key=lambda o: o["script_el"])
    assert [o["op"] for o in ops] == ["same", "absent", "absent", "absent", "same"]
    assert [heard_of(res, o) for o in res.ops if o["op"] == "added"] == ["예"]
    # 이웃 줄이 가까이 맞춰져 있으면 남긴다
    said = [SAID[0], "그 얘기는 나중에 다시 하도록 하죠.", "예", (SAID[3], {"gap": 60.0})]
    script, res = run(turns, speak(said))
    ops = sorted(by_el(res).values(), key=lambda o: o["script_el"])
    assert [o["op"] for o in ops] == ["same", "same", "same", "absent", "same"]


def test_changed_line_keeps_place_and_gets_diffs():
    """낱말을 바꿔 말한 줄은 changed 이고 낱말 범위는 이어진 한 덩어리다."""
    said = list(SAID)
    said[3] = "저는 집에서 그냥 책을 읽고 싶습니다."
    script, res = run(FILM, speak(said))
    op = by_el(res)[L["책"]]
    assert op["op"] == "changed" and 0 < op["text_score"] < 1
    assert heard_of(res, op) == "저는 집에서 그냥 책을 읽고 싶습니다."
    status, diffs = textnorm.verdict("저는 집에서 책을 읽고 싶어요.", heard_of(res, op))
    assert status == "changed" and any((d["script"], d["heard"]) == ("", "그냥") for d in diffs)   # 종류(kind)는 textnorm 시험이 본다
    assert res.stats["changed"] == 1 and 0.9 < res.match < 1.0


def test_script_holding_the_same_text_twice_still_aligns():
    """대본에 같은 내용이 두 번 들어 있으면 하나뿐인 조각이 없다. 그래도 한 벌은 맞추고 나머지는 absent 로 둔다."""
    twice = FILM + [(str(int(sc) + 3), typ, text) for sc, typ, text in FILM]
    script, res = run(twice, speak(SAID))
    ops = [o for o in res.ops if o["script_el"]]
    assert sorted(o["op"] for o in ops) == ["absent"] * 9 + ["same"] * 9 and res.stats["added"] == 0
    heard = sorted(heard_of(res, o) for o in ops if o["op"] == "same")
    assert heard == sorted(SAID)                                      # 말한 줄마다 대본 줄 하나씩


def test_line_ranges_do_not_overlap():
    script, res = run(FILM, speak(SAID))
    spans = sorted(timeline.op_range(o)[1:] for o in res.ops if o["asr"])
    assert all(a[1] < b[0] for a, b in zip(spans, spans[1:]))


def test_silent_turn_is_absent_and_left_out_of_the_match_rate():
    """말 없는 대사(점뿐)는 absent 로 두되 일치율 분모와 잘린 대사 통계에서 뺀다."""
    turns = FILM[:3] + [("1", "dialogue", ".......")] + FILM[3:]
    script, res = run(turns, speak(SAID))
    silent = next(e["id"] for e in script["elements"] if e["text"] == ".......")
    op = by_el(res)[silent]
    assert op["op"] == "absent" and op["asr"] is None and op["match"] == "text"
    assert res.stats["silent_turns"] == 1 and res.stats["absent"] == 1 and res.stats["absent_spoken"] == 0
    assert res.match == 1.0 and res.conf[op["item"]] is None


def test_match_rate_counts_exact_letters_over_spoken_script_letters():
    """일치율 = same·changed 줄의 맞은 글자 합 ÷ 말 있는 대본 대사 글자 합."""
    script, res = run(FILM, speak(SAID[:4] + SAID[6:]))
    total = sum(len(textnorm.key(t)) for t in SAID)
    cut = sum(len(textnorm.key(t)) for t in SAID[4:6])
    assert res.match == round((total - cut) / total, 3) == res.stats["match"]


def test_digits_find_their_place_but_are_changed():
    """받아쓰기가 숫자로 적어도 자리는 찾고, 판정은 changed(number)다."""
    turns = FILM[:2] + [("1", "dialogue", "삼백육호로 사팔팔오 번을 보내 주세요.")] + FILM[2:]
    said = [SAID[0], "306호로 4885 번을 보내 주세요."] + SAID[1:]
    script, res = run(turns, speak(said))
    el = next(e["id"] for e in script["elements"] if e["text"].startswith("삼백육호"))
    op = by_el(res)[el]
    assert op["op"] == "changed" and heard_of(res, op) == "306호로 4885 번을 보내 주세요."
    assert textnorm.verdict(script["elements"][2]["text"], heard_of(res, op))[0] == "changed"      # 숫자 표기만 달라도 changed


# ------------------------------------------------------------------ 다시 짝짓기, 비슷한 줄, 옮긴 씬
NAMES = "가나다라마바사아자차카타파하거너더러머버서어저처"
BIG = [f"{c}{c}씨는 내일 {c}{c}{c} 서류를 들고 창구로 오세요." for c in NAMES]     # 서로 다른 긴 줄 24개(낱말 168개)


def big_film(extra_turns, extra_said, at=10):
    """긴 영화 한가운데(at 번째 줄 뒤)에 대본 줄 extra_turns 와 한 말 extra_said 를 끼운 (turns, said)."""
    turns = [("1", "dialogue", t) for t in BIG[:at]] + [("1", "dialogue", t) for t in extra_turns] + \
            [("1", "dialogue", t) for t in BIG[at:]]
    return turns, BIG[:at] + list(extra_said) + BIG[at:]


def ops_of(script, res):
    """[(대본 글 또는 '', 상태, 들린 글)] 항목 차례."""
    text = {e["id"]: e["text"] for e in script["elements"]}
    return [(text.get(o["script_el"], ""), o["op"], heard_of(res, o) or "") for o in res.ops]


def test_repair_pairs_an_answer_said_with_another_word():
    """'예' 를 '네' 로 말한 줄은 absent + added 로 가르지 않고 그 줄로 본다(판정은 changed). 좁은 창에서만."""
    said = list(SAID)
    said[2] = "네."
    script, res = run(FILM, speak(said))
    op = by_el(res)[L["예1"]]
    assert op["op"] == "changed" and heard_of(res, op) == "네." and res.stats["added"] == 0
    # 잘린 대목(못 맞춘 줄이 많고 남은 말이 그 줄들을 다 말하기에 모자람) 앞의 '네' 는 짝짓지 않는다
    turns, said = big_film(["예."] + [f"영상에서 잘린 줄 {c}{c} 입니다 정말로요." for c in NAMES[:12]],
                           [("네", {"gap": 0.5})])
    script, res = run(turns, speak(said), duration=2000.0)
    rows = ops_of(script, res)
    assert ("예.", "absent", "") in rows and ("", "added", "네") in rows
    assert sum(st == "absent" for _, st, _ in rows) == 13


def test_repair_keeps_an_exact_short_line_whose_neighbours_are_cut():
    """글자까지 같은 짧은 줄은 이웃 줄이 안 맞아도, 그 말이 영화에 드물고 창 안 순서가 맞으면 그 줄로 본다."""
    turns, said = big_film(["이건 없는 줄.", "너 뭐야?", "저것도 없어."],
                           [("사장님! 빨리 가세요!", {"gap": 3.0}), ("너 뭐야?", {"gap": 4.0}), ("이 씨!", {"gap": 2.0})])
    script, res = run(turns, speak(said), duration=2000.0)
    rows = ops_of(script, res)
    assert ("너 뭐야?", "same", "너 뭐야?") in rows
    assert ("이건 없는 줄.", "absent", "") in rows and ("저것도 없어.", "absent", "") in rows
    assert [h for t, st, h in rows if st == "added"] == ["사장님!", "빨리 가세요!", "이 씨!"]
    # 어디에나 있는 말('예')은 같은 자리에서도 받아들이지 않는다(우연히 거기 있을 기대 횟수가 크다)
    turns, said = big_film(["이건 없는 줄.", "예.", "저것도 없어."],
                           [("사장님! 빨리 가세요!", {"gap": 3.0}), ("예", {"gap": 4.0}), ("이 씨!", {"gap": 2.0})])
    said = [s if isinstance(s, tuple) else "예 " + s for s in said]          # 영화 곳곳에 '예'
    script, res = run(turns, speak(said), duration=2000.0)
    assert ("예.", "absent", "") in ops_of(script, res)


def test_repair_pairs_a_line_said_in_part_and_misheard_short_lines():
    """줄의 한 문장만 말한 줄, 받아쓰기가 붙여 쓰거나 틀린 짧은 줄을 맞춘 줄 사이 창에서 순서대로 짝짓는다."""
    turns, said = big_film(["감기는 씨발. 끊어. 내가 전화해 볼께.", "엄마...", "응...", "약 먹어."],
                           [("감기는 시발 그려봐 새끼야", {"gap": 0.0}), ("엄마", {"gap": 25.0}), ("약먹어", {"gap": 4.0})])
    script, res = run(turns, speak(said), duration=2000.0)
    assert ops_of(script, res)[10:14] == [
        ("감기는 씨발. 끊어. 내가 전화해 볼께.", "changed", "감기는 시발 그려봐 새끼야"),   # 이 말이 앞 줄에 붙지 않는다
        ("엄마...", "same", "엄마"), ("응...", "absent", ""), ("약 먹어.", "same", "약먹어")]
    assert res.stats["added"] == 0 and res.stats["absent"] == 1
    # 자모가 닮은 짧은 줄('어디야' / '어디요')과 대답이 든 줄('예 오빠' / '네, 오빠')
    turns, said = big_film(["예 오빠...", "어디야."], [("네, 오빠.", {"gap": 20.0}), ("어디요?", {"gap": 0.4})])
    script, res = run(turns, speak(said), duration=2000.0)
    assert [(st, h) for _, st, h in ops_of(script, res)[10:12]] == [("changed", "네, 오빠."), ("changed", "어디요?")]
    assert res.stats["added"] == 0 and res.stats["absent"] == 0


def test_repair_takes_back_a_neighbours_edge_word_that_does_not_help_it():
    """맞춘 줄의 범위 가장자리에서 그 줄의 맞은 글자에 보탬이 안 되는 낱말은, 못 맞춘 이웃 줄과 닮았으면 그 줄로 옮긴다."""
    lines, _ = align.script_lines(S.tiny_script([("1", "dialogue", t) for t in (SAID[0], "감기는 걸렸니?", SAID[3])]))
    pw = align.PassWords("plain", S.say(glued("오늘 날씨가 정말 좋네요 감기는 걸렸니 저는 집에서 책을 읽고 싶어요"))["words"])

    def lm(li, a, b):
        span = pw.span_loc(a, b)
        hit = textnorm.lcs(lines[li].loc, span)
        return align.LineMatch(li, lines[li].el, a, b, hit, len(lines[li].loc), len(span), 1.0, "lines")

    got = {0: lm(0, 0, 4), 2: lm(2, 6, 10)}                          # 줄 0 이 다음 줄의 '감기는' 까지 갖고 있다
    align._repair(lines, pw, got)
    assert (got[0].w0, got[0].w1) == (0, 3) and (got[1].w0, got[1].w1, got[1].how) == (4, 5, "slot")


def test_similar_lines_longer_one_keeps_its_words():
    """비슷한 두 줄에서 짧은 뒤 줄이 3·4글자 닻으로 씨앗이 되어 앞 줄 말의 한가운데를 가져가지 않는다."""
    turns = [("1", "dialogue", t) for t in (
        "좋은 말할 때 일 나가. 욕 처먹고 나서 질질 짜지 말고.", "왜 대답이 없냐?", "오빠...",
        "아이씨... 너희 집 어디야. 내가 갈께.", "너희 집 어디냐고.", "예. 나갈께요.", "진작에 그러지. 오좆한테 전화해봐.")]
    said = ["좋은 말할 때 일 나가 욕 처먹고 나서 질질 짜지 말고", "왜 대답이 없어?", ("아이씨.. 야 너 집 어디냐? 내 집 갈게", {"gap": 4.0}),
            ("집이.", {"gap": 3.0}), ("네 나갈게요", {"gap": 3.0}), "진작에 그러지 오좆한테 전화해봐"]
    script, res = run(turns, speak(said))
    rows = ops_of(script, res)
    assert ("아이씨... 너희 집 어디야. 내가 갈께.", "changed", "아이씨.. 야 너 집 어디냐? 내 집 갈게") in rows
    assert ("너희 집 어디냐고.", "absent", "") in rows
    assert ("예. 나갈께요.", "changed", "네 나갈게요") in rows


def test_scattered_letters_do_not_match_a_mid_length_line(monkeypatch):
    """4~7글자 줄은 차례만 맞는 흩어진 글자로는 맞추지 않는다(이어진 두 글자나 자모 닮음이 있어야 한다)."""
    turns = [("1", "dialogue", t) for t in (SAID[0], "야, 야, 이형사.", SAID[3])]
    said = [SAID[0], ("야, 준공이야 가사면", {"gap": 2.0}), (SAID[3], {"gap": 2.0})]
    script, res = run(turns, speak(said))
    assert ops_of(script, res)[1:3] == [("야, 야, 이형사.", "absent", ""), ("", "added", "야, 준공이야 가사면")]
    assert not align._solid("야야이형사", "야준공이야가사면") and align._solid("아나이씨", "아참나씨")
    monkeypatch.setattr(align, "_solid", lambda a, b: True)          # 이 조건이 없으면 영상에 없는 줄이 대본에 없는 말과 짝이 된다
    script, res = run(turns, speak(said))
    assert ops_of(script, res)[1][1] == "changed"


def test_swapped_neighbour_lines_are_both_found():
    """씬 안에서 이웃한 두 줄의 순서만 바꿔 말해도 둘 다 제 말에 맞춘다(한 줄 absent + 같은 글 added 가 되지 않는다)."""
    lines = ["오늘 저녁에 망원동 사거리에서 만나기로 했잖아요", "밤늦게 죄송합니다 박집사님 지금 계신가요", "그런 사람 여기 안 삽니다",
             "경찰서에 가서 조서부터 쓰고 다시 얘기합시다"]
    script, res = run([("1", "dialogue", t) for t in lines], speak([lines[0], lines[2], lines[1], lines[3]]))
    assert [(t, st) for t, st, _ in sorted(ops_of(script, res))] == sorted((t, "same") for t in lines)
    assert res.stats["added"] == 0 and sum(it.get("moved", False) for it in res.items) == 1


def test_moved_scene_sharing_an_edge_letter_with_its_new_neighbour_is_found():
    """옮긴 씬의 블록은 양옆으로 같은 글자만큼 늘어나 뼈대 블록과 한 글자 겹칠 수 있다. 겹친 글자만 잘라 내고 남긴다."""
    film = [("1", "dialogue", "오늘 날씨가 정말 좋네요."), ("1", "dialogue", "그러게 말입니다 산책이나 갈까요"),
            ("2", "dialogue", "어제 본 영화는 어땠습니까?"), ("2", "dialogue", "생각보다 많이 지루했습니다."),
            ("2", "dialogue", "다음에는 다른 걸 봅시다."),
            ("3", "dialogue", "내일 아침 일찍 출발합시다."), ("3", "dialogue", "기차표는 미리 사 두었어요"),
            ("4", "dialogue", "저는 집에서 책을 읽고 싶습니다."), ("4", "dialogue", "그럼 다음 주에 다시 만나기로 하죠.")]
    text = [t for _, _, t in film]
    # 씬 2 를 씬 3 뒤로: 대본에서 씬 2 앞 줄의 끝 글자(요)와 영상에서 씬 2 앞 줄의 끝 글자(요)가 같다
    said = text[:2] + text[5:7] + text[2:5] + text[7:]
    assert textnorm.key(text[1])[-1] == textnorm.key(text[6])[-1]
    script, res = run(film, speak(said))
    assert [st for _, st, _ in ops_of(script, res)] == ["same"] * 9 and res.stats["added"] == 0
    assert {s["scene"]: s["how"] for s in res.scenes}["SC002"] == "moved"


def test_moved_cluster_is_found_even_when_the_heaviest_chain_is_only_scraps():
    """뼈대 밖 블록의 가장 무거운 사슬이 기준에 못 미치는 작은 묶음들뿐이어도, 사슬에 못 든 진짜 덩어리를 찾는다."""
    scraps = [(100, 5000, 9), (1000, 6000, 9), (2000, 7000, 9), (3000, 8000, 9)]      # 서로 멀리 떨어진 9글자 블록 넷
    cluster = [(5000, 100, 12), (5020, 120, 12)]                                      # 가까이 모인 24글자
    assert sum(b[2] for b in scraps) > sum(b[2] for b in cluster) >= align.MOVED_MIN
    assert align._moved(scraps + cluster, align._Spans(), align._Spans()) == [cluster]


def test_digits_and_abbreviations_find_short_lines():
    """숫자·영문 약어는 맞은편 글에 있는 읽는 꼴로 자리를 찾는다(고유어 수사, 영문 읽기). 짧은 줄도 absent 가 되지 않는다."""
    turns = [("1", "dialogue", t) for t in (SAID[0], "두 시.", "열두 시간이요?", "TV 꺼.", SAID[3])]
    script, res = run(turns, speak([SAID[0], "2시.", "12시간이요?", "티비 꺼.", SAID[3]]))
    assert [(st, h) for _, st, h in ops_of(script, res)] == [
        ("same", SAID[0]), ("changed", "2시."), ("changed", "12시간이요?"), ("changed", "티비 꺼."), ("same", SAID[3])]


# ------------------------------------------------------------------ 닻과 줄 받아들이기의 나머지 규칙
def test_anchors_of_one_line_far_apart_do_not_stretch_the_line_over_what_lies_between():
    """한 줄의 닻이 둘로 멀리 갈려 있으면(사이에 대본에 없는 말이 한참 낌) 한쪽만 그 줄로 삼는다. 그 줄의 범위가 사이의 말을
    다 삼키지 않는다(ANCHOR_GAP, ANCHOR_SKIP)."""
    head, tail = "오늘 저녁에 망원동 사거리에서 만나기로 했잖아", "경찰서에 가서 조서부터 쓰고 다시 얘기합시다"
    turns = [("1", "dialogue", t) for t in (SAID[0], head + " " + tail, SAID[3])]
    said = [SAID[0], head] + BIG[:8] + [tail, SAID[3]]                # 줄의 앞 절반, 대본에 없는 말 여덟 줄, 줄의 뒤 절반
    script = S.tiny_script(turns)
    lines, _ = align.script_lines(script)
    pw = align.PassWords("plain", speak(said)["words"])
    _, seg, _ = align._anchor_pairs("".join(ln.loc for ln in lines), pw.chars)
    assert set(seg) == {0}                                           # 두 절반의 닻이 모두 뼈대 사슬에 들었다(옮겨진 덩어리가 아니다)
    res = align.build_ops(script, S.asr_of(speak(said)), 2000.0)
    rows = ops_of(script, res)
    assert (head + " " + tail, "changed", head) in rows and ("", "added", tail) in rows
    assert sum(st == "added" for _, st, _ in rows) == 9 and [st for _, st, _ in rows][::len(rows) - 1] == ["same", "same"]


def test_long_script_holding_everything_twice_is_anchored_on_repeated_pieces():
    """대본에 같은 내용이 통째로 두 번 들어 있으면 하나뿐인 조각이 없다. 긴 대본에서는 줄 단위 맞추기만으로는 창(GAP_CAP 낱말)을
    넘는 곳을 못 맞춘다: 몇 번 나오는 긴 조각을 씨앗으로 쓴다(REPEAT_K, REPEAT_CAP)."""
    texts = [f"{a}{b}씨는 내일 {b}{a}{a} 서류를 들고 창구로 오세요." for a in NAMES for b in NAMES[:12]]
    assert len(texts) == 288 and len(texts) * 7 > 3 * align.GAP_CAP    # 씨앗이 없으면 가운데 600낱말 넘게가 어느 창에도 안 든다
    turns = [("1", "dialogue", t) for t in texts] + [("2", "dialogue", t) for t in texts]
    script, res = run(turns, speak(texts, gap=0.2), duration=3000.0)
    assert (res.stats["same"], res.stats["absent"], res.stats["added"]) == (288, 288, 0)
    # 한 벌이 통째로 닻으로 맞았다(씨앗이 없으면 줄 단위 맞추기와 순서 밖에서 되찾기로 몇 초 걸려 메운다: lines, moved)
    assert sorted(s["how"] for s in res.scenes) == ["anchors", "none"]


def test_part_of_a_long_line_is_accepted_on_one_long_anchor_block():
    """긴 줄의 앞머리만 말했어도, 이어진 닻 블록이 BLOCK_OK 글자 이상이면 그 줄로 받아들인다(맞은 비율은 R_BLOCK 만 넘으면 된다)."""
    long = "아버지가 방에 들어가신다고 어머니한테 전해 달라고 하셨는데 제가 그만 깜빡 잊어버렸습니다"
    turns = [("1", "dialogue", t) for t in (SAID[0], long, SAID[3])]
    plain = speak([SAID[0], "아버지가 방에 들어가신다고", SAID[3]])
    n, hit = len(textnorm.key(long)), len(textnorm.key("아버지가 방에 들어가신다고"))
    assert align.R_BLOCK <= hit / n < align.R_LONG and hit >= align.BLOCK_OK
    got = {m.line: m for m in align.match_pass(S.tiny_script(turns), plain["words"])}
    assert (got[1].w0, got[1].w1, got[1].how) == (4, 6, "anchors")    # 다시 짝짓기(slot)가 아니라 닻으로 맞았다
    assert not align._accept(n, hit, hit) and align._accept(n, hit, hit, align.BLOCK_OK)


def test_short_line_whose_matched_neighbours_are_far_in_time_is_dropped():
    """짧은 줄은 맞춘 이웃 대본 줄이 SHORT_NEAR 초 안에 있어야 남긴다(이웃이 맞았어도 멀면 우연히 같은 말일 수 있다)."""
    def film(gap):
        turns, said = big_film(["예."], [("예", {"gap": gap})])
        said = [s if isinstance(s, tuple) or k == 11 else "예 " + s for k, s in enumerate(said)]     # '예' 가 흔한 영화
        said[11] = (said[11], {"gap": gap})                           # 그 '예' 는 앞 줄에서도 뒤 줄에서도 gap 초 떨어져 있다
        return run(turns, speak(said), duration=3000.0)

    script, res = film(30.0)
    rows = ops_of(script, res)
    assert ("예.", "absent", "") in rows and ("", "added", "예") in rows
    script, res = film(5.0)
    assert ("예.", "same", "예") in ops_of(script, res) and res.stats["absent"] == 0


def test_line_candidates_are_no_longer_than_the_span_cap():
    """줄 하나가 맞을 수 있는 낱말 범위는 줄 글자 수 × SPAN_MAX_RATIO + SPAN_MAX_EXTRA 글자까지만 본다."""
    lines, _ = align.script_lines(S.tiny_script([("1", "dialogue", "가나 다라 마바 사아 자차")]))
    words = "가나 하하 호호 다라 히히 후후 마바 헤헤 허허 사아 흐흐 하하 자차".split()
    pw = align.PassWords("plain", S.say(glued(" ".join(words)))["words"])
    F = list(range(pw.n))
    cap = int(10 * align.SPAN_MAX_RATIO) + align.SPAN_MAX_EXTRA
    cands = align._line_cands(lines[0], pw, F, align._pieces(F))
    assert cands and max(len(pw.span_loc(F[b], F[c - 1])) for b, c, _ in cands) <= cap < len(pw.chars)


def test_lone_misheard_line_is_paired_on_a_lower_bar_only_between_close_neighbours():
    """맞춘 두 줄 사이에 못 맞춘 줄 하나와 남은 말 하나뿐이면 자모 닮음 기준을 SLOT_JAMO_MIN 까지 낮춘다. 두 줄 사이가
    SLOT_MAX_S 초를 넘으면(그 사이에 다른 말이 있었을 수 있다) 낮추지 않는다."""
    line, heard = "조용히 좀 하세요.", "조용 쫌 해"
    js = align._jamo_sim(textnorm.key(line), textnorm.key(heard))
    assert align.SLOT_JAMO_MIN <= js < align.SLOT_JAMO_MANY
    turns, said = big_film([line], [(heard, {"gap": 3.0})])
    script, res = run(turns, speak(said), duration=2000.0)
    assert (line, "changed", heard) in ops_of(script, res)
    turns, said = big_film([line], [(heard, {"gap": 20.0})])
    script, res = run(turns, speak(said), duration=2000.0)
    rows = ops_of(script, res)
    assert (line, "absent", "") in rows and ("", "added", heard) in rows


# ------------------------------------------------------------------ 남은 낱말
def leftover_case(script_lines, words, ranges, blocked=()):
    """(넓힌 ranges, added). script_lines: 말 있는 대사 글 목록(줄 번호 = 목록 차례), words: [(글, 시작, 끝)]."""
    lines, _ = align.script_lines(S.tiny_script([("1", "dialogue", t) for t in script_lines]))
    pw = align.PassWords("plain", S.say(words)["words"])
    return align.leftover(pw, lines, ranges, blocked)


def glued(text, t0=10.0, dur=0.3):
    """틈 없이 이어진 낱말들(받아쓰기 낱말 시각은 이어 붙는다): [(글, 시작, 끝)]."""
    return [(w, round(t0 + k * dur, 3), round(t0 + (k + 1) * dur, 3)) for k, w in enumerate(text.split())]


def test_leftover_attaches_short_words_to_the_adjacent_line():
    """한두 음절 덧말은 옆 줄에 붙어 있고 문장이 이어지는 꼴이면 그 줄에 넣는다. 쉼표 붙은 군말은 뒤 줄의 말머리로."""
    ranges, added = leftover_case(["오늘 가자.", "그래 가자."], [
        ("어,", 9.6, 9.8), ("오늘", 10.0, 10.3), ("가자", 10.3, 10.6), ("빨리", 10.7, 11.0),      # 줄 0 = 낱말 1~2
        ("좋아.", 12.0, 12.3), ("그래", 12.35, 12.6), ("가자.", 12.6, 12.9)],                      # 줄 1 = 낱말 5~6
        {0: (1, 2), 1: (5, 6)})
    assert ranges == {0: (0, 3), 1: (5, 6)}                          # '어,' 와 '빨리' 는 줄 0 에
    assert added == [(4, 4)]                                         # '좋아.' 는 문장 끝이라 따로


def test_leftover_does_not_swallow_a_long_run_without_punctuation():
    """받아쓰기가 부호를 안 찍고 낱말이 틈 없이 이어져도, 대본과 닮지 않은 긴 말은 줄에 붙이지 않는다(상한)."""
    words = glued("이 새끼야 넌 옆에 가만히 있어 조금만 이 새끼 조금만 가서 좋아 야 너 이리 와 봐 왜 그러세요 아저씨 가만 있어 봐 좀")
    ranges, added = leftover_case(["이 개새끼야!"], words, {0: (0, 1)})
    assert ranges == {0: (0, 1)}
    assert [a for a, _ in added][0] == 2 and added[-1][1] == len(words) - 1
    assert all(b - a + 1 <= align.ADDED_MAX_WORDS for a, b in added) and len(added) >= 2   # 긴 묶음은 나눈다
    # 글자가 다 맞은 줄 뒤의 세 글자 말도 붙이지 않는다(한도는 안 맞은 대본 글자 수 + ATTACH_EXTRA)
    ranges, added = leftover_case(["차 어디서 찾았어?"], glued("차 어디서 찾았어 망원동"), {0: (0, 2)})
    assert ranges == {0: (0, 2)} and added == [(3, 3)]


def test_leftover_keeps_the_other_speakers_short_answer_apart():
    """상대의 짧은 대답('응', '아니')은 부호가 없어도 글자가 다 맞은 줄에 붙이지 않는다. 쉼표로 이어지는 말머리는 붙인다."""
    words = glued("밥 먹었어 응 그럼 가자")
    ranges, added = leftover_case(["밥 먹었어?", "그럼 가자."], words, {0: (0, 1), 1: (3, 4)})
    assert ranges == {0: (0, 1), 1: (3, 4)} and added == [(2, 2)]
    words = glued("밥 먹었어? 응. 그럼 가자.")                          # 부호가 있어도 같다
    assert leftover_case(["밥 먹었어?", "그럼 가자."], words, {0: (0, 1), 1: (3, 4)}) == ({0: (0, 1), 1: (3, 4)}, [(2, 2)])
    words = glued("경찰에 신고했어 아니 미쳤냐 잡아 죽일 년들을")
    ranges, added = leftover_case(["경찰에 신고했어?", "미쳤냐. 잡아 죽일 년들을."], words, {0: (0, 1), 1: (3, 6)})
    assert added == [(2, 2)]
    # 쉼표가 붙어 뒤 줄로 이어지는 말머리('네, 어디세요')는 뒤 줄에 붙인다
    ranges, added = leftover_case(["어디세요?"], glued("네, 어디세요?"), {0: (1, 1)})
    assert ranges == {0: (0, 1)} and added == []
    # 대본 줄이 그 대답으로 시작하면(예 = 네) 근거가 있으니 붙인다
    ranges, added = leftover_case(["예, 확실해요."], glued("네 확실해요"), {0: (1, 1)})
    assert ranges == {0: (0, 1)} and added == []


def test_leftover_attaches_words_that_match_the_script_line():
    """붙였을 때 대본 줄과 더 닮아지는 말은 한도를 넘어도 붙인다(맞추기가 놓친 그 줄의 말). 자모로 닮은 것, 되풀이도."""
    line = "오늘 날씨가 정말 좋네요. 산책이나 갈까요?"
    ranges, added = leftover_case([line], glued("오늘 날씨가 정말 좋네요 산책이나 갈까요 그런데 저 사람은 누구예요"),
                                  {0: (0, 3)})
    assert ranges == {0: (0, 5)} and added == [(6, 9)]               # 대본에 있는 데까지만(뒤따르는 말이 얹혀 붙지 않는다)
    # 붙는 글자가 한도(안 맞은 대본 글자 수 + ATTACH_EXTRA)를 넘어도 절반 이상이 대본과 맞으면 붙인다
    ranges, added = leftover_case(["오늘 날씨가 정말 좋네요. 산책 갈까요?"],
                                  glued("오늘 날씨가 정말 좋네요 산책이나 한번 갈까요 그런데 누구세요"), {0: (0, 3)})
    assert ranges == {0: (0, 6)} and added == [(7, 8)]
    # 받아쓰기가 글자를 많이 틀려 글자로는 절반이 안 맞아도 소리(자모)가 닮았으면 붙인다
    heard = "겅찰써에 까써 조써부토 습씨다"
    assert 2 * textnorm.lcs(textnorm.key("경찰서에 가서 조서부터 씁시다"), textnorm.key(heard)) < len(textnorm.key(heard))
    ranges, added = leftover_case(["오늘 날씨가 정말 좋네요. 경찰서에 가서 조서부터 씁시다."],
                                  glued("오늘 날씨가 정말 좋네요 " + heard + " 그런데 누구세요"), {0: (0, 3)})
    assert ranges == {0: (0, 7)} and added == [(8, 9)]
    ranges, added = leftover_case(["야, 이리 와 봐."], glued("야 야 야 이리 와 봐"), {0: (2, 5)})
    assert ranges == {0: (0, 5)} and added == []                     # 줄 머리 낱말의 되풀이
    # 근거가 있으면 짧은 쉼(ADDED_GAP 이하) 너머의 낱말도 붙인다. 근거가 없는 말은 ATTACH_GAP 까지만
    words = [("야,", 10.0, 10.1), ("야,", 10.2, 10.3), ("뒤에", 10.75, 10.9), ("봐봐", 10.9, 11.2)]
    assert leftover_case(["야, 야. 뒤에 봐봐."], words, {0: (2, 3)}) == ({0: (0, 3)}, [])
    words = [("저,", 10.0, 10.1), ("그,", 10.2, 10.3), ("뒤에", 10.75, 10.9), ("봐봐", 10.9, 11.2)]
    assert leftover_case(["야, 야. 뒤에 봐봐."], words, {0: (2, 3)}) == ({0: (2, 3)}, [(0, 1)])
    words = [("야,", 10.0, 10.1), ("야,", 10.2, 10.3), ("뒤에", 11.2, 11.4), ("봐봐", 11.4, 11.7)]       # 쉼이 ADDED_GAP 을 넘으면
    assert leftover_case(["야, 야. 뒤에 봐봐."], words, {0: (2, 3)}) == ({0: (2, 3)}, [(0, 1)])


def test_leftover_budget_is_the_unmatched_script_text_on_that_side():
    """근거 없는 붙이기의 한도는 그 쪽(줄 꼬리·머리)에 아직 안 맞은 대본 글자 수 + ATTACH_EXTRA 이다."""
    line = "감기는 씨발. 끊어. 내가 전화해 볼께."
    words = glued("감기는 시발 그려봐 새끼야")
    ranges, added = leftover_case([line], words, {0: (0, 1)})
    assert ranges == {0: (0, 3)} and added == []                     # 줄 꼬리가 안 맞았다: 바꿔 말한 부분일 수 있다
    # 줄 머리는 다 맞았으니 앞에 온 네 글자 말은 붙이지 않는다
    ranges, added = leftover_case(["엄중호 어디 갔냐고."], glued("이거 왜요 엄주가 어디갔냐"), {0: (2, 3)})
    assert ranges == {0: (2, 3)} and added == [(0, 1)]
    # 쉼(ATTACH_GAP 초과)으로 떨어진 말은 붙이지 않는다
    ranges, added = leftover_case([line], [("감기는", 10.0, 10.3), ("시발", 10.3, 10.6), ("그려봐", 11.0, 11.3)], {0: (0, 1)})
    assert ranges == {0: (0, 1)} and added == [(2, 2)]
    # 글자가 다 맞은 한 글자 줄에 두 글자를 붙이면 닮음이 뚝 떨어진다(ATTACH_DROP)
    ranges, added = leftover_case(["예."], glued("예 그래"), {0: (0, 0)})
    assert ranges == {0: (0, 0)} and added == [(1, 1)]


def test_leftover_cuts_added_at_pauses_and_sentence_ends():
    """붙지 않은 남은 낱말은 쉼(0.8초 초과)이나 문장 끝 부호에서 잘라 added 로 만든다. 감탄사만 있어도 버리지 않는다."""
    ranges, added = leftover_case(["가자.", "가자."], [
        ("가자.", 10.0, 10.3),                                                                       # 줄 0
        ("누구세요?", 11.0, 11.4), ("저", 11.5, 11.6), ("옆집", 11.6, 11.9), ("사람인데요", 11.9, 12.4),
        ("아이고", 14.0, 14.4), ("...", 14.5, 14.6),
        ("가자", 20.0, 20.3)], {0: (0, 0), 1: (7, 7)})                                               # 줄 1
    assert ranges == {0: (0, 0), 1: (7, 7)}
    assert added == [(1, 1), (2, 4), (5, 6)]
    # 글자가 하나도 없는 묶음은 버린다
    assert leftover_case(["가자."], [("가자.", 10.0, 10.3), ("...", 12.0, 12.1), ("♪", 14.0, 14.1)], {0: (0, 0)})[1] == []


def test_leftover_skips_words_under_lines_chosen_from_another_pass():
    """다른 pass 에서 고른 줄의 시간대에 있던 낱말은 남은 낱말로 치지 않는다."""
    words = [("가자.", 10.0, 10.3), ("뭐라고", 12.0, 12.4), ("했지?", 12.4, 12.8)]
    assert leftover_case(["가자."], words, {0: (0, 0)})[1] == [(1, 2)]
    assert leftover_case(["가자."], words, {0: (0, 0)}, [(11.9, 12.9)])[1] == []


def test_leftover_does_not_give_a_filler_before_another_passes_line_to_the_line_before():
    """다른 pass 에서 고른 줄 바로 앞의 한두 음절 군말은 그 줄의 말머리일 수 있다. 그 줄에는 붙일 수 없으니(범위는 한 pass 안)
    근거 없이 앞 줄에 붙이지도 않고 added 로 둔다."""
    lines, _ = align.script_lines(S.tiny_script([("1", "dialogue", t) for t in ("오늘 날씨가 정말 좋네요 그렇죠.", "산책이나 갈까요?")]))
    pw = align.PassWords("plain", S.say(glued("오늘 날씨가 정말 음,"))["words"])
    assert align.leftover(pw, lines, {0: (0, 2)}) == ({0: (0, 3)}, [])      # 뒤에 아무 줄도 없으면: 군말은 앞 줄에 붙는다
    pw = align.PassWords("plain", S.say(glued("오늘 날씨가 정말 음, 산채기나 갈가요"))["words"])
    ranges, added = align.leftover(pw, lines, {0: (0, 2)}, skip=[False] * 4 + [True] * 2)      # 뒤 줄을 다른 pass 에서 골랐다
    assert ranges == {0: (0, 2)} and added == [(3, 3)]
    ranges, added = align.leftover(pw, lines, {0: (0, 2)}, blocked=[(11.2, 11.8)])             # 시간 구간으로 주어도 같다
    assert ranges == {0: (0, 2)} and added == [(3, 3)]


def test_words_not_in_the_script_become_added_lines():
    """대본에 없는 말: 한두 음절 덧말은 붙어 있는 줄에, 더 긴 말이나 떨어져 있는 말은 added 줄로."""
    said = ["오늘 날씨가 정말 좋네요 진짜", ("잠깐만 여기 좀 보세요.", {"gap": 2.0}), (SAID[1], {"gap": 2.0})] + SAID[2:]
    script, res = run(FILM, speak(said))
    ops = by_el(res)
    assert ops[L["날씨"]]["op"] == "changed" and heard_of(res, ops[L["날씨"]]) == "오늘 날씨가 정말 좋네요 진짜"
    assert ops[L["날씨"]]["text_score"] < 1
    added = [o for o in res.ops if o["op"] == "added"]
    assert [heard_of(res, o) for o in added] == ["잠깐만 여기 좀 보세요."]
    assert added[0]["script_el"] is None and added[0]["match"] is None and added[0]["decided_by"] == "rule"
    it = next(i for i in res.items if i["id"] == added[0]["item"])
    assert it["scene"] == "SC001" and it["src_el"] is None
    # 문장이 끝난 뒤에 이어진 말은 붙이지 않고 added 로 둔다
    script, res2 = run(FILM, speak([SAID[0] + " 진짜"] + SAID[1:]))
    assert by_el(res2)[L["날씨"]]["op"] == "same"
    assert [heard_of(res2, o) for o in res2.ops if o["op"] == "added"] == ["진짜"]
    # 글자가 다 맞은 줄 뒤의 긴 말은 부호가 없어도 붙이지 않는다
    script, res3 = run(FILM, speak(["오늘 날씨가 정말 좋네요 그렇지 않습니까"] + SAID[1:]))
    assert by_el(res3)[L["날씨"]]["op"] == "same"
    assert [heard_of(res3, o) for o in res3.ops if o["op"] == "added"] == ["그렇지 않습니까"]


# ------------------------------------------------------------------ pass 고르기
def decide_case(plain, ctx=None, rescue=None, alt=None):
    passes = [speak(plain)]
    if alt is not None:
        passes.append(speak(alt, pass_id="lv3"))                     # context 가 없는 둘째 pass = 대안
    if ctx is not None:
        passes.append(speak(ctx, pass_id="ctx", context="script_hotwords"))
    if rescue is not None:
        passes.append(speak(rescue, pass_id="rescue", context="full_window_stable"))
    return run(FILM, *passes)


def chosen(res, op):
    return op["candidates"][op["chosen"]]["src"]


def sounds_twice(res, asr):
    """같은 시각의 받아쓰기 낱말이 두 항목에 든 곳(한 번 난 소리는 한 줄에만 있어야 한다)."""
    index = timeline.asr_index(asr)
    seen, out = {}, []
    for op in res.ops:
        for w in timeline.op_words(index, op):
            k = (w["s"], w["e"])
            if seen.setdefault(k, op["item"]) != op["item"]:
                out.append((w["w"], seen[k], op["item"]))
    return out


def test_pass_roles_follow_the_context_not_the_id():
    """pass 의 구실은 context 로 가른다: null 이면 대본 없이 받아쓴 후보(plain 이 기준, 나머지는 대안)."""
    asr = S.asr_of(S.say([], "turbo"), S.say([], "plain"), S.say([], "hint", "script_hotwords"),
                   S.say([], "gaps", "full_window_stable"), S.say([], "odd", "something_new"), S.say([], "hint2", "script_window"))
    r = align.pass_roles(asr)
    assert (r.base, r.alts, r.ctx, r.rescue, r.unused) == ("plain", ("turbo",), "hint", "gaps", ("odd", "hint2"))
    assert [r.of(p) for p in ("plain", "turbo", "hint", "gaps", "odd")] == ["base", "alt", "ctx", "rescue", "unused"]
    r = align.pass_roles([("lv3", None), ("ctx", None), ("rescue", None)])      # id 가 ctx·rescue 여도 context 가 없으면 대안
    assert (r.base, r.alts, r.ctx, r.rescue) == ("lv3", ("ctx", "rescue"), None, None)
    assert align.pass_roles([("hint", "script_hotwords")]).base == "hint"       # 대본 없이 받아쓴 pass 가 없으면 맨 앞이 기준
    assert align.pass_roles([]).base == "plain"
    script, res = run(FILM, speak(SAID), speak(SAID, pass_id="odd", context="something_new"))
    assert res.method["roles"] == {"base": "plain", "alts": [], "ctx": None, "rescue": None, "unused": ["odd"]}
    assert res.method["passes"]["odd"]["role"] == "unused" and res.method["passes"]["odd"]["chosen_lines"] == 0


def test_decide_plain_when_plain_is_the_script():
    """기준 범위가 대본과 같으면 기준(rule). 대안이나 ctx 가 달라도 보지 않는다."""
    other = list(SAID)
    other[0] = "오늘 날씨가 참 좋네요."
    script, res = decide_case(SAID, other)
    op = by_el(res)[L["날씨"]]
    assert (op["op"], op["decided_by"], chosen(res, op)) == ("same", "rule", "asr:plain")
    assert [c["src"] for c in op["candidates"]] == ["asr:plain", "asr:ctx", "script"]
    assert op["candidates"][-1] == {"text": "오늘 날씨가 정말 좋네요.", "src": "script", "score": None}
    assert op["candidates"][0]["score"] == round(math.log(0.9), 4)
    script, res = decide_case(SAID, alt=other)
    op = by_el(res)[L["날씨"]]
    assert (op["op"], op["decided_by"], chosen(res, op)) == ("same", "rule", "asr:plain")
    assert [c["src"] for c in op["candidates"]] == ["asr:plain", "asr:lv3", "script"]
    assert res.method["decide"]["base_same"] == 9 and res.method["passes"]["lv3"]["chosen_lines"] == 0


def test_decide_alt_when_alt_is_the_script():
    """기준이 대본과 다르고 대안이 대본과 같으면 대안(rule: 둘 다 대본을 보지 않은 받아쓰기라 소리 점수로 가리지 않는다)."""
    plain = list(SAID)
    plain[0] = ("오늘 날시가 정말 좋네요.", {"p": 0.99})             # 확률이 높아도 상관없다
    alt = [(u, {"p": 0.5}) for u in SAID]
    script, res = decide_case(plain, alt=alt)
    op = by_el(res)[L["날씨"]]
    assert (op["op"], op["decided_by"], chosen(res, op)) == ("same", "rule", "asr:lv3")
    assert op["asr"] == ["lv3:0", "lv3:3"] and heard_of(res, op) == "오늘 날씨가 정말 좋네요."
    assert [c["text"] for c in op["candidates"]] == ["오늘 날시가 정말 좋네요.", "오늘 날씨가 정말 좋네요.", "오늘 날씨가 정말 좋네요."]
    assert not any(o["op"] == "added" for o in res.ops)             # 그 줄의 기준 낱말은 added 가 아니다(같은 소리를 두 번 담지 않는다)
    assert sum(chosen(res, o) == "asr:plain" for o in res.ops) == 8  # 나머지 줄은 기준
    d = res.method["decide"]
    assert (d["alt_same"], d["base_same"]) == (1, 8) and res.method["passes"]["lv3"] == {
        "role": "alt", "words": 29, "matched_lines": 9, "chosen_lines": 1, "added_lines": 0, "backsteps": 0}
    assert res.method["acoustic"] is None and res.method["roles"]["alts"] == ["lv3"]


def test_decide_alt_when_both_differ_and_alt_is_clearly_closer(monkeypatch):
    """둘 다 대본과 다르면 대본과 글자 닮음이 ALT_MARGIN 넘게 높은 쪽. 비슷하면 기준."""
    plain, alt = list(SAID), list(SAID)
    plain[3] = "저는 지베서 채글 일고 싶어요."                        # 네 글자 틀림
    alt[3] = "저는 집에서 책을 읽고 싶어오."                          # 한 글자 틀림
    script, res = decide_case(plain, alt=alt)
    op = by_el(res)[L["책"]]
    assert (op["op"], op["decided_by"], chosen(res, op)) == ("changed", "rule", "asr:lv3")
    assert heard_of(res, op) == "저는 집에서 책을 읽고 싶어오." and res.method["decide"]["alt_closer"] == 1
    assert res.stats["added"] == 0
    plain[3] = "저는 집에서 책을 익고 싶어요."                        # 둘 다 한 글자씩 틀림: 기준을 지킨다
    script, res = decide_case(plain, alt=alt)
    op = by_el(res)[L["책"]]
    assert chosen(res, op) == "asr:plain" and res.method["decide"]["base_kept"] == 1
    plain[3] = "저는 지베서 채글 일고 싶어요."
    monkeypatch.setattr(align, "ALT_MARGIN", 1.0)                    # 여유가 아주 크면 대안을 고르지 않는다
    script, res = decide_case(plain, alt=alt)
    assert chosen(res, by_el(res)[L["책"]]) == "asr:plain"


def test_decide_does_not_pick_text_the_recogniser_masked():
    """가려 적은 글(한글 옆의 X)이 든 후보는 다른 후보가 있으면 고르지 않는다. 다른 후보가 없으면 그대로 쓴다."""
    assert all(align.masked(t) for t in ("X됐다", "새X", "씨X새끼가!", "미친X야", "이 XX 봐라", "ㅅㄲ야", "뭐야 ㅅㅂ"))
    assert not any(align.masked(t) for t in ("DNA 검사", "A9요", "Xbox 샀어", "엑스레이"))
    plain, alt = list(SAID), list(SAID)
    plain[1] = "그러게요. 산채기나 갈가요?"                           # 세 글자 틀림
    alt[1] = "그러게요. 산책이나 갈X요?"                              # 한 글자를 가려 적음: 글자로는 더 닮았다
    script, res = decide_case(plain, alt=alt)
    assert chosen(res, by_el(res)[L["산책"]]) == "asr:plain"
    script, res = decide_case(alt, alt=plain)                        # 기준이 가려 적었으면 덜 닮았어도 대안
    op = by_el(res)[L["산책"]]
    assert chosen(res, op) == "asr:lv3" and heard_of(res, op) == "그러게요. 산채기나 갈가요?"
    assert res.method["decide"]["alt_unmasked"] == 1
    script, res = decide_case(alt)                                   # 다른 후보가 없으면 가려 적은 글이라도 쓴다
    assert heard_of(res, by_el(res)[L["산책"]]) == "그러게요. 산책이나 갈X요?"


def test_decide_keeps_base_when_alt_found_the_line_somewhere_else(monkeypatch):
    """대안은 기준 범위와 같은 소리 구간(겹치거나 틈이 SAME_NEAR_S 초 이하)일 때만 고른다."""
    plain = list(SAID)
    plain[0] = "오늘 날시가 정말 좋네요."                             # 11.0 ~ 12.2 초
    alt = [(SAID[0], {"at": 3.0}), (SAID[1], {"at": 13.2})] + SAID[2:]      # 같은 줄을 8초 앞에서 찾았다
    script, res = decide_case(plain, alt=alt)
    op = by_el(res)[L["날씨"]]
    assert (op["op"], chosen(res, op)) == ("changed", "asr:plain") and res.method["decide"]["alt_elsewhere"] == 1
    # 낱말 시각이 pass 마다 달라 전혀 겹치지 않아도, 틈이 작으면 같은 한마디다
    plain = SAID[:2] + ["네."] + SAID[3:]                             # 13.2+0.9+1.0 = 15.1 ~ 15.4 초
    alt = SAID[:2] + [("예.", {"at": 14.7})] + [(SAID[3], {"at": 16.4})] + SAID[4:]       # 14.7 ~ 15.0 초: 0.1초 떨어짐
    script, res = decide_case(plain, alt=alt)
    op = by_el(res)[L["예1"]]
    assert (op["op"], chosen(res, op), heard_of(res, op)) == ("same", "asr:lv3", "예.")
    assert res.stats["added"] == 0                                   # 기준의 '네.' 는 그 줄의 소리다: added 로 남지 않는다
    monkeypatch.setattr(align, "SAME_NEAR_S", -1.0)
    script, res = decide_case(plain, alt=alt)
    assert chosen(res, by_el(res)[L["예1"]]) == "asr:plain"


def test_decide_takes_alt_only_line_unless_it_holds_another_lines_sound():
    """기준이 못 맞춘 줄을 대안만 맞췄으면 받아들인다. 그 범위가 다른 줄이 가진 낱말과 겹치면 받아들이지 않는다."""
    plain = SAID[:3] + [("음 뭐 그러든가", {"p": 0.5})] + SAID[4:]
    script, res = decide_case(plain, alt=SAID)
    op = by_el(res)[L["책"]]
    assert (op["op"], op["decided_by"], chosen(res, op)) == ("same", "rule", "asr:lv3")
    assert [(c["src"], c["text"]) for c in op["candidates"][:2]] == [("asr:plain", "음 뭐 그러든가"), ("asr:lv3", SAID[3])]
    assert res.stats["added"] == 0 and res.method["decide"]["alt_only"] == 1      # 그 자리의 기준 낱말은 added 가 아니다
    plain = SAID[:3] + SAID[4:]                                      # 기준에서는 그 자리에 바로 다음 줄의 말이 있다
    alt = SAID[:3] + [(SAID[3], {"gap": 1.0})] + [(u, {"gap": 0.1}) for u in SAID[4:]]
    asr = S.asr_of(speak(plain), speak(alt, pass_id="lv3"))
    res = align.build_ops(S.tiny_script(FILM), asr, 600.0)
    op = by_el(res)[L["책"]]
    assert op["op"] == "absent" and res.method["decide"]["alt_only_rejected"] == 1
    assert sounds_twice(res, asr) == [] and timeline.cross_pass_overlaps(res.ops, timeline.asr_index(asr), 0.0) == []


def test_base_words_under_a_line_taken_from_another_pass_go_only_where_that_line_says_them():
    """대안으로 고른 줄을 기준도 맞췄을 때, 기준 범위의 낱말 가운데 대안 줄이 같은 글로 적은 것은 그 줄의 소리라 버린다
    (낱말 시각이 어긋나 시간으로는 안 겹쳐도). 대안 줄에 없는 말은 버리지 않고 added 로 남긴다."""
    turns = [("1", "dialogue", t) for t in (SAID[0], "그 미친놈이 쫓아오면 어떡해.", SAID[3])]
    head = [(w, 10.0 + 0.3 * k, 10.3 + 0.3 * k) for k, w in enumerate(SAID[0].split())]
    tail = [(w, 30.0 + 0.3 * k, 30.3 + 0.3 * k) for k, w in enumerate(SAID[3].split())]
    mid = [("그", 20.0, 20.2), ("미친놈이", 20.2, 20.8), ("쫓아오면", 20.8, 21.4)]
    plain = S.say(head + mid + [("어떻게", 21.4, 21.9), ("해?", 21.9, 22.2)] + tail)
    lv3 = S.say(head + mid + [("어떡해", 21.4, 21.95)] + tail, "lv3")     # 기준의 '해?' 는 대안 줄이 끝난 뒤에 있다
    assert not align._same_word(21.9, 22.2, 21.4, 21.95)
    script, res = run(turns, plain, lv3)
    op = by_el(res)["E0002"]
    assert (op["op"], chosen(res, op), heard_of(res, op)) == ("same", "asr:lv3", "그 미친놈이 쫓아오면 어떡해")
    assert res.stats["added"] == 0                                   # '해?' 가 따로 남으면 같은 소리가 두 번 담긴다
    turns[1] = ("1", "dialogue", "야. 그냥 시키는 대로 하라고.")
    plain = S.say(head + [("시키는대로", 20.0, 20.5), ("해", 20.5, 20.6), ("이", 20.6, 20.7), ("새끼야.", 20.7, 21.0)] + tail)
    lv3 = S.say(head + [("시키는", 20.0, 20.35), ("대로", 20.35, 20.5)] + tail, "lv3")
    script, res = run(turns, plain, lv3)
    op = by_el(res)["E0002"]
    assert (chosen(res, op), heard_of(res, op)) == ("asr:lv3", "시키는 대로")
    assert [heard_of(res, o) for o in res.ops if o["op"] == "added"] == ["해 이 새끼야."]     # 대안 줄에 없는 기준의 말은 남는다
    # 기준이 그 줄로 맞춘 범위 안의 낱말이라도, 대안 줄이 적지 않았고 시간도 안 겹치면 남긴다('힘 빼' 를 '임빼' 로 들은 뒤쪽 말)
    turns, said = big_film(["손 줘봐. 힘 빼."], [("손잡아.", {"gap": 1.0}), ("임빼", {"gap": 2.0})])
    plain = speak(said)
    lv3 = speak(BIG[:10] + [("손 줘봐.", {"gap": 1.0}), (BIG[10], {"gap": 3.0})] + BIG[11:], pass_id="lv3")
    lines, _ = align.script_lines(S.tiny_script(turns))
    core = {m.line: (m.w0, m.w1) for m in align.match_pass(lines, align.PassWords("plain", plain["words"]))}
    assert core[10] == (70, 71)                                      # 기준은 두 낱말을 다 그 줄로 맞췄다(다시 짝짓기)
    script, res = run(turns, plain, lv3, duration=2000.0)
    op = by_el(res)["E0011"]
    assert (chosen(res, op), heard_of(res, op)) == ("asr:lv3", "손 줘봐.")
    assert [heard_of(res, o) for o in res.ops if o["op"] == "added"] == ["임빼"]


def test_line_only_another_pass_heard_inside_the_pause_of_a_chosen_line_is_not_taken():
    """기준이 한 줄로 맞춘 범위(가운데 긴 쉼)의 쉼 안에서 대안만 다른 줄을 들었다. 받아들이면 두 줄의 시간이 겹친다
    (한 줄이 다른 줄 안에 들어앉는다): 받아들이지 않는다."""
    turns = [("1", "dialogue", t) for t in (SAID[0], "예. 지금요? 이 부장이 너 오랜다.", "저기 잠깐만 기다려 주세요.", SAID[3])]
    plain = [SAID[0], ("예. 지금요?", {"at": 20.0}), ("이 부장이 너 오랜다.", {"at": 26.0}), (SAID[3], {"at": 40.0})]
    alt = plain[:2] + [("저기 잠깐만 기다려 주세요.", {"at": 22.5})] + plain[2:]
    asr = S.asr_of(speak(plain), speak(alt, pass_id="lv3"))
    script = S.tiny_script(turns)
    res = align.build_ops(script, asr, 600.0)
    rows = ops_of(script, res)
    assert [st for _, st, _ in rows] == ["same", "same", "absent", "same"] and res.method["decide"]["alt_only_rejected"] == 1
    assert timeline.cross_pass_overlaps(res.ops, timeline.asr_index(asr), 0.0) == []


def test_short_line_only_the_alternate_matched_needs_a_chosen_neighbour_nearby():
    """대안만 맞춘 짧은 줄도 고른 이웃 줄이 가까이 있어야 받아들인다(대안 pass 안에서 이웃이던 줄이 빠졌을 수 있다)."""
    turns = [("1", "dialogue", t) for t in (SAID[0], SAID[1], "예.", SAID[3], SAID[4])]
    lines, _ = align.script_lines(S.tiny_script(turns))
    base = align.PassWords("plain", speak([SAID[0], (SAID[4], {"at": 100.0})])["words"])
    alt = align.PassWords("lv3", S.say([("예", 50.0, 50.3)], "lv3")["words"])
    passes = {"plain": base, "lv3": alt}
    m_base = {m.line: m for m in align.match_pass(lines, base)}
    assert sorted(m_base) == [0, 4]
    lone = align.LineMatch(2, lines[2].el, 0, 0, 1, 1, 1, 1.0, "lines")
    choice, stats = align.decide(lines, passes, {"plain": m_base, "lv3": {2: lone}})
    assert sorted(choice) == [0, 4] and stats["alt_only_rejected"] == 1 and stats["alt_only"] == 0
    near = align.PassWords("plain", speak([SAID[0], (SAID[1], {"at": 45.0}), (SAID[4], {"at": 100.0})])["words"])
    m_near = {m.line: m for m in align.match_pass(lines, near)}
    choice, stats = align.decide(lines, {"plain": near, "lv3": alt}, {"plain": m_near, "lv3": {2: lone}})
    assert sorted(choice) == [0, 1, 2, 4] and choice[2].pass_id == "lv3" and stats["alt_only"] == 1


def test_line_from_another_pass_does_not_share_a_sound_with_its_neighbour():
    """배우가 '향숙이' 를 두 번 말했고 기준은 둘 다, 다른 pass 는 한 번만 적었다. 다른 pass 의 줄을 고르면 그 줄이 앞 줄의
    끝 낱말과 같은 소리를 갖게 된다: 고르지 않는다(ctx 든 대안이든)."""
    turns = [("1", "dialogue", t) for t in (
        "오늘 아침에 시장에서 누구를 만났는지 알아?", "그래 너가 맨날 따라다니던 향숙이.",
        "향숙이 빤스를 그걸 얼굴에다 덮어 씌우던데.", "그 얘기는 이제 그만하고 밥이나 먹으러 가자.")]

    def seq(words, t0):
        out, t = [], t0
        for w in words:
            w, p = w if isinstance(w, tuple) else (w, 0.9)
            out.append((w, round(t, 3), round(t + 0.3, 3), p))
            t += 0.35
        return out, t

    p1, t = seq(turns[0][2].split(), 10.0)
    p2, t = seq(turns[1][2].split(), t + 1.0)
    p3, t3 = seq(["향숙이", "빤스를", "그걸", ("얼굴끮다", 0.4), "덮어", "씌우던데."], t + 0.5)
    p4, _ = seq(turns[3][2].split(), t3 + 1.0)
    # 다른 pass: 둘째 줄 끝의 '향숙이.' 를 빠뜨리고 셋째 줄 머리의 '향숙이' 를 그 시각(둘째 줄 끝)에 적었다. 틀린 낱말은 바르게
    c2 = p2[:-1]
    c3 = [("향숙이", *p2[-1][1:])] + [("얼굴에다", w[1], w[2], 0.9) if w[0] == "얼굴끮다" else w for w in p3[1:]]
    for pid, ctx in (("ctx", "script_hotwords"), ("lv3", None)):
        asr = S.asr_of(S.say(p1 + p2 + p3 + p4), S.say(p1 + c2 + c3 + p4, pid, ctx))
        res = align.build_ops(S.tiny_script(turns), asr, 60.0)
        assert sounds_twice(res, asr) == []
        assert all(o["asr"][0].startswith("plain:") for o in res.ops) and res.stats["added"] == 0
        d = res.method["decide"]
        assert d["ctx_clash" if pid == "ctx" else "alt_clash"] == 1


def test_decide_ctx_when_ctx_is_the_script_and_sound_backs_it():
    """plain 이 다르고 ctx 가 대본과 같으면, 같은 소리 구간이고 ctx 확률이 떨어지지 않을 때 ctx(acoustic)."""
    plain = list(SAID)
    plain[0] = ("오늘 날시가 정말 좋네요.", {"p": 0.6})
    script, res = decide_case(plain, SAID)
    op = by_el(res)[L["날씨"]]
    assert (op["op"], op["decided_by"], chosen(res, op)) == ("same", "acoustic", "asr:ctx")
    assert op["asr"][0].startswith("ctx:") and heard_of(res, op) == "오늘 날씨가 정말 좋네요."
    assert not any(o["op"] == "added" for o in res.ops)             # ctx 줄 밑의 plain 낱말은 added 가 아니다
    assert res.method["decide"]["ctx_same"] == 1 and "log-probability" in res.method["acoustic"]


def test_decide_keeps_plain_when_ctx_was_pulled_to_the_script():
    """배우가 다르게 말한 곳을 ctx 가 대본대로 적었으면(확률이 뚝 떨어짐) plain 을 지킨다(acoustic)."""
    plain = list(SAID)
    plain[0] = "오늘 날씨가 진짜 좋네요."
    ctx = list(SAID)
    ctx[0] = ("오늘 날씨가 정말 좋네요.", {"p": 0.2})
    script, res = decide_case(plain, ctx)
    op = by_el(res)[L["날씨"]]
    assert (op["op"], op["decided_by"], chosen(res, op)) == ("changed", "acoustic", "asr:plain")
    assert heard_of(res, op) == "오늘 날씨가 진짜 좋네요." and res.method["decide"]["ctx_kept_base"] == 1


def test_decide_keeps_plain_when_ctx_wrote_the_line_at_another_time():
    """ctx 범위가 plain 범위와 같은 소리 구간이 아니면(멀리 떨어진 시각) 확률이 받쳐 줘도 plain 을 지킨다."""
    plain = list(SAID)
    plain[0] = ("오늘 날시가 정말 좋네요.", {"p": 0.6})               # 11.0 ~ 12.2 초
    ctx = [(SAID[0], {"at": 3.0}), (SAID[1], {"at": 13.2})] + SAID[2:]
    script, res = decide_case(plain, ctx)
    op = by_el(res)[L["날씨"]]
    assert (op["op"], op["decided_by"], chosen(res, op)) == ("changed", "acoustic", "asr:plain")


def test_decide_scores_only_the_words_that_differ():
    """긴 줄에서 한 낱말만 끌려가도 잡는다: 평균은 두 pass 가 다르게 적은 낱말끼리만 낸다."""
    pw_a = align.PassWords("plain", S.say([("가", 0, 1, 0.9), ("나", 1, 2, 0.9), ("다", 2, 3, 0.9), ("라", 3, 4, 0.9)])["words"])
    pw_b = align.PassWords("ctx", S.say([("가", 0, 1, 0.9), ("나", 1, 2, 0.9), ("마", 2, 3, 0.2), ("라", 3, 4, 0.9)])["words"])
    la, lb = align._differing_logp(pw_a, (0, 3), pw_b, (0, 3))
    assert la == pytest.approx(math.log(0.9)) and lb == pytest.approx(math.log(0.2))
    la, lb = align._differing_logp(pw_a, (0, 3), pw_a, (0, 3))      # 다른 낱말이 없으면 줄 전체 평균
    assert la == lb == pytest.approx(math.log(0.9))


def test_decide_ctx_when_both_differ_and_ctx_is_closer():
    """둘 다 대본과 다르면 plain. ctx 가 대본에 더 가깝고 확률이 받쳐 주면 ctx."""
    plain = list(SAID)
    plain[3] = ("저는 지베서 채글 읽고 싶어요.", {"p": 0.6})
    ctx = list(SAID)
    ctx[3] = "저는 집에서 채글 읽고 싶어요."
    script, res = decide_case(plain, ctx)
    op = by_el(res)[L["책"]]
    assert (op["op"], op["decided_by"], chosen(res, op)) == ("changed", "acoustic", "asr:ctx")
    ctx[3] = ("저는 집에서 채글 읽고 싶어요.", {"p": 0.1})          # 확률이 안 받치면 plain
    plain[3] = ("저는 지베서 채글 읽고 싶어요.", {"p": 0.9})
    script, res = decide_case(plain, ctx)
    op = by_el(res)[L["책"]]
    assert (op["op"], op["decided_by"], chosen(res, op)) == ("changed", "acoustic", "asr:plain")


def test_decide_ctx_when_both_differ_and_ctx_is_much_more_certain(monkeypatch):
    """둘 다 대본과 다르고 ctx 가 대본에 더 가깝지도 않지만, 다르게 적은 낱말의 확률이 CTX_BETTER 넘게 높으면 ctx."""
    plain = list(SAID)
    plain[3] = ("저는 집에서 채글 읽고 싶어요.", {"p": 0.3})
    ctx = list(SAID)
    ctx[3] = ("저는 집에서 책을 일꼬 싶어요.", {"p": 0.9})           # 대본과의 닮음은 plain 과 같다(두 글자씩 틀림)
    lines, _ = align.script_lines(S.tiny_script(FILM))
    key = next(ln.key for ln in lines if ln.el == L["책"])
    assert textnorm.similarity(key, textnorm.key(ctx[3][0])) == textnorm.similarity(key, textnorm.key(plain[3][0]))
    script, res = decide_case(plain, ctx)
    op = by_el(res)[L["책"]]
    assert (op["decided_by"], chosen(res, op)) == ("acoustic", "asr:ctx") and res.method["decide"]["ctx_closer"] == 1
    monkeypatch.setattr(align, "CTX_BETTER", 1e9)
    script, res = decide_case(plain, ctx)
    assert chosen(res, by_el(res)[L["책"]]) == "asr:plain"


def test_decide_takes_ctx_only_line_when_sound_backs_it():
    """아무도 못 맞춘 줄을 ctx 만 맞췄으면: 글자가 닮고, 확률이 문턱 이상이고, 다른 줄과 겹치지 않을 때만."""
    plain = SAID[:3] + [("음 뭐 그러든가", {"p": 0.5})] + SAID[4:]
    script, res = decide_case(plain, SAID)
    op = by_el(res)[L["책"]]
    assert (op["op"], op["decided_by"], chosen(res, op)) == ("same", "acoustic", "asr:ctx")
    assert res.method["decide"]["ctx_only"] == 1
    assert [(c["src"], c["text"]) for c in op["candidates"][:2]] == [("asr:plain", "음 뭐 그러든가"), ("asr:ctx", SAID[3])]
    assert not any(o["op"] == "added" for o in res.ops)             # 그 시간대의 plain 낱말은 added 로 세지 않는다


def test_decide_rejects_ctx_only_line_when_plain_heard_something_else_clearly():
    """그 자리에 plain 이 대본에 없는 다른 말을 높은 확률로 받아썼으면, ctx 가 대본 줄을 낮은 확률로 적었어도 받아들이지 않는다
    (둘 다 맞춘 줄과 같은 기준: ctx 확률이 그 plain 낱말보다 tau 넘게 낮으면 안 된다)."""
    plain = SAID[:3] + [("저기요 잠깐만요 이것 좀 보세요.", {"p": 0.9})] + SAID[4:]
    ctx = SAID[:3] + [(SAID[3], {"p": 0.6})] + SAID[4:]
    script, res = decide_case(plain, ctx)
    op = by_el(res)[L["책"]]
    assert op["op"] == "absent" and res.method["decide"]["ctx_only_outvoted"] == 1
    assert [heard_of(res, o) for o in res.ops if o["op"] == "added"] == ["저기요 잠깐만요 이것 좀 보세요."]
    ctx[3] = (SAID[3], {"p": 0.9})                                   # ctx 확률도 그만큼 높으면 받아들인다
    script, res = decide_case(plain, ctx)
    assert chosen(res, by_el(res)[L["책"]]) == "asr:ctx"


def test_decide_rejects_hint_echo():
    """말 없는 자리에 ctx 가 대본 줄을 그대로 적은 것(힌트 되풀이, 낮은 확률)은 받아들이지 않는다."""
    plain = SAID[:3] + [(SAID[4], {"gap": 5.0})] + SAID[5:]         # '책' 줄은 말하지 않음(5초 침묵)
    ctx = SAID[:3] + [(SAID[3], {"p": 0.1}), (SAID[4], {"gap": 2.8})] + SAID[5:]
    script, res = decide_case(plain, ctx)
    op = by_el(res)[L["책"]]
    assert op["op"] == "absent" and op["asr"] is None
    assert res.method["decide"]["ctx_only_rejected"] == 1 and res.method["decide"]["ctx_only"] == 0


def test_decide_rejects_ctx_only_line_over_another_lines_words():
    """ctx 만 맞춘 줄의 시간대가 다른 줄의 낱말과 겹치면 받아들이지 않는다."""
    plain = SAID[:3] + SAID[4:]                                      # '책' 줄 없이 바로 다음 줄
    ctx = SAID[:3] + [(SAID[3], {"gap": 1.0})] + [(u, {"gap": 0.1}) for u in SAID[4:]]
    script = S.tiny_script(FILM)
    lines, _ = align.script_lines(script)
    passes = {p["id"]: align.PassWords(p["id"], p["words"], None, p["context"])
              for p in (speak(plain), speak(ctx, pass_id="ctx", context="script_hotwords"))}
    matches = {pid: {m.line: m for m in align.match_pass(lines, pw)} for pid, pw in passes.items()}
    li = next(ln.idx for ln in lines if ln.el == L["책"])
    assert li not in matches["plain"] and li in matches["ctx"]
    mc = matches["ctx"][li]
    assert any(passes["plain"].s[w] < passes["ctx"].end(mc.w0, mc.w1) and passes["plain"].e[w] > passes["ctx"].start(mc.w0, mc.w1)
               for w in range(passes["plain"].n))                    # 그 시간대에 다른 줄의 plain 낱말이 있다
    choice, stats = align.decide(lines, passes, matches)
    assert li not in choice and stats["ctx_only_rejected"] == 1


def test_same_word_is_judged_on_the_shorter_words_length():
    """서로 다른 pass 의 두 낱말이 같은 소리인가: 겹친 시간이 짧은 쪽 길이의 절반 이상. 시작이 앞으로 늘어난 낱말도 잡는다."""
    assert align._same_word(10.0, 10.4, 10.1, 10.5)                  # 0.3초 겹침, 짧은 쪽 0.4초
    assert align._same_word(8.0, 10.4, 10.0, 10.4)                   # 2.4초로 늘어난 낱말과 제 길이 낱말
    assert not align._same_word(10.0, 10.4, 10.3, 10.9)              # 0.1초 겹침: 이웃 낱말
    assert not align._same_word(10.0, 10.4, 10.4, 10.8)              # 맞닿기만 함
    assert align._same_word(10.2, 10.2, 10.0, 10.4) and not align._same_word(9.0, 9.0, 10.0, 10.4)   # 길이 0 인 낱말


def test_decide_measures_tau_and_scale_offset_on_the_film():
    """줄이 충분하면 tau 와 두 pass 의 확률 눈금 차이(ctx_bias)를 그 영화에서 잰다: ctx 확률이 늘 낮게 나와도 견줄 수 있다."""
    names = "가나다라마바사아자차카타파하거너더러머버서어저처"
    texts = [f"{c}씨는 {k + 2}번 창구에서 {c}{c} 서류를 받아 가세요." for k, c in enumerate(names)]
    turns = [("1", "dialogue", t) for t in texts]
    plain = [(t, {"p": 0.9}) for t in texts]
    plain[5] = (texts[5].replace("서류를", "설류를"), {"p": 0.9})
    ctx = [(t, {"p": 0.6}) for t in texts]                          # 같은 글인데 ctx 확률은 늘 0.6
    script, res = run(turns, speak(plain), speak(ctx, pass_id="ctx", context="script_hotwords"))
    d = res.method["decide"]
    assert d["tau_lines"] == len(texts) - 1 and d["tau"] == align.TAU_LO
    assert d["ctx_bias"] == round(math.log(0.6) - math.log(0.9), 4)
    op = by_el(res)["E0006"]
    assert (op["op"], op["decided_by"], chosen(res, op)) == ("same", "acoustic", "asr:ctx")
    ctx[5] = (texts[5], {"p": 0.3})                                  # 그 줄만 눈금 차이보다 훨씬 낮으면 plain
    script, res = run(turns, speak(plain), speak(ctx, pass_id="ctx", context="script_hotwords"))
    assert chosen(res, by_el(res)["E0006"]) == "asr:plain"


def test_plain_alone_is_enough():
    """asr.json 에 plain 만 있어도 돈다. 이름이 plain 이 아닌 pass 하나만 있어도 그것을 기준으로 삼는다."""
    script, res = run(FILM, speak(SAID))
    assert res.method["acoustic"] is None and set(res.method["passes"]) == {"plain"}
    assert res.method["decide"]["tau"] is None and res.method["roles"]["base"] == "plain"
    script, res = run(FILM, speak(SAID, pass_id="base"))
    assert all(o["asr"][0].startswith("base:") for o in res.ops) and res.method["base_pass"] == "base"
    script, res = run(FILM, S.say([]))
    assert res.match == 0.0 and all(o["op"] == "absent" for o in res.ops)
    script, res = run(FILM, speak(SAID, pass_id="ctx", context="script_hotwords"))      # 대본 없이 받아쓴 pass 가 없어도 돈다
    assert res.method["base_pass"] == "ctx" and res.match == 1.0


# ------------------------------------------------------------------ rescue
def rescue_pass(utts):
    return speak(utts, pass_id="rescue", context="full_window_stable")


def test_decide_takes_rescue_line_from_the_gap():
    """아무 pass 도 못 맞춘 줄을 rescue(받아쓰지 않은 틈)가 맞췄으면 받아들인다(rule)."""
    plain = SAID[:3] + [(SAID[4], {"gap": 5.0})] + SAID[5:]
    at = 10.0 + 1.0 + 4 * 0.3 + 1.0 + 3 * 0.3 + 1.0 + 0.3 + 1.0      # '예.' 다음 틈
    script, res = decide_case(plain, plain, rescue=[(SAID[3], {"at": at})])
    op = by_el(res)[L["책"]]
    assert (op["op"], op["decided_by"], op["asr"][0]) == ("same", "rule", "rescue:0")
    assert chosen(res, op) == "asr:rescue" and res.method["passes"]["rescue"]["chosen_lines"] == 1
    assert res.method["roles"]["rescue"] == "rescue" and res.stats["added"] == 0
    # 창 밖(다른 줄들 너머)에 있는 rescue 낱말은 그 줄로 쓰지 않는다
    script, res = decide_case(plain, plain, rescue=[(SAID[3], {"at": 300.0})])
    assert by_el(res)[L["책"]]["op"] == "absent"


def test_rescue_word_is_not_given_to_two_lines():
    """옮겨진 씬의 경계에서는 못 맞춘 줄 묶음의 창이 겹친다. 같은 rescue 낱말을 두 줄이 쓰지 않는다(한 번 난 소리는 한 줄)."""
    a = ["오늘 아침에 시장에서 누구를 만났는지 알아?", "글쎄 나는 전혀 짐작도 못 하겠는데.", "삼 년 전에 이사 간 옆집 아저씨를 만났어."]
    b = ["내일은 비가 많이 온다고 하니까 우산을 챙겨.", "일기예보가 맞은 적이 한 번도 없잖아요.", "그래도 혹시 모르니까 가방에 넣어 둬라."]
    c = ["저녁에는 다 같이 모여서 고기를 구워 먹자.", "좋아요 제가 장을 봐 올게요 걱정 마세요.", "상추하고 마늘도 잊지 말고 꼭 사 와야 한다."]
    rep = "죄송합니다 정말 죄송합니다 형님."
    turns = [("1", "dialogue", t) for t in a + [rep]] + [("2", "dialogue", t) for t in b + [rep]] + \
            [("3", "dialogue", t) for t in c]
    # 영화: 씬 2 가 맨 앞, 씬 1, [말 구간 검출이 놓친 '죄송합니다…' 한 번], 씬 3
    plain = speak(b + [(a[0], {"gap": 20.0})] + a[1:] + [(c[0], {"gap": 12.0})] + c[1:], gap=0.6)
    at = max(w["e"] for w in plain["words"][:len(" ".join(b + a).split())]) + 2.0
    script, res = run(turns, plain, rescue_pass([(rep, {"at": at})]), duration=300.0)
    used = [w for o in res.ops if o["asr"] and o["asr"][0].startswith("rescue:")
            for w in range(timeline.op_range(o)[1], timeline.op_range(o)[2] + 1)]
    assert used and len(used) == len(set(used)) and timeline.shared_words(res.ops) == []
    assert sum(o["op"] == "absent" for o in res.ops) == 1            # 두 '죄송합니다…' 가운데 한 줄만 들렸다


def test_rescue_line_that_lost_its_words_is_matched_again_with_what_is_left():
    """같은 rescue 낱말을 두 줄이 원하면 맞은 글자가 많은 줄이 갖고, 진 줄은 남은 낱말로 다시 찾는다(통째로 버리지 않는다)."""
    lines, _ = align.script_lines(S.tiny_script([("1", "dialogue", t) for t in (
        "오늘 아침에 시장에서 누구를 만났는지 알아?", "죄송합니다 형님.", "죄송합니다 정말 죄송합니다 형님.",
        "내일은 비가 많이 온다고 하니까 우산을 챙겨.")]))
    words = S.say([(w, 50.0 + 0.4 * k, 50.3 + 0.4 * k) for k, w in enumerate("죄송합니다 형님 죄송합니다 정말 죄송합니다 형님".split())],
                  "rescue", "full_window_stable")["words"]
    passes = {"plain": align.PassWords("plain", speak([lines[0].text, (lines[3].text, {"gap": 60.0})])["words"]),
              "rescue": align.PassWords("rescue", words, None, "full_window_stable")}
    roles = align.pass_roles([("plain", None), ("rescue", "full_window_stable")])
    choice = {m.line: align.Choice("plain", m.w0, m.w1, m.how, "rule") for m in align.match_pass(lines, passes["plain"])}
    assert sorted(choice) == [0, 3]
    calls = []
    real = align.match_window

    def window(lines_, todo, pw, F, cuts=frozenset()):
        calls.append((list(todo), list(F)))
        if len(calls) == 1:                          # 첫 판: 두 줄이 같은 낱말(2~5)을 원한다고 하자
            got = real(lines_, todo, pw, F, cuts)
            m = got[2]
            got[1] = align.LineMatch(1, lines_[1].el, 4, 5, 6, 7, 7, 0.9, "lines")
            return {1: got[1], 2: m}
        return real(lines_, todo, pw, F, cuts)

    align.match_window = window
    try:
        found = align._rescue_lines(lines, choice, passes, roles, align._Claims(passes), 200.0)
    finally:
        align.match_window = real
    assert len(calls) == 2 and calls[1] == ([1], [0, 1])             # 진 줄만, 남은 낱말로 다시
    assert {k: (m.w0, m.w1) for k, m in found.items()} == {1: (0, 1), 2: (2, 5)}


def test_rescue_line_does_not_run_across_speech_another_pass_wrote():
    """rescue 낱말은 다른 pass 의 말 사이사이에 흩어져 있다. 한 줄의 범위가 그 말을 건너뛰어 이어지지 않는다."""
    turns = [("1", "dialogue", t) for t in (SAID[0], "아, 왜 이래.", SAID[3])]
    plain = speak([SAID[0], ("누구세요 거기 누구 없어요", {"at": 20.0}), (SAID[3], {"at": 40.0})])
    rescue = rescue_pass([("아,", {"at": 15.0}), ("왜 이래?", {"at": 30.0})])      # 두 조각 사이에 plain 의 말이 있다
    script, res = run(turns, plain, rescue)
    rows = ops_of(script, res)
    assert ("아, 왜 이래.", "changed", "왜 이래?") in rows              # 뒤 조각만 그 줄이다(15초를 건너뛴 '아, 왜 이래?' 가 아니다)
    assert [h for _, st, h in rows if st == "added"] == ["아,", "누구세요 거기 누구 없어요"]      # plain 의 말이 사라지지 않는다
    rescue = rescue_pass([("아,", {"at": 30.0}), ("왜 이래?", {"at": 30.5})])      # 이어져 있으면 한 줄
    script, res = run(turns, plain, rescue)
    assert ("아, 왜 이래.", "same", "아, 왜 이래?") in ops_of(script, res)


def test_unused_rescue_words_become_added_lines_only_where_nothing_else_was_heard():
    """줄에 안 쓰인 rescue 낱말: 기준 pass 가 받아쓰지 않은 곳에 있고 어느 줄의 시간 범위와도 겹치지 않으면 added 로 만든다."""
    plain = speak([SAID[0], (SAID[1], {"at": 30.0}), (SAID[3], {"at": 50.0}), ("뭐라고 했어", {"at": 70.0})])
    rescue = S.say([("이쪽으로", 20.0, 20.4, 0.8), ("오세요.", 20.4, 20.8, 0.8),        # 아무도 못 들은 곳: added
                    ("그러게요.", 28.6, 30.3, 0.9),                                    # plain 의 '그러게요.'(30.0~30.3)와 같은 소리
                    ("저기요", 30.5, 30.7, 0.9),                                       # 둘째 줄(30.0~30.9)의 시간 범위 안
                    ("감사합니다.", 60.0, 60.4, 0.3),                                  # 헛소리 꼴, 낮은 확률: added 지만 conf 가 낮다
                    ("뭐라고", 69.0, 70.3, 0.9)],                                      # plain 의 added 낱말과 같은 소리
                   "rescue", "full_window_stable")
    asr = S.asr_of(plain, rescue, vad=[[10.0, 80.0]])
    asr["chunks"] = [{"start": 10.9, "end": 12.3, "lang": "ko"}, {"start": 29.9, "end": 31.0, "lang": "ko"},
                     {"start": 49.9, "end": 51.6, "lang": "ko"}, {"start": 69.9, "end": 70.7, "lang": "ko"}]
    script = S.tiny_script(FILM[:5])
    res = align.build_ops(script, asr, 600.0)
    added = [(heard_of(res, o), o["asr"]) for o in res.ops if o["op"] == "added"]
    assert added == [("이쪽으로 오세요.", ["rescue:0", "rescue:1"]), ("감사합니다.", ["rescue:4", "rescue:4"]),
                     ("뭐라고 했어", ["plain:12", "plain:13"])]
    assert sounds_twice(res, asr) == [] and timeline.cross_pass_overlaps(res.ops, timeline.asr_index(asr), 0.0) == []
    conf = {heard_of(res, o): res.conf[o["item"]] for o in res.ops if o["op"] == "added"}
    assert conf["이쪽으로 오세요."] == pytest.approx(0.8 * align.CONF_RESCUE) and conf["이쪽으로 오세요."] < 0.8     # 낱말 확률 평균 × rescue 계수
    assert conf["감사합니다."] == pytest.approx(0.3 * align.CONF_RESCUE * align.CONF_NOISE, abs=1e-3)
    assert res.method["passes"]["rescue"]["added_lines"] == 2 and res.method["passes"]["plain"]["added_lines"] == 1
    # 기준 pass 의 묶음(chunks) 안에 든 rescue 낱말은 기준 낱말과 겹치지 않아도 쓰지 않는다
    asr["chunks"][0]["end"] = 25.0
    res = align.build_ops(script, asr, 600.0)
    assert [heard_of(res, o) for o in res.ops if o["op"] == "added"] == ["감사합니다.", "뭐라고 했어"]


def test_leftover_rescue_words_join_the_rescue_line_by_the_same_attach_rules():
    """rescue 줄 옆의 남은 rescue 낱말은 기준 pass 와 같은 붙이기 규칙으로 그 줄에 붙는다."""
    plain = SAID[:3] + [(SAID[4], {"gap": 5.0})] + SAID[5:]
    at = 10.0 + 1.0 + 4 * 0.3 + 1.0 + 3 * 0.3 + 1.0 + 0.3 + 1.0
    script, res = decide_case(plain, rescue=[("어, " + SAID[3], {"at": at})])
    op = by_el(res)[L["책"]]
    assert (op["op"], op["asr"], heard_of(res, op)) == ("changed", ["rescue:0", "rescue:5"], "어, " + SAID[3])
    assert res.stats["added"] == 0


# ------------------------------------------------------------------ conf, 1차 맞춤
def test_conf_is_high_for_clean_lines_and_low_for_noise_and_doubtful_absents():
    said = SAID[:4] + [("감사합니다.", {"gap": 3.0, "p": 0.3})] + SAID[6:]
    script, res = run(FILM, speak(said), vad=[[10.0, 40.0]])
    ops = by_el(res)
    assert res.conf[ops[L["날씨"]]["item"]] == 0.95                 # 0.5 × 1.0 + 0.5 × 0.9
    noise = next(o for o in res.ops if o["op"] == "added")
    assert res.conf[noise["item"]] == pytest.approx(0.3 * align.CONF_NOISE) and res.conf[noise["item"]] < 0.2
    doubtful = res.conf[ops[L["영화"]]["item"]]                     # 그 자리에 설명 안 된 말이 있다
    script, res2 = run(FILM, speak(SAID[:4] + SAID[6:]))
    sure = res2.conf[by_el(res2)[L["영화"]]["item"]]
    assert doubtful < sure == align.CONF_ABSENT_TOP
    assert align.op_conf(script, S.asr_of(speak(SAID[:4] + SAID[6:])), res2.ops) == [res2.conf[o["item"]] for o in res2.ops]
    assert all(c is None or 0.0 <= c <= 1.0 for c in res.conf.values())


def test_conf_of_a_noise_shaped_line_is_cut_only_as_far_as_its_words_are_uncertain():
    """인사말·되풀이 꼴은 진짜 대사일 수 있다: 또렷이 들렸으면(낱말 확률이 높으면) 깎지 않는다. 자막 제작 문구는 늘 깎는다."""
    def added_conf(text, p):
        script, res = run(FILM, speak(SAID[:4] + [(text, {"gap": 3.0, "p": p})] + [(SAID[4], {"gap": 3.0})] + SAID[5:]))
        (op,) = [o for o in res.ops if o["op"] == "added"]
        assert heard_of(res, op) == text
        return res.conf[op["item"]]

    half = align.CONF_NOISE + (1 - align.CONF_NOISE) * 0.5
    assert added_conf("감사합니다.", 0.95) == 0.95 and added_conf("네, 네, 네, 네.", 0.95) == 0.95
    assert added_conf("감사합니다.", (align.NOISE_P_LO + align.NOISE_P_HI) / 2) == pytest.approx(0.7 * half, abs=1e-3)
    assert added_conf("감사합니다.", 0.3) == pytest.approx(0.3 * align.CONF_NOISE, abs=1e-3)
    assert added_conf("문 열어!", 0.3) == 0.3                        # 헛소리 꼴이 아니면 낱말 확률 평균 그대로
    # 자막 제작 문구 꼴은 added 로 만들지 않는다(method.noise_dropped, ``drop_noise``). 그런 줄이 든 ops 의 conf 식은 그대로 깎는다
    said = SAID[:4] + [("한글 자막 제공", {"gap": 3.0, "p": 0.95})] + [(SAID[4], {"gap": 3.0})] + SAID[5:]
    script, res = run(FILM, speak(said))
    n = sum(len(t.split()) for t in SAID[:4])
    assert [o for o in res.ops if o["op"] == "added"] == []
    assert res.method["noise_dropped"] == [{"asr": [f"plain:{n}", f"plain:{n + 2}"], "why": "credit"}]
    ops = res.ops + [{"item": "I9999", "op": "added", "script_el": None, "asr": [f"plain:{n}", f"plain:{n + 2}"],
                      "match": None, "decided_by": "rule"}]
    assert align.op_conf(script, S.asr_of(speak(said)), ops)[-1] == pytest.approx(0.95 * align.CONF_NOISE, abs=1e-3)


def test_conf_rises_when_two_passes_wrote_the_same_text_and_falls_when_they_differ():
    """대본 없이 받아쓴 다른 pass 가 그 줄에 같은 글을 적었으면 낱말 확률 몫을 올리고, 다른 글을 적었으면 낮춘다."""
    said = list(SAID)
    said[0] = ("오늘 날씨가 진짜 좋네요.", {"p": 0.6})
    sim = textnorm.similarity(textnorm.key(SAID[0]), textnorm.key(said[0][0]))

    def conf_of(*passes):
        script, res = run(FILM, *passes)
        op = by_el(res)[L["날씨"]]
        assert chosen(res, op) == "asr:plain" and op["op"] == "changed"
        return res.conf[op["item"]]

    alone = conf_of(speak(said))
    assert alone == pytest.approx(0.5 * sim + 0.5 * 0.6, abs=1e-3)
    agreed = conf_of(speak(said), speak(said, pass_id="lv3"))
    assert agreed == pytest.approx(0.5 * sim + 0.5 * (1 - 0.4 * align.CONF_AGREE), abs=1e-3) and agreed > alone
    other = list(SAID)
    other[0] = "오늘 날씨가 아주 좋네요."                             # 대본과의 닮음은 같고 글은 다르다
    differed = conf_of(speak(said), speak(other, pass_id="lv3"))
    assert differed == pytest.approx(alone * align.CONF_DISAGREE, abs=1e-3) and differed < alone
    # ctx 는 대본을 본 pass 라 같은 글을 적어도 올리지 않는다. 대안을 골랐고 기준이 다른 글이면 낮춘다
    assert conf_of(speak(said), speak(said, pass_id="ctx", context="script_hotwords")) == alone
    script, res = run(FILM, speak(said), speak(SAID, pass_id="lv3"))
    op = by_el(res)[L["날씨"]]
    assert chosen(res, op) == "asr:lv3" and res.conf[op["item"]] == pytest.approx(0.95 * align.CONF_DISAGREE, abs=1e-3)


def test_conf_of_a_weak_match_between_absent_lines_is_cut():
    """대본 앞뒤의 말 있는 대사가 모두 absent 인데 약하게 맞은 줄은 우연한 짝일 수 있다: conf 를 낮춘다."""
    turns = [("1", "dialogue", t) for t in (
        SAID[0], "영상에서 잘려 나간 앞 줄입니다.", "어제 저녁에 본 그 영화는 어땠습니까", "영상에서 잘려 나간 뒤 줄입니다.", SAID[3])]
    said = [SAID[0], ("어제 본 영화는 정말 어땠나요", {"gap": 3.0}), (SAID[3], {"gap": 3.0})]
    script, res = run(turns, speak(said))
    rows = ops_of(script, res)
    assert [st for _, st, _ in rows] == ["same", "absent", "changed", "absent", "same"]
    op = by_el(res)["E0003"]
    sim = op["text_score"]
    assert sim < align.LONELY_SIM
    lonely = res.conf[op["item"]]
    assert lonely == pytest.approx((0.5 * sim + 0.5 * 0.9) * align.CONF_LONELY, abs=2e-3) and lonely < 0.5 * sim + 0.5 * 0.9 - 0.05
    said[1:1] = [turns[1][2]]                                        # 앞 줄을 말했으면 깎지 않는다
    script, res = run(turns, speak(said))
    op = by_el(res)["E0003"]
    assert res.conf[op["item"]] == pytest.approx(0.5 * op["text_score"] + 0.5 * 0.9, abs=2e-3)


def test_absent_conf_counts_the_neighbours_unmatched_words_and_an_echo_of_the_line(monkeypatch):
    """absent 줄의 말이 옆 줄의 범위 안에 들어가 있으면(그 줄의 대본과 안 맞은 낱말) 영상에 없다고 믿기 어렵다."""
    turns = [("1", "dialogue", t) for t in (
        SAID[0], "그게 무슨 소리야 지금 장난해 그런 법이 어딨어.", "왜 말이 안 돼, 임마.", SAID[3])]
    said = [SAID[0], "그게 무슨 소리야 왜 말이 안 돼 인마 그런 법이 어딨어.", SAID[3]]      # 셋째 줄의 말이 둘째 줄 한가운데에
    script, res = run(turns, speak(said))
    rows = ops_of(script, res)
    assert [st for _, st, _ in rows] == ["same", "changed", "absent", "same"] and res.stats["added"] == 0
    absent = by_el(res)["E0003"]
    assert res.conf[absent["item"]] == round(align.CONF_ABSENT_TOP - align.CONF_ABSENT_DROP, 3)       # 0.35
    # 시간으로는 모자라도(필요한 시간이 아주 길다고 치면) 그 줄의 글이 옆 줄의 안 맞은 글자에 들어 있으면 낮춘다(메아리)
    monkeypatch.setattr(align, "SEC_PER_CHAR", 100.0)
    echo = textnorm.similarity("왜말이안돼임마", "왜말이안돼인마")
    conf = align.op_conf(script, S.asr_of(speak(said)), res.ops)[res.ops.index(absent)]
    assert conf == round(align.CONF_ABSENT_TOP - align.CONF_ABSENT_DROP * echo, 3) and echo >= align.ECHO_MIN
    monkeypatch.setattr(align, "ECHO_MIN", 2.0)                      # 메아리를 끄면 높아진다
    assert align.op_conf(script, S.asr_of(speak(said)), res.ops)[res.ops.index(absent)] > 0.9
    monkeypatch.undo()
    monkeypatch.setattr(align, "SEC_PER_CHAR", 100.0)
    monkeypatch.setattr(align, "ECHO_MIN_CHARS", 99)                 # 짧은 줄로 치면 메아리를 보지 않는다
    assert align.op_conf(script, S.asr_of(speak(said)), res.ops)[res.ops.index(absent)] > 0.9


def test_absent_conf_looks_a_little_outside_the_gap_between_its_neighbours(monkeypatch):
    """대본 순서 밖에서 말한 짧은 줄: 그 줄은 absent, 같은 말은 앞 줄보다 먼저 added 로 나온다. 앞뒤 맞춘 줄 사이만 보면
    설명 안 된 말이 없어 0.95 가 된다. 양쪽으로 ABSENT_PAD 초 넓혀 본다."""
    turns = [("1", "dialogue", t) for t in (SAID[0], SAID[1], "나와 봐.", SAID[3])]
    said = [SAID[0], "나와 봐.", SAID[1], SAID[3]]                    # 셋째 줄을 둘째 줄보다 먼저 말함
    script, res = run(turns, speak(said))
    rows = ops_of(script, res)
    assert ("나와 봐.", "absent", "") in rows and ("", "added", "나와 봐.") in rows
    absent = by_el(res)["E0003"]
    assert res.conf[absent["item"]] == round(align.CONF_ABSENT_TOP - align.CONF_ABSENT_DROP, 3)
    monkeypatch.setattr(align, "ABSENT_PAD", 0.0)
    assert align.op_conf(script, S.asr_of(speak(said)), res.ops)[res.ops.index(absent)] == align.CONF_ABSENT_TOP


def test_absent_conf_counts_speech_no_word_covers(monkeypatch):
    """말 구간(vad)인데 낱말이 덮지 못한 곳도 설명 안 된 말이다(FREE_MIN 초 이상). 필요한 시간은 줄 글자 × SEC_PER_CHAR."""
    said = SAID[:4] + [(SAID[6], {"gap": 10.0})] + SAID[7:]           # 씬 2(영화·지루)를 말하지 않음
    plain = speak(said)
    end = max(w["e"] for w in plain["words"][:13])                   # '책' 줄이 끝난 시각
    script, res = run(FILM, plain, vad=[[end, end + 0.6]])            # 그 뒤 0.6초가 받아쓰지 않은 말 구간
    ops = by_el(res)
    need = align.SEC_PER_CHAR * sum(len(textnorm.key(t)) for t in SAID[4:6])
    want = round(align.CONF_ABSENT_TOP - align.CONF_ABSENT_DROP * 0.6 / need, 3)
    assert 0.6 / need < 1 and res.conf[ops[L["영화"]]["item"]] == res.conf[ops[L["지루"]]["item"]] == want
    asr = S.asr_of(plain, vad=[[end, end + 0.6]])
    monkeypatch.setattr(align, "FREE_MIN", 0.7)                      # 그보다 짧은 틈은 말로 치지 않는다
    assert align.op_conf(script, asr, res.ops)[res.ops.index(ops[L["영화"]])] == align.CONF_ABSENT_TOP


def test_prealign_gives_ranges_and_windows():
    """1차 맞춤: 맞춘 줄은 낱말 범위와 제 구간, 못 맞춘 줄은 앞뒤 맞춘 줄 사이의 시간 창."""
    script = S.tiny_script(FILM)
    asr = S.asr_of(speak(SAID[:4] + SAID[6:]))
    pre = align.prealign(script, asr, 600.0)
    rows = {r["el"]: r for r in pre["lines"]}
    assert pre["pass"] == "plain" and len(pre["lines"]) == 9 and 0 < pre["match"] < 1
    hit = rows[L["날씨"]]
    assert hit["matched"] and hit["words"] == [0, 3] and hit["windows"] == [[hit["start"], hit["end"]]]
    miss = rows[L["영화"]]
    assert not miss["matched"] and miss["words"] is None and miss["start"] is None
    assert miss["windows"] == [[rows[L["책"]]["end"], rows[L["출발"]]["start"]]] == rows[L["지루"]]["windows"]
    pre = align.prealign(script, S.asr_of(speak(SAID[6:] + SAID[:4])), 600.0)       # 옮긴 씬의 경계: 창 둘
    assert len({r["el"]: r for r in pre["lines"]}[L["영화"]]["windows"]) == 2
    empty = align.prealign(script, S.asr_of(S.say([])), 600.0)
    assert all(r["windows"] == [[0.0, 600.0]] for r in empty["lines"])


# ------------------------------------------------------------------ align.json, 실행
def final_from(script, res, duration):
    """timeline 뼈대로 final.json 을 조립한다(다음 단계가 할 일의 가장 작은 꼴. validate 로 뼈대 약속을 확인)."""
    els = {e["id"]: e for e in script["elements"]}
    ops = {o["item"]: o for o in res.ops}
    items = []
    for it in res.items:
        if it["type"] != "dialogue":
            e = els[it["src_el"]]
            row = {"id": it["id"], "type": it["type"], "scene": it["scene"], "text": e["text"], "start": it["start"],
                   "end": it["end"], "timing": it["timing"], "src_el": it["src_el"]}
            if it["type"] == "onscreen":
                row["kind"] = e["kind"]
            items.append(row)
            continue
        e = els.get(it["src_el"])
        heard = it["heard"]
        status, diffs = textnorm.verdict(e["text"], heard) if e is not None and heard else (it["status"], [])
        assert status == it["status"]
        items.append({
            "id": it["id"], "type": "dialogue", "scene": it["scene"],
            "speakers": e["speakers"] if e else [], "speaker_from": "script" if e else "unknown",
            "how": e["how"] if e else "unknown", "lang": "ko", "heard": heard, "ko": heard,
            "ko_src": "same" if heard else None, "start": it["start"], "end": it["end"], "timing": it["timing"],
            "words": it["words"], "status": it["status"], "match": ops[it["id"]]["match"],
            "script_ref": {"el": e["id"], "text": e["text"]} if e else None, "diffs": diffs,
            "paren": e["paren"] if e else [], "sung": False, "overlap": None, "conf": res.conf[it["id"]],
            "src_el": it["src_el"]})
    spans = {s["id"]: s for s in timeline.scene_spans(script, res.items, duration)}
    scenes = [dict(sc, start=spans[sc["id"]]["start"], end=spans[sc["id"]]["end"], timing=spans[sc["id"]]["timing"])
              for sc in script["scenes"]]
    st = res.stats
    return {"schema": "sink.final/1",
            "film": {"id": "local_0001", "title": "시험", "title_orig": None, "year": None,
                     "video": {"sha1": "0" * 40, "duration": duration, "lang": "ko"},
                     "script": {"doc_id": script["doc_id"], "version": "unknown", "lang": "ko", "translation": False,
                                "match": res.match}},
            "made": {"pipeline": "0.1.0", "asr": "synth", "aligner": res.method["aligner"], "translator": None},
            "people": [{"id": p["id"], "name": p["name"], "aliases": p["aliases"]} for p in script["people"]],
            "scenes": scenes, "items": items,
            "quality": {"dialogue": st["dialogue"], "same": st["same"], "changed": st["changed"], "added": st["added"],
                        "absent": st["absent"], "speaker_unknown": st["added"], "mean_word_p": st["mean_word_p"]}}


@pytest.fixture
def proj(tmp_path, monkeypatch):
    """임시 프로젝트: 문서 fm1 = 영화 local_0001, 영상 칸이 찬 catalog 한 행."""
    for name, sub in (("ROOT", ""), ("DOCS", "docs"), ("FILMS", "films"), ("CACHE", "cache"), ("STATE", "cache/state"),
                      ("PROBLEMS", "cache/problems"), ("CATALOG", "catalog.csv"), ("QA", "qa")):
        monkeypatch.setattr(paths, name, tmp_path / sub if sub else tmp_path)
    row = {k: "" for k in catalog.FIELDS}
    row.update(doc_id="fm1", post_id="1", category="ko_film", files="raw/a.hwp", sha1="1" * 40, real_format="hwp5",
               title_raw="시험", title_norm="시험", script_lang="ko", is_translation="false", version_label="unknown",
               film_id="local_0001", film_title="시험", video_path="videos/local_0001.mp4", video_sha1="0" * 40,
               video_duration="600.000", video_lang="ko", is_primary="true", extract_quality="0.990", status="parsed")
    catalog.write_catalog([row])
    return tmp_path


def put_film(turns, *passes, vad=None):
    script = S.tiny_script(turns)
    util.write_json(paths.DOCS / "fm1" / "script.json", script)
    util.write_json(paths.FILMS / "local_0001" / "work" / "asr.json", S.asr_of(*passes, vad=vad))
    return script


ARGS = argparse.Namespace(cmd="align", docs=None, limit=None, force=False, workers=None, paths=[])


def test_run_writes_align_json_that_validates_and_updates_catalog(proj):
    """끝에서 끝까지: align.json 이 검사를 통과하고(스키마, 낱말 참조, asr_sha1), catalog 에 match 와 aligned 가 적힌다."""
    said = SAID[:3] + ["저는 집에서 그냥 책을 읽고 싶습니다.", ("누구세요?", {"gap": 2.0}), (SAID[6], {"gap": 2.0})] + SAID[7:]
    script = put_film(FILM + [("3", "dialogue", ".......")], speak(said))
    assert align.run(ARGS) == 0
    out = paths.FILMS / "local_0001" / "work" / "align.json"
    asr_p, script_p = out.parent / "asr.json", paths.DOCS / "fm1" / "script.json"
    assert validate.validate_align(out, asr_path=asr_p, script_path=script_p) == []
    data = util.read_json(out)
    assert list(data) == ["schema", "script_doc", "asr_sha1", "method", "ops", "scenes"]
    assert data["script_doc"] == "fm1" and data["asr_sha1"] == util.sha1_file(asr_p)
    assert {"anchor", "acoustic", "forced", "translate", "constants", "decide", "conf", "stats"} <= set(data["method"])
    assert data["method"]["translate"] is None and "forced alignment" in data["method"]["forced"]
    assert [o["item"] for o in data["ops"]] == sorted(o["item"] for o in data["ops"])
    assert sorted(o["op"] for o in data["ops"]) == ["absent"] * 3 + ["added"] + ["changed"] + ["same"] * 6
    row = catalog.load()[0]
    assert row["status"] == "aligned" and row["match"] == f"{data['method']['stats']['match']:.3f}"
    assert util.all_problems() == []
    # final 단계가 같은 뼈대를 다시 얻어 만든 final.json 도 검사를 통과하고 align.json 과 1:1 로 맞는다(C2)
    res = align.build_ops(script, util.read_json(asr_p), 600.0)
    assert res.ops == data["ops"] and res.scenes == data["scenes"]
    items = timeline.build_timeline(script, data["ops"], util.read_json(asr_p), 600.0)
    assert items == res.items and all(o["item"] == items[int(o["item"][1:]) - 1]["id"] for o in data["ops"])
    final_p = paths.FILMS / "local_0001" / "final.json"
    util.write_json(final_p, final_from(script, res, 600.0))
    assert validate.validate_final(final_p, script_p) == []
    assert validate.validate_align(out, final_path=final_p, asr_path=asr_p, script_path=script_p) == []
    covered = sorted(i["src_el"] for i in util.read_json(final_p)["items"] if i["src_el"])
    assert covered == sorted(e["id"] for e in script["elements"])    # 대본 요소가 빠짐없이 한 번씩


def test_run_is_byte_identical_and_skips_unchanged(proj, capsys):
    """같은 입력이면 align.json 바이트가 같고, 입력이 그대로면 건너뛴다."""
    put_film(FILM, speak(SAID[6:] + SAID[:4]), speak(SAID, pass_id="ctx"), vad=[[10.0, 30.0]])
    out = paths.FILMS / "local_0001" / "work" / "align.json"
    assert align.run(ARGS) == 0
    first = out.read_bytes()
    capsys.readouterr()
    assert align.run(ARGS) == 0 and "'skip': 1" in capsys.readouterr().out
    assert align.run(argparse.Namespace(**dict(vars(ARGS), force=True))) == 0
    assert out.read_bytes() == first
    a, b = (align.build_ops(S.tiny_script(FILM), util.read_json(out.parent / "asr.json"), 600.0) for _ in range(2))
    assert json.dumps(a.ops, ensure_ascii=False) == json.dumps(b.ops, ensure_ascii=False) and a.method == b.method
    # 받아쓰기가 바뀌면 다시 만든다
    util.write_json(out.parent / "asr.json", S.asr_of(speak(SAID)))
    assert align.run(ARGS) == 0 and out.read_bytes() != first
    assert util.read_json(out)["asr_sha1"] == util.sha1_file(out.parent / "asr.json")


def test_run_records_film_problems_without_script_text(proj):
    """일치율이 낮으면 version_mismatch, absent 가 많으면 absent_many. detail 에는 수치만 적는다."""
    put_film(FILM, speak(["전혀 다른 영화의 대사입니다.", "여기는 어디인가요?", SAID[0]]))
    assert align.run(ARGS) == 0
    probs = util.all_problems()
    assert sorted(p["problem"] for p in probs) == ["absent_many", "version_mismatch"]
    assert all(p["level"] == "film" and p["film_id"] == "local_0001" and p["doc_id"] == "fm1" for p in probs)
    assert all(p["path"] == "films/local_0001/work/align.json" for p in probs)
    said_text = "".join(SAID)
    assert not any(w in p["detail"] for p in probs for w in said_text.split() if len(w) > 2)
    assert catalog.load()[0]["status"] == "aligned"


def test_run_targets_only_primary_docs_with_video_and_asr(proj):
    """대상은 영상이 있고 대표 대본이며 asr.json 이 있는 영화뿐이다."""
    util.write_json(paths.DOCS / "fm1" / "script.json", S.tiny_script(FILM))
    assert align.select_docs(ARGS) == []                             # asr.json 없음
    put_film(FILM, speak(SAID))
    assert [d["doc_id"] for d in align.select_docs(ARGS)] == ["fm1"]
    assert align.select_docs(argparse.Namespace(**dict(vars(ARGS), docs="fm2"))) == []
    catalog.finalize(updates={"fm1": {"is_primary": False}})
    assert align.select_docs(ARGS) == []


def test_low_conf_items_are_capped():
    """항목 수준 asr_low_conf 는 conf 가 아주 낮은 것만, 영화당 상한까지."""
    turns = [("1", "dialogue", SAID[0])]
    said = [SAID[0]] + [(f"잡음{k} 소리{k} 말{k}.", {"gap": 2.0, "p": 0.05}) for k in range(align.ITEM_PROBLEMS_MAX + 10)]
    script, res = run(turns, speak(said), duration=2000.0)
    probs = align._problems("fm1", "local_0001", res)
    assert len(probs) == align.ITEM_PROBLEMS_MAX and {p["problem"] for p in probs} == {"asr_low_conf"}
    assert all(p["level"] == "item" and p["item_id"].startswith("I") and p["score"] < align.ITEM_LOW_CONF for p in probs)


FORCE = argparse.Namespace(**dict(vars(ARGS), force=True))


def test_run_keeps_the_previous_align_json_when_the_new_one_fails_the_check(proj, capsys):
    """검사를 못 넘은 align.json 은 쓰지 않는다: 앞선 정상 파일이 그대로 남고, 문제 기록에는 오류 코드별 수만 적힌다."""
    put_film(FILM, speak(SAID))
    assert align.run(ARGS) == 0
    out = paths.FILMS / "local_0001" / "work" / "align.json"
    first = out.read_bytes()
    bad = speak(SAID[:4] + SAID[6:])
    for w in bad["words"]:
        w["w"] = unicodedata.normalize("NFD", w["w"])               # 받아쓰기 낱말이 NFC 가 아니다: 검사 [E2] 에 걸린다
    util.write_json(out.parent / "asr.json", S.asr_of(bad))
    capsys.readouterr()
    assert align.run(ARGS) == 1
    err = capsys.readouterr().err
    assert out.read_bytes() == first and not list(out.parent.glob("*.check"))       # 앞선 파일 그대로, 임시 파일 없음
    (p,) = util.all_problems()
    assert (p["level"], p["problem"], p["film_id"], p["item_id"]) == ("film", "other", "local_0001", "")
    assert re.fullmatch(r"align\.json 검사 실패 (\d+)건\(\[E2\] \1\): align\.json 을 쓰지 않음", p["detail"])
    said = [w for t in SAID for w in t.split() if len(w) > 1]
    assert not any(w in unicodedata.normalize("NFC", p["detail"]) for w in said)    # 대사 글이 들어가지 않는다
    assert "[E2]" in err                                             # 화면에는 오류 글을 보여 준다
    # 단계 지문을 올리지 않았으니 다음 실행이 다시 맞춘다. 받아쓰기를 고치면 통과하고 문제 기록이 지워진다
    assert align.run(ARGS) == 1 and out.read_bytes() == first
    util.write_json(out.parent / "asr.json", S.asr_of(speak(SAID[:4] + SAID[6:])))
    assert align.run(ARGS) == 0 and out.read_bytes() != first and util.all_problems() == []


def test_run_records_an_exception_by_name_only(proj, monkeypatch, capsys):
    """맞추기가 예외로 끝나면 그 영화의 문제 기록을 예외 이름 한 줄로 갈아 치운다(예외 글은 화면에만)."""
    put_film(FILM, speak(["전혀 다른 영화의 대사입니다.", "여기는 어디인가요?", SAID[0]]))
    assert align.run(ARGS) == 0 and len(util.all_problems()) == 2

    def boom(*a, **k):
        raise ValueError("오늘 날씨가 정말 좋네요 에서 멈춤")

    monkeypatch.setattr(align, "build_ops", boom)
    capsys.readouterr()
    assert align.run(FORCE) == 1
    (p,) = util.all_problems()
    assert (p["level"], p["problem"]) == ("film", "other") and "ValueError" in p["detail"] and "날씨" not in p["detail"]
    assert "날씨" in capsys.readouterr().err


def test_ops_stay_in_item_order_past_ten_thousand_items():
    """ops 는 item 번호 차례다(글자 차례가 아니다: I10000 은 I9999 뒤)."""
    turns = [("1", "action", f"지문 {k}") for k in range(9998)] + [("1", "dialogue", SAID[0]), ("1", "dialogue", SAID[3])]
    script, res = run(turns, speak([SAID[0], SAID[3]]))
    assert [o["item"] for o in res.ops] == ["I9999", "I10000"]
    assert all(res.ops[it["op"]]["item"] == it["id"] for it in res.items if it["op"] is not None)


def test_word_times_that_go_backwards_are_counted_and_the_doubtful_line_is_reported(proj):
    """낱말 시작 시각이 거꾸로 가도 맞추기는 끝까지 한다(시각은 timeline 이 바로잡는다). pass 마다 뒤집힌 곳 수를 method 에
    적고, 바로잡아도 믿기 어려운 줄(1초 넘게 뒤집힘)은 final 단계와 같은 글로 항목 수준 문제를 남긴다."""
    from sinkprep import final

    plain = speak(SAID)
    ws = plain["words"]
    ws[2]["s"] = round(ws[1]["s"] - 0.1, 3)                          # 첫 줄: 0.1초 뒤집힘
    ws[10]["s"] = round(ws[9]["s"] - 1.5, 3)                         # '책' 줄: 1.5초 뒤집힘
    script = put_film(FILM, plain)
    assert align.run(ARGS) == 0
    out = paths.FILMS / "local_0001" / "work" / "align.json"
    assert validate.validate_align(out, asr_path=out.parent / "asr.json", script_path=paths.DOCS / "fm1" / "script.json") == []
    data = util.read_json(out)
    ops = {o["script_el"]: o for o in data["ops"]}
    assert [o["op"] for o in data["ops"]] == ["same"] * 9             # 글자와 판정은 그대로
    m = data["method"]
    assert m["passes"]["plain"]["backsteps"] == 2
    assert m["time_fix"] == {"lines": 2, "words": 2, "suspect": [ops[L["책"]]["item"]]}
    (p,) = util.all_problems()
    assert (p["level"], p["item_id"], p["problem"], p["detail"]) == ("item", ops[L["책"]]["item"], "other", align.TIME_SUSPECT_TEXT)
    assert align.TIME_SUSPECT_TEXT == final.TIME_SUSPECT_TEXT         # review 가 두 단계의 기록을 한 줄로 합친다
    res = align.build_ops(script, util.read_json(out.parent / "asr.json"), 600.0)
    line = next(it for it in res.items if it["src_el"] == L["책"])
    assert [w["s"] for w in line["words"]] == sorted(w["s"] for w in line["words"])      # 항목의 낱말 시각은 바로잡힌 값


# ------------------------------------------------------------------ 가짜 영화(진짜 대본)
@pytest.mark.skipif(not REAL_DOC.exists(), reason="docs/fm55890/script.json 이 없음")
def test_synthetic_film_from_a_real_script():
    """진짜 대본으로 만든 가짜 영화: 자리 정밀도·재현율, 힌트 되풀이, 검사 통과, 속도."""
    import eval_align_synth as E

    script = json.loads(REAL_DOC.read_text(encoding="utf-8"))
    others = [" ".join(reversed(e["text"].split())) for e in script["elements"]      # 대본에 없는 말: 어절을 뒤집은 줄
              if e["type"] == "dialogue" and len(e["text"].split()) >= 3][::5]
    mv = S.make_movie(script, others, seed=1)
    assert mv == S.make_movie(script, others, seed=1)               # 씨앗이 같으면 같은 영화
    truth, asr = mv["truth"], mv["asr"]
    t0 = time.perf_counter()
    res = align.build_ops(script, asr, truth["duration"])
    assert time.perf_counter() - t0 < 30
    r = E.score(truth, res, True)
    assert r["pos_p"] >= 0.99 and r["pos_r"] >= 0.9 and r["absent_r"] >= 0.98
    assert r["echo_taken"] <= 1 and r["pull_lost"] <= 1
    plain = align.build_ops(script, S.only_passes(asr, ("plain",)), truth["duration"])
    r = E.score(truth, plain, False)
    assert r["pos_p"] >= 0.99 and r["pos_r"] >= 0.9 and r["added_r"] >= 0.9
    # 차례와 낱말 범위의 약속
    errs = validate._Errs()
    validate._check_untimed_order(res.items, errs)
    assert errs.done() == []
    assert sorted(i["src_el"] for i in res.items if i["src_el"]) == \
        sorted(e["id"] for e in script["elements"] if e["type"] in timeline.ITEM_TYPES)
    for it in res.items:
        if it["type"] == "dialogue" and it["start"] is not None:
            assert it["start"] < it["end"] and it["heard"]
    spans = sorted((timeline.op_range(o) for o in plain.ops if o["asr"]), key=lambda x: x[1])
    assert all(a[2] < b[1] for a, b in zip(spans, spans[1:]))       # plain 낱말 범위는 겹치지 않는다
    # 대본에 없는 말이 대본 줄에 붙은 양과, 영상에 없는 줄(그 자리에 대본에 없는 말이 있음)을 맞췄다고 한 수
    assert r["ghost_n"] > 0 and r["ghost_paired"] <= max(2, r["ghost_n"] // 20)
    assert r["over_added"] <= 0.05 * sum(len(textnorm.key(asr["passes"][0]["words"][w]["w"])) for w in truth["added"])


@pytest.mark.skipif(not REAL_DOC.exists(), reason="docs/fm55890/script.json 이 없음")
def test_synthetic_film_does_not_give_away_where_lines_end():
    """가짜 받아쓰기는 실제처럼 부호를 빼먹고, 대본에 없는 말이 틈 없이 대사에 이어진다. 남은 낱말을 어느 줄에 붙일지를
    틈과 부호만으로 알 수 있으면(대본에 없는 말 앞뒤가 늘 0.35초 넘게 떨어져 있던 옛 생성기) 붙이기 규칙이 채점되지 않는다."""
    script = json.loads(REAL_DOC.read_text(encoding="utf-8"))
    others = [" ".join(reversed(e["text"].split())) for e in script["elements"]
              if e["type"] == "dialogue" and len(e["text"].split()) >= 3][::5]
    mv = S.make_movie(script, others, seed=1)
    words, truth = mv["asr"]["passes"][0]["words"], mv["truth"]
    added = set(truth["added"])
    assert 0.05 < sum(textnorm.ends_sentence(w["w"]) for w in words) / len(words) < 0.3     # 실제 추격자 0.18
    last = [words[tl["plain"][1]]["w"] for tl in truth["lines"].values() if tl["plain"]]
    assert 0.25 < sum(map(textnorm.ends_sentence, last)) / len(last) < 0.75                  # 실제 0.54
    edges = [b["s"] - a["e"] for a, b in zip(words, words[1:]) if (a["i"] in added) != (b["i"] in added)]
    assert len(edges) > 50 and sum(g <= align.ATTACH_GAP for g in edges) / len(edges) > 0.3  # 실제 0.48
    assert all(b["s"] >= a["s"] for a, b in zip(words, words[1:]))
    assert any(tl["ghost"] for tl in truth["lines"].values()) and any(tl["swapped"] for tl in truth["lines"].values())
    assert validate_asr_shape(mv["asr"])


@pytest.mark.skipif(not REAL_DOC.exists(), reason="docs/fm55890/script.json 이 없음")
def test_synthetic_film_with_a_second_recogniser_and_speech_only_the_rescue_pass_heard():
    """대안 pass(lv3)가 든 가짜 영화: 두 후보 가운데 실제 말에 가까운 쪽을 고르고, rescue 의 대본 밖 말만 added 로 만들고,
    어느 소리도 두 줄에 담지 않는다."""
    import eval_align_synth as E

    script = json.loads(REAL_DOC.read_text(encoding="utf-8"))
    others = [" ".join(reversed(e["text"].split())) for e in script["elements"]
              if e["type"] == "dialogue" and len(e["text"].split()) >= 3][::5]
    mv = S.make_movie(script, others, seed=1, alt=True)
    assert mv == S.make_movie(script, others, seed=1, alt=True)
    truth, asr = mv["truth"], mv["asr"]
    assert [(p["id"], p["context"]) for p in asr["passes"]] == [
        ("plain", None), ("lv3", None), ("ctx", "script_hotwords"), ("rescue", "full_window_stable")]
    assert validate_asr_shape(asr) and all(all(b["s"] >= a["s"] for a, b in zip(p["words"], p["words"][1:])) for p in asr["passes"])
    lv3 = asr["passes"][1]["words"]
    both = [tl for tl in truth["lines"].values() if tl["plain"] and tl["alt"]]
    assert len(both) > 200 and any(align.masked(w["w"]) for w in lv3)      # 대안 pass 는 욕설을 가려 적기도 한다('새X')
    assert 0.2 < sum(tl["plain_key"] == tl["alt_key"] for tl in both) / len(both) < 0.8      # 두 pass 는 서로 다른 곳에서 틀린다
    assert truth["rescue_added"] and truth["rescue_dup"] and any(tl["blurred"] for tl in truth["lines"].values())
    chunks = asr["chunks"]                                           # rescue 낱말은 기준 pass 의 묶음 밖에 있다(가운데 시각 기준)
    assert not any(c["start"] <= (w["s"] + w["e"]) / 2 <= c["end"] for w in asr["passes"][3]["words"] for c in chunks)
    a = S.only_passes(asr, ("plain", "lv3", "rescue"))
    res = align.build_ops(script, a, truth["duration"])
    r = E.score(truth, res, True, a)
    base = E.score(truth, align.build_ops(script, S.only_passes(asr, ("plain", "rescue")), truth["duration"]), True,
                   S.only_passes(asr, ("plain", "rescue")))
    assert r["pos_p"] >= 0.99 and r["pos_r"] >= base["pos_r"] and r["status_acc"] >= base["status_acc"] - 0.005
    assert r["pick_n"] > 100 and r["pick_acc"] >= 0.85 and r["heard_sim"] > base["heard_sim"] + 0.02
    assert r["radd_r"] >= 0.9 and r["radd_p"] >= 0.75 and r["radd_dup"] == 0
    assert r["cross"] == 0 and timeline.shared_words(res.ops) == []
    assert res.method["passes"]["lv3"]["chosen_lines"] > 50 and res.method["passes"]["rescue"]["added_lines"] > 10


def validate_asr_shape(asr):
    """낱말 번호가 0부터 이어지고 시각이 뒤집히지 않았는가(줄 경계를 넘어 붙인 낱말 포함)."""
    return all([w["i"] for w in p["words"]] == list(range(len(p["words"]))) and all(w["s"] <= w["e"] for w in p["words"])
               for p in asr["passes"])


# ------------------------------------------------------------------ 0.3.0: 같은 소리로 넓혀 고르기, 여러 pass 다시 짝짓기
HEAD = [(w, 10.0 + 0.3 * k, 10.3 + 0.3 * k) for k, w in enumerate(SAID[0].split())]          # 낱말 0~3
TAIL = [(w, 40.0 + 0.3 * k, 40.3 + 0.3 * k) for k, w in enumerate(SAID[3].split())]


def lmatch(lines, pw, li, a, b, how="anchors"):
    span = pw.span_loc(a, b)
    hit = textnorm.lcs(lines[li].loc, span)
    n, m = len(lines[li].loc), len(span)
    return align.LineMatch(li, lines[li].el, a, b, hit, n, m, round(2.0 * hit / (n + m), 4), how)


def three_lines(mid):
    """[SAID[0], mid, SAID[3]] 세 줄의 Line 목록."""
    return align.script_lines(S.tiny_script([("1", "dialogue", t) for t in (SAID[0], mid, SAID[3])]))[0]


def two_passes(lines, plain_mid, alt_mid, base_line, alt_line, base_core=None, alt_core=None):
    """머리 줄·가운데 줄·꼬리 줄로 된 plain·lv3 두 pass 와 맞춤(가운데 줄의 범위는 가운데 낱말 안의 번호)."""
    base = align.PassWords("plain", S.say(HEAD + plain_mid + TAIL)["words"])
    alt = align.PassWords("lv3", S.say(HEAD + alt_mid + TAIL, "lv3")["words"])
    passes = {"plain": base, "lv3": alt}

    def ms(pw, mid, rng):
        k = len(HEAD) + len(mid)
        return {0: lmatch(lines, pw, 0, 0, 3), 1: lmatch(lines, pw, 1, 4 + rng[0], 4 + rng[1]),
                2: lmatch(lines, pw, 2, k, k + len(TAIL) - 1)}

    matches = {"plain": ms(base, plain_mid, base_line), "lv3": ms(alt, alt_mid, alt_line)}
    cores = {"plain": ms(base, plain_mid, base_core or base_line), "lv3": ms(alt, alt_mid, alt_core or alt_line)}
    return passes, matches, cores


def test_decide_widens_the_base_range_over_head_and_tail_the_other_pass_put_in_the_line(monkeypatch):
    """두 pass 가 한 줄을 따로 맞췄을 때, 다른 pass 가 그 줄로 본 소리에 있는 이쪽 남은 낱말(줄 머리·꼬리를 바꿔 말한 말)은
    그 줄의 말로 넓혀 견준다. 넓히지 않으면 그 말이 그 줄 옆의 added 로 떨어진다(추격자 '일단 데리고 …보죠 … 되는데요')."""
    lines = three_lines("데려가 보죠. 진술하는 게 말이 되네.")
    plain = [("일단", 20.0, 20.3), ("데리고", 20.3, 20.6), ("가보죠", 20.6, 20.9), ("진술하는게", 20.9, 21.3),
             ("말해도", 21.3, 21.6), ("되는데요", 21.6, 21.9)]
    lv3 = [("일단", 20.0, 20.3), ("데리고", 20.3, 20.6), ("와보죠", 20.6, 20.9), ("뭐", 20.9, 21.0), ("진수라는", 21.0, 21.3),
           ("게", 21.3, 21.4), ("말이", 21.4, 21.6), ("되는데요?", 21.6, 21.9)]
    passes, matches, cores = two_passes(lines, plain, lv3, (2, 3), (0, 7))
    choice, _ = align.decide(lines, passes, matches, None, cores)
    assert (choice[1].pass_id, choice[1].w0, choice[1].w1) == ("plain", 4, 9)
    monkeypatch.setattr(align, "SPAN_UNION", 0)
    choice, _ = align.decide(lines, passes, matches, None, cores)
    assert (choice[1].pass_id, choice[1].w0, choice[1].w1) == ("plain", 6, 7)
    # 다른 pass 가 그 줄을 이쪽보다 SPAN_TRUST 넘게 못하게 맞췄으면(엉뚱한 범위일 수 있다) 근거로 쓰지 않는다
    monkeypatch.setattr(align, "SPAN_UNION", 1)
    monkeypatch.setattr(align, "SPAN_TRUST", 0.05)
    choice, _ = align.decide(lines, passes, matches, None, cores)
    assert (choice[1].pass_id, choice[1].w0, choice[1].w1) == ("plain", 6, 7)


def test_decide_widens_the_alternate_range_over_what_the_base_attached_and_picks_it():
    """대안을 고르면 기준이 붙인 꼬리('그려봐 새끼야')가 added 로 떨어지던 줄: 대안의 같은 소리 낱말로 대안 범위를 넓힌다."""
    lines = three_lines("감기는 씨발. 끊어. 내가 전화해 볼께.")
    plain = [("감기는", 20.0, 20.4), ("시발", 20.4, 20.8), ("그려봐", 20.8, 21.4), ("새끼야", 21.4, 21.7)]
    lv3 = [("감기는", 20.0, 20.4), ("씨발", 20.4, 20.8), ("그려봐", 21.0, 21.3), ("새끼야", 21.3, 21.6)]
    passes, matches, cores = two_passes(lines, plain, lv3, (0, 3), (0, 1), base_core=(0, 1))
    choice, stats = align.decide(lines, passes, matches, None, cores)
    assert (choice[1].pass_id, choice[1].w0, choice[1].w1) == ("lv3", 4, 7) and stats["alt_closer"] == 1
    assert passes["lv3"].text(4, 7) == "감기는 씨발 그려봐 새끼야"


def test_decide_counts_base_words_the_alternate_leaves_out_unless_it_heard_them_elsewhere(monkeypatch):
    """대안을 고르면 기준이 그 줄로 맞춘 낱말 가운데 대안 줄이 덮지 못하는 말은 added 로 떨어진다: 그 글자를 대안 쪽의
    안 맞은 글자로 센다('애 잘' 과 '얘 잘 봐'). 대안이 그 소리를 듣고도 어느 줄에도 넣지 않았으면(두 pass 가 엇갈림) 세지 않는다."""
    lines = three_lines("애 잘봐.")
    plain = [("얘", 20.0, 20.2), ("잘", 20.2, 20.4), ("봐", 20.4, 20.6)]
    lv3 = [("애", 20.0, 20.2), ("잘", 20.2, 20.4)]
    passes, matches, cores = two_passes(lines, plain, lv3, (0, 2), (0, 1))
    choice, _ = align.decide(lines, passes, matches, None, cores)
    assert (choice[1].pass_id, choice[1].w0, choice[1].w1) == ("plain", 4, 6)
    monkeypatch.setattr(align, "ALT_CHARGE", 0.0)
    choice, _ = align.decide(lines, passes, matches, None, cores)
    assert choice[1].pass_id == "lv3"
    monkeypatch.setattr(align, "ALT_CHARGE", 1.0)
    # 물음 뒤의 대답: 기준은 '없어요' 까지 그 줄로 맞췄고 대안은 그 말을 따로 적어 두었다
    lines = three_lines("음... 지영민씨, 여자 친구 있어요?")
    plain = [("그럼", 20.0, 20.2), ("여자친구로냐?", 20.2, 21.0), ("없어요", 21.6, 22.3)]
    lv3 = [("지금", 20.0, 20.2), ("여자친구는요?", 20.2, 21.0), ("없어요", 21.6, 22.2)]
    passes, matches, cores = two_passes(lines, plain, lv3, (0, 2), (0, 1), base_core=(1, 2))
    choice, _ = align.decide(lines, passes, matches, None, cores)
    assert (choice[1].pass_id, choice[1].w0, choice[1].w1) == ("lv3", 4, 5)       # 물음표 너머로 넓히지도 않는다
    lv3 = [("지금", 20.0, 20.2), ("여자친구는요?", 20.2, 21.0)]                     # 대안이 그 소리를 못 들었으면 센다
    passes, matches, cores = two_passes(lines, plain, lv3, (0, 2), (0, 1), base_core=(1, 2))
    choice, _ = align.decide(lines, passes, matches, None, cores)
    assert choice[1].pass_id == "plain"
    lv3 = [("지금", 20.0, 20.2), ("여자친구는요?", 20.2, 21.0), ("고맙습니다.", 21.6, 22.2)]     # 전혀 다른 글로 적었어도 센다
    passes, matches, cores = two_passes(lines, plain, lv3, (0, 2), (0, 1), base_core=(1, 2))
    choice, _ = align.decide(lines, passes, matches, None, cores)
    assert choice[1].pass_id == "plain"


def test_widening_needs_the_other_pass_text_and_stops_at_asides_and_question_turns():
    """넓히기의 근거: 다른 pass 가 이 범위 밖에 적은 글이 받쳐 주어야 한다(낱말 시각이 늘어나 옆 소리를 덮은 것만으로는 안 된다).
    바깥 끝의 대답·맞장구·군말은 대본 줄의 그 끝이 같은 말일 때만 둔다. 물음표를 넘어 차례가 바뀌는 곳은 넘지 않는다."""
    say = S.say
    plain = align.PassWords("plain", say([("하늘이", 10.0, 10.3), ("돕는다.", 10.3, 10.5), ("신고해야", 10.5, 10.9),
                                          ("되는", 10.9, 11.1)])["words"])
    lv3 = align.PassWords("lv3", say([("신고해야", 10.2, 10.9), ("되는", 10.9, 11.1)], "lv3")["words"])
    free = [True] * plain.n
    assert align._sounds_in(plain, 1, lv3, 0, 1)                    # 시간으로는 대안 줄의 소리 안이지만
    assert align._extend_over(plain, 2, 3, lv3, 0, 1, free, "경찰에 신고해야 되는 거 아녜요?") == (2, 3)
    words = [("예,", 10.0, 10.2), ("이", 10.2, 10.3), ("사람이", 10.3, 10.6), ("그", 10.6, 10.7)]
    plain = align.PassWords("plain", say(words[:3])["words"])
    lv3 = align.PassWords("lv3", say(words, "lv3")["words"])
    assert align._extend_over(plain, 2, 2, lv3, 0, 3, [True] * 3, "저 사람이 그...") == (1, 2)
    assert align._extend_over(plain, 2, 2, lv3, 0, 3, [True] * 3, "예, 이 사람이 그...") == (0, 2)
    assert align._extend_over(plain, 2, 2, lv3, 0, 3, [True, False, True], "예, 이 사람이 그...") == (2, 2)     # 다른 줄의 낱말
    assert align._extend_over(plain, 2, 2, lv3, 0, 3, [True] * 3, "네, 이 사람이 그...") == (0, 2)       # 예·네 는 같은 대답
    words = [("네,", 10.0, 10.2), ("예", 10.2, 10.4), ("그래요", 10.4, 10.8)]       # 그 대답을 이미 가진 범위에 또 붙이지 않는다
    plain = align.PassWords("plain", say(words)["words"])
    lv3 = align.PassWords("lv3", say(words, "lv3")["words"])
    assert align._extend_over(plain, 1, 2, lv3, 0, 2, [True] * 3, "예, 그래요.") == (1, 2)
    plain = align.PassWords("plain", say([("여자친구로냐?", 10.0, 10.8), ("없어요", 11.3, 11.9)])["words"])
    lv3 = align.PassWords("lv3", say([("여자친구는요?", 10.0, 10.7), ("없어요", 11.3, 11.9)], "lv3")["words"])
    assert align._extend_over(lv3, 0, 0, plain, 0, 1, [True, True], "여자 친구 있어요?") == (0, 0)
    assert align._extend_over(lv3, 0, 0, plain, 0, 1, [True, True], "여자 친구 있어요. 없어요.") == (0, 1)
    plain = align.PassWords("plain", say([("대답", 10.0, 10.3), ("안", 10.3, 10.4), ("해?", 10.4, 10.6),
                                          ("안양이요", 11.0, 11.6)])["words"])
    lv3 = align.PassWords("lv3", say([("대답", 10.0, 10.3), ("안", 10.3, 10.4), ("해?", 10.4, 10.6),
                                      ("안양은", 11.0, 11.6)], "lv3")["words"])
    assert align._extend_over(plain, 3, 3, lv3, 0, 3, [True] * 4, "아, 안양이요.") == (3, 3)      # 앞 사람의 물음
    words = [("저", 10.0, 10.2), ("안에", 10.2, 10.5), ("있는", 10.5, 10.8), ("그래.", 10.8, 11.1)]     # 맞장구도 상대의 말
    plain = align.PassWords("plain", say(words)["words"])
    lv3 = align.PassWords("lv3", say(words, "lv3")["words"])
    assert align._extend_over(plain, 0, 2, lv3, 0, 3, [True] * 4, "저 안에 있는...") == (0, 2)


def test_decide_does_not_widen_over_a_range_the_other_pass_matched_much_worse():
    """다른 pass 가 그 줄을 이쪽보다 SPAN_TRUST 넘게 못하게 맞췄으면(다시 짝짓기로 약하게 맞춘 엉뚱한 범위) 넓히는 근거로
    쓰지 않는다: 앞 사람의 말('세워 둔 차 빼')이 이 줄에 붙지 않는다."""
    lines = three_lines("아, 안양이요.")
    plain = [("세워", 20.0, 20.3), ("둔", 20.3, 20.5), ("차", 20.5, 20.7), ("빼", 20.7, 20.9), ("안양이요", 21.3, 21.9)]
    lv3 = [("세워", 20.0, 20.3), ("둔", 20.3, 20.5), ("차", 20.5, 20.7), ("빼", 20.7, 20.9), ("안양은", 21.3, 21.9)]
    passes, matches, cores = two_passes(lines, plain, lv3, (4, 4), (2, 4))
    choice, _ = align.decide(lines, passes, matches, None, cores)
    assert (choice[1].pass_id, choice[1].w0, choice[1].w1) == ("plain", 8, 8)


def test_decide_looks_again_at_an_alternate_that_clashed_with_a_line_decided_later():
    """대안이 다른 줄(기준 범위)과 소리가 겹쳐 못 골랐는데, 그 줄이 뒤에서 대안으로 바뀌었으면 다시 본다(줄 차례에 매이지 않게)."""
    lines = align.script_lines(S.tiny_script([("1", "dialogue", t) for t in (
        SAID[0], "저 열쇠 갖고 더 뒤져 봐.", "아니 왜요?", SAID[3])]))[0]
    mid_p = [("너도", 20.0, 20.3), ("열쇠", 20.3, 20.6), ("가지고", 20.6, 20.9), ("더", 20.9, 21.2), ("뛰자", 21.2, 21.5),
             ("아니", 21.5, 21.7), ("왜요", 21.7, 22.0)]
    mid_a = [("여자", 20.0, 20.3), ("열쇠", 20.3, 20.6), ("가지고", 20.6, 20.9), ("더", 20.9, 21.2), ("뒤져", 21.2, 21.5),
             ("아니", 21.5, 21.7), ("왜요?", 21.7, 22.0)]
    base = align.PassWords("plain", S.say(HEAD + mid_p + TAIL)["words"])
    alt = align.PassWords("lv3", S.say(HEAD + mid_a + TAIL, "lv3")["words"])
    m_base = {0: lmatch(lines, base, 0, 0, 3), 1: lmatch(lines, base, 1, 4, 7), 2: lmatch(lines, base, 2, 8, 10),
              3: lmatch(lines, base, 3, 11, 15)}
    m_alt = {0: lmatch(lines, alt, 0, 0, 3), 1: lmatch(lines, alt, 1, 4, 8), 2: lmatch(lines, alt, 2, 9, 10),
             3: lmatch(lines, alt, 3, 11, 15)}
    choice, stats = align.decide(lines, {"plain": base, "lv3": alt}, {"plain": m_base, "lv3": m_alt})
    assert [(choice[li].pass_id, choice[li].w0, choice[li].w1) for li in (1, 2)] == [("lv3", 4, 8), ("lv3", 9, 10)]
    assert stats["alt_clash"] == 0 and stats["alt_closer"] == 1 and stats["alt_same"] == 1


def test_decide_breaks_an_exact_tie_by_sound():
    """두 후보의 글자 닮음이 똑같으면 자모 닮음이 뚜렷이 높은 쪽('성모병원이요' 에 '정보병원이요' 와 '전부병원이요')."""
    lines = three_lines("성모병원이요! 전화드릴께요!")
    plain = [("전부병원이요", 20.0, 20.6), ("가세", 20.6, 20.9), ("연락할게요.", 20.9, 21.5)]
    lv3 = [("정보병원이요,", 20.0, 20.6), ("가서", 20.6, 20.9), ("연락할게요", 20.9, 21.5)]
    passes, matches, cores = two_passes(lines, plain, lv3, (0, 2), (0, 2))
    choice, stats = align.decide(lines, passes, matches, None, cores)
    assert choice[1].pass_id == "lv3" and stats["alt_jamo"] == 1


def test_repair_across_passes_pairs_answers_the_rescue_pass_heard():
    """pass 를 다 고른 뒤 고른 줄 사이 창에서 rescue 낱말까지 보고 다시 짝짓는다('예.' 를 rescue 의 '네' 로)."""
    plain = SAID[:2] + [(SAID[3], {"gap": 1.0})] + SAID[4:]          # '예.' 를 기준은 못 들었다
    script, res = decide_case(plain, rescue=[("네", {"at": 14.5})])
    op = by_el(res)[L["예1"]]
    assert (op["op"], op["asr"], heard_of(res, op)) == ("changed", ["rescue:0", "rescue:0"], "네")
    assert res.stats["added"] == 0 and res.method["decide"]["across_slot"] == 1
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(align, "ACROSS_REPAIR", 0)
        script, res = decide_case(plain, rescue=[("네", {"at": 14.5})])
        assert by_el(res)[L["예1"]]["op"] == "absent" and res.stats["added"] == 1


def test_repair_across_takes_an_answer_or_a_closing_word_said_alone_but_not_a_common_head_word():
    """좁은 창에서 대본 줄의 첫 낱말(대답)이나 끝 낱말만 말한 줄: '예? 하하... 나 이거. 참.' 을 '네?' 로, '노량진... 왜?' 를
    '왜' 로. 줄 머리의 '너', '이거' 같은 낱말 하나로는 짝짓지 않는다. 되묻기는 '요' 만 달라도 같은 말('뭐요?' / '뭐?')."""
    for line, said, want in (("예? 하하... 나 이거. 참.", "네?", "changed"), ("노량진... 왜?", "왜", "changed"),
                             ("뭐요?", "뭐?", "changed"), ("너 이거 절대로 쓰면 안 된다.", "너", "absent")):
        turns, utts = big_film([line], [(said, {"gap": 2.0})])
        script, res = run(turns, speak(utts), duration=2000.0)
        assert (line, want) in [(t, st) for t, st, _ in ops_of(script, res)], (line, said)
    assert align._polite_pair("뭐요", "뭐") and align._polite_pair("왜", "왜요") and not align._polite_pair("예요", "예")


def test_common_short_answer_is_taken_only_near_a_matched_neighbour(monkeypatch):
    """우연히 그 창에 있을 기대 횟수가 SLOT_CHANCE 를 넘어도 SLOT_CHANCE_NEAR 이하면, 앞뒤 맞춘 줄 가운데 하나에서
    SHORT_NEAR 초 안에 있을 때만 받는다."""
    monkeypatch.setattr(align, "SLOT_CHANCE", 0.0)
    monkeypatch.setattr(align, "SLOT_CHANCE_NEAR", 1.0)
    for gap, want in ((3.0, "changed"), (30.0, "absent")):
        turns, said = big_film(["예."], [("네", {"gap": gap})])
        said[11] = (said[11], {"gap": gap})
        script, res = run(turns, speak(said), duration=3000.0)
        assert ("예.", want) in [(t, st) for t, st, _ in ops_of(script, res)], gap
    monkeypatch.setattr(align, "SLOT_CHANCE_NEAR", 0.0)
    turns, said = big_film(["예."], [("네", {"gap": 3.0})])
    script, res = run(turns, speak(said), duration=3000.0)
    assert ("예.", "absent") in [(t, st) for t, st, _ in ops_of(script, res)]


def test_answer_common_in_the_film_is_paired_when_said_right_after_its_neighbour():
    """'예' 가 흔한 영화(기대 횟수가 SLOT_CHANCE 와 SLOT_CHANCE_NEAR 사이): 바로 옆 줄 3초 뒤의 '네' 는 그 줄이고, 앞뒤 줄에서
    30초 떨어진 '네' 는 아니다."""
    for gap, want in ((3.0, "changed"), (30.0, "absent")):
        turns, said = big_film(["예."], [("네", {"gap": gap})])
        said = [s if isinstance(s, tuple) or k == 11 else "예 " + s for k, s in enumerate(said)]
        said[11] = (said[11], {"gap": gap})
        script, res = run(turns, speak(said), duration=3000.0)
        assert ("예.", want) in [(t, st) for t, st, _ in ops_of(script, res)], gap


def test_repair_does_not_run_a_line_across_a_cut():
    """다시 짝짓기의 후보 범위는 cuts(가상 pass 에서 pass 가 바뀌는 곳)를 건너 이어지지 않는다."""
    lines = three_lines("거기 있었어?")
    pw = align.PassWords("v", S.say(HEAD + [("거기", 20.0, 20.3), ("있었어?", 20.4, 20.8)] + TAIL)["words"])
    for cuts, want in ((frozenset(), (4, 5)), (frozenset({5}), (5, 5))):
        got = {0: lmatch(lines, pw, 0, 0, 3), 2: lmatch(lines, pw, 2, 6, 10)}
        align._repair(lines, pw, got, cuts)
        assert (got[1].w0, got[1].w1) == want, cuts


def test_repair_across_does_not_join_words_of_two_passes_in_one_line():
    """한 줄의 범위는 한 pass 안에서 이어진 낱말이다: rescue '거기' 와 기준 '있었어?' 를 한 줄로 묶지 않고 끝 낱말로만 짝짓는다."""
    turns = [("1", "dialogue", t) for t in (SAID[0], "거기 있었어?", SAID[3])]
    plain = speak([SAID[0], ("있었어?", {"at": 15.0}), (SAID[3], {"at": 20.0})])
    script, res = run(turns, plain, rescue_pass([("거기", {"at": 14.4})]))
    rows = ops_of(script, res)
    assert ("거기 있었어?", "changed", "있었어?") in rows and ("", "added", "거기") in rows


def test_rescue_word_written_again_just_before_the_same_sound_is_not_added():
    """rescue 낱말 바로 뒤(RESCUE_ECHO_S 안)에 다른 pass 의 낱말이 같은 소리로 시작하면, 그 소리를 시작만 늘려 한 번 더
    적은 것이다('첫' 뒤의 '첫번째'): added 로 만들지 않는다. 다른 소리로 시작하면 그대로 둔다."""
    plain = speak([SAID[0], ("첫번째 계단 밑", {"at": 20.7}), (SAID[3], {"at": 40.0})])
    for word, kept in (("첫", False), ("저기", True)):
        rescue = S.say([(word, 20.0, 20.6, 0.8)], "rescue", "full_window_stable")
        asr = S.asr_of(plain, rescue)
        res = align.build_ops(S.tiny_script([("1", "dialogue", t) for t in (SAID[0], SAID[3])]), asr, 600.0)
        added = [heard_of(res, o) for o in res.ops if o["op"] == "added"]
        assert (word in added) == kept and "첫번째 계단 밑" in added


def test_base_words_the_chosen_alternate_wrote_the_same_way_after_attaching_are_not_added_again():
    """대안 줄을 고르면, 기준이 그 줄에 붙였던 꼬리 가운데 대안 줄이 같은 글로 적은 낱말('해')은 시각이 안 겹쳐도 그 줄의
    소리다(added 로 또 담지 않는다). 대안 줄에 없는 말('이 씨!')은 added 로 남는다."""
    turns = [("1", "dialogue", t) for t in (SAID[0], "아 씨발! 조용하라고!", SAID[3])]
    plain = S.say(HEAD + [("아,", 20.0, 20.2), ("시발", 20.2, 20.5), ("조용히", 20.5, 20.9), ("해,", 20.95, 21.1),
                          ("이", 21.1, 21.2), ("씨!", 21.2, 21.4)] + TAIL)
    lv3 = S.say(HEAD + [("아,", 20.0, 20.2), ("시발!", 20.2, 20.5), ("조용히", 20.5, 20.75), ("해!", 20.75, 20.9)] + TAIL, "lv3")
    script, res = run(turns, plain, lv3)
    op = by_el(res)["E0002"]
    assert (chosen(res, op), heard_of(res, op)) == ("asr:lv3", "아, 시발! 조용히 해!")
    assert [heard_of(res, o) for o in res.ops if o["op"] == "added"] == ["이 씨!"]
    assert sounds_twice(res, S.asr_of(plain, lv3)) == []


# ------------------------------------------------------------------ 0.4.0: LLM 짝 판정(align_llm), 헛소리 묶음 빼기
from sinkprep import align_llm, llm  # noqa: E402

_S_ROW = re.compile(r"^S(\d+) \[(\w+)\] [^:]*: (.*)$")
_H_ROW = re.compile(r"^\d+\.\d: (.*)$")
_E_ROW = re.compile(r"^E(\d+) = S(\d+) (head|tail): \[(\d+)\]\.\.\[(\d+)\]$")
_H_WORD = re.compile(r"\[(\d+)\](\S+)")


class Judge:
    """가짜 claude(align_pair): 요청 글을 읽어 정해 둔 짝을 번호로만 답한다.
    pairs: {대본 글: (들린 낱말 글, kind[, conf])}, joins: {대본 글: 그 줄에 붙는 낱말 글}, noise: [헛소리 낱말 글],
    hook(답, 낱말 [(번호, 글)]) 이 답을 바꿀 수 있다. 정해 두지 않은 줄은 none, 후보는 no."""
    name = "fake"

    def __init__(self, pairs=None, joins=None, noise=(), conf="sure", hook=None):
        self.pairs, self.joins, self.noise, self.conf, self.hook = pairs or {}, joins or {}, noise, conf, hook
        self.calls = 0
        self.users = []

    def __call__(self, req, model, effort, images, user):
        assert req.task == align_llm.TASK and req.prompt_version == align_llm.PROMPT_VERSION
        self.calls += 1
        self.users.append(user)
        rows, words, edges = {}, [], []
        for row in user.split("\n"):
            if m := _S_ROW.match(row):
                rows[int(m.group(1))] = (m.group(2), m.group(3))
            elif m := _E_ROW.match(row):
                edges.append(tuple(int(m.group(k)) if k != 3 else m.group(k) for k in (1, 2, 3, 4, 5)))
            elif m := _H_ROW.match(row):
                words += [(int(n), w) for n, w in _H_WORD.findall(m.group(1))]

        def find(text, lo=1, hi=10 ** 9):
            toks, seq = text.split(), [x for x in words if lo <= x[0] <= hi]
            for k in range(len(seq) - len(toks) + 1):
                if [w for _, w in seq[k:k + len(toks)]] == toks:
                    return seq[k][0], seq[k + len(toks) - 1][0]
            return None

        out = {"lines": [], "edges": [], "noise": []}
        for s, (tag, text) in sorted(rows.items()):
            if tag != "UNMATCHED":
                continue
            want = self.pairs.get(text)
            r = find(want[0]) if want else None
            out["lines"].append({"s": s, "kind": want[1], "from": r[0], "to": r[1],
                                 "conf": want[2] if len(want) > 2 else self.conf} if r else
                                {"s": s, "kind": "none", "conf": "sure"})
        for e, s, side, a, b in edges:
            want = self.joins.get(rows[s][1])
            r = find(want, a, b) if want else None
            out["edges"].append({"e": e, "joins": "yes", "from": r[0], "to": r[1], "conf": self.conf} if r else
                                {"e": e, "joins": "no", "conf": "sure"})
        for t in self.noise:
            if r := find(t):
                out["noise"].append({"from": r[0], "to": r[1], "conf": "sure"})
        if self.hook:
            out = self.hook(out, words) or out
        return {"structured_output": out, "result": json.dumps(out), "modelUsage": {model: {}}}


class NoCall:
    """부르면 안 되는 백엔드."""
    name = "nocall"

    def __call__(self, *a, **k):
        raise AssertionError("캐시만 쓰는 실행이 claude 를 불렀다")


@pytest.fixture
def judged(tmp_path, monkeypatch):
    """LLM 캐시를 임시 폴더로(cache/llm), 실행 상태는 시험마다 새로."""
    monkeypatch.setattr(paths, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(llm, "BACKOFF", (0.01, 0.01, 0.01))
    llm.reset_run()
    yield tmp_path
    llm.set_backend(None)
    llm.reset_run()


def judge_run(turns, said, judge=None, *, calls=True, habits=None, pair=True):
    """가짜 claude 를 끼우고(judge) 맞춘다: (script, AlignResult)."""
    llm.set_backend(judge if judge is not None else NoCall())
    script = S.tiny_script(turns)
    asr = S.asr_of(speak(said)) if isinstance(said, list) else said
    return script, align.build_ops(script, asr, 2000.0, habits=habits, llm_pair=pair, llm_calls=calls, workers=1)


PARA = big_film(["그만 좀 해."], [("됐다 됐어 이제", {"gap": 2.0})])


def test_llm_pair_takes_a_paraphrased_line_the_rules_left_absent(judged, monkeypatch):
    """규칙이 absent 와 added 로 가른 바꿔 말한 줄을 claude 가 그 낱말로 짝지으면 그 줄로 본다(decided_by llm, 판정은
    글로 다시: changed). conf 는 같은 식에 CONF_LLM 을 곱한 값이다."""
    turns, said = PARA
    script, rule = judge_run(turns, said, pair=False)
    assert ("그만 좀 해.", "absent", "") in ops_of(script, rule) and ("", "added", "됐다 됐어 이제") in ops_of(script, rule)
    assert rule.method["llm_pair"] is None
    judge = Judge(pairs={"그만 좀 해.": ("됐다 됐어 이제", "paraphrase")})
    script, res = judge_run(turns, said, judge)
    assert judge.calls == 1
    rows = ops_of(script, res)
    assert ("그만 좀 해.", "changed", "됐다 됐어 이제") in rows and not [r for r in rows if r[1] == "added"]
    op = next(o for o in res.ops if o["op"] == "changed")
    assert op["decided_by"] == "llm" and [c["src"] for c in op["candidates"]] == ["asr:plain", "script"]
    m = res.method["llm_pair"]
    assert (m["windows"], m["requests"], m["asked_lines"], m["accepted"], m["kinds"]) == \
        (1, 1, 1, {"lines": 1, "head": 0, "tail": 0}, {"paraphrase": 1})
    assert (m["model"], m["prompt"], m["task"]) == (llm.MODEL, align_llm.PROMPT_VERSION, "align_pair")
    assert m["rejected"] == {} and m["skipped_windows"] == 0 and len(m["keys"]) == 1
    assert res.llm_run["called"] == 1 and "run" not in m
    assert sum(o["decided_by"] == "llm" for o in res.ops) == 1
    conf = res.conf[op["item"]]
    monkeypatch.setattr(align, "CONF_LLM", 1.0)
    _, again = judge_run(turns, said, None, calls=False)                   # 캐시에서
    assert again.conf[op["item"]] == pytest.approx(conf / 0.8, abs=2e-3) and conf < again.conf[op["item"]]


def test_llm_pair_uses_only_the_cache_without_calls_and_rules_stand_without_cache(judged):
    """부르지 않는 실행(--llm 없음, all)은 캐시에 있는 판정만 쓴다: 백엔드를 부르지 않고, 캐시가 있으면 부른 실행과 같은
    결과, 없으면 규칙 결과 그대로(ops 가 같다). 같은 캐시면 결과가 바이트까지 같다."""
    turns, said = PARA
    llm.reset_run(offline=False)                                       # llm 실행 상태가 '부름' 이어도 부르지 않는다
    script, none = judge_run(turns, said, None, calls=False)
    _, rule = judge_run(turns, said, pair=False)
    assert json.dumps(none.ops, ensure_ascii=False) == json.dumps(rule.ops, ensure_ascii=False)
    m = none.method["llm_pair"]
    assert m["skipped_requests"] == m["requests"] == 1 and m["skipped_windows"] == m["windows"] == 1
    assert none.llm_run["missing"] == 1 and m["keys"] == []
    judge = Judge(pairs={"그만 좀 해.": ("됐다 됐어 이제", "paraphrase")})
    _, called = judge_run(turns, said, judge)
    _, cached = judge_run(turns, said, None, calls=False)
    _, cached2 = judge_run(turns, said, None, calls=False)
    assert judge.calls == 1 and cached.llm_run["cached"] == 1
    dump = [json.dumps([r.ops, r.method], ensure_ascii=False) for r in (called, cached, cached2)]
    assert dump[0] == dump[1] == dump[2]
    assert called.method["llm_pair"]["decisions_sha1"] != m["decisions_sha1"]
    _, again = judge_run(turns, said, judge)                           # 부르는 실행도 캐시에 있으면 다시 부르지 않는다
    assert judge.calls == 1 and again.llm_run == {"cached": 1, "stopped": None, "seconds": 0.0, "errors": []}


def test_llm_pair_ignores_unsure_and_none_answers(judged):
    """sure 가 아닌 판정과 none 은 적용하지 않고 수만 센다."""
    turns, said = big_film(["그만 좀 해.", "밥은 먹었어?"], [("됐다 됐어 이제", {"gap": 2.0}), ("식사는 하셨나", {"gap": 1.0})])
    judge = Judge(pairs={"그만 좀 해.": ("됐다 됐어 이제", "paraphrase", "unsure")})
    script, res = judge_run(turns, said, judge)
    assert sorted(r for r in ops_of(script, res) if r[1] != "same") == [
        ("", "added", "됐다 됐어 이제"), ("", "added", "식사는 하셨나"), ("그만 좀 해.", "absent", ""), ("밥은 먹었어?", "absent", "")]
    m = res.method["llm_pair"]
    assert m["answers"] == {"none": 1 + 2, "unsure": 1, "answered": 2 + 2} and m["accepted"]["lines"] == 0


def test_llm_pair_drops_judgements_that_break_the_structure(judged):
    """구조 제약을 어긴 판정은 버리고 까닭별로 센다: 두 조각에 걸친 범위(segment), 다른 틈의 낱말(gap), 두 줄이 같은
    낱말(overlap), 대본 차례와 시각 차례가 어긋남(order)."""
    # segment: 5초 넘는 쉼으로 갈린 두 조각에 걸친 범위
    turns, said = big_film(["그만 좀 해."], [("됐다 됐어", {"gap": 2.0}), ("이제 됐다니까", {"gap": 6.0})])
    script, res = judge_run(turns, said, Judge(pairs={"그만 좀 해.": ("됐다 됐어 이제 됐다니까", "paraphrase")}))
    assert ("그만 좀 해.", "absent", "") in ops_of(script, res) and res.method["llm_pair"]["rejected"] == {"segment": 1}
    # gap: 다른 틈(앞뒤 맞춘 줄이 다른 곳)의 낱말. 제 틈의 낱말을 받은 줄은 그대로 받는다
    turns = [("1", "dialogue", t) for t in BIG[:5] + ["그만 좀 해."] + BIG[5:10] + ["밥은 먹었어?"] + BIG[10:]]
    said = BIG[:5] + [("됐다 됐어 이제", {"gap": 2.0})] + BIG[5:10] + [("식사는 하셨나", {"gap": 2.0})] + BIG[10:]
    judge = Judge(pairs={"그만 좀 해.": ("식사는 하셨나", "paraphrase"), "밥은 먹었어?": ("식사는 하셨나", "paraphrase")})
    script, res = judge_run(turns, said, judge)
    rows = ops_of(script, res)
    assert ("그만 좀 해.", "absent", "") in rows and ("밥은 먹었어?", "changed", "식사는 하셨나") in rows
    assert res.method["llm_pair"]["rejected"] == {"gap": 1} and res.method["llm_pair"]["windows"] == 2
    # overlap: 두 줄이 같은 낱말을 가지면 둘 다 버린다
    turns, said = big_film(["그만 좀 해.", "밥은 먹었어?"], [("됐다 됐어 이제", {"gap": 2.0}), ("식사는 하셨나", {"gap": 1.0})])
    judge = Judge(pairs={"그만 좀 해.": ("됐다 됐어", "partial"), "밥은 먹었어?": ("됐어 이제", "partial")})
    script, res = judge_run(turns, said, judge)
    assert res.method["llm_pair"]["rejected"] == {"overlap": 2} and res.stats["absent"] == 2

    def forget():                                                # 같은 요청이라 캐시에 남은 앞 판정을 지운다
        for p in (paths.CACHE / "llm").glob("*.json"):
            p.unlink()

    # order: 대본 차례로 앞 줄이 시각으로 뒤 낱말을 가지면 그 틈의 새 줄을 모두 버린다
    forget()
    judge = Judge(pairs={"그만 좀 해.": ("식사는 하셨나", "paraphrase"), "밥은 먹었어?": ("됐다 됐어 이제", "paraphrase")})
    script, res = judge_run(turns, said, judge)
    assert res.method["llm_pair"]["rejected"] == {"order": 2} and res.stats["absent"] == 2
    forget()
    judge = Judge(pairs={"그만 좀 해.": ("됐다 됐어 이제", "paraphrase"), "밥은 먹었어?": ("식사는 하셨나", "paraphrase")})
    script, res = judge_run(turns, said, judge)
    assert [r for r in ops_of(script, res) if r[1] != "same"] == [
        ("그만 좀 해.", "changed", "됐다 됐어 이제"), ("밥은 먹었어?", "changed", "식사는 하셨나")]
    assert res.method["llm_pair"]["rejected"] == {} and res.method["llm_pair"]["accepted"]["lines"] == 2


def test_llm_pair_does_not_give_a_line_a_sound_another_pass_already_chose(judged, monkeypatch):
    """다른 pass 에서 고른 줄과 같은 소리인 범위(``_Claims.clashes``)는 버린다(clash)."""
    turns, said = PARA
    armed = {}
    real_ask, real_clash = align_llm.ask, align._Claims.clashes

    def ask(*a, **k):
        armed["on"] = True                                     # 받아들이기 단계에서만 같은 소리로 보이게
        return real_ask(*a, **k)

    monkeypatch.setattr(align_llm, "ask", ask)
    monkeypatch.setattr(align._Claims, "clashes", lambda self, *a: armed.get("on", False) or real_clash(self, *a))
    script, res = judge_run(turns, said, Judge(pairs={"그만 좀 해.": ("됐다 됐어 이제", "paraphrase")}))
    assert ("그만 좀 해.", "absent", "") in ops_of(script, res) and res.method["llm_pair"]["rejected"] == {"clash": 1}


def test_llm_pair_joins_a_head_that_fell_off_the_line(judged):
    """맞춘 줄과 같은 pass 에서 바로 이어진 added 조각(받아쓰기가 뭉갠 줄 머리 '근데' -> '그게', 쉼이 길어 규칙은 붙이지
    않음)이 그 줄의 머리라고 하면 줄 범위를 넓힌다(decided_by llm). 줄에 닿지 않는 낱말만 붙이라고 하면 버린다(touch)."""
    turns, said = big_film(["근데 왜 안 오지?"], [("그게", {"gap": 2.0}), ("왜 안 오지?", {"gap": 2.5})])
    script, rule = judge_run(turns, said, pair=False)
    assert ("근데 왜 안 오지?", "changed", "왜 안 오지?") in ops_of(script, rule) and ("", "added", "그게") in ops_of(script, rule)
    script, res = judge_run(turns, said, Judge(joins={"근데 왜 안 오지?": "그게"}))
    rows = ops_of(script, res)
    assert ("근데 왜 안 오지?", "changed", "그게 왜 안 오지?") in rows and not [r for r in rows if r[1] == "added"]
    assert by_el(res)["E0011"]["decided_by"] == "llm"
    assert res.method["llm_pair"]["accepted"] == {"lines": 0, "head": 1, "tail": 0}
    # 줄에 닿지 않는 낱말(사이에 다른 낱말이 있음)만 붙이라는 판정
    turns, said = big_film(["자 근데 왜 안 오지?"], [("가 그게", {"gap": 2.0}), ("왜 안 오지?", {"gap": 2.5})])
    script, rule = judge_run(turns, said, pair=False)
    assert ("", "added", "가 그게") in ops_of(script, rule)
    script, res = judge_run(turns, said, Judge(joins={"자 근데 왜 안 오지?": "가"}))
    assert res.method["llm_pair"]["rejected"] == {"touch": 1} and ("", "added", "가 그게") in ops_of(script, res)


def test_llm_noise_lowers_the_conf_of_an_added_line_but_keeps_it(judged):
    """claude 가 헛소리로 본 added 낱말은 지우지 않고 conf 만 CONF_NOISE 배로 낮춘다(method.llm_pair.noise 로 남아 final 도
    같은 값을 얻는다)."""
    turns, said = big_film([], [("파란 하늘 위로 노래가", {"gap": 2.0, "p": 0.6})])
    script, rule = judge_run(turns, said, pair=False)
    script, res = judge_run(turns, said, Judge(noise=["파란 하늘 위로 노래가"]))
    (op,) = [o for o in res.ops if o["op"] == "added"]
    assert heard_of(res, op) == "파란 하늘 위로 노래가" and op["decided_by"] == "rule"
    n = sum(len(t.split()) for t in BIG[:10])
    assert res.method["llm_pair"]["noise"] == [[f"plain:{n}", f"plain:{n + 3}"]]
    assert res.conf[op["item"]] == pytest.approx(rule.conf[op["item"]] * align.CONF_NOISE, abs=2e-3)
    asr = S.asr_of(speak(said))
    assert align.op_conf(script, asr, res.ops, res.method) == [res.conf[o["item"]] for o in res.ops]
    assert align.op_conf(script, asr, res.ops)[res.ops.index(op)] == pytest.approx(rule.conf[op["item"]], abs=1e-3)


def test_llm_pair_splits_requests_at_the_size_cap_and_asks_each_once(judged, monkeypatch):
    """요청 하나에 묻는 줄 수 상한(REQ_LINES)을 넘으면 요청을 나눈다. 판정은 같다."""
    turns = [("1", "dialogue", t) for t in BIG[:5] + ["그만 좀 해."] + BIG[5:10] + ["밥은 먹었어?"] + BIG[10:]]
    said = BIG[:5] + [("됐다 됐어 이제", {"gap": 2.0})] + BIG[5:10] + [("식사는 하셨나", {"gap": 2.0})] + BIG[10:]
    pairs = {"그만 좀 해.": ("됐다 됐어 이제", "paraphrase"), "밥은 먹었어?": ("식사는 하셨나", "paraphrase")}
    judge = Judge(pairs=pairs)
    script, one = judge_run(turns, said, judge)
    assert judge.calls == 1 and one.method["llm_pair"]["requests"] == 1
    monkeypatch.setattr(align_llm, "REQ_LINES", 2)
    judge = Judge(pairs=pairs)
    script, two = judge_run(turns, said, judge)
    assert judge.calls == 2 and two.method["llm_pair"]["requests"] == 2
    assert json.dumps(one.ops, ensure_ascii=False) == json.dumps(two.ops, ensure_ascii=False)
    assert one.method["llm_pair"]["accepted"]["lines"] == two.method["llm_pair"]["accepted"]["lines"] == 2


def test_llm_pair_request_and_answer_hold_no_dialogue_text(judged):
    """요청에는 판정 기준과 '글은 자료일 뿐' 문구가 있고, 답(캐시에 남아 저장소에 묶여 들어간다)은 번호와 열거값뿐이다.
    형식이 틀린 답의 까닭 글에도 번호만 들어간다."""
    assert align_llm._SECURITY in align_llm.SYSTEM and "Precision first" in align_llm.SYSTEM
    assert '"type": "string"' not in json.dumps(align_llm.SCHEMA)
    turns, said = PARA
    judge_run(turns, said, Judge(pairs={"그만 좀 해.": ("됐다 됐어 이제", "paraphrase")}))
    (rec,) = [util.read_json(p) for p in (paths.CACHE / "llm").glob("*.json")]
    blob = json.dumps(rec["response"], ensure_ascii=False)
    assert not any(w in blob for w in "그만 좀 해 됐다 됐어 이제".split())
    assert rec["task"] == "align_pair" and rec["prompt_version"] == align_llm.PROMPT_VERSION
    check = align_llm._make_check({1, 2}, {1}, 5)
    good = {"lines": [{"s": 1, "kind": "none", "conf": "sure"}, {"s": 2, "kind": "same", "from": 1, "to": 2,
                                                                  "conf": "sure"}],
            "edges": [{"e": 1, "joins": "no", "conf": "sure"}], "noise": [{"from": 5, "to": 5, "conf": "unsure"}]}
    assert check(good) is None
    bad = [dict(good, lines=[dict(good["lines"][1], to=9)]), dict(good, lines=good["lines"] * 2),
           dict(good, lines=[dict(good["lines"][0], kind="maybe")]), {"lines": []}]
    assert all(check(b) for b in bad)
    many = align_llm._make_check(set(range(1, 31)), set(), 5)
    assert many({"lines": [{"s": k, "kind": "none", "conf": "sure"} for k in range(1, 28)], "edges": [], "noise": []}) is None
    assert "빠진" in many({"lines": [{"s": k, "kind": "none", "conf": "sure"} for k in range(1, 20)], "edges": [],
                           "noise": []})


def test_drop_noise_leaves_out_credits_and_the_films_habit_phrases(judged):
    """added 로 남을 묶음 가운데 자막 제작 문구 꼴과 그 영화의 버릇 글귀(asr.film_habits)와 닮은 이어진 낱말 범위는 added 로
    만들지 않고 method.noise_dropped 에 남긴다. 버릇 글귀가 없으면 그 범위는 added 로 남는다."""
    turns, said = big_film([], [("아 다음 영상에서 만나요! 진짜", {"gap": 2.0})])
    habits = {"다음영상에서만나요": 37}
    script, res = judge_run(turns, said, pair=False, habits=habits)
    n = sum(len(t.split()) for t in BIG[:10])
    assert [heard_of(res, o) for o in res.ops if o["op"] == "added"] == ["아", "진짜"]
    assert res.method["noise_dropped"] == [{"asr": [f"plain:{n + 1}", f"plain:{n + 3}"], "why": "habit"}]
    script, res = judge_run(turns, said, pair=False)
    assert [heard_of(res, o) for o in res.ops if o["op"] == "added"] == ["아 다음 영상에서 만나요!", "진짜"]
    assert res.method["noise_dropped"] == []
    pw = align.PassWords("plain", S.say([("한글", 1.0, 1.2), ("자막", 1.2, 1.4), ("제공", 1.4, 1.6), ("..욕봤다", 1.6, 2.0),
                                         ("다음", 2.5, 2.7), ("영상에서", 2.7, 3.0), ("만나요.", 3.0, 3.3)])["words"])
    kept, dropped = align.drop_noise({"plain": pw}, [("plain", 0, 6)], habits)
    assert kept == [("plain", 3, 3)]
    assert dropped == [{"asr": ["plain:0", "plain:2"], "why": "credit"}, {"asr": ["plain:4", "plain:6"], "why": "habit"}]
    assert align.drop_noise({"plain": pw}, [("plain", 0, 6)], {"전혀다른글귀입니다": 9}) == \
        ([("plain", 3, 6)], [{"asr": ["plain:0", "plain:2"], "why": "credit"}])
    # 문구 뒤에 부호 없이 이어 받아써진 대사는 남긴다. 문구 바로 뒤의 짝말('by 이름', '감사합니다')은 문구와 함께 뺀다
    for words, want_kept, want_drop in (
            (["한글", "자막", "제공", "눋머니", "홱식", "몰릐지나?"], [(3, 5)], (0, 2)),
            (["한글자막", "by", "한효정"], [], (0, 2)),
            (["시청해주셔서", "감사합니다.", "그래서"], [(2, 2)], (0, 1)),
            (["아이", "구독과", "좋아요", "부탁드립니다"], [(0, 0)], (1, 3))):
        pw = align.PassWords("plain", S.say([(w, 1.0 + k, 1.5 + k) for k, w in enumerate(words)])["words"])
        kept, dropped = align.drop_noise({"plain": pw}, [("plain", 0, len(words) - 1)], None)
        assert kept == [("plain", a, b) for a, b in want_kept], words
        assert dropped == [{"asr": [f"plain:{want_drop[0]}", f"plain:{want_drop[1]}"], "why": "credit"}], words


def test_run_llm_calls_only_with_the_align_llm_option_and_remakes_when_the_cache_changes(proj, monkeypatch, capsys):
    """'align --llm' 만 claude 를 부른다(all 과 옵션 없는 align 은 캐시만). 적용한 판정의 해시가 단계 지문에 들어가서
    캐시가 생기면 다시 만들고, 같은 캐시면 건너뛰고 바이트가 같다. 캐시를 지우면 규칙 결과로 돌아간다. align.json 과
    final.json 이 검사를 통과한다. 버릇 글귀를 읽을 수 없으면(받아쓰기 캐시 없음) 조용히 넘어가고, 글귀가 바뀌면 다시 만든다."""
    from sinkprep import final

    turns, said = PARA
    script = put_film(turns, speak(said))
    out = paths.FILMS / "local_0001" / "work" / "align.json"
    judge = Judge(pairs={"그만 좀 해.": ("됐다 됐어 이제", "paraphrase")})
    llm.set_backend(judge)
    try:
        assert align.run(ARGS) == 0 and judge.calls == 0
        rule = out.read_bytes()
        assert util.read_json(out)["method"]["noise_dropped"] == []
        assert align.run(argparse.Namespace(**dict(vars(ARGS), cmd="all", llm=True))) == 0 and judge.calls == 0
        capsys.readouterr()
        assert align.run(argparse.Namespace(**dict(vars(ARGS), llm=True))) == 0 and judge.calls == 1
        assert "claude 호출 1번" in capsys.readouterr().out
        data = util.read_json(out)
        asr_p, script_p = out.parent / "asr.json", paths.DOCS / "fm1" / "script.json"
        assert validate.validate_align(out, asr_path=asr_p, script_path=script_p) == []
        assert [o["decided_by"] for o in data["ops"] if o["op"] == "changed"] == ["llm"]
        assert data["method"]["llm_pair"]["accepted"]["lines"] == 1 and "run" not in data["method"]["llm_pair"]
        judged_bytes = out.read_bytes()
        capsys.readouterr()
        assert align.run(ARGS) == 0 and "'skip': 1" in capsys.readouterr().out and out.read_bytes() == judged_bytes
        assert align.run(FORCE) == 0 and out.read_bytes() == judged_bytes and judge.calls == 1
        assert final.run(ARGS) == 0
        final_p = paths.FILMS / "local_0001" / "final.json"
        assert validate.validate_final(final_p, script_p) == []
        assert validate.validate_align(out, final_path=final_p, asr_path=asr_p, script_path=script_p) == []
        it = next(i for i in util.read_json(final_p)["items"] if i["src_el"] == "E0011")
        assert it["status"] == "changed" and it["heard"] == "됐다 됐어 이제"
        for p in (paths.CACHE / "llm").glob("*.json"):
            p.unlink()
        assert align.run(ARGS) == 0 and out.read_bytes() == rule
        monkeypatch.setattr(align, "film_habits", lambda film_id: {"됐다됐어이제": 5})
        assert align.run(ARGS) == 0 and util.read_json(out)["method"]["noise_dropped"] == [
            {"asr": [f"plain:{sum(len(t.split()) for t in BIG[:10])}", f"plain:{sum(len(t.split()) for t in BIG[:10]) + 2}"],
             "why": "habit"}]
        assert validate.validate_align(out, asr_path=asr_p, script_path=script_p) == []
    finally:
        llm.set_backend(None)
        llm.reset_run()
    assert script["elements"][10]["text"] == "그만 좀 해."
