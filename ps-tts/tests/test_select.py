"""select.py: 거르기(길이·뜻·말투), PS-Comet 고르기, fallback."""
from __future__ import annotations

import pytest

from pstts import select
from pstts.config import Config
from pstts.records import Candidate, LineItem, Target
from pstts.select import (add_dtw, assess, english_numbers, fallback, korean_problem, missing_numbers,
                          number_readings, ps_comet, ranked, source_seconds)
from pstts.textnorm import ko_syllable_count

CFG = Config()


def target(n: int = 6, tol: int = 2, en: int = 3, basis: str = "relative") -> Target:
    return Target(n=n, tol=tol, en_syllables=en, basis=basis, seconds_est=n / CFG.ko_rate)


def cand(text: str = "가", syllables: int = 6, dtw: float | None = None, comet: float | None = None,
         labse: float | None = None) -> Candidate:
    return Candidate(text=text, origin="llm", round=1, syllables=syllables, dtw=dtw, comet=comet, labse=labse)


# ---------------------------------------------------------------- 한국어인지

@pytest.mark.parametrize("text", ["내가 반 낼게.", "제임스, 가자!", "James 씨, 가자!", "삼 번 승강장입니다."])
def test_korean_text_is_fine(text: str) -> None:
    assert korean_problem(text) == ""


def test_untranslated_or_mostly_latin_is_not_korean() -> None:
    assert korean_problem("I'll pay half.") == "한국어 아님: 한글이 없음"
    assert korean_problem("") == "한국어 아님: 한글이 없음"
    assert korean_problem("OK Google 가자").startswith("한국어 아님: 영문")


# ---------------------------------------------------------------- 수 확인

@pytest.mark.parametrize("text, expect", [
    ("The last train leaves from platform 3 in ten minutes.", [("3", 3), ("ten", 10)]),
    ("Twenty-five dollars and two hundred and fifty people.", [("Twenty-five", 25), ("two hundred and fifty", 250)]),
    # 'A hundred' 혼자는 과장일 때가 많아 세지 않는다. 시각 7:40 은 7 과 40
    ("A hundred times, 1,500 won, 3.5 seconds, at 7:40.", [("1,500", 1500), ("3.5", 3), ("7", 7), ("40", 40)]),
    ("ten thirty", [("ten", 10), ("thirty", 30)]),
    ("one hundred and five", [("one hundred and five", 105)]),
    ("No one knows. The one I love.", []),
    ("Hundreds of people, the first time, a dozen eggs.", []),
    ("Be there at 10:00.", [("10", 10)]),                                  # :00 은 세지 않는다
    ("It cost 1.5 million dollars.", [("1.5 million", 1_500_000)]),
    ("I've told you a thousand times! He's one in a million.", []),
    ("We're open 24/7. It's a nine-to-five job.", []),
    ("It happened in nineteen eighty-four.", [("nineteen eighty-four", 1984)]),
    ("I have nineteen twenty-dollar bills.", [("nineteen", 19), ("twenty", 20)]),   # 해 신호 낱말이 없으면 따로
])
def test_english_numbers(text: str, expect: list) -> None:
    assert [(num.shown, num.value) for num in english_numbers(text)] == expect


def test_english_numbers_extra_readings() -> None:
    (hour, half) = english_numbers("Meet me at 7:30.")
    assert "반" in half.extra and "반" not in hour.extra
    assert "반" in english_numbers("Seven-thirty tomorrow.")[1].extra
    assert {"구일일"} <= english_numbers("Call 911!")[0].extra
    assert {"공공칠", "영영칠"} <= english_numbers("Agent 007.")[0].extra
    assert "구십" in english_numbers("Back in the 1990s.")[0].extra
    assert "일점오" in english_numbers("1.5 million")[0].extra


def test_number_readings() -> None:
    assert number_readings(3) == {"삼", "셋", "세", "석"}
    assert {"이십오", "스물", "스무"} <= number_readings(25)
    assert number_readings(1500) == {"천오백"}
    assert number_readings(0) == {"영", "공"}


PLATFORM = "The last train to Busan leaves from platform 3 in ten minutes."


@pytest.mark.parametrize("en, ko, missing", [
    (PLATFORM, "부산행 마지막 열차는 십 분 뒤 출발합니다.", ["3"]),
    (PLATFORM, "부산행 막차는 삼 번 승강장에서 십 분 뒤 출발합니다.", []),
    (PLATFORM, "부산행 막차는 제3 승강장에서 십 분 뒤 출발합니다.", []),
    (PLATFORM, "부산행 막차는 제삼 승강장에서 십 분 뒤 출발합니다.", []),
    ("I waited 3 hours.", "세 시간 기다렸어.", []),
    ("I waited 3 hours.", "3시간 기다렸어.", []),          # 후보의 숫자는 한자어로 읽어 본다
    ("I waited 3 hours.", "한참 기다렸어.", ["3"]),
    ("Give me twenty-five dollars.", "스물다섯 달러 줘.", []),
    ("Give me twenty-five dollars.", "돈 좀 줘.", ["twenty-five"]),
    ("Meet me at 7:40.", "일곱 시 사십 분에 보자.", []),
    ("No one knows.", "아무도 몰라.", []),
    # 2026-11-22 검토에서 맞는 번역인데 '빠짐' 으로 나온 꼴들
    ("Call 911!", "구일일에 전화해!", []),
    ("Be there at 10:00.", "열 시까지 와.", []),
    ("Meet me at 7:30.", "일곱 시 반에 만나.", []),
    ("Meet me at 3:30.", "세 시 반에 만나.", []),
    ("Seven-thirty tomorrow.", "내일 일곱 시 반.", []),
    ("Agent 007 is here.", "공공칠 요원이 왔어.", []),
    ("It cost 1.5 million dollars.", "백오십만 달러나 들었어.", []),
    ("It cost 1.5 million dollars.", "일점오 백만 달러가 들었어.", []),
    ("I've had COVID-19 twice.", "코로나19에 두 번 걸렸어.", []),
    ("It happened in nineteen eighty-four.", "천구백팔십사 년에 일어났어.", []),
    ("Back in the 1990s, we had nothing.", "90년대엔 가진 게 없었어.", []),
    ("I've told you a thousand times!", "내가 몇 번을 말했어!", []),
    ("We're open 24/7.", "우리는 언제나 열려 있어요.", []),
    ("It's a nine-to-five job.", "평범한 직장 일이야.", []),
    # 남은 한계(README): 현지화한 수는 여전히 '빠짐' 이다. 그래서 숫자 조건은 기본으로 끈다
    ("Call 911!", "119에 전화해!", ["911"]),
    ("I'll give you ten bucks.", "만 원 줄게.", ["ten"]),
])
def test_missing_numbers(en: str, ko: str, missing: list[str]) -> None:
    assert missing_numbers(en, ko) == missing


def test_number_check_is_opt_in() -> None:
    item = LineItem("a", PLATFORM, register="하십시오체")
    dropped = Candidate("부산행 마지막 열차는 십 분 뒤 출발합니다.", "llm", 1)
    assess([dropped], item, target(19, 2), [0.804], CFG)
    assert CFG.check_numbers is False
    assert dropped.passed and dropped.ok_numbers is None and dropped.reasons == []

    on = Config(check_numbers=True)
    dropped = Candidate("부산행 마지막 열차는 십 분 뒤 출발합니다.", "llm", 1)
    kept = Candidate("부산행 막차는 삼 번 승강장에서 십 분 뒤 출발합니다.", "llm", 1)
    assess([dropped, kept], item, target(19, 2), [0.804, 0.764], on)
    # 숫자 조건은 뜻 조건(LaBSE, 식 7)과 따로 적는다. 그래서 통과 후보가 없을 때 식 6·7 을 채운 후보로 남는다
    assert dropped.ok_numbers is False and dropped.ok_meaning is True and dropped.iso_ok
    assert dropped.reasons == ["숫자: 3 빠짐"] and not dropped.passed
    assert kept.passed and kept.ok_numbers is True


# ---------------------------------------------------------------- assess

def test_assess_fills_scores_and_korean_reasons() -> None:
    item = LineItem("a", "I'll pay half.", register="반말")
    texts = ["내가 반 낼게.", "제가 그 비용의 절반을 부담하겠습니다.", "반은 내가 낼게.", "I'll pay half."]
    cands = [Candidate(t, "llm", 1) for t in texts]
    out = assess(cands, item, target(6, 2), [0.9, 0.88, 0.7496, 1.0], CFG)
    assert out is not None and len(out) == 4

    ok, long_formal, low_labse, english = cands
    assert (ok.syllables, ok.passed, ok.reasons, ok.register_found) == (5, True, [], ["반말"])
    assert long_formal.syllables == ko_syllable_count(long_formal.text) == 16
    assert long_formal.reasons == ["길이 16음절(목표 6±2)", "말투: 반말 요청인데 하십시오체 문장"]
    assert (long_formal.ok_length, long_formal.ok_meaning, long_formal.ok_register) == (False, True, False)
    # 0.7496 은 둘째 자리로 내려서 적는다(0.75 < 0.75 처럼 보이지 않게)
    assert low_labse.reasons == ["뜻 유사도 0.74 < 0.75"]
    assert (low_labse.ok_meaning, low_labse.passed) == (False, False)
    # 번역 안 한 영어는 LaBSE 1.0 이어도 뜻 조건에서 떨어진다
    assert english.labse == 1.0 and english.ok_meaning is False
    assert "한국어 아님: 한글이 없음" in english.reasons


def test_assess_length_bounds_are_inclusive() -> None:
    item = LineItem("a", "I'll pay half.")
    texts = ("반 낼게.", "반은 낼게.", "반은 내가 낼게 정말.", "절반은 내가 낼게 정말.")
    three, four, eight, nine = (Candidate(t, "llm", 1) for t in texts)
    assess([three, four, eight, nine], item, target(6, 2), [0.9] * 4, CFG)
    assert (three.syllables, four.syllables, eight.syllables, nine.syllables) == (3, 4, 8, 9)
    assert (three.ok_length, four.ok_length, eight.ok_length, nine.ok_length) == (False, True, True, False)
    assert three.reasons == ["길이 3음절(목표 6±2)"]
    assert nine.reasons == ["길이 9음절(목표 6±2)"]


def test_assess_register_requests() -> None:
    polite, casual, formal = "제가 반 낼게요.", "내가 반 낼게.", "제가 반을 내겠습니다."
    cases = {
        "미지정": (True, True, True),
        "반말": (False, True, False),
        "해요체": (True, False, True),
        "하십시오체": (False, False, True),
    }
    for reg, expect in cases.items():
        cands = [Candidate(t, "llm", 1) for t in (polite, casual, formal)]
        assess(cands, LineItem("a", "I'll pay half.", register=reg), target(6, 3), [0.9] * 3, CFG)
        assert tuple(c.ok_register for c in cands) == expect, reg
    cands = [Candidate(casual, "llm", 1)]
    assess(cands, LineItem("a", "I'll pay half.", register="해요체"), target(6, 3), [0.9], CFG)
    assert cands[0].reasons == ["말투: 해요체 요청인데 반말 문장"]


def test_assess_register_check_off_keeps_detection() -> None:
    cfg = Config(check_register=False)
    c = Candidate("내가 반 낼게.", "llm", 1)
    assess([c], LineItem("a", "I'll pay half.", register="하십시오체"), target(), [0.9], cfg)
    assert c.ok_register is True and c.register_found == ["반말"] and c.passed


def test_assess_needs_one_score_per_candidate() -> None:
    with pytest.raises(ValueError):
        assess([Candidate("가", "llm", 1)], LineItem("a", "Go."), target(), [], CFG)


# ---------------------------------------------------------------- ps_comet

def test_ps_comet_normalises_and_inverts_dtw() -> None:
    a, b, c = cand("가", dtw=10.0, comet=0.2), cand("나", dtw=20.0, comet=0.9), cand("다", dtw=15.0, comet=0.5)
    best = ps_comet([a, b, c], CFG, lip=True, target=target())
    assert (a.dtw_inv, b.dtw_inv, c.dtw_inv) == (1.0, 0.0, 0.5)
    assert a.score == pytest.approx(1.6 * 1.0 + 0.4 * 0.2)
    assert b.score == pytest.approx(0.4 * 0.9)
    assert c.score == pytest.approx(1.6 * 0.5 + 0.4 * 0.5)
    assert best is a


def test_ps_comet_without_lip_uses_comet_only() -> None:
    a, b = cand("가", dtw=10.0, comet=0.2), cand("나", dtw=20.0, comet=0.9)
    best = ps_comet([a, b], CFG, lip=False, target=target())
    assert best is b
    assert (a.score, b.score) == (0.2, 0.9)
    assert a.dtw_inv is None and b.dtw_inv is None


def test_ps_comet_all_equal_dtw_gives_one() -> None:
    a, b = cand("가", dtw=7.0, comet=0.1), cand("나", dtw=7.0, comet=0.3)
    best = ps_comet([a, b], CFG, lip=True)
    assert (a.dtw_inv, b.dtw_inv) == (1.0, 1.0)
    assert best is b


def test_ps_comet_single_candidate() -> None:
    a = cand("가", dtw=3.0, comet=-0.2)
    assert ps_comet([a], CFG, lip=True) is a
    assert a.dtw_inv == 1.0 and a.score == pytest.approx(1.6 - 0.08)


def test_ps_comet_tie_breaks() -> None:
    # 점수가 같으면 COMET 높은 것: 1.6 x 1.0 + 0.4 x 0.0 = 1.6 x 0.75 + 0.4 x 1.0
    a, b, c = cand("가", dtw=0.0, comet=0.0), cand("나", dtw=1.0, comet=1.0), cand("다", dtw=4.0, comet=0.5)
    assert ps_comet([a, b, c], CFG, lip=True) is b
    # 점수·COMET 이 같으면 목표 음절 수에 가까운 것
    far, near = cand("가", syllables=8, dtw=5.0, comet=0.5), cand("나", syllables=6, dtw=5.0, comet=0.5)
    assert ps_comet([far, near], CFG, lip=True, target=target(6)) is near
    # 다 같으면 먼저 나온 것
    first, second = cand("가", dtw=5.0, comet=0.5), cand("나", dtw=5.0, comet=0.5)
    assert ps_comet([first, second], CFG, lip=True, target=target(6)) is first


def test_ps_comet_alpha_beta_change_the_winner() -> None:
    lip_like, meaning_like = cand("가", dtw=10.0, comet=0.1), cand("나", dtw=20.0, comet=0.9)
    assert ps_comet([lip_like, meaning_like], CFG, lip=True) is lip_like
    assert ps_comet([lip_like, meaning_like], Config(alpha=0.0, beta=1.0), lip=True) is meaning_like


def test_ps_comet_errors() -> None:
    with pytest.raises(ValueError):
        ps_comet([], CFG, lip=True)
    with pytest.raises(ValueError):
        ps_comet([cand("가", dtw=1.0, comet=None)], CFG, lip=True)
    with pytest.raises(ValueError):
        ps_comet([cand("가", dtw=None, comet=0.5)], CFG, lip=True)
    assert ps_comet([cand("가", dtw=None, comet=0.5)], CFG, lip=False).score == 0.5


# ---------------------------------------------------------------- fallback, ranked

def _assessed(item: LineItem, texts: list[str], labse: list[float], t: Target, cfg: Config = CFG,
              rounds: list[int] | None = None) -> list[Candidate]:
    """후보를 만들어 assess 한다. 차수 0 은 모델 1 초벌, 1 이상은 모델 2 후보."""
    rounds = rounds or [1] * len(texts)
    cands = [Candidate(x, "llm" if r else "model1", r) for x, r in zip(texts, rounds)]
    assess(cands, item, t, labse, cfg)
    return cands


def test_selection_pool_tiers() -> None:
    item = LineItem("a", "I'll pay half.", register="해요체")
    # 통과 후보가 있으면 그것
    cands = _assessed(item, ["반 낼게요.", "반 낼게.", "제가 그 비용의 절반을 부담하겠습니다."], [0.9, 0.9, 0.95],
                      target(5, 2))
    kind, group = select.selection_pool(cands)
    assert kind == select.POOL_PASS and [c.text for c in group] == ["반 낼게요."]
    # 말투만 못 채운 후보(식 6·7 은 채움)가 있으면, LaBSE 가 더 높아도 길이가 넘친 초벌 대신 그 가운데서 고른다
    cands = _assessed(item, ["반 낼게.", "내가 반 낼게.", "제가 그 비용의 절반을 부담하겠습니다."], [0.85, 0.8, 0.95],
                      target(5, 2), rounds=[1, 1, 0])
    kind, group = select.selection_pool(cands)
    assert kind == select.POOL_ISO and [c.text for c in group] == ["반 낼게.", "내가 반 낼게."]
    for c, comet in zip(group, (0.1, 0.2)):
        c.comet, c.dtw = comet, 10.0
    assert ps_comet(group, CFG, lip=True, target=target(5, 2)).text == "내가 반 낼게."
    # 식 6·7 을 함께 채운 후보도 없으면 논문 규칙: LaBSE 가 가장 높은 후보 하나
    cands = _assessed(item, ["반 낼게.", "제가 그 비용의 절반을 부담하겠습니다."], [0.7, 0.95], target(5, 2))
    kind, group = select.selection_pool(cands)
    assert kind == select.POOL_LABSE and [c.text for c in group] == ["제가 그 비용의 절반을 부담하겠습니다."]
    assert select.selection_pool([]) == (select.POOL_NONE, [])


def test_selection_pool_drops_draft_when_it_is_the_comet_reference() -> None:
    item = LineItem("a", "I'll pay half.", register="반말")
    cands = _assessed(item, ["내가 반 낼게.", "반은 내가 낼게."], [0.9, 0.9], target(5, 2), rounds=[0, 1])
    assert [c.text for c in select.selection_pool(cands)[1]] == ["내가 반 낼게.", "반은 내가 낼게."]
    kind, group = select.selection_pool(cands, draft_is_reference=True)
    assert kind == select.POOL_PASS and [c.text for c in group] == ["반은 내가 낼게."]
    # 초벌만 통과했으면 초벌을 그대로 둔다
    only = _assessed(item, ["내가 반 낼게.", "제가 반 낼게요."], [0.9, 0.9], target(5, 2), rounds=[0, 1])
    assert [c.text for c in select.selection_pool(only, draft_is_reference=True)[1]] == ["내가 반 낼게."]


def test_unspecified_register_follows_model2_majority() -> None:
    item = LineItem("a", "Then why didn't you call from a friend's phone?")
    texts = ["그럼 친구 전화를 왜 안 했어요?", "그럼 친구 폰으로 왜 안 걸었어?", "친구 전화로라도 하지 그랬어?",
             "그럼 왜 친구 폰으로 전화 안 했어?", "왜 친구 전화 빌려서 안 했어요?"]
    cands = _assessed(item, texts, [0.9] * 5, target(12, 4), rounds=[0, 1, 1, 1, 1])
    assert all(c.ok_register for c in cands)                     # 미지정이라 assess 는 모두 통과시킨다
    assert select.line_register(cands) == "반말"                  # 모델 2 후보: 반말 3, 해요체 1(초벌은 세지 않음)
    assert select.apply_line_register(item, cands, CFG) == "반말"
    assert [c.ok_register for c in cands] == [False, True, True, True, False]
    assert cands[0].reasons == ["말투: 미지정 줄인데 모델 2 후보 다수(반말)와 다름"]
    # 말투를 정한 줄, 말투 조건을 끈 설정, 다수를 정하지 못한 줄은 그대로 둔다
    polite_item = LineItem("b", item.source_en, register="해요체")
    fixed = _assessed(polite_item, texts[:2], [0.9] * 2, target(12, 4))
    assert select.apply_line_register(polite_item, fixed, CFG) is None
    assert select.apply_line_register(item, cands, Config(check_register=False)) is None
    tie = _assessed(item, texts[1:3] + [texts[4], texts[0]], [0.9] * 4, target(12, 4))
    assert select.line_register(tie) is None and select.apply_line_register(item, tie, CFG) is None
    assert all(c.ok_register for c in tie)


def test_fallback_picks_highest_labse_first_on_ties() -> None:
    a, b, c = cand("가나", labse=0.7), cand("다라", labse=0.72), cand("마바", labse=0.72)
    assert fallback([a, b, c]) is b


def test_fallback_skips_untranslated_english() -> None:
    english, korean = cand("I'll pay half.", labse=1.0), cand("반 낼게.", labse=0.6)
    assert fallback([english, korean]) is korean
    assert fallback([english]) is english       # 한국어 후보가 없으면 그대로
    with pytest.raises(ValueError):
        fallback([cand("가", labse=None)])


def test_ranked_puts_scored_passing_first() -> None:
    item = LineItem("a", "I'll pay half.", register="반말")
    texts = ["내가 반 낼게.", "반은 내가 낼게.", "제가 반 낼게요.", "절반만 낼게 진짜 정말로 그래."]
    cands = [Candidate(t, "llm", 1) for t in texts]
    assess(cands, item, target(6, 2), [0.9, 0.85, 0.95, 0.99], CFG)
    passing = [c for c in cands if c.passed]
    assert [c.text for c in passing] == texts[:2]
    passing[0].comet, passing[1].comet = 0.1, 0.8
    ps_comet(passing, CFG, lip=False, target=target(6))
    assert [c.text for c in ranked(cands, target(6))] == [texts[1], texts[0], texts[3], texts[2]]


def test_ranked_puts_scored_iso_group_first() -> None:
    # 통과 후보가 없어 식 6·7 묶음을 PS-Comet 으로 골랐으면, 그 묶음이 점수 순으로 먼저 나온다
    item = LineItem("a", "I'll pay half.", register="해요체")
    cands = _assessed(item, ["제가 그 비용의 절반을 부담하겠습니다.", "반 낼게.", "내가 반 낼게."], [0.99, 0.8, 0.85],
                      target(5, 2))
    kind, group = select.selection_pool(cands)
    assert kind == select.POOL_ISO
    group[0].comet, group[1].comet = 0.9, 0.1
    ps_comet(group, CFG, lip=False, target=target(5, 2))
    assert [c.text for c in ranked(cands, target(5, 2))] == ["반 낼게.", "내가 반 낼게.", "제가 그 비용의 절반을 부담하겠습니다."]


# ---------------------------------------------------------------- DTW

def test_source_seconds_by_basis() -> None:
    item = LineItem("a", "I'll pay half.", seconds=1.5)
    assert source_seconds(item, target(4, en=3, basis="relative"), CFG) == pytest.approx(3 / CFG.en_rate)
    assert source_seconds(item, target(12, en=3, basis="seconds"), CFG) == 1.5
    t = target(6, en=3, basis="explicit")
    assert source_seconds(item, t, CFG) == pytest.approx(t.seconds_est)


def test_add_dtw_real_text() -> None:
    item = LineItem("a", "I'll pay half.")
    cands = [Candidate(t, "llm", 1) for t in ("내가 반 낼게.", "절반은 제가 다 낼게요.", "!!!")]
    assert add_dtw(item, target(4), cands, CFG) is True
    assert cands[0].dtw is not None and cands[0].dtw > 0
    assert cands[1].dtw is not None and cands[1].dtw > 0
    assert cands[2].dtw is None                  # 음절이 없는 후보는 잴 수 없다


def test_add_dtw_without_source_syllables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(select, "english_syllables", lambda text: [])
    c = Candidate("내가 반 낼게.", "llm", 1)
    assert add_dtw(LineItem("a", "..."), target(1), [c], CFG) is False
    assert c.dtw is None
