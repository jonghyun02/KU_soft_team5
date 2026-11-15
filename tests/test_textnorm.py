"""sinkprep.textnorm: same 판정 열쇠, 자리 찾기용 글, 차이 조각의 종류."""
import pytest

from sinkprep import textnorm as T


def kinds(script: str, heard: str) -> list[str]:
    return [d["kind"] for d in T.diffs(script, heard)]


def pieces(script: str, heard: str) -> list[tuple[str, str, str]]:
    return [(d["kind"], d["script"], d["heard"]) for d in T.diffs(script, heard)]


# ------------------------------------------------------------------ 열쇠
def test_key_drops_spaces_punctuation_and_case():
    """열쇠는 NFC·소문자·글자만: 띄어쓰기·문장부호·대소문자만 다른 글은 열쇠가 같다."""
    assert T.key("집에 있었는데요.") == T.key("집에있었는데요!") == "집에있었는데요"
    assert T.key("TV 꺼, 씨발놈아...") == "tv꺼씨발놈아"
    assert T.key("가가") == "가가"                       # 풀어 쓴 자모도 NFC 로 모은다
    assert T.key("3층 #12-A") == "3층12a"
    assert T.key(None) == "" and T.key(".......") == "" and T.key("(…) ♪ ~") == ""


@pytest.mark.parametrize("a, b", [
    ("아ㅡ 진짜", "아 진짜"),                             # 'ㅡ'(U+3161)를 줄표로 친 것
    ("아ㅡㅡ 진짜 ㅡㅡ;", "아 진짜"),
    ("형사ㆍ경찰 다 불러", "형사 경찰 다 불러"),           # 아래아 가운뎃점(U+318D)
    ("그래ㅤ알았어", "그래 알았어"),                      # 한글 채움 문자(U+3164)
    ("ㅋㅋㅋ 진짜?", "진짜"),
    ("ＴＶ 꺼", "TV 꺼"),                                 # 전각 영문
    ("１２시에 와", "12시에 와"),                          # 전각 숫자
    ("누구세요？", "누구세요?"),
    ("사랑해❤️", "사랑해"),                               # 이모지 뒤 변형 선택자
    ("김○○ 씨", "김 씨"),
    ("O.K.", "OK"),
    ("A/S 센터", "AS센터")])
def test_key_ignores_symbols_that_are_not_speech(a, b):
    """글자 범주에 들지만 말이 아닌 기호(낱자모, 채움 문자, 변형 선택자)와 전각 꼴은 판정을 바꾸지 않는다."""
    assert T.key(a) == T.key(b) and T.verdict(a, b) == ("same", [])


def test_key_of_jamo_only_text_is_empty():
    """낱자모뿐인 대사(ㅠ.ㅠ)는 말할 글자가 없다."""
    assert T.key("ㅜ.ㅜ;;;;") == "" and not T.sayable("ㅠ.ㅠ") and not T.sayable("ㅋㅋㅋ")


def test_key_keeps_the_marks_that_change_a_number():
    """숫자에 붙은 부호: 소수점은 '점', 숫자 뒤 % 는 그대로, 숫자 줄 사이의 경계는 SEP. 자릿수 쉼표만 없는 것으로 친다."""
    assert T.key("1.5배") == "1점5배" != T.key("15배")
    assert T.key("100%") == "100%" != T.key("100") and T.key("100 %") == "100%"
    assert T.key("2, 3일") == "2" + T.SEP + "3일" != T.key("23일")
    assert T.key("3,000원") == T.key("3000원") == "3000원"
    assert T.key("1,234,567원") == "1234567원"
    assert T.key("7,8월") == "7" + T.SEP + "8월"               # 세 자리가 아니면 자릿수 쉼표가 아니다
    assert T.key("했어.3시에") == "했어3시에"                    # 숫자 사이가 아닌 마침표는 문장부호다
    assert T.key("2008.2.14") == T.key("2008 2 14") == f"2008{T.SEP}2{T.SEP}14"     # 마침표로 이어진 숫자 줄은 소수가 아니다
    assert T.key("50% 할인") == "50%할인" and T.key("% 는") == "는"


@pytest.mark.parametrize("a, b", [
    ("2, 3일", "2 3일"), ("2, 3일", "2-3일"), ("2, 3일", "2~3일"),
    ("016-9265-4885", "016 9265 4885"), ("016-9265-4885", "016.9265.4885 "), ("4885. 4885.", "4885 4885"),
    ("3,000원", "3000원")])
def test_number_run_boundary_does_not_depend_on_the_punctuation(a, b):
    """숫자 줄 사이에 낀 것이 공백이든 쉼표든 붙임표든 경계 하나다: 문장부호만 다른 글은 same 이다."""
    assert T.verdict(a, b) == ("same", [])


@pytest.mark.parametrize("a, b", [
    ("1.5배", "15배"), ("2, 3일 걸려", "23일 걸려"), ("100%", "100"), ("1 2 3", "123"), ("21-15번지", "2115번지")])
def test_different_numbers_are_never_same(a, b):
    """숫자 사이 부호를 지우면 같아지는 다른 수는 changed 다."""
    status, d = T.verdict(a, b)
    assert status == "changed" and d


def test_sayable_tells_silent_turns():
    """말 없는 대사(점·괄호뿐)는 맞출 글자가 없다."""
    assert T.sayable("예.") and not T.sayable(".......") and not T.sayable("") and not T.sayable(None)


# ------------------------------------------------------------------ 숫자·영문 읽기
@pytest.mark.parametrize("digits, want", [
    ("4885", "사팔팔오"), ("016", "공일육"), ("306", "삼백육"), ("15", "십오"), ("110", "백십"),
    ("3", "삼"), ("20", "이십"), ("0", "공"), ("12345", "일이삼사오")])
def test_read_number_reads_by_digit_or_by_place(digits, want):
    """4자리 이상이거나 0 으로 시작하면 한 자리씩, 아니면 자릿수 읽기."""
    assert T.read_number(digits) == want


@pytest.mark.parametrize("digits, after, first, also", [
    ("12", "시간", "열두", {"십이", "열둘", "일이"}),            # 고유어로 세는 말 앞
    ("12", "", "십이", {"열두"}),
    ("3", "명", "세", {"삼", "셋"}),
    ("6", "개월", "육", {"여섯"}),                                # 개월은 한자어로
    ("20", "살", "스무", {"이십", "스물"}),
    ("1500", "이나", "천오백", {"일오공공"}),                     # 00 으로 끝나는 네 자리는 양
    ("2004", "년", "이천사", {"이공공사", "이영영사"}),
    ("4885", "", "사팔팔오", {"사천팔백팔십오"}),                 # 번호
    ("016", "", "공일육", {"영일육"}),
    ("112", "", "백십이", {"일일이"}),
    ("15000", "원", "만오천", {"일오공공공"}),
    ("0", "점", "영", {"공"})])
def test_number_readings_puts_the_likely_reading_first(digits, after, first, also):
    """읽는 꼴 후보: 자릿수 읽기, 고유어 수사, 한 자리씩. 뒤따르는 말로 순서를 정한다."""
    got = T.number_readings(digits, after)
    assert got[0] == first and also <= set(got) and len(got) == len(set(got))


def test_loc_makes_digits_and_reading_the_same_place():
    """자리 찾기용 글에서는 숫자와 읽는 꼴이 같다. 판정용 열쇠에서는 다르다."""
    assert T.loc("306호요.") == T.loc("삼백육호요") == "삼백육호요"
    assert T.loc("016-9265-4885") == "공일육구이육오사팔팔오"
    assert T.loc("１２층") == "십이층"                               # 전각 숫자
    assert T.key("306호요.") != T.key("삼백육호요")
    assert T.loc("숫자 없는 글") == T.key("숫자 없는 글")
    assert T.loc(None) == "" and T.loc("...") == ""


def test_loc_default_reading_ignores_the_words_around():
    """맞은편 글을 주지 않으면 앞뒤 말과 상관없는 기본 읽기다: 대본 줄('12 시간')과 받아쓰기 낱말('12')이 같게 읽힌다."""
    assert T.loc("12시간") == T.loc("12") + "시간" == "십이시간"
    assert T.loc("1500이나") == "일오공공이나" and T.loc("2004년") == "이공공사년"
    assert T.loc("2, 3일") == "이삼일" and T.loc("3,000원") == "삼공공공원"      # 쉼표: 숫자 줄 경계 / 자릿수
    assert T.loc("1.5배") == "일점오배" and T.loc("0.5초") == "영점오초" and T.loc("1.25") == "일점이오"
    assert T.loc("100%") == "백퍼센트"


@pytest.mark.parametrize("text, said", [
    ("1500이나", "천오백이나"), ("5000원", "오천원"), ("10000원", "만원"), ("1000만원", "천만원"), ("2004년", "이천사년"),
    ("112", "일일이"), ("119", "일일구"), ("893", "팔구삼"), ("1시", "한시"), ("2명", "두명"), ("3개", "세개"),
    ("20살", "스무살"), ("12시간", "열두시간"), ("12시 30분", "열두시삼십분"), ("9시 4분", "아홉시사분"),
    ("1.5배", "일점오배"), ("0.5초", "영점오초"), ("2, 3일", "이삼일"), ("7,8월", "칠팔월"),
    ("21-15번지", "이십일다시십오번지"), ("893-1번지", "팔구삼에일번지"), ("016-9265-4885", "공일육구이육오사팔팔오"),
    ("100%", "백프로"), ("100%", "백퍼센트"), ("306호", "삼백육호"), ("4885번", "사팔팔오번"), ("3번", "세번"),
    ("TV 꺼", "티비꺼"), ("TV 꺼", "티브이꺼"), ("DNA 검사", "디엔에이검사"), ("OK", "오케이"), ("CCTV", "씨씨티비")])
def test_loc_picks_the_reading_found_in_the_other_text(text, said):
    """맞은편 글에 있는 읽는 꼴을 고른다: 금액·연도·한 자리씩 읽는 번호·고유어 수사·소수점·붙임표·영문 약어."""
    assert T.loc(text, said) == said


def test_loc_does_not_trust_a_short_reading_without_what_follows():
    """뒤에 붙여 볼 글자가 없으면 두 글자 읽기는 믿지 않는다(영화 한 편 글에는 우연히 있다)."""
    assert T.loc("18", "오늘은일팔청춘하나") == "십팔"                # '일팔' 이 맞은편에 있어도 기본 읽기
    assert T.loc("꼬마집사1호가", "하나둘셋") == "꼬마집사일호가"
    assert T.loc("893", "팔구삼에") == "팔구삼"                       # 세 글자 읽기는 믿는다
    assert T.loc("TV", "tv를꺼") == "tv" and T.loc("stand", "에스티") == "stand"


def test_lcs_and_similarity_count_exact_letters():
    assert T.lcs("너어제어디있었어", "어제밤에어디있었냐고") == 6
    assert T.similarity("가나다", "가나다") == 1.0 and T.similarity("", "") == 1.0
    assert T.similarity("가나다", "라마바") == 0.0 and T.lcs("", "가") == 0


def test_jamo_string_spells_out_syllables():
    assert T.jamo("가") == (0, 0, 0) and T.jamo("a") is None
    assert len(T.jamo_string("값a")) == 4 and T.jamo_string("a1") == "a1"


# ------------------------------------------------------------------ 표기 차이
@pytest.mark.parametrize("a, b", [
    ("됐어", "됬어"), ("안", "않"), ("되요", "돼요"), ("웬일이야", "왠일이야"), ("거예요", "거에요"), ("몇일", "며칠"),
    ("괜찮아", "괜찬아"), ("괜찮니", "괜찬니"), ("좋아", "조아"), ("많이", "마니"), ("싫어", "시러"), ("같이", "가치"),
    ("없어", "업써"), ("어떡해", "어떻게"), ("있다가", "이따가"), ("좆나", "존나"), ("합니다", "함니다"), ("택시", "택씨"),
    ("네가", "내가"), ("애가", "얘가"), ("걔들", "개들"), ("같고", "갔고"), ("희망", "히망")])
def test_spelling_like_same_sound(a, b):
    """소리 나는 대로 고치면 같은 글: 받침 대표음, 받침이 뒤로 넘어감, ㅎ, 된소리·콧소리 되기, 가르지 않는 모음."""
    assert T.spelling_like(a, b) and T.spelling_like(b, a)


@pytest.mark.parametrize("a, b", [
    ("할게", "할께"), ("올거야", "올꺼야"), ("거야", "꺼야"), ("할걸", "할껄"), ("알어", "알아"), ("맞었어", "맞았어"),
    ("같애", "같아"), ("같앴어", "같았어"), ("뭐라구", "뭐라고"), ("하자구요", "하자고요"), ("했구", "했고"),
    ("갖구", "갖고"), ("해가지구", "해가지고"), ("그리구", "그리고"), ("일루", "일로"), ("별루", "별로"),
    ("그랬어여", "그랬어요"), ("니가", "네가"), ("니들", "네들"), ("니", "네"), ("임마", "인마"), ("씨발", "시발"),
    ("씨발놈아", "씨팔놈아"), ("쫌", "좀"), ("쪼끔", "조금"), ("이것", "이거"), ("것", "거")])
def test_spelling_like_known_variants(a, b):
    """같은 말의 굳은 다른 표기: 된소리로 적은 거·게, 입말 꼴 어/아·구/고·루/로, 니/네, 임마/인마."""
    assert T.spelling_like(a, b) and T.spelling_like(b, a)


@pytest.mark.parametrize("a, b", [
    ("너", "나"), ("안", "난"), ("죽여", "죽어"), ("물", "불"), ("가자", "가지"), ("밥", "밤"), ("돈", "돌"), ("딸이", "달이"),
    ("간다", "한다"), ("형사님", "행사님"), ("상무님", "장모님"), ("누가", "누나"), ("경찰을", "경찰은"),
    ("사장님", "사정님"), ("했니", "했네"), ("지영민", "지용민"), ("초소", "초수"), ("괜찮니", "괜창니"),
    ("내가", "니가"),                                                  # 니·네·내 를 한 꼴로 접지 않는다
    ("가자", "가짜"), ("아까", "아가"), ("오빠", "오바"), ("한때", "한데"), ("인사", "인싸"), ("조개", "쪼개"),
    ("서", "써"), ("꺼내", "거네"),                                    # 된소리는 뜻을 가른다
    ("입구", "입고"), ("가구", "가고"), ("식구", "식고"), ("하루", "하로"), ("자두", "자도"),
    ("상어", "상아"), ("영어", "영아"), ("갔어", "갔아"), ("뭐여", "뭐요"), ("니트", "네트"),
    ("진짜", "정말"), ("가", "가"), ("가나", "가"), ("ab", "ac"), ("", "가")])
def test_spelling_like_rejects_other_words(a, b):
    """뜻이 갈리는 차이와 받아쓰기의 비슷한 소리 오류는 표기 차이가 아니다."""
    assert not T.spelling_like(a, b) and not T.spelling_like(b, a)


def test_spelling_like_takes_word_starts():
    """first 로 낱말 경계를 주면 낱말 끝·낱말 머리 규칙이 그 경계를 따른다."""
    assert T.spelling_like("안돼", "않돼", [True, True])              # 않 / 돼: 낱말 끝 받침은 대표음
    assert not T.spelling_like("안돼", "않돼")                        # 한 낱말이면 않돼 = 안퇘
    assert T.spelling_like("일루와", "일로와", [True, False, True]) and not T.spelling_like("일루와", "일로와")


# ------------------------------------------------------------------ 군말·헛소리 꼴
@pytest.mark.parametrize("word, want", [
    ("어", True), ("음", True), ("어어어", True), ("아이", True), ("아이이", True), ("에이", True), ("아이고", True),
    ("으음", True), ("아휴", True), ("어휴", True), ("에이씨", True), ("하", True), ("뭐", True), ("그", True), ("야", True),
    ("뭐야", False), ("저자", False), ("그게", False), ("", False), ("진짜", False), ("엄", False), ("네", False)])
def test_is_filler_takes_whole_words_only(word, want):
    """낱말 하나가 통째로 군말 꼴일 때만 참이다(늘인 꼴 포함). 자리와 부호는 보지 않는다."""
    assert T.is_filler(word) is want


def test_noise_shape_names_asr_filler_text():
    """받아쓰기가 말 없는 곳에 적는 꼴: 자막 문구, 인사말만, 한 낱말 네 번 이상."""
    assert T.noise_shape(["감사합니다."]) == "greeting"
    assert T.noise_shape(["감사합니다.", "감사합니다."]) == "greeting"
    assert T.noise_shape(["한글", "자막", "제공"]) == "credit"
    assert T.noise_shape(["한글자막", "by", "한효정"]) == "credit"
    assert T.noise_shape(["네", "네", "네", "네"]) == "repeat"
    assert T.noise_shape(["네", "네"]) is None and T.noise_shape(["자막", "좀", "켜", "봐"]) is None
    assert T.noise_shape(["...", "!"]) is None
    assert T.noise_shape(["stand", "by"]) is None and T.noise_shape(["감사합니다", "선생님"]) is None


def test_ends_sentence():
    assert T.ends_sentence("했어.") and T.ends_sentence("뭐?") and T.ends_sentence("야!\"") and T.ends_sentence("뭐?!")
    assert not T.ends_sentence("어,") and not T.ends_sentence("그래서") and not T.ends_sentence("(웃음)")
    assert not T.ends_sentence("음...") and not T.ends_sentence("그게…")      # 말줄임표는 말이 이어지는 꼴


# ------------------------------------------------------------------ 판정
def test_verdict_same_when_only_spacing_and_punctuation_differ():
    """띄어쓰기·문장부호·대소문자만 다르면 same 이고 차이 조각이 없다(D4)."""
    assert T.verdict("집에있었는데요.", "집에 있었는데요!") == ("same", [])
    assert T.verdict("TV 꺼.", "tv 꺼") == ("same", [])
    assert T.verdict("“어디야?”", "어디야") == ("same", [])
    assert T.verdict("예…… 죄송합니다, 상무님……", "예 죄송합니다 상무님") == ("same", [])


@pytest.mark.parametrize("s, h", [
    ("예.", "네."), ("가", "가나"), ("가나", "나"), ("abc", "abd"), ("예 예", "예"), ("1.5배", "15배"), ("2, 3일", "23일"),
    ("100%", "100"), ("016-9265-4885", "01692654885"), ("1 2 3", "123"), ("그 점은", "그건"), ("3시", "세 시")])
def test_verdict_changed_always_has_diffs(s, h):
    """열쇠가 다르면 changed 이고 차이 조각이 1개 이상이다(D4). 조각마다 한쪽은 글이 있다."""
    status, d = T.verdict(s, h)
    assert status == "changed" and d and all(set(x) == {"kind", "script", "heard"} for x in d)
    assert all(x["script"] or x["heard"] for x in d)
    assert all(x["kind"] in ("filler", "repeat", "word", "number", "spelling") for x in d)


def test_verdict_invariants_hold_on_odd_inputs():
    """가장자리 글(숫자 부호뿐, 낱자모, 다른 문자의 숫자, 빈 글)에서도 same 은 열쇠 같음과 같고, changed 면 조각이 있다."""
    atoms = ["", "%", ".5", "5.", "1..2", "1,", ",000", "3,00", "١٢٣", "②", "1.5.2", "0.0", "-1", "1 - 2", "3 %", "1%2", "A.1",
             "점", "1점5", "·", "1·2", "…", "ㅋㅋ", "️", "TV", "X됐다", "그,", "어어", "야!", "2시", "두 시", "100%", "백 프로",
             "4885", "사팔팔오", "016-9265-4885", "2, 3일", "23일"]
    for a in atoms:
        assert T.verdict(a, a) == ("same", []) and T.key(T.key(a)) == T.key(a)
        for b in atoms:
            status, d = T.verdict(a, b)
            assert (status == "same") == (T.key(a) == T.key(b)) and bool(d) == (status == "changed")
            assert all((x["script"] or x["heard"]) and x["kind"] in ("filler", "repeat", "word", "number", "spelling") for x in d)
            assert isinstance(T.loc(a, T.key(b)), str) and isinstance(T.spelling_like(T.key(a), T.key(b)), bool)


# ------------------------------------------------------------------ 차이 조각: 군말
def test_diffs_filler():
    """더해지거나 빠진 말이 군말뿐이면 filler. 낱말 첫 글자가 아니라 앞 낱말로 잡는다."""
    assert T.verdict("어디 가?", "어 어디 가?") == ("changed", [{"kind": "filler", "script": "", "heard": "어"}])
    assert T.diffs("어디 가?", "어, 음... 어디 가?") == [{"kind": "filler", "script": "", "heard": "어, 음..."}]
    assert T.diffs("아, 그래요?", "그래요?") == [{"kind": "filler", "script": "아,", "heard": ""}]
    assert pieces("누구세요?", "누구세요 어?") == [("filler", "", "어?")]
    assert pieces("글쎄요.", "음 글쎄요 음.") == [("filler", "", "음"), ("filler", "", "음.")]
    assert pieces("아, 안양이요.", "안양이요") == [("filler", "아,", "")]            # 뒤 낱말의 앞머리와 겹쳐도 군말


@pytest.mark.parametrize("extra", ["아휴", "어휴", "으음", "하...", "에이씨", "아이고", "어어", "아이,", "오...", "그...", "저,", "뭐..."])
def test_diffs_filler_takes_interjections_and_marked_words(extra):
    """감탄사와 늘인 꼴은 군말이다. 뜻 있는 낱말과 꼴이 같은 것(그, 저, 뭐, 아이, 오)은 쉼표·말줄임표가 붙었을 때만."""
    assert pieces("잠깐만요.", f"{extra} 잠깐만요.") == [("filler", "", extra)]


@pytest.mark.parametrize("s, h", [
    ("그 사람 어디 갔어?", "사람 어디 갔어?"), ("저 사람 누구야?", "사람 누구야?"),                # 가리키는 말
    ("그 여자가 아니라 저 여자야.", "그 여자가 아니라 여자야."),
    ("뭐 먹을래?", "먹을래?"), ("저 좀 도와주세요.", "좀 도와주세요."),                            # 묻는 말, 나
    ("아이 어디 갔어?", "어디 갔어?"), ("오 형사 불러.", "형사 불러."), ("오 분만 기다려.", "분만 기다려."),
    ("빨리 자.", "빨리."),                                                                      # 잠자다
    ("야! 거기 서!", "거기 서!"), ("돈 내놔 야.", "돈 내놔."), ("뭐야 이게", "야 뭐야 이게"),      # 부르는 말
    ("제 말은요.", "그러니까 제 말은요."), ("잠깐만요.", "저기 잠깐만요."), ("알겠습니다.", "네 알겠습니다.")])
def test_diffs_meaningful_words_are_not_filler(s, h):
    """군말 꼴이어도 가리키는 말·묻는 말·부르는 말·뜻 있는 낱말로 쓰인 것은 word 다."""
    assert kinds(s, h) == ["word"]


def test_diffs_filler_at_line_start_and_filler_swapped_for_filler():
    """'자' 는 줄 맨 앞이면 부호 없이도 군말이다. 군말이 다른 군말로 바뀐 것도 filler."""
    assert pieces("자 지금 몇 시야?", "지금 몇 시야?") == [("filler", "자", "")]
    assert pieces("아... 됐어.", "에이, 됐어.") == [("filler", "아...", "에이,")]
    assert pieces("아유... 됐어.", "아이고, 됐어.") == [("filler", "아유...", "아이고,")]
    assert kinds("그 차", "저 차") == ["word"]                        # 가리키는 말끼리 바뀐 것은 군말이 아니다


# ------------------------------------------------------------------ 차이 조각: 되풀이·더듬기
def test_diffs_repeat():
    """더해진 말이 옆 낱말과 같거나 다음 낱말의 앞소리면 repeat."""
    assert T.diffs("진짜 좋아", "진짜 진짜 좋아") == [{"kind": "repeat", "script": "", "heard": "진짜"}]
    assert T.diffs("딸이 있어요.", "따, 딸이 있어요.") == [{"kind": "repeat", "script": "", "heard": "따,"}]
    assert T.diffs("따, 딸이 있어요.", "딸이 있어요.") == [{"kind": "repeat", "script": "따,", "heard": ""}]
    assert pieces("그, 그게 왜 궁금한 거야?", "그게 왜 궁금한 거야?") == [("repeat", "그,", "")]      # 군말 꼴이어도 더듬기가 먼저
    assert pieces("모르겠어요.", "모 모르 모르겠어요.") == [("repeat", "", "모 모르")]
    assert pieces("왜 그래.", "왜 왜 왜 그래.") == [("repeat", "", "왜 왜")]


@pytest.mark.parametrize("s, h", [
    ("문 열어!", "문 열어 문 열어!"), ("이리 와.", "이리 와 이리 와."), ("가지 마!", "가지 마 가지 마!"),
    ("차 빼라고!", "차 빼라고 차 빼라고!"), ("빨리 가 빨리 가!", "빨리 가!"), ("하늘이 돕는다.", "하늘이 돕는다 하늘이 돕는다."),
    ("미안해요.", "미안해요 미안해요."), ("야 이 새끼야.", "야 이 새끼야 야 이 새끼야."), ("됐어.", "됐어 됐어 됐어.")])
def test_diffs_repeat_of_several_words(s, h):
    """여러 낱말을 통째로 되풀이한 것도 repeat."""
    assert kinds(s, h) == ["repeat"]


@pytest.mark.parametrize("s, h", [
    ("이 인간아!", "인간아!"), ("인간아!", "이 인간아!"), ("이 일은 내가 할게.", "일은 내가 할게."), ("내가 가 볼게.", "내가 볼게."),
    ("누나 나 간다.", "누나 간다."), ("다 달라고 해.", "달라고 해."), ("차 창문 열어.", "창문 열어."), ("나 나갈게.", "나갈게."),
    ("사람이 이 시간에 왜 와?", "사람이 시간에 왜 와?"), ("너 넘어졌어?", "넘어졌어?"), ("안 안아줘.", "안아줘."),
    ("밥 다 먹었다 다 치워.", "밥 다 먹었다 치워."), ("소 손질해.", "손질해."), ("새 생각 났어.", "생각 났어.")])
def test_diffs_one_syllable_words_are_not_stutter(s, h):
    """한 음절 낱말이 앞말 끝이나 뒷말 머리와 우연히 겹친 것은 더듬기가 아니다."""
    assert kinds(s, h) == ["word"]


def test_diffs_stutter_added_in_heard_text():
    """받아쓰기에 더해진 한 음절 앞머리는 부호 없이도 더듬기로 본다(대본에서 빠진 것은 쉼표·말줄임표가 있어야)."""
    for h in ("나 나는 몰라요.", "사 살려주세요.", "살 살려주세요.", "살려 살려주세요."):
        assert kinds(h.split(" ", 1)[1], h) == ["repeat"]
    assert kinds("가, 가지 마.", "가지 마.") == ["repeat"] and kinds("가 가지 마.", "가지 마.") == ["word"]


def test_diffs_lengthened_sound_is_repeat():
    """끝소리를 늘인 것(야 / 야아)은 repeat."""
    for s, h in (("야!", "야아!"), ("네.", "네에."), ("엄마!", "엄마아!"), ("어.", "어어.")):
        assert pieces(s, h) == [("repeat", s, h)]


def test_diffs_one_sided_piece_is_split_by_kind():
    """한쪽에만 있는 말은 낱말마다 종류를 매겨 같은 종류끼리 따로 낸다: 군말은 군말, 되풀이는 되풀이, 낱말은 낱말."""
    assert pieces("어... 일 년... 일 년 정도면", "1년 정도면") == [
        ("filler", "어...", ""), ("repeat", "일 년...", ""), ("number", "일 년", "1년")]
    assert pieces("진짜 좋네.", "아 근데 진짜 좋네.") == [("filler", "", "아"), ("word", "", "근데")]
    assert pieces("가자.", "어, 음... 자 가자.") == [("filler", "", "어, 음..."), ("word", "", "자")]


def test_diffs_repeat_needs_a_copy_that_stays():
    """되풀이는 남은 말과 같은 말이 더 있을 때다. 같은 말 두 번이 통째로 빠진 것은 빠진 낱말이다."""
    assert pieces("저게... 야! 야! 아 나 참.", "아 나 참") == [("word", "저게... 야! 야!", "")]
    assert pieces("문 열어 문 열어 문 열어!", "문 열어!") == [("repeat", "문 열어 문 열어!", "")]     # 앞의 것이 남고 뒤 두 번이 되풀이
    assert pieces("야 야 야 이리 와.", "야 이리 와.") == [("repeat", "야 야", "")]


# ------------------------------------------------------------------ 차이 조각: 숫자
def test_diffs_number():
    """같은 수를 달리 적은 것만 number(읽는 꼴이 같아도 changed 다)."""
    assert T.verdict("삼백육호요.", "306호요.") == ("changed", [{"kind": "number", "script": "삼백육호요.", "heard": "306호요."}])
    assert T.diffs("사팔팔오", "4885")[0]["kind"] == "number"
    assert T.diffs("이십일다시십오번지요.", "21-15번지요")[0]["kind"] == "number"
    assert T.diffs("3층 가요", "3층 가세요") == [{"kind": "word", "script": "가요", "heard": "가세요"}]   # 숫자는 같은 곳


@pytest.mark.parametrize("s, h", [
    ("세 시에 와.", "3시에 와."), ("두 명이요.", "2명이요."), ("천만 원.", "1000만 원."), ("오백만 원 줄게.", "500만 원 줄게."),
    ("이천사년에", "2004년에"), ("공일육 구이육오 사팔팔오", "016-9265-4885"), ("스물한 살이요.", "21살이요."), ("서른이요.", "30이요."),
    ("일점오 배", "1.5배"), ("백 프로야.", "100%야."), ("100 퍼센트", "100%"), ("열두 시 반", "12시 반"), ("둘째 주", "2째 주"),
    ("112에 신고해.", "일일이에 신고해."), ("KBS 9시 뉴스", "KBS 아홉 시 뉴스"), ("5만원짜리", "오만 원짜리"), ("0.5초", "영점오초"),
    ("제1조", "제일조"), ("천이 넘는데", "1000이 넘는데"), ("이십일 다시 십오", "21-15"), ("이삼 일 걸려", "2, 3일 걸려"),
    ("016-9265-4885", "01692654885"), ("4885. 4885.", "48854885")])
def test_diffs_number_is_the_same_number_written_differently(s, h):
    """읽는 꼴 ↔ 아라비아 숫자(자릿수 읽기, 한 자리씩, 고유어 수사, 소수점, %), 숫자 줄을 나눈 자리만 다른 것."""
    assert kinds(s, h) == ["number"]


@pytest.mark.parametrize("s, h", [
    ("오백만 원 줄게.", "300만 원 줄게."), ("3시에 만나.", "4시에 만나."), ("2004년에", "2014년에"), ("천이 넘는데", "2000이 넘는데"),
    ("두 시간 뒤에 봐.", "세 시간 뒤에 봐."), ("삼백육호로 와", "307호로 와"), ("사팔팔오 맞죠?", "488 맞죠?"),
    ("3시에 와.", "저녁에 와."), ("번호가 뭐야", "번호가 306이야"),
    ("2명 죽었어.", "죽었어."), ("죽었어.", "2명 죽었어."),                                       # 숫자가 든 말이 빠지거나 더해짐
    ("1.5배", "15배"), ("2, 3일 걸려", "23일 걸려"), ("100%", "100")])                             # 부호를 지우면 같아지는 다른 수
def test_diffs_changed_or_added_numbers_are_word(s, h):
    """수가 실제로 다르거나 숫자가 낀 말이 더해지고 빠진 것은 word."""
    assert kinds(s, h) == ["word"]


def test_diffs_number_does_not_swallow_other_words():
    """숫자가 든 덩어리에 딴 말이 섞이지 않게 나눈다. 섞인 조각은 word 다."""
    assert pieces("그 새끼 전에도 그랬단다.", "야 18 좋댔다 그 새끼 전에도 그랬단다.") == [("word", "", "야 18 좋댔다")]
    assert pieces("야, 지가 그랬대.", "여기서 딱 5분 거리야 지가 그랬대.") == [("word", "야,", "여기서 딱 5분 거리야")]
    assert pieces("열두 시간 안에 받아야 된다고.", "말이야 12시간 안에 받아야 된다고") == [
        ("word", "", "말이야"), ("number", "열두 시간", "12시간")]
    assert pieces("삼십 분만 기다려.", "30분만 기다려요.") == [("number", "삼십 분만", "30분만"), ("word", "기다려.", "기다려요.")]
    assert pieces("3층이에요", "삼층이요") == [("word", "3층이에요", "삼층이요")]


def test_diffs_number_words_anchor_on_their_reading():
    """숫자가 든 낱말은 맞은편의 읽는 꼴 낱말과 짝이 된다: 줄 전체가 한 덩어리로 뭉개지지 않는다."""
    assert pieces("아홉시 사분... 사팔팔오... 들어봤는데. 사팔팔오... 사팔팔오...", "아 4885 4885") == [
        ("word", "아홉시 사분... 사팔팔오... 들어봤는데.", "아"), ("number", "사팔팔오...", "4885"), ("number", "사팔팔오...", "4885")]
    assert pieces("그거 펴. 공일육 구이육오에 사팔팔오. 읊어봐.", "016 -9265 -4885 물 뽑아") == [
        ("word", "그거 펴.", ""), ("number", "공일육", "016"), ("number", "구이육오에", "-9265"),
        ("number", "사팔팔오.", "-4885"), ("word", "읊어봐.", "물 뽑아")]
    assert pieces("공일육인가 공일일인가?", "011인가 016인가?") == [                  # 차례가 뒤바뀐 수는 같은 수가 아니다
        ("word", "공일육인가", "011인가"), ("word", "공일일인가?", "016인가?")]
    assert pieces("하나 둘 셋", "1, 2, 3") == [("number", "하나", "1,"), ("number", "둘 셋", "2, 3")]


# ------------------------------------------------------------------ 차이 조각: 표기
def test_diffs_spelling():
    """같은 말의 다른 표기면 spelling."""
    assert T.diffs("그랬어요.", "그랬어여.") == [{"kind": "spelling", "script": "그랬어요.", "heard": "그랬어여."}]
    assert T.diffs("안 돼요.", "않 돼요.") == [{"kind": "spelling", "script": "안", "heard": "않"}]
    assert pieces("내가 할 거야.", "내가 할 꺼야.") == [("spelling", "거야.", "꺼야.")]
    assert pieces("지금 올 꺼야. 오면 얘기해.", "지금 올 거야 오면 얘기해") == [("spelling", "꺼야.", "거야")]
    assert pieces("니가 뭘 알어?", "네가 뭘 알아?") == [("spelling", "니가", "네가"), ("spelling", "알어?", "알아?")]
    assert pieces("태워 갖구 왔어.", "태워갖고 왔어") == [("spelling", "태워 갖구", "태워갖고")]


@pytest.mark.parametrize("s, h", [
    ("됐어.", "됬어."), ("할게.", "할께."), ("거야.", "꺼야."), ("네가 했어?", "니가 했어?"), ("네가 했어?", "내가 했어?"),
    ("애가 어디 갔어?", "얘가 어디 갔어?"), ("안 되요.", "안 돼요."), ("지금 보내드릴께요.", "지금 보내드릴게요."),
    ("뭐라고?", "뭐라구?"), ("하고 싶어.", "하구 싶어."), ("몇일 걸려?", "며칠 걸려?"), ("웬일이야.", "왠일이야."),
    ("거예요.", "거에요."), ("이거 뭐야?", "이것 뭐야?"), ("괜찮아.", "괜찬아."), ("같아.", "같애."), ("씨발.", "시발."),
    ("씨발.", "씨팔."), ("좆까.", "조까."), ("택시 불러.", "택씨 불러."), ("캘리포니아 모텔", "켈리포니아 모텔"),
    ("미진이 어디 있어?", "미지니 어디 있어?"), ("가시라구요.", "가시라고요."), ("안 맞어.", "안맞아"), ("임마.", "인마?"),
    ("있다가 봐.", "이따가 봐."), ("어떡해.", "어떻게.")])
def test_diffs_spelling_pairs(s, h):
    assert kinds(s, h) == ["spelling"]


@pytest.mark.parametrize("s, h", [
    ("너 죽었어.", "나 죽었어."), ("안 가.", "난 가."), ("죽여!", "죽어!"), ("물 줘.", "불 줘."), ("가자.", "가지."),
    ("밥 먹어.", "밤 먹어."), ("돈 어딨어?", "돌 어딨어?"), ("딸이 있어요.", "달이 있어요."), ("간다.", "한다."),
    ("형사님.", "행사님."), ("이리 와.", "이리 봐."), ("누가 그랬어?", "누나 그랬어?"), ("손 들어.", "돈 들어."),
    ("상무님.", "장모님."), ("사장님 오셨어요.", "사정님 오셨어요."), ("했니?", "했네?"), ("지영민 씨.", "지용민 씨."),
    ("경찰을 불러", "경찰은 불러"), ("찾았어.", "찾아서"), ("거요.", "거야?"), ("들어가자.", "들어가죠"),      # 조사·어미
    ("괜찮니?", "괜창니?"), ("초소 앞", "초수 앞"), ("횡설수설만", "횡설수절만"),                           # 비슷한 소리 오류
    ("그런데 말이야.", "근데 말이야."), ("이야기 좀 해.", "얘기 좀 해."), ("TV 꺼.", "티비 꺼."),
    ("내가 했어?", "니가 했어?")])
def test_diffs_other_words_particles_and_mishearings_are_word(s, h):
    """뜻이 다른 낱말, 조사·어미 바뀜, 받아쓰기의 비슷한 소리 오류, 줄임말은 spelling 이 아니라 word."""
    assert kinds(s, h) == ["word"]


# ------------------------------------------------------------------ 차이 조각: 묶기
def test_diffs_word():
    """낱말이 바뀌거나 빠지거나 더해지면 word. 보여 주는 글은 양쪽 어절이다."""
    assert T.diffs("나 간다 진짜", "나 간다 정말") == [{"kind": "word", "script": "진짜", "heard": "정말"}]
    assert T.diffs("너 어제 어디 있었어?", "어제 밤에 어디 있었냐고.") == [
        {"kind": "word", "script": "너", "heard": ""},
        {"kind": "word", "script": "", "heard": "밤에"},
        {"kind": "word", "script": "있었어?", "heard": "있었냐고."}]
    assert T.diffs("했어", "했어야") == [{"kind": "word", "script": "했어", "heard": "했어야"}]      # 낱말 안에 더해진 글자
    assert pieces("뭐야 이게", "뭐야 뭐 이게") == [("word", "", "뭐")]      # 부호 없는 '뭐' 는 군말도, 앞 낱말의 되풀이도 아니다


def test_diffs_keep_unchanged_words_out_of_the_pieces():
    """양쪽에 그대로 있는 낱말은 조각에 넣지 않는다: 낱말을 먼저 짝짓고 그 사이만 글자로 견준다."""
    assert pieces("서울 시장이란 양반이... 이 시간에...", "서울시장 날량거리. 이 시간의") == [
        ("word", "서울 시장이란 양반이...", "서울시장 날량거리."), ("word", "시간에...", "시간의")]
    assert pieces("도망을 가서 요즘 정말 힘들어요.", "도망을 가서요 요즘 정말 힘들어요.") == [("word", "가서", "가서요")]
    assert pieces("저게... 야! 야! 아 나 미친년 성질은...", "아 나 저년 저 성질 저거 아이씨") == [
        ("word", "저게... 야! 야!", ""), ("word", "미친년", "저년 저"), ("word", "성질은...", "성질 저거 아이씨")]
    d = pieces("꼭 해드릴 테니까 좀 살려주십쇼. 걔들 그 돈만 회수해도", "걔들 땡겨간 돈 그 돈만 회수해도")
    assert d == [("word", "꼭 해드릴 테니까 좀 살려주십쇼.", ""), ("word", "", "땡겨간 돈")]


def test_diffs_split_a_dropped_word_from_a_changed_word():
    """빠지거나 더해진 낱말과 바뀐 낱말이 낱말 경계에서 이어지면 따로 낸다."""
    assert pieces("아, 진짜 좋네.", "정말 좋네.") in (
        [("filler", "아,", ""), ("word", "진짜", "정말")], [("word", "아, 진짜", "정말")])
    assert pieces("그 돈만 회수해도... 상무님.", "그 돈만 회수해도 저 사모님한테 바로")[-1][0] == "word"


def test_diffs_merge_takes_heaviest_kind():
    """같은 어절에 닿는 차이는 하나로 합치고 종류는 가장 무거운 것(word > number > spelling > repeat > filler)."""
    d = T.diffs("아,상무님어떻게삼일만에그돈을해드려요...", "상무님 어떻게 3일 만에 그 돈을 해드려요")
    assert len(d) == 1 and d[0]["kind"] == "number"                 # 띄어쓰기 없는 대본: 줄 전체가 한 어절
    d = T.diffs("자,잠깐만요.따,딸이있어요.", "잠깐만요 딸이 있어요")
    assert len(d) == 1 and d[0]["kind"] == "repeat"
    d = T.diffs("3층이에요", "삼층이요")
    assert len(d) == 1 and d[0]["kind"] == "word"


def test_diffs_pieces_are_whole_words_in_text_order():
    """조각 글은 온 어절이고 글 순서대로다. 조각을 뺀 나머지는 양쪽 열쇠가 같다."""
    cases = [("예... 죄송합니다, 상무님... 애들이 둘씩이나 도망을 가서 요즘 정말 힘들어요. 걔들 땡겨준 돈만 천이 넘는데...",
              "장모님 죄송합니다. 저 우리 애들이 둘씩이나 도망을 가서요 요즘 정말 힘들어요 예 아니 걔들 땡겨준 돈만 해도 그게 돈이 천이 넘는데"),
             ("제가 다 죽였어요. 아홉 명... 아니 열두 명.", "제가 다 죽였어요 9명 아니 12명"),
             ("불었어요! 팔구삼에 일번지! 팔구삼에 일번지요!", "불었어요. 망원동 893 -1번지요."),
             ("야, 야. 뒤에 담배 떨어졌지? 뒤에. 봐봐.", "야, 야, 야, 야, 뒤에 봐봐, 뒤에, 뒤에 저 담배 떨어져 있지?")]
    for s, h in cases:
        d = T.diffs(s, h)
        sw, hw = s.split(), h.split()
        rest_s, rest_h, ps, ph = list(sw), list(hw), 0, 0
        for x in d:
            for words, rest, text in ((sw, rest_s, x["script"]), (hw, rest_h, x["heard"])):
                for w in text.split():
                    assert w in words
                    rest.remove(w)
            if x["script"]:
                ps = s.index(x["script"].split()[0], ps)
            if x["heard"]:
                ph = h.index(x["heard"].split()[0], ph)
        assert sorted(T.key(" ".join(rest_s))) == sorted(T.key(" ".join(rest_h)))
