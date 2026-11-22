"""말투 판정(반말, 해요체, 하십시오체).

논문에 없는 단계다. 주제의 '말투 살리기' 때문에 더했다.
형태소 분석기 없이 문장 끝 어미 규칙으로 판정한다. 끝 어미가 없는 조각(이름, 명사, 감탄사)은 '불명' 이다.

판정 순서
1. 문장 나누기: . ? ! ~ 와 줄바꿈. 말줄임표(… ..)는 말 사이 쉼으로 보고 나누지 않는다.
   문장부호 뒤에 공백이나 글 끝이 와야 나눈다(3.5초, "가자!"라고 같은 데서 나누지 않음).
2. 쉼표로 마디를 나누고 뒤에서부터 본다. 끝 어미가 분명한 마디(강함)가 나오면 그 말투로 정한다.
   그래서 '알겠습니다, 사장님.' 은 하십시오체, '고마워요, 정말.' 은 해요체다.
   쉼표 앞 마디의 이음 어미(-고, -는데, -니까, -어서, -며)는 문장 끝이 아니라서 말투로 치지 않는다.
3. 강한 마디가 없을 때 마지막 마디가 왜·뭐·진짜 같은 반말 낱말 하나뿐이면(약함) 반말이다.
하오체(-소, -구려, -시오)는 네 갈래에 없어서 높임 쪽인 하십시오체로 묶는다.
"""
from __future__ import annotations

import re

from .records import REGISTERS

LABELS = ("반말", "해요체", "하십시오체", "불명")

_NONE, _WEAK, _STRONG = 0, 1, 2

_TERMINATORS = ".?!~…。？！～"
_CLOSERS = "\"'”’」』)]）〉》"
_PAREN = re.compile(r"\([^()]*\)|（[^（）]*）|\[[^\[\]]*\]")   # (웃음) 같은 지문
_COMMA = re.compile(r"[,，、]")
_HANGUL = re.compile(r"[가-힣]+")
_EDGE = re.compile(r"^\W+|\W+$")

# 종성 번호(0 = 받침 없음, 4 = ㄴ, 17 = ㅂ, 18 = ㅄ, 20 = ㅆ, 27 = ㅎ), 중성 번호(11 ㅚ, 16 ㅟ, 19 ㅢ, 20 ㅣ).
# 유니코드 한글 음절 배열 순서다.
_JONG_N, _JONG_B = 4, 17
# 하오체 '-소' 앞 받침: 했소, 겠소, 없소, 그렇소, 좋소(명사 장소·주소·미소는 받침이 없거나 다름)
_JONG_BEFORE_SO = {18, 20, 27}
# 받침 없는 음절 뒤 '-어' 가 동사 어미인 경우: 쉬어, 뛰어, 되어, 띄어, 비어, 이어
_EO_AFTER_OPEN = {11, 16, 19, 20}

# 문장(마디) 전체가 이것뿐이면 감탄사·부름말이라 말투가 없다
_NEUTRAL_WHOLE = set("""
아 아아 아하 와 와아 우와 우와아 와우 오 오오 오호 자 자자 음 음음 흠 흐음 헐 헉 어머 어머나 아이고 아이구 어이구 에구 에고
에휴 아휴 어휴 세상에 맙소사 젠장 제기랄 이런 저런 아야 앗 엇 엥 에이 휴 후 하하 호호 히히 허허 크크 쉿 쉬 이야 오케이 땡큐
제발 으악 악 꺄 아이참 참 저기 여기 거기 짠 얍 영차 으쌰 브라보 할렐루야 아멘 만세 대박 어라 어랏 쯧 쳇 칫 흥 흑 캬
""".split())

_YES = re.compile(r"(?:네|예)+")                  # 네, 예, 네네(대답)
_POLITE_ANSWERS = {"아니요", "아뇨", "아니오"}

# 문장 전체가 이것이면 반말(강함): 대답·인사·사과
_CASUAL_STRONG = set("응 응응 엉 웅 야 야야 어이 그래 그래그래 안녕 됐어 됐다 미안 고마워 잘가 그치 아냐 몰라 알았어 좋아 "
                     "싫어 맞아".split())
# 마디 전체가 이것이면 반말(약함): 존댓말 끝에 덧붙인 부사일 수도 있어서 앞 마디가 이긴다
_CASUAL_WEAK = set("""
왜 뭐 뭘 진짜 정말 누구 누가 어디 언제 어떻게 그럼 잠깐 잠깐만 그만 나 너 내가 네가 니가 나도 너도 나는 너는 난 넌
글쎄 물론 그래서 근데 그런데 그러니까 그니까 아니 설마
""".split())
# 여러 낱말 마디의 마지막 낱말이 이것이면 반말(아, 응 / 이봐, 야 / 범인은 바로 너)
_CASUAL_TOKENS = {"응", "야", "안녕", "너", "나"}

# '-요' 로 끝나지만 명사인 낱말
_NOUN_YO = set("필요 중요 수요 강요 동요 주요 소요 요요 개요".split())

# 반말 어미와 끝 글자가 같은 명사·이름·호칭·감탄사(낱말 전체가 같을 때만)
_NEUTRAL_WORDS = set("""
언니 누나 할머니 어머니 외할머니 시어머니 아버지 할아버지 외할아버지 시아버지 아가 왕자
어머나 아이고 어이고 에고 아야 우와 와아
영어 단어 언어 국어 용어 숙어 속어 상어 문어 연어 오징어 붕어 잉어 장어 숭어 고등어 광어 농어 은어 인어
한국어 중국어 일본어 외국어 독일어 스페인어 아랍어 베트남어 태국어 모국어 표준어 목적어 서술어 유의어 반의어
참여 급여 기여 부여 수여 잉여 주여 증여 대여 관여 투여
시야 마야 분야 광야
새해 올해 오해 이해 손해 방해 동해 서해 남해 공해 재해 화해 견해
기와 샤워 타워 파워
노래 미래 원래 고래 모래 빨래 거래 오래 재래 도래 유래 외래 본래 장래
군대 반대 침대 무대 세대 시대 기대 상대 최대 부대 고대 현대 절대 대대 교대 연대 학대 접대 환대 일대 거대 확대
의대 공대 법대 해병대 소방대 구조대 특공대 경비대
그게 이게 저게 가게 무게 지게 집게 대게 꽃게
강아지 돼지 바지 휴지 편지 반지 이미지 메시지 페이지 소시지 오렌지 송아지 망아지 엄지 목적지
뭐든지 언제든지 어디든지 누구든지 조지 앤지
의자 남자 여자 과자 모자 상자 기자 부자 피자 감자 숫자 글자 환자 이자 투자 혼자 저자 독자 학자 액자 원자 제자
탁자 주전자 손자 효자 신자 과학자 범죄자 피해자 가해자 용의자 목격자 관리자 사용자 참가자 운전자 보호자 배우자
후보자 지원자 책임자 담당자
카메라 콜라 고릴라 오로라 코브라 드라큘라 사라 노라 클라라 라라 헤라 아우라 미라 카라 나라 사쿠라 로라 세라
소라 보라 사하라 마스카라
지니 미니 비키니 토니 제니 조니 대니 케니 보니 로니 애니 페니 써니 스테파니 멜라니 브리트니 레니 시드니 버니 코니 바니
소냐 타냐 아까 까까
하나 바나나 언제나 어디나 아무나 누구나 아무거나 나나 한나 안나 유나 루나 모아나 미나 아테나 티나 니나 애나 레나
엘레나 헬레나 다이애나
동네 우리네
바다 캐나다 플로리다 그라나다 네바다 버뮤다 우간다 르완다 아젠다 린다 아만다 미란다 브렌다 완다
최고 사고 경고 광고 창고 금고 신고 망고 레고 고고 원고 재고 휴고 마고 디에고 산티아고 시카고 콩고 탱고
장군 해군 공군 육군 아군 적군 국군 대군 지원군 반란군 연합군 정부군
배려 고려 우려 격려 장려 수려
달러 컬러 딜러 킬러 스릴러 트레일러 롤러 텔러 쿨러
안내 실내 시내 국내 사내 이내 교내 원내
어깨 참깨 들깨
스피커 해커 조커 마커 스티커 로커 사커 메이커 쿠커
벌써 뭔가 어딘가 언젠가 누군가 무언가
""".split())

# 반말 끝 글자(낱말 길이 상관없음). 해체 줄임꼴(해 돼 줘 봐 쳐 켜 려 …)과 해라체·감탄·의문 어미다.
_CASUAL_FINALS = set("해 돼 줘 봐 와 워 둬 놔 쏴 춰 꿔 숴 쳐 켜 꺼 져 텨 뎌 벼 셔 려 러 내 깨 빼 커 떠 써 여 "
                     "지 자 라 니 냐 게 래 대 데 까 나 다 고 군 걸 며 렴".split())
# 반말 끝 여러 글자
_CASUAL_SUFFIXES = ("지마", "거든", "구나", "구먼", "구만", "더라", "지롱", "든지", "든가", "어때",
                    "아파", "고파", "슬퍼", "기뻐", "예뻐", "바빠", "나빠")
# '-어' 로 끝나도 언어 이름인 꼴(라틴어, 포르투갈어, 스웨덴어, 몽골어, 한국어, 일본어)
_LANGUAGE_TAILS = ("국어", "본어", "일어", "인어", "남어", "랍어", "틴어", "갈어", "덴어", "골어", "준어")
# '-가' 로 끝나는 반말: 나가, 들어가, 올라가, 내려가, 가져가, 물러가, 건너가, 다가가, 가다가
_GA_ENDINGS = ("나가", "어가", "아가", "라가", "려가", "져가", "러가", "너가", "다가")
# '-서' 로 끝나는 이음 어미(문장 끝에 오면 반말: 그래서?, 했다면서?)
_SEO_LINKING = ("어서", "아서", "해서", "와서", "봐서", "워서", "래서", "라서", "여서", "쳐서", "져서", "러서",
                "려서", "춰서", "면서")
# '-서' 로 끝나는 반말 전체(명령 일어서, 물러서 포함)
_SEO_ENDINGS = _SEO_LINKING + ("일어서", "물러서")
# 쉼표 앞에서는 문장 끝이 아닌 이음 어미(배고프니까, 밥 먹어요 / 제가 했는데, 혹시…)
_LINKING_ENDINGS = ("니까", "고", "데", "며", "다가", "거든", "든지", "든가") + _SEO_LINKING
# 그래도 문장을 끝내는 강조·인용 꼴(했다니까, 힐디 / 알았다고, 엄마)
_QUOTE_ENDINGS = ("다니까", "라니까", "냐니까", "자니까", "다고", "라고", "냐고", "자고")


def _jong(ch: str) -> int:
    return (ord(ch) - 0xAC00) % 28


def _jung(ch: str) -> int:
    return (ord(ch) - 0xAC00) // 28 % 21


def _is_pause(run: str) -> bool:
    """말줄임표(…, .., ...)인지. 물음표·느낌표가 섞이면 문장 끝이다."""
    return all(c in ".…" for c in run) and ("…" in run or run.count(".") >= 2)


def _terminator(sentence: str) -> str:
    """문장 끝 문장부호. 뒤에 붙은 닫는 따옴표·이모지 같은 기호는 건너뛴다."""
    end = len(sentence)
    while end > 0 and not sentence[end - 1].isalnum() and sentence[end - 1] not in _TERMINATORS:
        end -= 1
    start = end
    while start > 0 and sentence[start - 1] in _TERMINATORS:
        start -= 1
    return sentence[start:end]


def _sentences(text: str) -> list[tuple[str, str]]:
    """(문장, 끝 문장부호) 목록. 글자·숫자가 하나도 없는 조각(이모지만 등)은 버린다."""
    pieces: list[str] = []
    for line in _PAREN.sub(" ", text).splitlines():
        start, i, n = 0, 0, len(line)
        while i < n:
            if line[i] not in _TERMINATORS:
                i += 1
                continue
            j = i
            while j < n and line[j] in _TERMINATORS:
                j += 1
            k = j
            while k < n and line[k] in _CLOSERS:
                k += 1
            if (k == n or line[k].isspace()) and not _is_pause(line[i:j]):
                pieces.append(line[start:k])
                start = k
            i = k
        pieces.append(line[start:])
    out = []
    for piece in pieces:
        piece = piece.strip()
        if any(ch.isalnum() for ch in piece):
            out.append((piece, _terminator(piece)))
    return out


def _last_word(clause: str) -> tuple[str, bool, str]:
    """(마지막 한글 낱말, 그 토큰이 한글 낱말뿐인지, 바로 앞 토큰의 한글 낱말)."""
    tokens = [t for t in clause.split() if _HANGUL.search(t)]
    if not tokens:
        return "", False, ""
    tok = tokens[-1]
    word = _HANGUL.findall(tok)[-1]
    pure = _EDGE.sub("", tok) == word
    prev = _HANGUL.findall(tokens[-2])[-1] if len(tokens) >= 2 else ""
    return word, pure, prev


def _formal(word: str) -> bool:
    """하십시오체 끝: -ㅂ니다, -ㅂ니까, -ㅂ니다만, -ㅂ디다, -ㅂ시다, -십시오, -소서, -ㅂ쇼.

    하오체(-시오, -소, -구려)도 여기에 넣는다. '니까' 는 앞 음절 받침이 ㅂ 일 때만(합니까, 습니까).
    받침 ㅂ 이 없는 '-니까' 는 반말 이음 어미다(그러니까, 했으니까).
    """
    if len(word) >= 3 and word[-2:] in ("니다", "니까", "디다", "디까", "시다") and _jong(word[-3]) == _JONG_B:
        return True
    if len(word) >= 4 and word.endswith("니다만") and _jong(word[-4]) == _JONG_B:
        return True
    if word.endswith(("시오", "소서", "구려")):
        return True
    if len(word) >= 2 and word[-1] == "소" and _jong(word[-2]) in _JONG_BEFORE_SO:
        return True
    return len(word) >= 2 and word[-1] == "쇼" and _jong(word[-2]) == _JONG_B


def _polite(word: str, pure: bool) -> bool:
    """해요체 끝: -요(명사 필요·중요 등 제외), -죠."""
    if word.endswith("요"):
        return not (pure and word in _NOUN_YO)
    return word.endswith("죠")


def _casual_end(word: str) -> bool:
    """반말 어미로 끝나는지."""
    if word.endswith(_CASUAL_SUFFIXES):
        return True
    last = word[-1]
    before = word[-2] if len(word) >= 2 else ""
    if last in _CASUAL_FINALS:
        return True
    if last == "아":        # 좋아, 잡아, 있잖아, 서연아(부름). 받침 없는 음절 뒤 '아'(줄리아, 마리아)는 이름이 많다
        return bool(before) and _jong(before) != 0
    if last == "어":        # 먹어, 했어, 쉬어, 뛰어
        if word.endswith(_LANGUAGE_TAILS):
            return False
        return bool(before) and (_jong(before) != 0 or _jung(before) in _EO_AFTER_OPEN)
    if last == "야":        # 뭐야, 거야, 민지야. 받침 뒤 '야'(분야)는 명사
        return not before or _jong(before) == 0
    if last == "네":        # 좋네, 하네. 한 글자 '네' 는 대답이라 앞에서 해요체로 처리
        return bool(before)
    if last == "가":        # 빨리 가, 나가, 그런가, 아닌가
        return word == "가" or word.endswith(_GA_ENDINGS) or (bool(before) and _jong(before) == _JONG_N)
    if last == "서":        # 거기 서, 일어서, 그래서?
        return word == "서" or word.endswith(_SEO_ENDINGS)
    if last == "타":        # 어서 타
        return word == "타" or word.endswith("올라타")
    return False


def _before_trailing_name(clause: str) -> str | None:
    """쉼표 없이 이름·호칭이 뒤에 붙은 마디('안녕 짐', '고마워요 민수 씨')에서 앞 낱말의 말투.

    마지막 토큰 앞 두 토큰까지만 본다. '-아/-어' 꼴은 '살아 있는 전설' 처럼 이음꼴일 수 있어서 보지 않고,
    하십시오체·해요체 어미와 반말 낱말(안녕, 그래, 알았어 …)만 받는다.
    """
    tokens = [t for t in clause.split() if _HANGUL.search(t)]
    for tok in reversed(tokens[-3:-1]):
        word = _HANGUL.findall(tok)[-1]
        if _EDGE.sub("", tok) != word:
            continue
        if _formal(word):
            return "하십시오체"
        if _polite(word, True):
            return "해요체"
        if word in _CASUAL_STRONG or word in _CASUAL_TOKENS:
            return "반말"
    return None


def _clause_label(clause: str, is_last: bool, term: str) -> tuple[str, int]:
    """쉼표로 나눈 마디 하나의 (말투, 세기)."""
    whole = "".join(_HANGUL.findall(clause))
    if not whole:
        return "불명", _NONE
    if whole == "어":
        # '어.' 는 반말 대답, '어?' '어!' '어…' 는 놀람·머뭇거림 감탄사
        plain = term == "" or term in (".", "。") or set(term) <= {"~", "～"}
        return ("반말", _STRONG) if is_last and plain else ("불명", _NONE)
    if whole in _NEUTRAL_WHOLE:
        return "불명", _NONE
    if _YES.fullmatch(whole) or whole in _POLITE_ANSWERS:
        return "해요체", _STRONG
    word, pure, prev = _last_word(clause)
    if pure and (_YES.fullmatch(word) or word in _POLITE_ANSWERS):
        return "해요체", _STRONG
    if _formal(word):
        return "하십시오체", _STRONG
    if _polite(word, pure):
        return "해요체", _STRONG
    if whole in _CASUAL_STRONG:
        return "반말", _STRONG
    if whole in _CASUAL_WEAK:
        return "반말", _WEAK
    if pure and word in _CASUAL_TOKENS:
        return "반말", _STRONG
    if word == "마" and prev:    # 하지 마, 가지 마, 걱정 마 (따로 띄운 '마' 는 금지 명령)
        return "반말", _STRONG
    if not (pure and word in _NEUTRAL_WORDS):     # 명사·이름·호칭이면 어미로 보지 않는다
        if not is_last and word.endswith(_LINKING_ENDINGS) and not word.endswith(_QUOTE_ENDINGS):
            return "불명", _NONE
        if _casual_end(word):
            return "반말", _STRONG
    before_name = _before_trailing_name(clause)
    if before_name is not None:
        return before_name, _STRONG
    return "불명", _NONE


def _sentence_label(sentence: str, term: str) -> str:
    clauses = [c for c in _COMMA.split(sentence) if any(ch.isalnum() for ch in c)]
    if not clauses:
        return "불명"
    labels = [_clause_label(c, i == len(clauses) - 1, term) for i, c in enumerate(clauses)]
    for label, strength in reversed(labels):
        if strength == _STRONG:
            return label
    label, strength = labels[-1]
    return label if strength == _WEAK else "불명"


def detect(text: str) -> list[str]:
    """문장마다 '반말' | '해요체' | '하십시오체' | '불명' 을 돌려준다."""
    return [_sentence_label(s, term) for s, term in _sentences(text)]


def check(text: str, requested: str) -> tuple[bool, list[str], str]:
    """요청한 말투에 맞는지 본다. (통과 여부, 문장별 판정, 탈락 까닭)

    미지정은 늘 통과. 반말 요청은 해요체·하십시오체 문장이 있으면 탈락. 해요체 요청은 반말 문장이 있으면 탈락.
    하십시오체 요청은 반말 문장이 있거나, 하십시오체 문장 없이 해요체 문장만 있으면 탈락(섞임은 허용).
    '불명' 문장은 어느 말투와도 어긋나지 않는다.
    """
    requested = requested or "미지정"
    if requested not in REGISTERS:
        raise ValueError(f"알 수 없는 말투입니다: {requested} ({', '.join(REGISTERS)} 중 하나)")
    found = detect(text)
    if requested == "미지정":
        return True, found, ""
    if requested == "반말":
        bad = [r for r in ("해요체", "하십시오체") if r in found]
    elif "반말" in found:
        bad = ["반말"]
    elif requested == "하십시오체" and "하십시오체" not in found and "해요체" in found:
        return False, found, "말투: 하십시오체 요청인데 해요체 문장만 있음"
    else:
        bad = []
    if bad:
        return False, found, f"말투: {requested} 요청인데 {'·'.join(bad)} 문장"
    return True, found, ""
