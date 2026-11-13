"""구조 파싱의 글자 도구: 정규식, 언어 판정, 화자 꼬리표, 괄호 지시, 줄 잇기.

parse.py 가 쓴다. 쪽 배치(x·y)는 보지 않고 글자만 본다. 모두 결정적이고 상태가 없다.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

# ---------------------------------------------------------------- 글자 정리

_WS_MAP = str.maketrans({
    " ": " ", " ": " ", " ": " ", "　": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", " ": " ", "​": "", "﻿": "", "꞉": ":",
    "：": ":", "（": "(", "）": ")", "‐": "-", "‑": "-", "­": "",
})
_RX_SPACES = re.compile(r"[ \t]+")
_RX_REV_TAIL = re.compile(r"(?:[ \t]+\*{1,3})+\s*$")      # 오른쪽 여백 수정 표시 '*'
_RX_REV_HEAD = re.compile(r"^\*{1,3}[ \t]+(?=\S)")


def norm(s: str) -> str:
    """분석용 글: 특수 공백을 보통 공백으로, 쌍점 변형을 ':' 로. 탭은 남긴다."""
    return unicodedata.normalize("NFC", s.translate(_WS_MAP))


def strip_rev(s: str) -> str:
    """줄 끝·앞에 붙은 수정 표시 '*' 떼기(글 사이의 '*' 는 둔다)."""
    s = _RX_REV_TAIL.sub("", s)
    return _RX_REV_HEAD.sub("", s)


def flat(s: str) -> str:
    """탭·여러 공백을 공백 하나로."""
    return _RX_SPACES.sub(" ", s).strip()


def is_hangul(c: str) -> bool:
    o = ord(c)
    return 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F


def char_script(c: str) -> str:
    o = ord(c)
    if 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
        return "ko"
    if 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF:
        return "kana"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF:
        return "han"
    if 0x0400 <= o <= 0x04FF:
        return "cyr"
    if c.isalpha():
        if c.isascii() or 0x00C0 <= o <= 0x024F:
            return "lat"
        return "other"
    if c.isdigit():
        return "digit"
    return ""


_RX_SCRIPTS = (
    ("ko", re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")),
    ("kana", re.compile(r"[぀-ヿㇰ-ㇿ]")),
    ("han", re.compile(r"[一-鿿㐀-䶿豈-﫿]")),
    ("cyr", re.compile(r"[Ѐ-ӿ]")),
    ("lat", re.compile(r"[A-Za-zÀ-ÖØ-öø-ɏ]")),
    ("digit", re.compile(r"\d")),
)


def script_counts(s: str) -> Counter:
    """글자 종류별 개수(char_script 와 같은 분류를 정규식으로 빠르게)."""
    c = Counter()
    for k, rx in _RX_SCRIPTS:
        n = len(rx.findall(s))
        if n:
            c[k] = n
    return c


def letters(s: str) -> int:
    return sum(1 for ch in s if ch.isalpha())


# ---------------------------------------------------------------- 언어

_STOP = {
    "en": set("the and of to a in is you that it he was for on are with as his they i at be this have from "
              "or one had by but not what all were we when your can said there an she her him my me no "
              "don't it's i'm we're just out up into looks".split()),
    "fr": set("le la les de des du un une et est que qui il elle je tu nous vous ne pas dans sur pour "
              "avec au aux ce cette son sa ses mais ou on se lui leur c'est j'ai qu'il".split()),
    "es": set("el la los las de del y que en un una es no se por con para su al lo como pero mas sí yo "
              "tu usted está estoy qué".split()),
    "de": set("der die das und ist nicht ich du er sie es ein eine zu den mit auf für von dem sich auch "
              "wir ihr was wie".split()),
    "it": set("il lo la gli le di che e è un una per non in con su mi ti si sono io tu lui lei ma come".split()),
}
_RX_WORD = re.compile(r"[A-Za-zÀ-ÿ']+")


def latin_lang(text: str) -> str | None:
    words = [w.lower() for w in _RX_WORD.findall(text)]
    if len(words) < 3:
        return None
    best, bn = None, 0
    for lg, st in _STOP.items():
        n = sum(1 for w in words if w in st)
        if n > bn:
            best, bn = lg, n
    return best if bn >= max(2, len(words) // 12) else None


def doc_lang(texts: list[str]) -> tuple[str, str]:
    """(문서 언어, 라틴 글자일 때 쓸 언어)."""
    c = Counter()
    step = max(1, len(texts) // 1500)
    for t in texts[::step]:
        c.update(script_counts(t))
    total = sum(v for k, v in c.items() if k != "digit") or 1
    sample = " ".join(texts[:: max(1, len(texts) // 400)])
    lat = latin_lang(sample) or "en"
    if c["ko"] / total >= 0.25:
        return "ko", lat
    if c["kana"] / total >= 0.2:
        return "ja", lat
    if c["han"] / total >= 0.3:
        return "zh", lat
    if c["cyr"] / total >= 0.3:
        return "ru", lat
    if c["lat"] / total >= 0.3:
        return lat, lat
    return "unknown", lat


def text_lang(s: str, doc: str, doc_latin: str) -> str:
    """글 하나의 언어. 글자 종류가 문서 언어와 같으면 문서 언어."""
    c = script_counts(s)
    c.pop("digit", None)
    tot = sum(c.values())
    if not tot:
        return doc
    k, n = c.most_common(1)[0]
    if k == "ko":
        return "ko"
    if k == "kana" or (k == "han" and c["kana"]):
        return "ja"
    if k == "han":
        return "ko" if doc == "ko" and c["ko"] else "zh"
    if k == "cyr":
        return "ru"
    if k == "lat":
        if doc not in ("ko", "ja", "zh", "ru", "unknown"):
            if tot >= 10:
                lg = latin_lang(s)
                if lg and lg != doc:
                    return lg
            return doc
        lg = latin_lang(s) if tot >= 12 else None
        return lg or (doc_latin if doc_latin != "en" and tot >= 12 else "en")
    return doc


# ---------------------------------------------------------------- 씬 머리

KO_TIME = (r"(?:한낮|낮|대낮|밤|한밤|한밤중|밤중|심야|늦은\s?밤|아침|이른\s?아침|저녁|초저녁|새벽|이른\s?새벽|오후|늦은\s?오후|오전|"
           r"정오|자정|황혼|해\s?질\s?녘|해질녁|석양|노을|일몰|일출|동틀\s?녘|여명|다음\s?날|다음날|며칠\s?후|"
           r"몇\s?시간\s?후|잠시\s?후|시간\s?경과|계속|연속|같은\s?시각|동시|"
           r"DAY|NIGHT|D|N|M|E|DAWN|DUSK)")
EN_TIME = (r"(?:DAY|NIGHT|MORNING|EVENING|AFTERNOON|DUSK|DAWN|SUNSET|SUNRISE|TWILIGHT|MIDNIGHT|NOON|MAGIC HOUR|"
           r"LATER|MOMENTS LATER|A MOMENT LATER|CONTINUOUS|CONT(?:'|’)?D|SAME|SAME TIME|PRE-?DAWN|LATE NIGHT|"
           r"EARLY MORNING|LATE AFTERNOON|EARLY EVENING|NIGHTFALL|DAYBREAK|FLASHBACK|PRESENT|JOUR|NUIT|SOIR|"
           r"MATIN|AUBE|CR[ÉE]PUSCULE|APR[ÈE]S-MIDI|D[ÍI]A|NOCHE)")
_NUM = r"(?P<num>\d{1,4}(?:\s?\.?\s?[A-Za-z](?![A-Za-z]))?(?:\s?-\s?\d{1,3}[A-Za-z]?)?)"

RX_H_INTEXT = re.compile(
    r"^(?:(?P<num>\d{1,4}[A-Z]{0,3}(?:-\d{1,3}[A-Z]?)?)\s*[.):\-]?\s+)?"
    r"(?P<ie>(?:INT|EXT|I/E|E/I|INT\.?\s?/\s?EXT|EXT\.?\s?/\s?INT|INT\.?\s?-\s?EXT|INTERIOR|EXTERIOR|EST|ESTABLISHING)"
    r"(?:\.|:|\b))\s*(?P<rest>.*)$")
RX_H_SHARP = re.compile(r"^\s*(?:S\s*[#＃]|[#＃]\s*(?:씬|신|S)?|S\s*/|S\.(?=\s*\d)|(?:씬|신|장면)\s*[#＃]?)\s*"
                        + _NUM + r"\s*(?:[.,:)\]]|-(?!\d))?\s*(?P<rest>.*)$")
# 번호 없는 'S#⇥병실 (밤)'(오버 더 레인보우): 문서 안에서 5번 이상 되풀이될 때만 씬 머리(parse._accept_headings)
RX_H_SHARP_NONUM = re.compile(r"^\s*S\s*[#＃]\s+(?P<rest>[^\d\s.,:)\]].{0,59})$")
RX_H_SCENEWORD = re.compile(r"^\s*(?:SCENE|Scene|SC\.?)\s*" + _NUM + r"\s*(?:[.:)\-]|/+)?\s*(?P<rest>.*)$")
RX_H_P = re.compile(r"^\s*(?P<num>P\s?\d{1,3}[A-Za-z]?)\s*[.)]\s*(?P<rest>.+)$")
RX_H_CUT = re.compile(r"^\s*(?P<num>C\s*[#＃]\s*\d{1,3}[A-Za-z]?)\s*[.:)]?\s*(?P<rest>.*)$")
RX_H_NUMDOT = re.compile(r"^\s*" + _NUM + r"\s*(?:[.)](?!\d)|-(?=\s|[^\d\s])|\s{2,}|\t|(?=[#＃]))\s*(?P<rest>\S.*)$")
RX_H_NUMONLY = re.compile(r"^\s*(?:S\s*[#＃]|[#＃]|씬|신)?\s*" + _NUM + r"\s*\.?\s*$")
RX_H_KO_NONUM = re.compile(
    r"^(?P<rest>[^.?!…\"“”'‘’()]{2,40}?)\s*(?:(?:/|–|-|,)\s*(?P<t1>" + KO_TIME + r")\s*\.?"
    r"|[(\[]\s*(?:실내|실외|내부|외부|내|외|INT|EXT|I|E)?[\s.,/]*(?P<t2>" + KO_TIME + r")\s*[)\]])\s*$")
RX_H_EN_NOINT = re.compile(r"^(?P<rest>[A-Z0-9][A-Z0-9 '’.,&/()\-]{2,60}?)\s+[-–—]\s+(?P<t>" + EN_TIME + r")\s*$")
RX_TIME_TAIL = re.compile(
    r"(?:\s*(?:[-–—/,.]|\s)\s*|\s*[(\[]\s*)(?P<t>(?:(?:실내|실외|내부|외부|내|외|INT|EXT)\s*[.,/]?\s*)?"
    r"(?:" + KO_TIME + "|" + EN_TIME + r"))\s*[)\]]?\s*\.?\s*$", re.I)
RX_INTEXT_WORD = re.compile(r"^(?:INT|EXT|I/E|INT\.?/EXT|EXT\.?/INT|INTERIOR|EXTERIOR)\.?\s*[-–]?\s*", re.I)

FLAG_RX = {
    "montage": re.compile(r"몽타[주쥬]|MONTAGE|SERIES OF SHOTS|몽따주", re.I),
    "flashback": re.compile(r"회상|플래[시쉬]\s?백|플래쉬|플래시|FLASH\s?BACK|FLASHBACK|과거[.\s/)-]|\(과거\)|^과거$|RETOUR EN ARRI", re.I),
    "intercut": re.compile(r"인터컷|교차\s?편집|교차|INTER\s?CUT|INTERCUT", re.I),
    "dream": re.compile(r"꿈|상상|환상|DREAM|FANTASY|NIGHTMARE|R[ÊE]VE", re.I),
}


def norm_scene_number(s: str) -> str:
    return re.sub(r"[\s.]", "", s).upper().replace("＃", "#")


RX_H_NUMBOTH = re.compile(r"^(?P<num>\d{1,4}[A-Z]{0,3}(?:-\d{1,3}[A-Z]?)?)\s+(?P<rest>\S.*?)\s+(?P=num)\s*\*?$")


def match_heading(t: str) -> tuple[str, str | None, str] | None:
    """(형식, 번호 문자열|None, 나머지 글). t 는 flat 된 글."""
    if len(t) > 120:
        return None
    m = RX_H_NUMBOTH.match(t)
    if m and len(m.group("rest")) <= 80 and not RX_H_INTEXT.match(m.group("rest")) \
            and m.group("rest").upper() == m.group("rest"):
        return "numboth", norm_scene_number(m.group("num")), m.group("rest").strip()
    m = RX_H_INTEXT.match(t)
    if m:
        rest = m.group("ie") + (" " + m.group("rest") if m.group("rest") else "")
        num = m.group("num")
        rest, tail_num = _strip_tail_num(rest)
        return "intext", norm_scene_number(num) if num else (tail_num or None), rest.strip()
    m = RX_H_SHARP.match(t)
    if m:
        return "sharp", norm_scene_number(m.group("num")), m.group("rest").strip()
    m = RX_H_SHARP_NONUM.match(t)
    if m:
        return "sharp_nonum", None, m.group("rest").strip()
    m = RX_H_SCENEWORD.match(t)
    if m:
        return "sceneword", norm_scene_number(m.group("num")), m.group("rest").strip()
    m = RX_H_P.match(t)
    if m:
        return "p", norm_scene_number(m.group("num")), m.group("rest").strip()
    m = RX_H_CUT.match(t)
    if m:
        return "cut", norm_scene_number(m.group("num")), m.group("rest").strip()
    m = RX_H_NUMDOT.match(t)
    if m:
        rest = m.group("rest").strip()
        rest = re.sub(r"^[#＃]\s*", "", rest)
        rest, _ = _strip_tail_num(rest)
        m2 = RX_H_INTEXT.match(rest)
        return ("numdot_ie" if m2 else "numdot"), norm_scene_number(m.group("num")), rest
    m = RX_H_EN_NOINT.match(t)
    if m and not t.endswith(("TO:", "TO.")):
        return "en_noint", None, t
    m = RX_H_KO_NONUM.match(t)
    if m and any(is_hangul(c) for c in m.group("rest")):
        return "ko_nonum", None, t
    return None


def _strip_tail_num(rest: str) -> tuple[str, str | None]:
    m = re.search(r"\s+(\d{1,4}[A-Z]{0,3}(?:-\d{1,3}[A-Z]?)?)\s*\*?\s*$", rest)
    if m and not re.search(r"(?:19|20)\d\d\s*$", rest):
        return rest[: m.start()], norm_scene_number(m.group(1))
    return rest, None


_RX_EN_TIME_WORD = re.compile(r"\b" + EN_TIME + r"\b")


def split_place_time(rest: str) -> tuple[str | None, str | None]:
    s = rest.strip().strip("<>[]").strip()
    s = re.sub(r"\s+\d{1,4}[A-Z]?\s*$", "", s)
    # 'BEDROOM - THE NEXT MORNING', '거리 / 늦은 오후': 마지막 구분자 뒤 덩어리에 시간 낱말이 있으면 통째로 시간
    parts = re.split(r"\s+[-–—]\s+", s)
    if len(parts) >= 2 and len(parts[-1]) <= 30 and _RX_EN_TIME_WORD.search(parts[-1]):
        tail = parts[-1]
        head = s[: len(s) - len(tail)].rstrip(" -–—")
        return (RX_INTEXT_WORD.sub("", head).strip(" .-–—/,:") or None), tail.strip(" .")
    m = RX_TIME_TAIL.search(s)
    time = None
    if m and m.start() > 0:
        time = m.group("t").strip(" .")
        s = s[: m.start()]
    place = RX_INTEXT_WORD.sub("", s).strip(" .-–—/,:")
    return (place or None), (time or None)


def heading_flags(text: str) -> list[str]:
    return [k for k, rx in FLAG_RX.items() if rx.search(text)]


# ---------------------------------------------------------------- 전환·화면 글자·이어짐·쪽번호

RX_TRANSITION = re.compile(
    r"^[<\[(]?\s*(?:"
    r"(?:SMASH |MATCH |JUMP |HARD |QUICK |TIME |STRAIGHT |FAST |FLASH )?CUT(?:S)?(?: BACK)?(?: TO)?(?: BLACK| WHITE| SCENE)?"
    r"|(?:SLOW |QUICK )?DISSOLVE(?:S)?(?: TO)?(?: BLACK| WHITE)?|(?:SLOW )?FADE(?:S)?(?: ?- ?| )(?:IN|OUT|UP|DOWN|TO(?: BLACK| WHITE)?)(?: TO BLACK)?"
    r"|FADE ?IN|FADE ?OUT|WIPE(?: TO)?|IRIS (?:IN|OUT)|BACK TO(?: PRESENT| SCENE| REALITY)?|INTERCUT(?: WITH)?|"
    r"SMASH TO BLACK|BLACKOUT|BLACK OUT|THE END|END|END CREDITS|ROLL CREDITS|OVERLAP|LAP DISSOLVE|"
    r"FONDU(?: ENCHA[IÎ]N[ÉE]| AU NOIR)?|NOIR|CUT|COUPE|"
    r"F\.?\s?[IO]\.?|O\.?\s?L\.?|D\.?\s?S\.?|디졸브(?:\s?투)?|오버\s?랩|암전|화면\s?전환|장면\s?전환|컷\s?투|커트\s?투|컷|"
    r"페이드\s?(?:인|아웃)|화이트\s?아웃|블랙\s?아웃|화이트\s?인|와이프|점프\s?컷|점프|플래시\s?컷|끝|막|타이틀\s?아웃"
    r")\s*[>\])]?\s*[:.]?\s*[>\])]?\s*$", re.I)
RX_TRANSITION_TO = re.compile(r"^[A-Z][A-Z .'’-]{1,30} TO[:.]?$")
RX_ONSCREEN = re.compile(
    r"^\s*[<\[(]?\s*(?P<kw>하단\s?자막|화면\s?자막|자막|메인\s?타이틀|타이틀|인서트|INSERT(?: CUT)?|SUPER(?:IMPOSE[D]?)?|"
    r"TITLE(?: CARD)?|MAIN TITLE|CARD|SUBTITLES?|SUBTITLED|CHYRON|CAPTION|LEGEND|ON SCREEN|TEXT ON SCREEN|ON THE SCREEN|"
    r"TEXT|CARTON|SOUS-TITRES?|INCRUSTATION|PROJECTED)"
    r"\s*(?P<sep>[:：>\]\-–—)]|\s|$)\s*(?P<body>.*)$", re.I)
_ONSCREEN_KIND = (
    (re.compile(r"자막|SUPER|SUBTITLE|CHYRON|CAPTION|LEGEND|SOUS|INCRUST", re.I), "subtitle"),
    (re.compile(r"인서트|INSERT", re.I), "insert"),
    (re.compile(r"타이틀|TITLE|CARD|CARTON", re.I), "title"),
)
RX_CONTINUED = re.compile(r"^(?:\d{1,4}[A-Z]?\s+)?\(?\s*(?:CONTINUED|CONTINUÉ|SUITE|CONT['’]?D|CONT\.)\s*\)?\s*[:.]?\s*"
                          r"(?:\(\s*\d+\s*\))?\s*(?:\d{1,4}[A-Z]?)?\s*$", re.I)
RX_CONT_MARKER = re.compile(r"^\(\s*(?:MORE|CONT['’]?D|CONT\.?|CONTINUED|계속|이어서|SUITE|PLUS)\s*\)$|^[-–—]?\s*계\s?속\s*[-–—]?$"
                            r"|^\(?\s*MORE\s*\)?$", re.I)
RX_PAGE_NUM = re.compile(r"^(?:[-–—]\s*\d{1,3}\s*[-–—]|\d{1,3}\s*/\s*\d{1,3}|(?:page|pg\.?|p\.)\s*\d{1,3}(?:\s*(?:/|of)\s*\d{1,3})?"
                         r"|\(\s*\d{1,3}\s*\)|\d{1,3}\s*쪽|\d{1,3}\.)$", re.I)
RX_BARE_NUM = re.compile(r"^\d{1,4}[A-Z]{0,2}$")
RX_REVISION = re.compile(r"^[*＊]+(?:\s*[*＊]+)*$")
RX_SEPARATOR = re.compile(r"^(?:[-=~_*·•.]\s*){3,}$")


def match_transition(t: str) -> bool:
    if len(t) > 40:
        return False
    return bool(RX_TRANSITION.match(t) or RX_TRANSITION_TO.match(t))


def match_onscreen(t: str) -> tuple[str, str] | None:
    if t.startswith("{") and len(t) >= 3:
        return "subtitle", t.strip("{} ")                      # 녹취 대본의 화면 자막 '{SEPTEMBER 15, 2008}'
    m = RX_ONSCREEN.match(t)
    if not m:
        return None
    kw = m.group("kw")
    sep = m.group("sep")
    body = m.group("body").strip()
    # 영어 낱말은 대문자일 때만(문장 안 'Text' 같은 것 막기), 'TEXT' 는 쌍점이 있어야
    if kw.isascii() and kw != kw.upper() and not (not body and kw.upper() in ("SUBTITLES", "SUBTITLE", "INSERT", "TITLE",
                                                                           "TITLE CARD", "SUPER", "CAPTION")):
        return None
    if kw.isascii() and kw.upper() in ("TEXT", "CARD", "LEGEND", "PROJECTED") and sep not in ":-–—":
        return None
    if not kw.isascii() and sep not in (":", "：", ">", "]", ")", "-", "–", "—", "") and t.lstrip()[:1] not in "<[(":
        # '자막 1992년.' 처럼 공백 뒤 글은 받되, '자막이' 같은 토씨는 이미 정규식에서 빠짐
        if not body:
            return None
    for rx, kind in _ONSCREEN_KIND:
        if rx.search(kw):
            return kind, body
    return "other", body


# ---------------------------------------------------------------- 화자

RX_TAIL = re.compile(r"\s*[(（\[]([^()（）\[\]]{1,40})[)）\]]\s*$")
_CUE_CLEAN = re.compile(r"[\s.·'’\-_:;,]")


def cue_key(s: str) -> str:
    return _CUE_CLEAN.sub("", s).lower()


HOW_CUES: list[tuple[str, re.Pattern]] = [
    ("phone", re.compile(r"전화|핸드폰|휴대폰|핸폰|수화기|통화|phone|telephone|filtered|filter|^f$|^fil$|t[ée]l[ée]phone")),
    ("radio", re.compile(r"무전|워키|walkie|^radio$|overradio|intoradio|onradio|intercom|인터폰|^rt$|headset|헤드셋|인이어|earpiece")),
    ("broadcast", re.compile(r"방송|^tv$|ontv|티비|텔레비전|라디오|스피커|확성기|안내|뉴스|speaker|^pa$|overpa|onpa|broadcast|"
                             r"announcer|television|megaphone|bullhorn|loudspeaker|radiotv|ontelevision")),
    ("video", re.compile(r"화면|영상|비디오|모니터|녹화|녹음|테이프|video|tape|monitor|onscreen|recorded|recording|onfilm|"
                         r"onlaptop|computer|youtube|skype|facetime|webcam|playback|archive")),
    ("inner", re.compile(r"속으로|속마음|마음의소리|마음속|생각|독백|innervoice|thought|thinking|inhead|internal|tohimself|toherself")),
    ("narration", re.compile(r"^na$|^n$|^nar$|^narr$|나레이션|내레이션|나래이션|해설|narrat|^voixoff$|^vo\)?$|^v$")),
    ("off", re.compile(r"^e$|^os$|^oc$|^off$|^oov$|소리|목소리|offscreen|offcamera|^hc$|horschamp|^voix|voiceover|^ost?$|^오프$|^off\b")),
]
LANG_CUES: list[tuple[str, re.Pattern]] = [
    ("en", re.compile(r"영어|영문|^英|english|anglais|ingl[ée]s|englisch|^eng$")),
    ("ja", re.compile(r"^일어|일본어|일본말|^日|japanese|japonais")),
    ("zh", re.compile(r"중국어|중국말|북경어|광동어|^中|chinese|mandarin|cantonese|chinois")),
    ("ru", re.compile(r"러시아어|러시아말|노어|russian|russe")),
    ("de", re.compile(r"독일어|독어|german|allemand|deutsch")),
    ("fr", re.compile(r"불어|프랑스어|french|fran[çc]ais")),
    ("es", re.compile(r"스페인어|spanish|espagnol|espa[ñn]ol")),
    ("it", re.compile(r"이탈리아어|italian|italien")),
    ("ko", re.compile(r"한국어|한국말|조선말|korean|cor[ée]en")),
    ("vi", re.compile(r"베트남어|vietnamese")),
    ("ar", re.compile(r"아랍어|arabic|arabe")),
    ("pt", re.compile(r"포르투갈어|portuguese|portugais")),
    ("la", re.compile(r"라틴어|^latin$|inlatin")),
    ("he", re.compile(r"히브리어|hebrew")),
    ("mn", re.compile(r"몽골어|mongolian")),
    ("hi", re.compile(r"힌디어|hindi")),
    ("th", re.compile(r"태국어|^thai$|inthai")),
]
RX_CONTD = re.compile(r"^(?:cont['’]?d|cont|continued|continuing|계속|계속해서|이어서|suite|contd)$", re.I)
RX_SUNG = re.compile(r"노래|노랫|흥얼|콧노래|합창|랩|열창|부른다|부르며|가사|sing|sung|chant|humm|hums|croon|serenad|chorus|"
                     r"chantant|chante|lyrics|[♪♬♩♫]", re.I)
RX_OVERLAP = re.compile(r"동시에|동시|겹쳐|overlap|simultan|same time|en même temps|ensemble|together", re.I)
GROUP_NAMES = {"일동", "모두", "다같이", "다 같이", "다함께", "함께", "모두들", "전원", "사람들", "군중", "관객들", "학생들",
               "ALL", "EVERYONE", "EVERYBODY", "CROWD", "BOTH", "TOGETHER", "THE CROWD", "ALL OF THEM", "TOUS",
               "ENSEMBLE", "CHORUS", "OMNES"}


def split_speaker(raw: str) -> tuple[str, list[str]]:
    """'천세황(무전소리)' -> ('천세황', ['무전소리']). 꼬리 괄호를 모두 뗀다."""
    s = raw.strip().rstrip(":：").strip()
    m = re.fullmatch(r"<\s*([^<>]{1,20}?)\s*>(.*)", s)
    if m and not m.group(2).strip(" :："):
        s = m.group(1)                                          # '<우주>' 표기
    tails: list[str] = []
    for _ in range(4):
        m = RX_TAIL.search(s)
        if not m or m.start() == 0:
            break
        tails.insert(0, m.group(1).strip())
        s = s[: m.start()].rstrip()
    s = s.rstrip(":：").strip()
    # 괄호 없이(또는 반쪽 괄호로) 붙은 꼬리: 'NAME CONT'D', 'ALFIE O.S', 'GILPIN CONT’D)'
    for _ in range(3):
        m = re.search(r"\s*\(?\b(CONT['’]?D|CONT\.|V\.\s?O\.?|O\.\s?S\.?|O\.\s?C\.?)\)?$", s)
        if not m or m.start() == 0:
            break
        tails.append(m.group(1))
        s = s[: m.start()].strip()
    # 글자층이 겹쳐 나온 'BELLA (CONT’D)BELLA' -> 'BELLA'
    m = re.fullmatch(r"(.+?)\s*\(?CONT['’]?D\)?\s*\1", s)
    if m:
        tails.append("CONT'D")
        s = m.group(1).strip()
    s = re.sub(r"(?:\s+[.·…]+|·+)$", "", s).strip()           # '애자 .··', '미영·' (붙은 마침표 하나는 둔다)
    return s, tails


def how_of(cues: list[str]) -> str | None:
    """괄호 꼬리표들 -> how. 'V.O.' 는 'vo' 로 돌려 나중에 문서 안에서 정한다."""
    found: list[str] = []
    for c in cues:
        k = cue_key(c)
        if not k:
            continue
        if k in ("vo", "v.o", "voiceover", "vooff") or k.startswith("vo") and len(k) <= 4:
            found.append("vo")
            continue
        for how, rx in HOW_CUES:
            if rx.search(k):
                found.append(how)
                break
    if not found:
        return None
    for pref in ("phone", "radio", "broadcast", "video", "inner", "narration", "vo", "off"):
        if pref in found:
            return pref
    return found[0]


def lang_of(cues: list[str]) -> str | None:
    for c in cues:
        k = cue_key(c)
        if not k or len(k) > 24:
            continue
        for lg, rx in LANG_CUES:
            m = rx.search(k)
            if m and (m.end() - m.start()) * 2 >= len(re.sub(r"(소리|subtitled|speaking|in|en|parle|말로|로)", "", k)):
                return lg
    return None


def is_cont_cue(c: str) -> bool:
    return bool(RX_CONTD.match(c.strip().replace(" ", "")))


_NAME_STOP_LAT = {
    "CUT TO", "FADE IN", "FADE OUT", "THE END", "END", "CONTINUED", "MORE", "BLACK", "LATER", "MONTAGE",
    "INTERCUT", "FLASHBACK", "END FLASHBACK", "BACK TO SCENE", "ANGLE ON", "CLOSE ON", "CLOSE UP", "CLOSEUP", "POV",
    "INSERT", "SUPER", "TITLE", "SILENCE", "DARKNESS", "BEAT", "PAUSE", "OMITTED", "OMIT", "SCENE OMITTED", "THE NEXT DAY",
    "NIGHT", "DAY", "MORNING", "BACK TO PRESENT", "END MONTAGE", "END OF MONTAGE", "SERIES OF SHOTS", "TITLE CARD",
    "SUBTITLE", "SUBTITLES", "SUBTITLED", "WIDE SHOT", "TRACKING SHOT", "CREDITS", "END CREDITS", "OPENING CREDITS", "BLACK SCREEN",
    "FIN", "NOIR", "SUITE", "PLUS", "THE", "A", "AND", "OK", "OKAY", "YES", "NO", "WHAT", "HEY", "OH", "HUH", "UH",
    "SOUND", "MUSIC", "LIGHTNING", "BANG", "BOOM", "CRASH", "SFX", "VFX", "PRESENT DAY", "TO", "STOP", "HELP",
    "I", "ME", "MY", "WE", "US", "HE", "SHE", "IT", "YOU", "HI", "SO", "BUT", "OR", "IF", "AS", "AT", "ON", "IN", "OF",
    "THEN", "NOW", "MEANWHILE", "SUDDENLY", "AGAIN", "SLOW MOTION", "SILENT", "QUICK SHOTS", "SHOTS", "REACTION",
    "NOTE", "NOTES",
}
_NAME_STOP_KO = {
    "자막", "인서트", "몽타주", "몽타쥬", "회상", "플래시백", "플래쉬백", "암전", "디졸브", "오버랩", "끝", "시간경과", "시간 경과",
    "점프", "계속", "화면전환", "장면전환", "등장인물", "등장 인물", "인물", "타이틀", "메인타이틀", "컷", "네", "예", "응", "어",
    "아", "음", "흠", "자", "야", "왜", "뭐", "그래", "아니", "아니요", "아뇨", "글쎄", "정말", "진짜", "그럼", "근데", "그런데",
    "그리고", "하지만", "그래서", "여보세요", "제발", "잠깐", "누구", "어디", "언제", "엄마야", "으악", "헉", "휴", "에이", "이런",
    "참", "저기", "저", "나", "너", "우리", "그", "이", "또", "다시", "지금", "오늘", "내일", "어제", "밤", "낮", "아침", "저녁",
    "새벽", "현재", "과거", "플래시", "교차", "인터컷", "프롤로그", "에필로그", "엔딩", "오프닝", "THE END", "소리", "음악",
    "효과음", "정적", "침묵",
}
_RX_NAME_LAT = re.compile(r"^[A-ZÀ-ÞŒ0-9][A-ZÀ-ÞŒ0-9 .'’\-&#/,]*$")
_RX_KO_ENDING = re.compile(r"(?:습니다|니다|어요|아요|에요|예요|해요|세요|군요|네요|지요|죠|는데|다\.|다$|요$|까$|냐$|니$|자$)")


_RX_TITLE_WORD = re.compile(r"^(?:[A-ZÀ-Þ][a-zà-ÿ'’\-]+(?:-[A-Za-zà-ÿ]+)?|[A-ZÀ-Þ]{1,4}\.?|Mc[A-Z][a-z]+|Mac[A-Z][a-z]+|"
                            r"O'[A-Z][a-z]+|de|van|von|da|di|le|la|del|du|#?\d{1,2})$")
_TITLE_STOP = set("Yes No Okay Ok Oh Hey Hi Hello Thanks Thank You Sorry Please Wait Look Listen Well Good Great Right "
                  "What Why How Who Where When Really Sure Fine Nothing Never Always Maybe Stop Help Come Go Now Here There "
                  "Bye Goodbye Damn Shit Fuck God Jesus Silence Pause Beat Later Continued More Fade Cut End Black "
                  "Night Day Morning Evening Interior Exterior Insert Montage Flashback Title Subtitle Subtitles Super Then "
                  "And But So Or The A An It He She They We I Me My Your His Her Our".split())


def title_name_like(core: str) -> bool:
    """'Moss', 'Carla Jean', 'Sang-hyun' 처럼 첫 글자만 대문자인 이름 줄(이름 칸 x·빈도로 다시 거른다)."""
    s = core.strip()
    w = s.split()
    if not 1 <= len(w) <= 3 or len(s) > 30:
        return False
    if not all(_RX_TITLE_WORD.match(x) for x in w):
        return False
    if w[0] in _TITLE_STOP or s in _TITLE_STOP:
        return False
    return any(c.islower() for c in s)


def name_like(core: str, *, block: bool = False) -> bool:
    """화자 이름처럼 생겼나(꼬리 괄호를 뗀 이름). block=True 면 이름 줄(이름만 있는 줄) 기준으로 더 엄격하게."""
    s = core.strip()
    if not s or len(s) > 40:
        return False
    if s[0] in "(-–—*#<[{\"'“‘.…·,!?~" or re.search(r"[:：{}\[\]|\\!?;@=+]", s):
        return False
    if re.fullmatch(r"(?:[SC]|씬|신|컷)\s*[#＃]\s*\d{0,4}[A-Za-z]?", s):
        return False                                           # 'S#', 'C#1': 씬·컷 표지는 이름이 아니다
    if re.fullmatch(r"[A-Z]{1,2}\d{1,4}[A-Z]?|\d{1,4}[A-Z]{1,2}", s) or re.match(r"\d+\s*[.)]", s) \
            or re.search(r"(?:^|[^A-Za-z])(?:INT|EXT)(?:[^A-Za-z]|$)", s) or s.endswith(("-", "–", "/")):
        return False                                           # 'A116', '12A' 같은 씬 번호, 씬 머리 조각
    if re.search(r"[.?!…~,;\"“”'’]$", s) and not re.search(r"\b(?:JR|SR|DR|MR|MRS|MS|ST|LT|SGT|COL|CAPT|GEN|PROF)\.$", s, re.I):
        return False
    if re.search(r"[\"“”]|https?:|www\.", s):
        return False
    sc = script_counts(s)
    nlet = sc["ko"] + sc["lat"] + sc["kana"] + sc["han"] + sc["cyr"]
    if nlet == 0:
        return bool(re.fullmatch(r"\d{2,4}", s))          # '112(전화)' 같은 번호 화자
    if sc["ko"] >= sc["lat"]:
        if sc["ko"] > (8 if block else 12) or len(s.split()) > 3:
            return False
        if flat(s) in _NAME_STOP_KO:
            return False
        if block and _RX_KO_ENDING.search(s) and sc["ko"] >= 3:
            return False
        return True
    words = s.split()
    if len(words) > 5:
        return False
    up = sum(1 for c in s if c.isupper())
    low = sum(1 for c in s if c.islower())
    if block:
        if low > 0 and up < 2 * low:
            return False
        if not _RX_NAME_LAT.match(s.replace("Mc", "MC").replace("Mac", "MAC").replace("De", "DE")) and low > 1:
            return False
    if flat(s).upper() in _NAME_STOP_LAT:
        return False
    return True


def name_key(core: str) -> str:
    """같은 사람인지 볼 열쇠: 공백·문장부호 빼고 대문자."""
    return re.sub(r"[\s.·'’`\-_:]", "", core).upper()


def display_name(core: str) -> str:
    s = flat(core).rstrip(":：").strip()
    parts = s.split(" ")
    if len(parts) > 1 and all(len(p) == 1 and is_hangul(p) for p in parts):
        return "".join(parts)                                  # '만  식' -> '만식'
    return s


_MULTI_SEPS = tuple(re.compile(p) for p in (r"\s*/\s*", r"\s*&\s*", r"\s*,\s*", r"\s*·\s*", r"\s+AND\s+", r"\s+and\s+",
                                              r"\s*\+\s*", r"\s+와\s+|\s+과\s+"))


def split_multi(core: str) -> list[str]:
    """'JASPER/VIRGINIA', '임숭재/임사홍', 'A & B', 'A, B' -> 이름 목록(나눌 수 없으면 [core])."""
    if not any(c in core for c in "/&,·+") and " and " not in core.lower() and " 와 " not in core and " 과 " not in core:
        return [core]
    for sep in _MULTI_SEPS:
        parts = [p for p in sep.split(core) if p.strip()]
        if len(parts) >= 2 and all(name_like(p) for p in parts):
            return parts
    return [core]


# ---------------------------------------------------------------- 괄호 지시

def split_parens(s: str) -> list[tuple[str, str]]:
    """글을 ('text'|'paren', 조각) 목록으로. 짝 없는 '(' 는 글로 둔다. 안쪽 괄호도 하나로 묶는다."""
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in "(（":
            depth, j = 0, i
            while j < n:
                if s[j] in "(（":
                    depth += 1
                elif s[j] in ")）":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j < n:
                if buf:
                    out.append(("text", "".join(buf)))
                    buf = []
                out.append(("paren", s[i + 1: j]))
                i = j + 1
                continue
        buf.append(c)
        i += 1
    if buf:
        out.append(("text", "".join(buf)))
    return out


def assemble(pieces: list[tuple[str, str]]) -> tuple[str, list[dict]]:
    """조각들 -> (괄호 뺀 글, [{text, at}]). 공백은 하나로, at 은 결과 글 안 위치."""
    out: list[str] = []
    olen = 0
    parens: list[dict] = []
    last_space = True
    for kind, s in pieces:
        if kind == "paren":
            parens.append({"text": flat(s), "at": olen})
            continue
        for ch in s:
            if ch in " \t\n\r":
                if not last_space:
                    out.append(" ")
                    olen += 1
                    last_space = True
            else:
                out.append(ch)
                olen += 1
                last_space = False
    text = "".join(out)
    if text.endswith(" "):
        text = text[:-1]
    for p in parens:
        p["at"] = min(p["at"], len(text))
    # 괄호 앞 공백 정리: at 이 공백 바로 뒤를 가리키면 그대로 둔다(다음 낱말 시작 위치)
    return text, parens


# ---------------------------------------------------------------- 줄 잇기

_KO_CONT_START = re.compile(r"^(?:[다요죠네까지고며면서는은을를이가의에도로와과게만]|[가-힣][.?!,…~])")


def join_lines(parts: list[str]) -> str:
    """여러 줄 글을 하나로. 한글 낱말 중간에서 끊긴 줄('벗는'+'다.')은 공백 없이 잇는다."""
    out = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not out:
            out = p
            continue
        a, b = out[-1], p[0]
        if is_hangul(a) and is_hangul(b):
            first_tok = p.split(" ", 1)[0]
            glue = len(first_tok.rstrip(".?!,…~")) <= 1 and bool(_KO_CONT_START.match(p))
            out = out + ("" if glue else " ") + p
        elif a == "-" and len(out) >= 2 and out[-2].isalpha() and b.islower():
            out = out + p
        else:
            out = out + " " + p
    return out
