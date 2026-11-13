"""구조 파싱: docs/{doc_id}/lines.jsonl -> docs/{doc_id}/script.json (명세 4.4.2, 까다로운 경우 4.4.3).

하나의 코드로 한국식(S#·번호. 장소 / 낮, '이름 : 대사', '이름⇥대사', 이름 줄 아래 대사)과 할리우드식(INT./EXT.,
대문자 이름 줄, 들여쓴 대사, 괄호 줄)을 함께 다룬다. 문서마다
  1) 쪽 장식(머리말·꼬리말·쪽번호·CONTINUED·(MORE)·수정 표시 *)을 dropped 로 빼고 (parse_layout.find_decor)
  2) x 분포로 지문 여백과 들여쓰기 단계를 찾고 (parse_layout.analyze_x)
  3) 씬 머리 형식을 문서 안 빈도·번호 차례로 받아들이고, 첫 씬 머리 앞을 front 로
  4) 화자 이름표(인물 어휘)를 높은 확신 표기(탭·쌍점·이름 줄)에서 모은 뒤 낮은 확신 표기(공백 하나)에 쓴다
  5) 줄을 차례로 보며 요소를 만든다(지문 문단·대사 이어짐은 들여쓰기·y 간격·글 모양으로 묶음)
  6) 쪽 넘김으로 쪼개진 대사 합치기, V.O. 해석, 동시 대사 묶음, 인물표(보수적 별칭), 씬 목록, 품질 지표.
결정적이며(LLM 없음) 문서 하나에서 예외가 나면 그 문서는 모든 줄을 unknown 으로 내고 문제를 기록한다.
"""
from __future__ import annotations

import re
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict

from . import paths, util
from . import parse_layout as PL
from . import parse_text as T

STAGE = "parse"
VERSION = "0.2.3"
PARSER_NAME = "sink-parse"

TH_TYPED = 0.85
TH_DIALOGUE = 0.15
TH_SPEAKERLESS = 0.05
# 말 없는 대사(이름 줄만 잡히고 말은 지문으로 간 경우)와 앞 대사에서 이어받은 추정 화자 비율의 기준(구현 중 보완 2026-11-13).
# 말 없는 대사는 괄호 지시만 있는 대사 등 정상인 경우도 있어 10% 를 넘을 때만 올린다.
TH_EMPTY_DIALOGUE = 0.10
TH_INFERRED = 0.10
MIN_DIALOGUE_FOR_RATIO = 20   # 대사가 이보다 적으면(조각·짧은 문서) 위 두 비율로 판정하지 않음

_TERMINAL = tuple(".?!…\"'”’)>]~」』")
# '*김상두 : 대사' 처럼 이름 앞에 별표·점을 찍는 대본도 있다
_RX_COLON = re.compile(r"^[*＊•·]?\s*([^:\t]{1,30}?)\s*:\s*(.*)$")
# 쌍점 대신 쌍반점·빗금: '감찰반장; 대사'(킬리만자로), '왕/ 대사', '조내시(o.s)/ 대사'(음란서생). 뒤에 공백이 있어야 함
_RX_SEMI_SLASH = re.compile(r"^[*＊•·]?\s*([^:;/\t]{1,30}?)\s*(?:;|/(?!/))\s+(\S.*)$")
_RX_SPACE2 = re.compile(r"^(\S(?:[^\t]{0,18}?\S)?)\s{2,}(\S.*)$")
_RX_QUOTE = re.compile(r"^([^\s\"“”'‘’()]{1,12}(?:\s*\([^()]{1,20}\))?)\s*(\"|“|‘|'')(.+)$")
_RX_ANGLE = re.compile(r"^<\s*([^<>]{1,20}?)\s*>\s*[:：]?\s*(\S.*)$")
_RX_SPACE1 = re.compile(r"^(\S{1,15}?(?:\s*\([^()]{1,20}\))?)\s+(\S.*)$")
# 촬영 크기 표기('영일 cu', '주희 : bs', '영일, 주희 fs'): 이름 뒤가 이것으로 시작하면 대사가 아니라 화면 설명
_RX_SHOT = re.compile(r"^(?:e?cu|bcu|mcu|bs|ms|mls|fs|ws|ls|els|ks|pov|[1-9]s|full\s*shot)\s*(?:[,/]\s*\S.*)?$",
                      re.IGNORECASE)
# 엑셀 표 대본 행: 첫 칸이 씬 번호, 둘째 칸이 장소('1⇥웨딩거리⇥영일 cu')
_RX_XLS_HEAD = re.compile(r"^(\d{1,3}[A-Za-z]?)(?:\.0)?\t([^\t]{1,40})(?:\t.*)?$")
# 이름 앞 글머리표 하나('*장달수⇥대사', '●민수 : 대사'). 문서 안에서 되풀이될 때만 뗀다(_candidates).
# '[김반장] 대사'(날 보러 와요). 되풀이될 때만(_bracketed). 표지 낱말은 이름이 아니다.
_RX_BRACKET = re.compile(r"^\[\s*([^\[\]]{1,12}?)\s*\]\s*[:：]?\s*(\S.*)$")
_RX_BRACKET_META = re.compile(r"^\s*(?:페이지|쪽|제목|부제|자막|인서트|회상|플래[시쉬]\s?백|몽타[주쥬]|효과음?|음악|장면|씬|컷|"
                              r"S\s*#|C\s*#|F\s*\d+|\d+|page|title|insert|sfx|music|note)\s*$", re.I)
_RX_BULLET = re.compile(r"^([*＊※●○◆◇■□▶·\-])[ \t]*(?=\S)")
# 글자 사이를 띄운 이름('진 수 (웃으며) 대사', '연 희 아냐.', 'P⇥D⇥대사'). 문서 안 통계로 거른다(_letterspaced).
_RX_LSP2 = re.compile(r"^([가-힣])[ \t]([가-힣])(?=[ \t(]|$)[ \t]*(.*)$")
_RX_LSP3 = re.compile(r"^([가-힣])[ \t]([가-힣])[ \t]([가-힣])(?=[ \t(]|$)[ \t]*(.*)$")
_RX_LSP_LAT = re.compile(r"^([A-Z])[ \t]([A-Z])(?:[ \t]([A-Z]))?(?=[ \t(]|$)[ \t]*(.*)$")
_LSP_STOP = set("이 그 저 또 한 두 세 네 몇 첫 새 옛 온 각 매 전 후 안 밖 앞 뒤 위 왜 뭐 예 응 다 더 덜 꼭 잘 못 좀 막 곧 늘 "
                "볼 할 갈 올 될 줄 알 쓸 살 아 어 오 우 에 음 흠 허 하 호 히 후 와 야 자 으 흐 쾅 쿵 탁 툭 퍽 펑 휙 쿡 킥 똑 딱 "
                "쏴 촥 착 척 뚝 뻥 꽝 빵 삐 띠 띵 딩 둥 쨍 쩍 툭 쓱 슥 휘 훅 핑 팍 탕 텅 쿨 컥 켁 큭".split())
_RX_HOWPREFIX = re.compile(r"^(소리|목소리|E|NA|Na|N|V\.?O\.?|O\.?S\.?)\s*>\s*")
_SURNAMES = set("김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민진나지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용예봉경사부가복태목형피두감음빈동온호범좌팽승간상갈단견당화창")
_SPACE_STOP = set("갑자기 다시 그때 그 때 이때 이 때 순간 잠시 이윽고 한편 문득 결국 조용히 천천히 그리고 하지만 그러나 계속 모두 함께 "
                  "역시 이미 아직 또 또다시 다들 서로 멀리 가까이 그대로 마침 바로 곧 한참 점점 슬며시 살며시 가만히 여전히 오직 "
                  "어느 한 두 세 그 이 저 그런 이런 저런 무슨 어떤 아무 모든 다른 옆 앞 뒤 위 아래 안 밖 속 곳 때 날 밤 낮 "
                  "화면 카메라 음악 소리 다음 이어 이어서 동시에 한쪽 저쪽 이쪽 사이 이번 지금 오늘 내일 어제 거리 방 집 "
                  # 대사 첫 낱말로 흔한 대명사·말버릇('난 몰라', '내가 할게')
                  "난 나 내가 나는 나도 내 넌 너 너는 너도 니가 네가 니 제가 저는 전 저도 우린 우리 우리가 당신 "
                  "뭔가 뭐 뭘 그건 이건 저건 그게 이게 저게 그걸 이걸 근데 그냥 이제 정말 진짜 혹시 아마 그래 그럼 "
                  "그래서 왜 어떻게 언제 누가 누구 무슨 아니 아냐 글쎄 제발 잠깐 어서 빨리 여기 거기 저기".split())
_EN_ROLE = set("MAN WOMAN GIRL BOY KID CHILD DOCTOR NURSE OFFICER COP GUARD ANNOUNCER REPORTER ATTENDANT DRIVER VOICE "
               "SOLDIER AGENT WAITER WAITRESS BARTENDER CLERK PRIEST JUDGE LAWYER DETECTIVE SERGEANT CAPTAIN PILOT "
               "TEACHER STUDENT MOTHER FATHER MOM DAD SISTER BROTHER WIFE HUSBAND FRIEND STRANGER CUSTOMER MANAGER "
               "SECRETARY ASSISTANT OPERATOR DISPATCHER NARRATOR CROWD ALL EVERYONE BOTH OTHERS".split())
_VOICE_WORDS = {"소리", "목소리"}
_RX_SENT_END = re.compile(r"(?:[ㄴ는]다|있다|없다|했다|한다|이다|였다|았다|었다)\.?$")
_ROLE_WORDS = set("형사 반장 팀장 기자 교수 선생 사장 회장 의사 경찰 부장 과장 대리 실장 국장 소장 원장 검사 판사 변호사 박사 기사 대장 "
                  "중사 하사 상사 병장 소위 중위 대위 소령 중령 대령 장군 신부 수녀 스님 할매 할배 아저씨 아줌마 엄마 아빠 누나 언니 "
                  "오빠 형님 동생 남자 여자 소년 소녀 청년 노인 아이 주인 손님 직원 기장 선장 대위 순경 경장 경사 경위 경감 계장 "
                  "주임 이사 상무 전무 사원 비서 간호사 학생".split())


# ================================================================ 글 모양 점수

_KO_SPEECH_END = re.compile(r"(?:요|야|어|아|지|네|니|냐|까|래|게|걸|군|데|라|자|죠|오|마|해|봐|줘|거든|잖아|구나|세요|시오|"
                            r"니다|네요|군요|대|래요|다니까|다고|냐고|라고|잖아요)\s*[.?!~…]*\s*$")
_KO_ACTION_END = re.compile(r"(?:[가-힣]*[ㄴ는]다|한다|있다|없다|된다|진다|보인다|이다|했다|였다|았다|었다|온다|간다|본다|선다|낸다|"
                            r"든다|운다|친다|린다|킨다|켠다|뜬다|준다|는다|난다|단다|탄다|잔다|찬다|민다|쥔다|싼다|편다|한다)\s*\.?\s*$")
_KO_PRON = re.compile(r"(?:^|\s)(?:너|니가|네가|너는|너도|넌|당신|난|내가|나는|나도|저는|제가|전|우리|저희|여보|이봐|야,|자,|아니|"
                      r"그래|응|네,|예,|어머|아이고|에이|임마|인마|형님|선배님|오빠|언니|엄마|아빠)(?:\s|$|[,.!?])")


def ko_style(s: str) -> float:
    """한국어 글이 대사 같으면 +, 지문 같으면 -."""
    s = s.strip()
    if not s:
        return 0.0
    body = re.sub(r"\([^()]*\)", " ", s).strip() or s
    sc = 0.0
    if re.search(r"[?!]", body):
        sc += 1.5
    if re.search(r"[~…]|\.\.\.?$|\.\.\.", body):
        sc += 0.7
    if _KO_SPEECH_END.search(body):
        sc += 1.0
    if _KO_PRON.search(body):
        sc += 1.0
    if _KO_ACTION_END.search(body) and not re.search(r"[?!~]", body):
        sc -= 1.5
    if re.search(r"[가-힣]{2,}\.$", body) and not _KO_SPEECH_END.search(body) and not _KO_ACTION_END.search(body):
        sc -= 0.5                               # '…녹음기.' 같은 이름씨로 끝나는 지문
    if re.search(r"\([^()]{2,}\)\s*\S", s) and not s.startswith("("):
        sc += 0.8                               # 말 사이에 든 연기 지시
    return sc


def wrapped(prev: str, width: int) -> bool:
    """앞 줄이 문장 중간에서 끊겼나(끝 문장부호 없음 또는 칸을 거의 다 채움)."""
    p = prev.rstrip()
    if not p:
        return False
    if not p.endswith(_TERMINAL) and not p.endswith(","):
        return True
    return width > 0 and len(p) >= 0.92 * width and not p.endswith((".", "?", "!", "…"))


# ================================================================ 한 문서 파싱

class DocParser:
    def __init__(self, doc_id: str, rows: list[dict]):
        self.doc_id = doc_id
        self.rows = rows
        self.notes: list[str] = []

    # ------------------------------------------------------------ 바깥
    def parse(self) -> dict:
        L = self.L = PL.make_lines(self.rows)
        self.drop: dict[int, str] = PL.find_decor(L)
        n_sub = sum(1 for w in self.drop.values() if w == "subtitle_cue")
        if n_sub:
            self.notes.append(f"자막 파일 꼴(SRT·WebVTT): 화살표 시각 줄과 자막 번호 {n_sub}줄을 dropped(timecode·subtitle_cue)")
        active = [ln.i for ln in L if ln.i not in self.drop]
        self.lay = PL.analyze_x(L, active)
        self.lang, self.lang_lat = T.doc_lang([L[i].f for i in active])
        self.tol2 = self.lay.get("tol2", 0.012)
        self.centered = bool(self.lay.get("centered"))
        lens = sorted(len(L[i].f) for i in active) or [0]
        self.width = lens[int(0.9 * (len(lens) - 1))]
        gaps = [ln.dy for ln in L if ln.dy is not None and ln.i not in self.drop]
        self.para_spacing = bool(gaps) and sum(1 for g in gaps if g >= 1.5) >= 0.08 * len(gaps)
        self._candidates(active)
        self._accept_headings(active)
        active = [i for i in active if i not in self.drop]
        self.active = active
        self.pos = {i: k for k, i in enumerate(active)}
        self.fe = 0
        self._vocab()
        self._front_end()
        els = self._label()
        els = self._merge_split_dialogue(els)
        return self._build(els)

    # ------------------------------------------------------------ 줄 후보
    def _candidates(self, active: list[int]) -> None:
        bullet: dict[str, list[tuple[int, tuple]]] = defaultdict(list)
        bracket: list[tuple[int, str, str]] = []
        stopped: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        we_seen = 0
        for i in active:
            ln = self.L[i]
            f = ln.f
            ex = ln.extra
            h = T.match_heading(f)
            if not h and "\t" in ln.t:
                # '10⇥SHOT - PODIUM of AMPHITHEATRE': 탭이 번호 뒤 구분자인 촬영 번호 줄(영어 대문자 장면 설명)
                h2 = T.match_heading(ln.t.strip())
                if h2 and h2[0] == "numdot" and _shot_slug(h2[2]):
                    h = h2
            if ln.src == "xls":
                m = _RX_XLS_HEAD.match(ln.t)
                h = ("xls_row", T.norm_scene_number(m.group(1)), m.group(2).strip()) if m else None
            if h:
                ex["head"] = h
            if T.match_transition(f):
                ex["trans"] = True
            o = T.match_onscreen(f)
            if o:
                ex["onscr"] = o
            sp = self._inline_split(ln.t)
            mb = _RX_BRACKET.match(ln.t.strip())
            if mb and not _RX_BRACKET_META.match(mb.group(1)) and not re.fullmatch(r"[\d\s.\-]+", mb.group(2)):
                bracket.append((i, mb.group(1).strip(), mb.group(2).strip()))
            if sp and sp[2] in ("tab", "colon") and re.fullmatch(r"우리\d{1,2}", T.name_key(T.split_speaker(sp[0])[0])):
                we_seen += 1                                   # '우리1 : …' 은 이미 이름으로 읽힌다(세기만)
            if not sp:
                mc = _RX_COLON.match(ln.t.strip())
                if mc and mc.group(2).strip():
                    k = T.name_key(T.split_speaker(mc.group(1))[0])
                    grp = "voice" if k in _VOICE_WORDS else ("we" if re.fullmatch(r"우리\d{0,2}", k) else None)
                    if grp:
                        stopped[grp].append((i, mc.group(1).strip(), mc.group(2).strip()))
            if sp:
                ex["inl"] = sp
            else:
                m = _RX_BULLET.match(ln.t)
                if m:
                    sp = self._inline_split(ln.t[m.end():])
                    if sp and sp[2] in ("tab", "colon"):
                        bullet[m.group(1)].append((i, sp))
            ex["garbage"] = self._garbage(ln)
        # 글머리표 붙은 이름('*장달수⇥대사', '*함상욱(E)⇥…'): 같은 표가 붙은 '이름⇥대사' 꼴이 문서에서 되풀이될 때만
        # (8줄 이상, 두 번 이상 나온 이름 3개 이상) 표를 떼고 이름으로 본다. 인물표·메모의 글머리표는 되풀이가 적다.
        for mk, items in bullet.items():
            names = Counter(T.name_key(T.split_speaker(sp[0])[0]) for _, sp in items)
            if len(items) >= 8 and sum(1 for c in names.values() if c >= 2) >= 3:
                for i, sp in items:
                    self.L[i].extra["inl"] = sp
                self.notes.append(f"대사 표기: 글머리표 '{mk}' 붙은 이름 {len(items)}줄(표를 떼고 이름으로 봄)")
        self._bracketed(bracket)
        # 막아 둔 낱말이 쌍점 화자로 되풀이되는 문서(꽃잎): '소리 : …'·'목소리 : …' 는 누군지 적지 않은 화면 밖 목소리
        # (화자 없음, how=off), '우리 : …'·'우리4:' 는 무리 화자. 3번 이상일 때만. 소리 묘사('소리 : 천둥소리')는 뺀다.
        for grp, items in stopped.items():
            if len(items) + (we_seen if grp == "we" else 0) < 3:
                continue
            n = 0
            for i, left, rest in items:
                if grp == "voice" and (len(rest) < 4 or re.search(r"(?:소리|음|음향|효과음)\s*[.…]*$", rest)):
                    continue
                self.L[i].extra["inl"] = (left, rest, "colon")
                n += 1
            if n:
                self.notes.append(f"대사 표기: '{'소리/목소리' if grp == 'voice' else '우리'} : 대사' {n}줄"
                                  + ("(화자 없음, off)" if grp == "voice" else "(무리 화자)"))
        self._letterspaced(active)

    def _bracketed(self, cands: list[tuple[int, str, str]]) -> None:
        """'[김반장] 대사'(날 보러 와요): 대괄호 이름이 3번 이상 되풀이되면 이름. 그런 줄이 20개 이상이면 덜 나온 이름도
        이름처럼 생겼으면 받는다. '[페이지] 001', '[제목]' 같은 표지는 뺀다."""
        self.bracket_mode = False
        cnt = Counter(T.name_key(T.split_speaker(nm)[0]) for _, nm, _ in cands)
        def good(nm: str) -> bool:
            core = T.split_speaker(nm)[0]
            return self._speaker_ok(nm, "colon") and not T.how_of([core]) and not T.lang_of([core]) \
                and not T.match_transition(core) and not T.match_onscreen(core) and len(core) > 1

        ok = [(i, nm, rest) for i, nm, rest in cands if cnt[T.name_key(T.split_speaker(nm)[0])] >= 3 and good(nm)]
        if len(ok) < 20:
            return                                      # '[자료화면]', '[SLOW-MOTION]' 같은 몇 줄짜리 표지는 대사 표기가 아님
        ok += [(i, nm, rest) for i, nm, rest in cands if cnt[T.name_key(T.split_speaker(nm)[0])] < 3 and good(nm)]
        self.bracket_mode = True
        for i, nm, rest in ok:
            self.L[i].extra["inl"] = (nm, rest, "bracket")
        self.notes.append(f"대사 표기: 대괄호 이름 [이름] {len(ok)}줄")

    def _letterspaced(self, active: list[int]) -> None:
        """글자 사이를 띄운 이름('진 수 대사', '경 우⇥대사', 'P⇥D⇥대사'): 고정폭 이름 칸 때문에 두 글자 이름을 한 칸씩
        띄운 대본. 붙인 이름이 줄 첫머리에 3번 이상 나오고, 붙인 꼴이 문서 다른 곳에 낱말로(토씨 붙어도) 2번 이상
        나오고, 첫 글자가 지시어·부사·소리흉내가 아닐 때만 이름으로 본다. 세 글자는 두 글자로 안 될 때만."""
        if self.lang != "ko":
            return
        L = self.L
        c2: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        c3: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        lat: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        for i in active:
            t = L[i].t
            if "head" in L[i].extra:
                continue
            m = _RX_LSP3.match(t)
            if m and m.group(4).strip() and m.group(1) not in _LSP_STOP and len({m.group(1), m.group(2)}) == 2:
                c3[m.group(1) + m.group(2) + m.group(3)].append((i, " ".join(m.group(1, 2, 3)), m.group(4).strip()))
            m = _RX_LSP2.match(t)
            if m and m.group(3).strip() and m.group(1) not in _LSP_STOP and m.group(1) != m.group(2):
                c2[m.group(1) + m.group(2)].append((i, m.group(1) + " " + m.group(2), m.group(3).strip()))
            m = _RX_LSP_LAT.match(t)
            if m and m.group(4).strip() and T.is_hangul(next((c for c in m.group(4) if c.isalpha()), "a")):
                lat["".join(g for g in m.group(1, 2, 3) if g)].append((i, "".join(g for g in m.group(1, 2, 3) if g),
                                                                      m.group(4).strip()))
        if not c2 and not c3 and not lat:
            return
        text = "\n".join(L[i].f for i in active)

        def word_count(k: str) -> int:
            return len(re.findall(r"(?<![가-힣])" + re.escape(k) + r"(?![가-힣])|(?<![가-힣])" + re.escape(k)
                                  + r"(?=[이가은는을를의도와과에랑야아씨님한로만께])", text))

        ok2 = {k for k, v in c2.items() if len(v) >= 3 and word_count(k) >= 2}
        ok3 = {k for k, v in c3.items() if len(v) >= 3 and k[:2] not in ok2 and word_count(k) >= 2}
        acc: dict[int, tuple] = {}
        for k in ok3:
            for i, raw, rest in c3[k]:
                acc[i] = (raw, rest, "lsp")
        for k in ok2:
            for i, raw, rest in c2[k]:
                acc.setdefault(i, (raw, rest, "lsp"))
        if ok2 or ok3:
            for k, v in lat.items():
                if len(v) >= 3:
                    for i, raw, rest in v:
                        acc.setdefault(i, (raw, rest, "lsp"))
        for i, sp in acc.items():
            L[i].extra["inl"] = sp
        if acc:
            self.notes.append(f"대사 표기: 글자 사이를 띄운 이름 {len(acc)}줄({', '.join(sorted(ok2 | ok3))[:200]})")

    def _garbage(self, ln) -> bool:
        f = ln.f
        if ln.conf is not None and ln.conf < 0.5:
            return True
        nlet = T.letters(f)
        if f.count("�") + sum(1 for c in f if 0xE000 <= ord(c) <= 0xF8FF) >= max(1, 0.3 * len(f)):
            return True
        if nlet == 0:
            return not re.fullmatch(r"[.?!…~\-–—*·,'\"“”‘’()\s]+", f) and not f.strip().isdigit()
        if self.__dict__.get("lang") == "ko":
            lat1 = sum(1 for c in f if 0xC0 <= ord(c) <= 0xFF)
            if lat1 >= 0.4 * nlet and nlet >= 4:
                return True
        return False

    def _speaker_ok(self, left: str, sep: str) -> bool:
        core, tails = T.split_speaker(left)
        if not core:
            return False
        parts = T.split_multi(core)
        if not all(T.name_like(p) for p in parts):
            return False
        if any(c.isascii() and c.isalpha() for c in core) and not any(T.is_hangul(c) for c in core):
            words = core.split()
            if len(words) > 3:
                return False
            if sep in ("colon", "space2", "tab") and not all(w[:1].isupper() or not w[:1].isalpha() for w in words):
                return False
            if sep in ("space", "quote") and (core.upper() != core or T.letters(core) < 2):
                return False
        return True

    def _inline_split(self, t: str):
        sp = self._inline_split_raw(t)
        if sp and _RX_SHOT.match(sp[1]):
            return None          # '영일 cu': 이름 뒤가 촬영 크기 표기면 대사가 아니라 화면 설명
        return sp

    def _inline_split_raw(self, t: str):
        s = t.strip()
        m = _RX_ANGLE.match(s)
        if m and self._speaker_ok(m.group(1), "colon"):
            return m.group(1).strip(), m.group(2).strip(), "angle"
        if "\t" in s:
            left, right = s.split("\t", 1)
            left = left.strip().rstrip(":").strip()
            right = right.strip().lstrip(":\t ").strip()
            m = re.fullmatch(r"\(([^()]{1,10})\)", left)
            if m and right and T.how_of([m.group(1)]):
                return left, right, "cue"              # '(소리) ⇥ 정수씨, …' 이름 없이 들리는 방식만 적힌 대사
            if re.fullmatch(r"\d{1,4}[A-Z]?", left) and _shot_slug(right):
                return None                            # '10⇥SHOT - PODIUM': 촬영 번호 줄은 대사가 아니다
            if left and right and len(left) <= 30 and self._speaker_ok(left, "tab"):
                return left, right, "tab"
        m = _RX_COLON.match(s)
        if m:
            left, right = m.group(1).strip(), m.group(2).strip()
            if left and not (left[-1:].isdigit() and right[:1].isdigit()) and self._speaker_ok(left, "colon"):
                return left, right, "colon"
        m = _RX_SEMI_SLASH.match(s)
        if m:
            left, right = m.group(1).strip(), m.group(2).strip()
            if left and not left[-1:].isdigit() and self._speaker_ok(left, "colon"):
                return left, right, "colon"
        s1 = s.replace("\t", "  ")
        m = _RX_SPACE2.match(s1)
        if m and self._speaker_ok(m.group(1), "space2"):
            return m.group(1).strip(), m.group(2).strip(), "space2"
        m = _RX_QUOTE.match(s1)
        if m and self._speaker_ok(m.group(1), "quote"):
            return m.group(1).strip(), (m.group(2) + m.group(3)).strip(), "quote"
        m = _RX_SPACE1.match(s1)
        if m and self._speaker_ok(m.group(1), "space"):
            return m.group(1).strip(), m.group(2).strip(), "space"
        return None

    # ------------------------------------------------------------ 씬 머리
    def _accept_headings(self, active: list[int]) -> None:
        L = self.L
        cands = [(i, L[i].extra["head"]) for i in active if "head" in L[i].extra]
        # '4. 지웅 모 가게/골목길. 오전.' 처럼 시간 낱말로 끝나는 번호 씬 머리는 하나만 있어도 받는다
        for k, (i, h) in enumerate(cands):
            if h[0] == "numdot" and self._numdot_shape(L[i], h) and len(h[2]) <= 50 \
                    and T.split_place_time(h[2])[1] \
                    and (L[i].cls in ("M", "L") or self.centered and L[i].ind <= 0.06):
                h2 = ("numdot_t",) + tuple(h[1:])
                L[i].extra["head"] = h2
                cands[k] = (i, h2)
        styles = Counter(h[0] for _, h in cands)
        strong = {"intext", "sharp", "sceneword", "p", "numdot_ie", "numboth", "numdot_t", "xls_row"}
        ok_style = {s for s in styles if s in strong}

        def lead_int(num):
            m = re.search(r"\d+", num or "")
            return int(m.group()) if m else None

        nd = [(i, lead_int(h[1])) for i, h in cands if h[0] in ("numdot", "numdot_t")]
        nd = [(i, v) for i, v in nd if v is not None and self._numdot_shape(L[i], L[i].extra["head"])]
        if len(nd) >= 3 and len({v for _, v in nd}) >= 3:    # '202. 찬석에게 다가온다.'(인물 202) 같은 같은 번호 되풀이는 아님
            inc = sum(1 for (a, va), (b, vb) in zip(nd, nd[1:]) if 0 <= vb - va <= 10)
            if inc >= 0.5 * (len(nd) - 1):
                ok_style.add("numdot")
        n_main = sum(styles[s] for s in ok_style)
        if "cut" in styles and n_main < 3:
            ok_style.add("cut")
        # INT./EXT. 도 번호도 없는 문서: 여백에 놓인 대문자 짧은 줄('MOSS'S TRAILER')을 씬 머리로
        if n_main < 5 and self.lang not in ("ko", "ja", "zh"):
            slug = []
            for k, i in enumerate(active):
                ln = L[i]
                f = ln.f
                if ln.cls not in ("M", "L") or "head" in ln.extra or "trans" in ln.extra or "onscr" in ln.extra:
                    continue
                if not (3 <= len(f) <= 60) or T.letters(f) < 3 or f.upper() != f or f.endswith((":", ".", "!", "?", ",")):
                    continue
                if f.startswith(("(", "\"", "“", "'", "-")) or re.search(r"\d{3,}", f):
                    continue
                nj = L[active[k + 1]] if k + 1 < len(active) else None
                if nj is None or nj.cls not in ("M", "L") or nj.f.upper() == nj.f:
                    continue
                slug.append(i)
            # 인물 이름 줄(들여쓰기 없는 문서)도 같은 모양이다: 자주 되풀이되면 장소가 아니라 이름
            rep = Counter(T.name_key(L[i].f) for i in slug)
            if slug and len(rep) < 0.6 * len(slug):
                slug = []
            slug = [i for i in slug if rep[T.name_key(L[i].f)] <= 3]
            if len(slug) >= 8:
                for i in slug:
                    L[i].extra["head"] = ("slug", None, L[i].f)
                cands = [(i, L[i].extra["head"]) for i in active if "head" in L[i].extra]
                ok_style.add("slug")
                self.notes.append(f"씬 머리: INT./EXT.·번호가 없어 여백의 대문자 짧은 줄 {len(slug)}개를 씬 머리로 봄")
        if styles.get("ko_nonum"):
            ok_style.add("ko_nonum")
        if styles.get("sharp_nonum", 0) >= 5:
            ok_style.add("sharp_nonum")
        if styles.get("en_noint"):
            ok_style.add("en_noint")
        # 번호. 씬 머리는 문서 안에서 같은 x 에 놓인다: 후보 x 의 봉우리에서 먼 줄은 빼낸다
        nd_x = None
        if "numdot" in ok_style and nd and Counter(L[i].src for i in active).most_common(1)[0][0] != "ocr":
            cl = PL.clusters_x([L[i].rx for i, _ in nd], max(self.tol2, 0.01))
            cx, cn = max(cl, key=lambda c: c[1])
            if cn >= 0.7 * len(nd):
                nd_x = cx
        acc: list[int] = []
        for i, h in cands:
            st = h[0]
            ln = L[i]
            if st not in ok_style:
                continue
            if st == "numdot" and not self._numdot_shape(ln, h):
                continue
            if st == "numdot" and nd_x is not None and abs(ln.rx - nd_x) > 2 * self.tol2:
                continue
            if st in ("ko_nonum", "en_noint") and (ln.cls not in ("L", "M") or len(ln.f) > 45 or "inl" in ln.extra
                                                    and ln.extra["inl"][2] in ("tab", "colon", "lsp")):
                continue
            if st in ("sharp", "sharp_nonum", "cut", "sceneword", "p") and len(ln.f) > 100:
                continue
            if st == "sharp" and styles[st] < 3 and len(h[2]) > 12 and _RX_SENT_END.search(h[2]):
                continue                               # 지문 속 씬 참조('S#84 회상에서 틀린 부분을 바로잡는 장면들이다.')
            if ln.cls == "I" and ln.ind > 0.12 and st not in ("intext", "numdot_ie"):
                continue
            acc.append(i)
        # 가운데 정렬·들여쓴 번호 씬 머리 묶음(공동경비구역 JSA, 와이키키 브라더스): 들여쓰기 검사로 대부분 빠졌어도
        # 문서 전체에서 번호가 차례로 늘어나는 한 줄 묶음이면 받는다. 차례에 맞지 않는 같은 형식 줄('S#14의…')은 뺀다.
        series, sstyles = self._series_heads(cands, lead_int)
        if series and sum(1 for i in acc if L[i].extra["head"][0] in sstyles) < 0.5 * len(series):
            keep = [i for i in acc if L[i].extra["head"][0] not in sstyles or i in series]
            acc = sorted(set(keep) | series, key=lambda i: self.L[i].n)
            self.notes.append(f"씬 머리: 들여쓰거나 가운데 놓인 번호 씬 머리 묶음 {len(series)}개를 받음")
        # 번호가 앞쪽에서 1 로 다시 시작하고 그 앞 번호들 사이가 아주 짧으면(줄 몇 개) 인물표·설정 번호(4.4.3 #10)
        nums = [(i, lead_int(L[i].extra["head"][1])) for i in acc
                if L[i].extra["head"][1] and re.fullmatch(r"\d+", L[i].extra["head"][1])]
        nums = [(i, v) for i, v in nums if v is not None]
        Na = max(1, len(active))
        pos = {i: k for k, i in enumerate(active)}
        cut_at = None
        hi = -1
        for k, (i, v) in enumerate(nums):
            if pos[i] > 0.3 * Na:
                break
            after = [w for _, w in nums[k + 1: k + 4]]
            if k and v <= 1 and hi >= 3 and len(after) >= 2 and after == sorted(after) and after[-1] <= v + 8:
                prev_pos = [pos[j] for j, _ in nums[:k]]
                gaps = [b - a for a, b in zip(prev_pos, prev_pos[1:])]
                if gaps and statistics.median(gaps) <= 4:
                    cut_at = i
            hi = max(hi, v)
        if cut_at is not None:
            before = {i for i, v in nums if pos[i] < pos[cut_at]}
            acc = [i for i in acc if i not in before]
            self.notes.append(f"씬 번호가 {L[cut_at].n}번 줄에서 1로 다시 시작: 앞 번호 {len(before)}개는 앞머리(인물표·설정)로 봄")
        self.heads = set(acc)
        self.head_styles = Counter(L[i].extra["head"][0] for i in acc)
        # 맨 번호 줄(여백 씬 번호): 옆 씬 머리와 같은 번호면 dropped, 씬 머리에 번호가 없으면 씬에 붙인다
        self.head_attach: dict[int, list[int]] = defaultdict(list)
        self.head_num: dict[int, str] = {}
        act = [i for i in active if i not in self.drop]
        for k, i in enumerate(act):
            ln = L[i]
            if i in self.heads or not T.RX_BARE_NUM.match(ln.f):
                continue
            num = T.norm_scene_number(ln.f)
            for j in (act[k + 1] if k + 1 < len(act) else None, act[k - 1] if k else None):
                if j is None or j not in self.heads:
                    continue
                hn = L[j].extra["head"][1]
                if hn == num:
                    self.drop[i] = "scene_number_margin"
                elif hn is None and j == (act[k + 1] if k + 1 < len(act) else None):
                    self.head_attach[j].append(i)
                    self.head_num[j] = num
                break
        # 두 줄로 나뉜 씬 머리: 앞 줄이 '–' 로 끝나거나 뒷 줄이 장소 없는 'EXT+INT/JOUR' 조각이면 한 씬 머리로
        for k, i in enumerate(act[:-1]):
            j = act[k + 1]
            if i not in self.heads or j not in self.heads or L[i].pk != L[j].pk and L[i].file != L[j].file:
                continue
            hj = L[j].extra["head"]
            pj, _ = T.split_place_time(hj[2]) if hj[2] else (None, None)
            dangling = re.search(r"[-–—,/]$", L[i].f)
            if dangling or (hj[1] is None and (not pj or len(pj) <= 3)):
                self.heads.discard(j)
                self.head_attach[i].append(j)
        # 번호만 있는 씬 머리('씬6', '#12') 다음 줄이 장소면 함께 묶는다
        for k, i in enumerate(act):
            if i not in self.heads or k + 1 >= len(act):
                continue
            h = L[i].extra["head"]
            if h[2]:
                continue
            j = act[k + 1]
            nj = L[j]
            if j in self.heads or len(nj.f) > 45 or nj.extra.get("inl") or nj.cls == "I" or "trans" in nj.extra:
                continue
            if re.search(r"[.?!]$", nj.f) and not re.search(r"(?:낮|밤|DAY|NIGHT)\s*[).]?$", nj.f):
                continue
            self.head_attach[i].append(j)

    def _series_heads(self, cands, lead_int) -> tuple[set[int], set[str]]:
        """같은 형식(번호. / S# 번호.)의 씬 머리 후보가 10개 이상이고 이웃 번호 차이가 80% 이상 0~3 이면 그 묶음.
        한 줄 50자 이하, 문장으로 끝나지 않는 후보만 세고, 앞뒤 어느 한쪽 번호와 차례가 맞는 줄만 고른다."""
        L = self.L
        out: set[int] = set()
        styles: set[str] = set()
        for members in (("numdot", "numdot_t"), ("sharp",)):
            cs = []
            for i, h in cands:
                if h[0] not in members or not self._series_shape(L[i], h):
                    continue
                v = lead_int(h[1])
                if v is not None:
                    cs.append((i, v))
            if len(cs) < 10:
                continue
            steps = [b - a for (_, a), (_, b) in zip(cs, cs[1:])]
            if sum(1 for d in steps if 0 <= d <= 3) < 0.8 * len(steps):
                continue
            for k, (i, v) in enumerate(cs):
                if (k and 0 <= v - cs[k - 1][1] <= 3) or (k + 1 < len(cs) and 0 <= cs[k + 1][1] - v <= 3):
                    out.add(i)
            styles.update(members)
        return out, styles

    @staticmethod
    def _series_shape(ln, h) -> bool:
        rest = h[2]
        if not rest or len(ln.f) > 50 or not (rest[0].isalpha() or rest[0] in "<[(#＃"
                                              or re.match(r"\d+[가-힣A-Za-z]", rest)):
            return False
        if re.match(r"^[가-힣A-Za-z0-9 ]{1,12}\s*:\s*\S", rest) or _RX_SENT_END.search(rest):
            return False
        return not (re.search(r"[.?!…~]$", rest) and not re.search(r"(?:낮|밤|DAY|NIGHT)\s*[).]?$", rest))

    def _numdot_shape(self, ln, h) -> bool:
        rest = h[2]
        if not rest or len(rest) > 80:
            return False
        if not (rest[0].isalpha() or rest[0] in "<[#＃") or re.match(r"^[가-힣A-Za-z0-9 ]{1,12}\s*:\s*\S", rest):
            return False
        if re.search(r"(?:[ㄴ는]다|있다|없다|했다|한다|이다)\.?$", rest) and len(rest) > 25:
            return False
        if ln.cls == "I" and ln.ind > 0.1:
            return False
        v = re.search(r"\d+", h[1] or "")
        return bool(v) and int(v.group()) <= 999

    # ------------------------------------------------------------ 앞머리
    def _front_end(self) -> None:
        """앞머리(front) 끝 위치. 기본은 첫 씬 머리. 그 앞에 진짜 대사(같은 이름이 되풀이되는 주고받기)가 있으면
        대사가 시작되는 쪽(또는 덩어리)부터 본문으로 본다. 인물표('수현: 29세…')는 이름이 한 번씩만 나와 걸러진다."""
        act, L = self.active, self.L
        Na = len(act)
        heads = sorted(self.pos[i] for i in self.heads if i in self.pos)
        h0 = heads[0] if heads else None
        dpos = [k for k, i in enumerate(act) if i in self.blk_ok or i in self.spkless or self._inline_ok(i)]

        def is_talk(ks: list[int]) -> bool:
            names = []
            for k in ks:
                i = act[k]
                if i in self.spkless:
                    names.append("?")
                elif i in self.blk_ok:
                    sh = self.blk_ok[i]
                    names.append(T.name_key(sh[1] if sh[0] != "dual" else "/".join(sh[1])))
                else:
                    names.append(T.name_key(T.split_speaker(L[i].extra["inl"][0])[0]))
            return len(ks) >= 3 and len(set(names)) <= 0.7 * len(names)

        def near_h0(k: int) -> bool:
            a, b = L[act[k]], L[act[h0]]
            if a.page is None or b.page is None:
                return h0 - k <= 60
            return a.file == b.file and b.page - a.page <= 1

        if h0 is not None and h0 <= max(0.3 * Na, 60):
            # 표지 쪽의 '이름 같은 줄'(BEST ORIGINAL SCREENPLAY / Written by)은 대사가 아니다: 첫 씬 머리와 같은 쪽이나
            # 바로 앞 쪽의 대사만 본다
            before = [k for k in dpos if k < h0 and near_h0(k)]
            if not is_talk(before[:12]):
                k = h0 - 1                       # 첫 씬 머리 바로 앞의 FADE IN 같은 전환은 앞머리가 아니다
                while k >= 0 and k >= h0 - 2 and "trans" in L[act[k]].extra:
                    k -= 1
                self.fe = k + 1
                return
            d0 = before[0]
        else:
            d0 = dpos[0] if dpos and dpos[0] <= 0.15 * Na else None
            if d0 is None:
                for k in range(int(0.15 * Na)):
                    if "trans" in L[act[k]].extra:
                        self.fe = k
                        return
                self.fe = 0
                return
        self.fe = self._content_start(d0)
        self.notes.append(f"앞머리: 첫 씬 머리 전에 대사가 있어 {L[act[self.fe]].n if self.fe < Na else '-'}번 줄부터 본문으로 봄"
                          if self.fe else "앞머리: 없음(첫 줄부터 본문)")

    def _content_start(self, d0: int) -> int:
        act, L = self.active, self.L
        for k in range(d0, -1, -1):
            if "trans" in L[act[k]].extra and k < d0:
                return k
        p0 = L[act[0]].pk
        pd = L[act[d0]].pk
        if pd[1] != -1:
            if pd != p0:
                # 대사가 나온 쪽보다 앞선 쪽 중 표지(첫 쪽)만 앞머리로
                for k in range(d0 + 1):
                    if L[act[k]].pk != p0:
                        return k
            # 같은 쪽: 맨 위 가운데 정렬된 짧은 줄(제목)만 앞머리
            k = 0
            while k < d0 and L[act[k]].cls in ("I", "R") and len(L[act[k]].f) <= 60 and act[k] not in self.blk_ok:
                k += 1
            return k
        # 쪽 없는 글: 첫 대사 앞 큰 빈칸(3줄 이상)까지 거슬러 올라가 거기서부터
        k = d0
        while k > 0 and d0 - k < 40:
            dy = L[act[k]].dy
            if dy is not None and dy >= 3:
                return k
            k -= 1
        return 0 if d0 < 40 else d0

    # ------------------------------------------------------------ 인물 어휘
    def _vocab(self) -> None:
        L, act = self.L, self.active
        body = [i for i in act if i not in self.heads]
        inl_cnt: Counter = Counter()
        sep_cnt: Counter = Counter()
        for i in body:
            ex = L[i].extra
            if "head" in ex and i in self.heads or "inl" not in ex:
                continue
            spk, rest, sep = ex["inl"]
            if sep in ("space", "cue"):
                continue
            core = T.split_speaker(spk)[0]
            for p in T.split_multi(core):
                inl_cnt[T.name_key(p)] += 1
            sep_cnt[sep] += 1
        # 이름 줄 모양: 짧은 이름만 있는 줄 + 다음 줄이 들여쓴 줄
        blk_cnt: Counter = Counter()
        blk_x: list[float] = []
        nxt = {i: (act[k + 1] if k + 1 < len(act) else None) for k, i in enumerate(act)}
        self.nxt = nxt
        self.prv = {i: (act[k - 1] if k else None) for k, i in enumerate(act)}
        shape: dict[int, tuple] = {}
        title_cnt: Counter = Counter()
        flat_cnt: Counter = Counter()
        flat_shape: dict[int, tuple] = {}
        for i in body:
            ln = L[i]
            ex = ln.extra
            if i in self.heads or "trans" in ex:
                continue
            j = nxt[i]
            if j is None or L[j].file != ln.file:
                continue
            nj = L[j]
            indented = ln.cls in ("I", "L") or (ln.cls == "R" and ln.rx < 0.62)
            if indented:
                dual = self._dual_names(ln)
                if dual:
                    if "\t" in nj.t or nj.cls in ("I", "M"):
                        shape[i] = ("dual", dual)
                    continue
            if len(ln.f) > 45:
                continue
            paren_name = False
            if re.fullmatch(r"\(([^()]{1,12})\)", ln.f):
                core, tails, paren_name = ln.f[1:-1].strip(), [], True
            else:
                core, tails = T.split_speaker(ln.f)
            if not core:
                continue
            kind = "name" if T.name_like(core, block=True) else ("title" if T.title_name_like(core) else None)
            if kind is None or j in self.heads or "trans" in nj.extra:
                continue
            if nj.dy is not None and nj.dy > 3.5:
                continue
            # 가운데 정렬 대사 문서: 대사 첫 줄이 글 길이에 따라 여백 근처·왼쪽에서 시작한다. 가운데 놓인 이름 줄 바로
            # 아래(빈 줄 없이) 줄이면 x 와 상관없이 이름 줄 모양으로 본다.
            cen = self.centered and ln.rx > self.lay.get("M", 0.0) + 0.15 and (nj.dy is None or nj.dy < 1.6) \
                and nj.pk == ln.pk
            if indented and (nj.cls in ("I", "R") or cen):
                shape[i] = (kind, core, tails, paren_name)
                if not paren_name:
                    for p in T.split_multi(core):
                        (blk_cnt if kind == "name" else title_cnt)[T.name_key(p)] += 1
                    if kind == "name":
                        blk_x.append(ln.rx)
            elif ln.cls == "M" and nj.cls in ("M", "I") and not paren_name and (nj.dy is None or nj.dy < 1.5) \
                    and (ln.dy is None or ln.dy >= 1.5 or not self.para_spacing) and "inl" not in ex \
                    and not T.name_like(T.split_speaker(nj.f)[0], block=True):
                flat_shape[i] = (kind, core, tails, False)
                flat_cnt[T.name_key(core)] += 1
        # 첫 글자만 대문자인 이름 줄: 같은 이름이 세 번 이상, 이름 칸 x 가 한곳일 때만
        title_wait: dict[int, tuple] = {}
        if title_cnt:
            tx = [L[i].rx for i, sh in shape.items() if sh[0] == "title"]
            tmed = statistics.median(tx)
            for i, sh in list(shape.items()):
                if sh[0] != "title":
                    continue
                k = T.name_key(sh[1])
                if title_cnt[k] >= 3 and abs(L[i].rx - tmed) <= 0.06:
                    blk_cnt[k] += 1
                    blk_x.append(L[i].rx)
                    shape[i] = ("name",) + sh[1:]
                else:
                    title_wait[i] = ("name",) + sh[1:]
                    del shape[i]
        if title_wait and blk_x:
            nx = statistics.median(blk_x)
            for i, sh in title_wait.items():
                if abs(L[i].rx - nx) <= 0.03 and title_cnt[T.name_key(sh[1])] >= 2:
                    shape[i] = sh
                    blk_cnt[T.name_key(sh[1])] += 1
        self.collapsed = False
        if not self.centered and self.para_spacing:
            self._collapsed_cues(body, shape, blk_cnt, blk_x)
        n_blk = sum(blk_cnt.values())
        self.block_mode = n_blk >= max(8, 0.02 * len(body))
        # 들여쓰기 없는 문서(txt): 빈 줄 뒤 이름 줄 + 바로 아래 대사
        self.flat_mode = False
        if not self.block_mode and flat_cnt:
            good = {k for k, c in flat_cnt.items() if c >= 3}
            n_flat = sum(flat_cnt[k] for k in good)
            if n_flat >= max(10, 0.03 * len(body)):
                self.flat_mode = True
                for i, sh in flat_shape.items():
                    if T.name_key(sh[1]) in good:
                        shape[i] = ("flat",) + sh[1:]
                        blk_cnt[T.name_key(sh[1])] += 1
                self.notes.append(f"대사 표기: 들여쓰기 없는 이름 줄 {n_flat}(빈 줄 앞까지 대사)")
                n_blk = sum(blk_cnt.values())
        n_inl = sum(sep_cnt.values())
        self.inline_mode = n_inl >= max(8, 0.02 * len(body))
        small = len(body) < 200
        vocab: dict[str, int] = {}
        for k, c in inl_cnt.items():
            if c >= 2 or small:
                vocab[k] = vocab.get(k, 0) + c
        for k, c in blk_cnt.items():
            if c >= 2 or small:
                vocab[k] = vocab.get(k, 0) + c
        self.vocab = vocab
        self.name_x = statistics.median(blk_x) if blk_x else None
        # 공백 하나로 붙은 이름('재혁 한밤중에…'): 다른 표기가 드물 때만 줄 첫 낱말 빈도로 이름을 찾는다
        sp_cnt: Counter = Counter()
        sp_style: dict[str, list[float]] = defaultdict(list)
        part_cnt: Counter = Counter()
        for i in body:
            ln = L[i]
            ex = ln.extra
            if "inl" in ex and ex["inl"][2] == "space" and i not in self.heads:
                core = T.split_speaker(ex["inl"][0])[0]
                k = T.name_key(core)
                sp_cnt[k] += 1
                if self.lang == "ko":
                    sp_style[k].append(ko_style(ex["inl"][1]))
            m = re.match(r"^([가-힣]{1,5})(이|가|은|는|을|를|의|에게|와|과|도|,)\s", ln.f)
            if m:
                part_cnt[T.name_key(m.group(1))] += 1
        self.space_vocab: set[str] = set()
        # 띄운 이름(lsp) 문서는 세 글자 이름을 공백 하나로 적는다('여고생 대사'): 공백 하나 표기 찾기를 끄지 않는다
        strong_n = n_blk + n_inl - sep_cnt.get("lsp", 0)
        for k, c in sp_cnt.items():
            if k in vocab and c >= 2:
                self.space_vocab.add(k)
                continue
            if strong_n >= 0.03 * len(body) or c < 4 or k in _SPACE_STOP:
                continue
            if c < 2 * part_cnt.get(k, 0) and c < 15:
                continue
            if T.is_hangul(k[-1:]) and k[-1] in "이가은는을를의도와과로에" and k[:-1] in sp_cnt and len(k) >= 3:
                continue
            st = sp_style.get(k)
            if st and statistics.mean(st) < 0.3:
                continue                       # 뒤 글이 대사 같지 않다(지문의 첫 낱말)
            self.space_vocab.add(k)
        if self.space_vocab and not (self.block_mode or self.inline_mode):
            self.notes.append(f"대사 표기: 이름 뒤 공백 하나(이름 {len(self.space_vocab)}개를 줄 첫 낱말 빈도로 찾음)")
        # 이름 줄 받아들이기
        self.blk_ok: dict[int, tuple] = {}
        for i, sh in shape.items():
            if sh[0] == "dual":
                self.blk_ok[i] = sh
                continue
            _, core, tails, paren_name = sh
            keys = [T.name_key(p) for p in T.split_multi(core)]
            ln = L[i]
            near_x = self.name_x is not None and abs(ln.rx - self.name_x) <= max(0.03, 2 * self.tol2)
            if paren_name:
                if keys[0] in vocab:
                    self.blk_ok[i] = sh
                continue
            if all(k in vocab for k in keys) or (near_x and self.block_mode):
                self.blk_ok[i] = sh
        # 이름 줄인데 대사 줄이 빠진 경우(OCR 누락): 이름 칸 x 에 있는 자주 나오는 이름
        if self.block_mode and self.name_x is not None:
            for i in body:
                ln = L[i]
                if i in self.blk_ok or i in self.heads or len(ln.f) > 25 or ln.cls != "I":
                    continue
                if abs(ln.rx - self.name_x) > max(0.02, 1.5 * self.tol2):
                    continue
                core, tails = T.split_speaker(ln.f)
                if core and T.name_like(core, block=True) and vocab.get(T.name_key(core), 0) >= 3:
                    self.blk_ok[i] = ("name", core, tails, False)
        # 지문 칸: 씬 머리 바로 다음 줄이 가장 많이 놓이는 x
        after_head = [L[nxt[i]].rx for i in self.heads if i in nxt and nxt[i] is not None
                      and nxt[i] not in self.heads and "inl" not in L[nxt[i]].extra and nxt[i] not in shape]
        self.act_x = None
        if len(after_head) >= 5:
            cl = PL.clusters_x(after_head, self.lay.get("tol", 0.008))
            cx, cn = max(cl, key=lambda c: c[1])
            if cn >= 0.6 * len(after_head):
                self.act_x = cx
        # 매달린 들여쓰기 칸: '이름⇥대사' 다음 줄들이 놓이는 x. 그 x 의 줄 대부분이 대사 뒤에만 나오면 이어짐 칸이다.
        # (x, 허용치) 목록. 글자층 문서는 여백 바로 옆(공백 한 칸, 0.004~tol2)의 가는 매달림 칸도 본다(안녕! 유에프오).
        self.cont_cols: list[tuple[float, float]] = []
        inl = {i for i in body if self._inline_ok(i)}
        if inl:
            tol = self.lay.get("tol", 0.008)
            hang = [L[nxt[i]].rx for i in inl if nxt[i] is not None and L[nxt[i]].rx > L[i].rx + self.tol2
                    and nxt[i] not in inl and nxt[i] not in self.heads]
            groups = [(h, c, 2 * tol) for h, c in (PL.clusters_x(hang, tol) if hang else [])]
            if Counter(L[i].src for i in body).most_common(1)[0][0] not in ("ocr", "txt"):
                fine = [L[nxt[i]].x for i in inl if nxt[i] is not None and L[nxt[i]].x is not None
                        and L[i].x is not None and 0.004 < L[nxt[i]].x - L[i].x <= self.tol2
                        and nxt[i] not in inl and nxt[i] not in self.heads]
                groups += [(h, c, 0.002) for h, c in (PL.clusters_x(fine, 0.0015) if fine else [])]
            for h, c, ht in groups:
                if c < 3:
                    continue
                if ht > 0.002 and self.act_x is not None and abs(h - self.act_x) <= 2 * tol:
                    continue
                if ht <= 0.002 and self.act_x is not None and abs(h - self.act_x) <= ht:
                    continue
                xs = (lambda ln: ln.rx) if ht > 0.002 else (lambda ln: ln.x if ln.x is not None else -1.0)
                at = [i for i in body if abs(xs(L[i]) - h) <= ht]
                chain = 0
                for i in at:
                    # 같은 x 줄이 이어진 덩어리의 바로 앞 줄이 '이름⇥대사' 줄이어야 이어짐
                    p = self.prv.get(i)
                    while p is not None and abs(xs(L[p]) - h) <= ht:
                        p = self.prv.get(p)
                    if p is not None and p in inl:
                        chain += 1
                if at and chain >= 0.7 * len(at):
                    self.cont_cols.append((h, ht))
        modes = []
        if self.block_mode:
            modes.append(f"이름 줄 {n_blk}")
        sep_names = {"tab": "이름⇥대사", "colon": "이름 : 대사", "space2": "이름  대사", "quote": "이름 따옴표 대사",
                     "lsp": "띄운 이름 대사", "bracket": "[이름] 대사"}
        for s, c in sep_cnt.most_common():
            modes.append(f"{sep_names.get(s, s)} {c}")
        self.notes.append("대사 표기: " + (", ".join(modes) if modes else "뚜렷한 표기 없음")
                          + f"; 지문 여백 x={self.lay.get('M')}, 단계 {self.lay.get('levels')[:6]}"
                          + (f", 쪽 밀림 맞춤 {self.lay.get('shifted_pages')}쪽" if self.lay.get("shifted_pages") else ""))
        # 이름 없는 대사 칸 줄(이름 줄 문서에서 지문 뒤 바로 대사 칸에 놓인 줄): 화자 없는 대사로 시작
        self.spkless: set[int] = set()
        self.dlg_x = None
        if self.block_mode:
            dx = [L[nxt[i]].rx for i, sh in self.blk_ok.items() if sh[0] == "name" and nxt[i] is not None
                  and not L[nxt[i]].f.startswith("(")]
            if dx:
                self.dlg_x = statistics.median(dx)
                for i in body:
                    ln = L[i]
                    p = self.prv.get(i)
                    if ln.cls != "I" or i in self.blk_ok or i in self.heads or p is None:
                        continue
                    if abs(ln.rx - self.dlg_x) > 2 * self.tol2 or ln.f.startswith("("):
                        continue
                    pl = L[p]
                    if pl.cls in ("M", "L") and p not in self.heads and pl.file == ln.file \
                            and "trans" not in ln.extra and "onscr" not in ln.extra:
                        self.spkless.add(i)

    def _collapsed_cues(self, body: list[int], shape: dict, blk_cnt: Counter, blk_x: list[float]) -> None:
        """대사 칸이 지문 여백으로 눌린 영어 대본(The Graduate, Apocalypse Now, The Last Samurai): 대문자 이름 줄 아래
        대사가 들여쓰지 않고 여백에 놓인다. 빈 줄 뒤 대문자 이름 줄 + 바로 아래(한 줄 간격, 들여쓴 이름은 빈 줄 하나까지)
        대문자 아닌 줄인데 그 다음 줄이 여백에 있는 경우가 20번 이상이면 그런 문서로 보고 이름 줄로 받는다.
        들여쓴 이름은 두 번 이상, 여백의 이름은 들여쓴 이름으로도 나온 이름만. 대사는 문단 간격 전까지(_gap_cont)."""
        L, nxt = self.L, self.nxt
        M = self.lay.get("M", 0.0)
        cand: list[tuple[int, str, list, bool]] = []
        for i in body:
            ln = L[i]
            j = nxt[i]
            if j is None or i in shape or i in self.heads or "trans" in ln.extra or "onscr" in ln.extra:
                continue
            if len(ln.f) > 40 or "inl" in ln.extra and ln.extra["inl"][2] in ("tab", "colon"):
                continue
            core, tails = T.split_speaker(ln.f)
            if not core or core.upper() != core or not any(c.isascii() and c.isalpha() for c in core) \
                    or not T.name_like(core, block=True):
                continue
            nj = L[j]
            if nj.file != ln.file or j in self.heads or "trans" in nj.extra or nj.f.upper() == nj.f:
                continue
            if not (ln.dy is None or ln.dy >= 1.5 or ln.pg_first or (i - 1) in self.drop):
                continue                                # 이름 줄 앞은 빈 줄(문단 간격)이나 쪽 장식
            indented = ln.rx > M + 0.03
            if nj.pk == ln.pk and (nj.dy is None or nj.dy > (2.2 if indented else 1.3)):
                continue
            cand.append((i, T.name_key(core), [core, tails], indented))
        flat_next = sum(1 for i, *_ in cand if L[nxt[i]].cls in ("M", "L"))
        if flat_next < 20:
            return
        keys = Counter(k for _, k, _, ind in cand if ind)
        acc = [(i, k, core, tails) for i, k, (core, tails), ind in cand
               if (ind and keys[k] >= 2) or (not ind and (keys[k] >= 2 or blk_cnt.get(k, 0) >= 2))]
        n_norm = sum(1 for sh in shape.values() if sh[0] == "name")
        if len(acc) < 0.1 * (n_norm + len(acc)):
            return                                      # 대부분 보통 이름 줄인 문서: 몇 줄 때문에 문단 간격 잇기를 켜지 않음
        n = 0
        for i, k, core, tails in acc:
            shape[i] = ("name", core, tails, False)
            blk_cnt[k] += 1
            n += 1
        if n:
            self.collapsed = True
            self.notes.append(f"대사 표기: 대사가 지문 여백에 놓인 대문자 이름 줄 {n}개(대사는 문단 간격 전까지)")

    def _dual_names(self, ln):
        if "\t" not in ln.t or ln.cls not in ("I", "R", "M"):
            return None
        parts = [p.strip() for p in ln.t.split("\t") if p.strip()]
        if len(parts) != 2:
            return None
        cores = [T.split_speaker(p)[0] for p in parts]
        if not all(c and T.name_like(c, block=True) for c in cores):
            return None
        if all(any(ch.isalpha() and ch.isascii() for ch in c) and c.upper() == c for c in cores):
            return parts
        return None

    def _inline_ok(self, i: int) -> bool:
        ln = self.L[i]
        sp = ln.extra.get("inl")
        if not sp:
            return False
        spk, rest, sep = sp
        if sep == "cue":
            return self.inline_mode or self.block_mode or len(self.active) < 200
        core = T.split_speaker(spk)[0]
        keys = [T.name_key(p) for p in T.split_multi(core)]
        if sep == "space":
            if not all(k in self.space_vocab or k in self.vocab and self.vocab[k] >= 3 for k in keys):
                return False
            if self.lang == "ko" and ko_style(rest) <= -1.5 and not rest.startswith("("):
                return False
            if not all(k in self.space_vocab for k in keys) and self.block_mode:
                return False
            return True
        if all(k in self.vocab for k in keys):
            return True
        if sep in ("tab", "colon") and self.inline_mode and len(core) <= 12:
            return True
        return False

    # ------------------------------------------------------------ 차례대로 요소 만들기
    def _label(self) -> list[dict]:
        L, act = self.L, self.active
        els: list[dict] = []
        cur: dict | None = None
        skip: set[int] = set()
        # 씬 머리 앞에 놓여 그 씬 머리에 붙는 줄(여백 씬 번호 등)은 차례가 와도 따로 요소로 만들지 않는다.
        # 앞머리 안의 씬 머리는 씬으로 만들지 않으므로 제외(그 줄은 앞머리로 남음).
        pre: set[int] = set()
        for k, i in enumerate(act):
            if k >= self.fe and i in self.heads:
                pre.update(j for j in self.head_attach.get(i, []) if self.pos.get(j, -1) < k)

        def new(tp: str, i: int, **kw) -> dict:
            e = {"type": tp, "lines": [i]}
            e.update(kw)
            els.append(e)
            return e

        for k, i in enumerate(act):
            if i in skip or i in pre:
                continue
            ln = L[i]
            ex = ln.extra
            prev = L[cur["lines"][-1]] if cur else None
            if k < self.fe:
                if cur and cur["type"] == "front" and self._same_para(prev, ln, front=True):
                    cur["lines"].append(i)
                else:
                    cur = new("front", i)
                continue
            if i in self.heads:
                cur = new("scene", i)
                for j in self.head_attach.get(i, []):
                    if j != i and (j in pre or j not in skip):
                        cur["lines"].append(j)
                        skip.add(j)
                        pre.discard(j)
                cur["lines"].sort(key=lambda x: self.pos.get(x, 0))
                continue
            in_dlg = cur is not None and cur["type"] == "dialogue"
            strong_inl = bool(ex.get("inl")) and ex["inl"][2] in ("tab", "colon", "angle", "lsp", "bracket") \
                and self._inline_ok(i)
            if in_dlg and (self.centered or self.collapsed) and cur["mode"] == "block" and i not in self.blk_ok \
                    and not ex.get("trans") and not strong_inl and self._gap_cont(cur, ln):
                cur["lines"].append(i)                 # 가운데 정렬·눌린 대사 칸: x 와 상관없이 문단 간격 전까지
                continue
            if in_dlg and cur["mode"] in ("block", "dual") and ln.cls == "I" and i not in self.blk_ok \
                    and not ex.get("trans") and not strong_inl and self._block_cont(cur, ln):
                cur["lines"].append(i)
                continue
            if in_dlg and cur["mode"] == "dual" and "\t" in ln.t and i not in self.blk_ok and not ex.get("trans") \
                    and i not in self.heads and ln.file == L[cur["lines"][-1]].file:
                cur["lines"].append(i)                 # 좌우 두 칸: 탭으로 이어진 줄은 여백에 있어도 대사
                continue
            if in_dlg and cur["mode"] == "flat" and i not in self.blk_ok and not self._starts_something(i) \
                    and ln.file == L[cur["lines"][-1]].file \
                    and (ln.dy is None or ln.dy < 1.5 or not self.para_spacing):
                cur["lines"].append(i)
                continue
            if ex.get("trans") and not (in_dlg and ln.cls == "I" and cur["mode"] == "block" and len(ln.f) > 12):
                cur = new("transition", i)
                continue
            if ex.get("onscr") and not (in_dlg and ln.cls == "I" and cur["mode"] == "block"):
                kind, body = ex["onscr"]
                cur = new("onscreen", i, kind=kind)
                closer = {"<": ">", "[": "]", "{": "}"}.get(ln.f[:1])
                if closer and not ln.f.rstrip().endswith(closer):
                    cur["open"], cur["closer"] = True, closer      # 괄호가 닫힐 때까지(최대 6줄)
                elif not body:
                    cur["open"], cur["closer"] = True, None        # 'SUPER:' 다음 줄(따옴표 글이면 이어서)
                continue
            if cur is not None and cur["type"] == "onscreen" and cur.get("open") and len(cur["lines"]) < 6 \
                    and not self._starts_something(i):
                cur["lines"].append(i)
                cl = cur.get("closer")
                if (cl and ln.f.rstrip().endswith(cl)) or (not cl and not ln.f.startswith(("\"", "“", "'"))):
                    cur["open"] = False
                continue
            if i in self.blk_ok:
                sh = self.blk_ok[i]
                if sh[0] == "dual":
                    cur = new("dialogue", i, mode="dual", spk_raw=" / ".join(sh[1]), names=sh[1], speech=[])
                else:
                    cur = new("dialogue", i, mode="flat" if sh[0] == "flat" else "block", spk_raw=ln.f, names=None,
                              speech=[], paren_name=sh[3])
                continue
            if self._inline_ok(i):
                spk, rest, sep = ex["inl"]
                cur = new("dialogue", i, mode="inline", sep=sep, spk_raw=spk, names=None, speech=[(i, rest)])
                continue
            if i in self.spkless and not in_dlg:
                cur = new("dialogue", i, mode="block", spk_raw=None, names=None, speech=[], noname=True)
                continue
            if in_dlg and self._cont(cur, ln, k):
                cur["lines"].append(i)
                continue
            if ex.get("garbage"):
                if cur and cur["type"] == "unknown":
                    cur["lines"].append(i)
                else:
                    cur = new("unknown", i)
                continue
            if cur and cur["type"] == "action" and self._same_para(prev, ln):
                cur["lines"].append(i)
            else:
                cur = new("action", i)
        return els

    def _starts_something(self, i: int) -> bool:
        ex = self.L[i].extra
        return i in self.heads or i in self.blk_ok or "trans" in ex or "onscr" in ex or self._inline_ok(i)

    def _gap_cont(self, cur: dict, ln) -> bool:
        """가운데 정렬 대사·눌린 대사 칸 문서의 대사 잇기: 같은 쪽에서 줄 간격이 한 줄(1.5배 이하)이면 잇는다
        (눌린 칸 문서는 이름 줄 바로 다음에 빈 줄 하나까지). 쪽이 바뀌면 이름 줄만 있거나 앞 줄이 문장 중간에서
        끊겼을 때만(가운데 정렬 문서는 새 줄이 지문 여백에 있지 않을 때만)."""
        prev = self.L[cur["lines"][-1]]
        first = len(cur["lines"]) == 1 and not cur.get("noname")
        if ln.file != prev.file:
            return False
        if ln.pk != prev.pk:
            if first:
                return True
            return wrapped(prev.f, self.width) and (self.collapsed or abs(ln.rx - self.lay.get("M", 0.0)) > 0.003)
        if ln.dy is None:
            return False
        return ln.dy <= (2.2 if first and self.collapsed and prev.rx > self.lay.get("M", 0.0) + 0.03 else 1.5)

    def _block_cont(self, cur: dict, ln) -> bool:
        prev = self.L[cur["lines"][-1]]
        if ln.file != prev.file:
            return False
        if cur["mode"] == "dual":
            return "\t" in ln.t or (ln.dy is not None and ln.dy < 1.5)
        if ln.dy is not None and ln.dy > 2.6 and len(cur["lines"]) > 1:
            # 큰 빈칸 뒤 들여쓴 줄: 이름 없는 새 덩어리일 수 있으나 대사 칸이면 이어짐으로 본다
            return abs(ln.rx - self.L[cur["lines"][-1]].rx) <= 0.08
        return True

    def _cont(self, cur: dict, ln, k: int) -> bool:
        """대사 이어짐 판정(점수). 들여쓰기·앞 줄 끊김·글 모양·y 간격·이름+토씨를 함께 본다."""
        L = self.L
        prev = L[cur["lines"][-1]]
        first = L[cur["lines"][0]]
        if ln.file != prev.file:
            return False
        f = ln.f
        sp = self._inline_split_raw(ln.t)
        if sp and _RX_SHOT.match(sp[1]):
            return False                        # '영일 cu' 촬영 크기 표기 줄은 대사에 잇지 않음(화면 설명)
        mb = _RX_BRACKET.match(f)
        if mb and (_RX_BRACKET_META.match(mb.group(1)) or self.bracket_mode):
            return False                        # '[페이지] 001', '[제목] …' 같은 대괄호 표지 줄은 잇지 않음
        if cur["mode"] in ("block", "dual", "flat"):
            # 이름 줄 덩어리 대사는 들여쓴 줄(flat 은 빈 줄 전까지)만 잇는다(_label 에서 처리).
            return False
        sc = 0.0
        lead = first.rx
        tol = self.lay.get("tol", 0.008)
        at_act = self.act_x is not None and abs(ln.rx - self.act_x) <= 2 * tol and abs(self.act_x - lead) > 2 * tol
        if any(abs((ln.rx if ht > 0.002 else (ln.x if ln.x is not None else -1.0)) - h) <= ht
               for h, ht in self.cont_cols):
            sc += 3.0
        elif at_act:
            sc -= 2.0                           # 씬 머리 바로 뒤 지문이 놓이는 칸(대사 시작 칸과 다름)
        elif self.cont_cols and ln.rx > lead + self.tol2:
            sc -= 0.5                           # 들여썼지만 이어짐 칸이 아님
        elif ln.cls == "I" or ln.rx > lead + self.tol2:
            sc += 1.5 if ln.rx > lead + self.tol2 else 0.5
        elif ln.cls in ("M", "L") and abs(ln.rx - lead) <= self.tol2:
            sc -= 0.3
        if f.startswith("("):
            m = re.match(r"^\(([^()]*)\)\s*(.*)$", f)
            if m and m.group(2):
                sc += 2.0
            elif m:
                inner = m.group(1)
                has_name = any(T.name_key(w) in self.vocab for w in re.findall(r"[가-힣A-Za-z]{2,}", inner)[:3])
                if len(inner) <= 15 and not has_name:
                    sc += 1.0
                elif ln.rx <= lead + self.tol2:
                    sc -= 1.5
            else:
                sc -= 2.0                       # 여러 줄에 걸친 괄호 지문
        if wrapped(prev.f, self.width) and not prev.f.startswith("("):
            sc += 2.0
        nlet = T.letters(f)
        if nlet == 0:
            sc += 1.0
        if self.lang == "ko":
            st = ko_style(f)
            if not f.endswith(_TERMINAL) and k + 1 < len(self.active):
                nx = L[self.active[k + 1]]
                if not self._starts_something(nx.i):
                    st = max(st, ko_style(nx.f))
            sc += max(-2.0, min(2.0, st))
            m = re.match(r"^([가-힣]{1,6}?)(?:이|가|은|는|을|를|의|에게|와|과|도|,)(?:\s|$)", f)
            if m and T.name_key(m.group(1)) in self.vocab:
                sc -= 2.0
        else:
            if re.match(r"^[a-z]", f):
                sc += 1.0
        if ln.dy is not None and ln.dy >= 1.6 and self.para_spacing:
            sc -= 1.5
        if prev.pk != ln.pk and not wrapped(prev.f, self.width):
            sc -= 0.5
        if len(cur["lines"]) >= 8:
            sc -= 0.5
        return sc > 0.5

    def _same_para(self, prev, ln, front: bool = False) -> bool:
        if prev is None or prev.file != ln.file:
            return False
        if front:
            if prev.pk != ln.pk:
                return False
            return ln.dy is None or ln.dy < 1.6
        if abs(ln.rx - prev.rx) > 2 * self.tol2 and not (ln.rx > prev.rx and wrapped(prev.f, self.width)):
            return False
        if ln.dy is not None and ln.dy >= 1.6:
            return False
        if ln.f.startswith(("-", "–", "—", "·", "*", "•")) and not prev.f.endswith(("-", ",")):
            return False
        if prev.pk != ln.pk:
            return wrapped(prev.f, self.width)
        if self.para_spacing:
            return True
        return wrapped(prev.f, self.width)

    # ------------------------------------------------------------ 쪽 넘김으로 쪼개진 대사
    def _merge_split_dialogue(self, els: list[dict]) -> list[dict]:
        out: list[dict] = []
        for e in els:
            if e["type"] == "dialogue" and e["mode"] == "block" and out and e.get("spk_raw"):
                p = out[-1]
                if p["type"] == "dialogue" and p["mode"] in ("block", "inline") and p.get("spk_raw"):
                    core, tails = T.split_speaker(e["spk_raw"])
                    pcore = T.split_speaker(p["spk_raw"])[0]
                    if T.name_key(core) == T.name_key(pcore) and (any(T.is_cont_cue(t) for t in tails)):
                        a, b = p["lines"][-1], e["lines"][0]
                        between = [x for x in range(a + 1, b)]
                        pb = self.L[a].pk != self.L[b].pk
                        more = any(self.drop.get(x) == "cont_marker" for x in between)
                        if (more or pb) and all(x in self.drop for x in between):
                            self.drop[b] = "cont_marker"
                            p["lines"].extend(e["lines"][1:])
                            p.setdefault("merged", 0)
                            p["merged"] += 1
                            if len(e["lines"]) == 1:
                                pass
                            continue
            out.append(e)
        return out

    # ------------------------------------------------------------ 결과 만들기
    def _build(self, els: list[dict]) -> dict:
        L = self.L
        scenes: list[dict] = []
        out_els: list[dict] = []
        scene_id = None
        dlg_meta: list[tuple[dict, dict]] = []
        last_dlg: dict | None = None        # 같은 씬 바로 앞 대사(화자 이어받기용)
        gap = 0                             # 그 뒤 지문·화면 글자 요소 수
        n_inherit = 0
        for e in els:
            lines = [x for x in e["lines"] if x not in self.drop]
            if not lines:
                continue
            e["lines"] = lines
            tp = e["type"]
            o: dict = {"id": None, "type": tp}
            if tp in ("action", "onscreen", "unknown"):
                gap += 1
            elif tp != "dialogue":
                last_dlg = None
            if tp == "scene":
                hi = next((x for x in lines if x in self.heads), lines[0])
                st, num, rest = L[hi].extra["head"]
                num = num or self.head_num.get(hi)
                text = T.flat(" ".join(L[x].f for x in lines))
                if st in ("ko_nonum", "en_noint"):
                    place_src = L[hi].f
                elif rest:
                    place_src = rest
                    more = [L[x].f for x in lines if x != hi and "head" in L[x].extra]
                    if more:                            # 두 줄로 나뉜 씬 머리
                        place_src = T.flat(rest.rstrip(" -–—") + " - " + " ".join(more))
                else:
                    place_src = T.flat(" ".join(L[x].f for x in lines if x != hi and not T.RX_BARE_NUM.match(L[x].f)))
                place, tm = T.split_place_time(place_src) if place_src else (None, None)
                sid = f"SC{len(scenes) + 1:03d}"
                scenes.append({"id": sid, "number": num or None, "heading": text, "place": place, "time": tm,
                               "flags": T.heading_flags(text)})
                scene_id = sid
                o.update(scene=sid, text=text)
            elif tp == "dialogue":
                o.update(scene=scene_id)
                o.update(self._dialogue_fields(e))
                # 화자 이어받기: 이름 줄 문서에서 이름 줄 없이 대사 칸에 놓인 덩어리가 같은 씬 앞 대사와 지문·화면 글자
                # 요소 2개 이하로만 떨어져 있으면 그 대사의 화자가 말을 잇는 것(작가가 이름 줄을 되풀이하지 않음)
                if e.get("noname") and self.block_mode and last_dlg is not None and gap <= 2 \
                        and last_dlg["_names"] and last_dlg.get("scene") == scene_id:
                    o["_names"] = list(last_dlg["_names"])
                    o["speaker_inferred"] = True
                    n_inherit += 1
                dlg_meta.append((o, e))
                last_dlg, gap = o, 0
            elif tp == "onscreen":
                first = L[lines[0]]
                kind, body = first.extra.get("onscr") or ("other", first.f)
                parts = [body] + [L[x].f for x in lines[1:]]
                txt = T.join_lines(parts).strip().strip("<>[]").strip()
                txt = re.sub(r"^[:：\-–—]\s*", "", txt).rstrip(">]").strip()
                o.update(scene=scene_id, kind=kind, text=txt)
            else:
                txt = T.join_lines([L[x].f for x in lines])
                o.update(scene=scene_id if tp != "front" else None, text=txt)
                if tp == "action" and scene_id and len(txt) <= 30:
                    fl = T.heading_flags(txt)
                    if fl and re.match(r"^[<\[(]?\s*\S{0,6}\s*(?:몽타[주쥬]|회상|플래[시쉬]\s?백|인터컷|교차|꿈|"
                                       r"MONTAGE|FLASHBACK|INTERCUT|DREAM|SERIES OF SHOTS)", txt, re.I):
                        sc = scenes[-1]
                        for f in fl:
                            if f not in sc["flags"]:
                                sc["flags"].append(f)
            o["lines"] = [L[x].n for x in lines]
            out_els.append(o)
        if n_inherit:
            self.notes.append(f"화자 이어받기: 이름 줄 없이 지문 뒤 대사 칸에 이어진 대사 {n_inherit}개는 앞 대사의 화자"
                              "(speaker_inferred)")
        self._resolve_vo(dlg_meta)
        self._overlap(out_els, dlg_meta)
        people = self._people(dlg_meta)
        for k, o in enumerate(out_els, 1):
            o["id"] = f"E{k:04d}"
        # 키 차례를 스키마 예시처럼
        out_els = [self._order(o) for o in out_els]
        dropped = self._dropped()
        front = self._front_info(out_els)
        quality = self._quality(out_els, dropped)
        return {
            "schema": "sink.script/1",
            "doc_id": self.doc_id,
            "parser": {"name": PARSER_NAME, "version": VERSION},
            "lang": self.lang,
            "front": front,
            "people": people,
            "scenes": scenes,
            "elements": out_els,
            "dropped": dropped,
            "quality": quality,
        }

    @staticmethod
    def _order(o: dict) -> dict:
        keys = ["id", "type", "scene", "kind", "speakers", "speaker_inferred", "speaker_raw", "how", "lang", "text",
                "raw", "paren", "gloss", "sung", "overlap", "lines"]
        return {k: o[k] for k in keys if k in o}

    # ------------------------------------------------------------ 대사 칸
    def _dialogue_fields(self, e: dict) -> dict:
        L = self.L
        lines = e["lines"]
        raw = "\n".join(T.norm(L[x].raw).strip() for x in lines)
        tails: list[str] = []
        if e["mode"] == "inline":
            spk_raw = e["spk_raw"]
            first_rest = e["speech"][0][1]
            speech_parts = [first_rest] + [L[x].t for x in lines[1:]]
            if e.get("sep") == "cue":
                core, tails = "", [spk_raw.strip()[1:-1]]
                names = []
            else:
                core, tails = T.split_speaker(spk_raw)
                names = [core]
        elif e["mode"] == "dual":
            spk_raw = e["spk_raw"]
            names = []
            for p in e["names"]:
                c, tl = T.split_speaker(p)
                names.append(c)
                tails += tl
            speech_parts = [L[x].t.replace("\t", " / ") for x in lines[1:]]
        elif e.get("noname"):
            spk_raw = None
            names = []
            speech_parts = [L[x].t for x in lines]
        else:
            spk_raw = L[lines[0]].f
            if e.get("paren_name"):
                core = spk_raw.strip()[1:-1].strip()
                tails = ["(괄호 이름)"]
            else:
                core, tails = T.split_speaker(spk_raw)
            names = [core]
            speech_parts = [L[x].t for x in lines[1:]]
        speech_parts = [T.flat(p) for p in speech_parts]
        speech = T.join_lines(speech_parts)
        two_in_line = False
        if e["mode"] == "inline":
            # 한 줄에 두 화자: '민우 : 괜찮습… / 장사장 : (동시에) 자, 펴.'
            m = re.search(r"\s/\s*([^/:()]{1,12}?)\s*(\([^()]{1,15}\))?\s*:\s*", speech)
            if m and T.name_key(T.split_speaker(m.group(1))[0]) in self.vocab:
                names.append(T.split_speaker(m.group(1))[0])
                speech = speech[: m.start()] + " / " + (m.group(2) or "") + " " + speech[m.end():]
                two_in_line = True
        how = None
        m = _RX_HOWPREFIX.match(speech)
        if m:
            how = T.how_of([m.group(1)])
            speech = speech[m.end():]
        speech = re.sub(r"^[\"“‘']{1,2}(.*?)[\"”’']{0,2}$", r"\1", speech) if e.get("sep") == "quote" else speech
        pieces = T.split_parens(speech)
        text, parens = T.assemble(pieces)
        # 화자
        spk_names: list[str] = []
        for c in names:
            spk_names.extend(T.split_multi(c) if c else [])
        voice_only = bool(spk_names) and all(T.name_key(T.display_name(c)) in _VOICE_WORDS for c in spk_names)
        if voice_only or e.get("sep") == "cue":
            spk_names = []                      # '소리 : …', '소 리 …': 누군지 적지 않은 화면 밖 목소리
            # '목소리 : (재민의) …', '(소리)⇥(문수의) …': 맨 앞 괄호의 '누구의' 가 아는 인물이면 그 인물
            m = re.fullmatch(r"\s*([가-힣]{1,6})\s*의\s*", parens[0]["text"]) if parens and parens[0]["at"] == 0 else None
            if m and T.name_key(m.group(1)) in self.vocab:
                spk_names = [m.group(1)]
            named = [t for t in tails if T.name_key(t) in self.vocab and not T.how_of([t])]
            if voice_only and named:
                spk_names = [named[0]]                  # '소리(요시마치): …', '소리(역도산): …'

        cue_tails = [t for t in tails if t != "(괄호 이름)"]
        how = how or T.how_of(cue_tails)
        if voice_only:
            how = how or "off"
        if e.get("paren_name"):
            how = how or "narration"
        lang = T.lang_of(cue_tails)
        sung = any(T.RX_SUNG.search(t) for t in cue_tails)
        overlap_cue = any(T.RX_OVERLAP.search(t) for t in cue_tails)
        # 괄호 지시에서 how·lang·노래·동시
        for p in parens:
            ptxt = p["text"]
            if p["at"] == 0 and how is None:
                h2 = T.how_of([ptxt]) if len(ptxt) <= 16 else None
                if h2:
                    how = h2
            if lang is None and len(ptxt) <= 24:
                lang = T.lang_of([ptxt])
            if T.RX_SUNG.search(ptxt):
                sung = True
            if T.RX_OVERLAP.search(ptxt) and len(ptxt) <= 30:
                overlap_cue = True
        if re.search(r"[♪♬♩♫]", speech):
            sung = True
        # 병기 번역: '와. / Woah.' 또는 'Sir! …(심각한…)'
        gloss = None
        tl = T.text_lang(text, self.lang, self.lang_lat) if text else self.lang
        if " / " in text:
            a, b = text.split(" / ", 1)
            la = T.text_lang(a, self.lang, self.lang_lat)
            lb = T.text_lang(b, self.lang, self.lang_lat)
            if a.strip() and b.strip() and la != lb and T.letters(a) and T.letters(b):
                if la == self.lang and lb != self.lang:
                    a, b, la, lb = b, a, lb, la
                text, gloss, tl = a.strip(), b.strip(), la
                parens = [dict(p, at=min(p["at"], len(text))) for p in parens]
        if tl != self.lang and parens and gloss is None:
            keep = []
            gl = []
            for p in parens:
                # 병기 번역은 외국어 글 뒤에 붙는다(맨 앞 괄호는 연기 지시: '(가픈 숨을 몰아쉬며) Sir! …')
                if 0 < p["at"] and p["at"] >= len(text) - 1 and T.letters(p["text"]) >= 3 \
                        and T.text_lang(p["text"], self.lang, self.lang_lat) == self.lang \
                        and len(p["text"]) >= 0.3 * max(1, len(text[: p["at"]])):
                    gl.append(p["text"])
                else:
                    keep.append(p)
            if gl:
                gloss = " ".join(gl)
                parens = keep
        if lang is None:
            lang = tl if text else self.lang
        if how is None:
            if any(re.fullmatch(r"(?i)나레이션|내레이션|나래이션|NARRATOR|NARRATION|해설|NA", T.flat(c)) for c in spk_names):
                how = "narration"
        return {
            "speakers": [], "_names": spk_names,
            "speaker_raw": spk_raw, "how": how or "on", "lang": lang, "text": text, "raw": raw,
            "paren": parens, "gloss": gloss, "sung": bool(sung), "overlap": None, "_overlap_cue": overlap_cue,
            "_dual": e["mode"] == "dual" or two_in_line,
        }

    def _resolve_vo(self, dlg: list[tuple[dict, dict]]) -> None:
        on_in_scene: set[tuple] = set()
        for o, _ in dlg:
            if o["how"] != "vo":
                for nm in o["_names"]:
                    on_in_scene.add((o.get("scene"), T.name_key(nm)))
        n_off = n_nar = 0
        for o, _ in dlg:
            if o["how"] == "vo":
                if any((o.get("scene"), T.name_key(nm)) in on_in_scene for nm in o["_names"]):
                    o["how"] = "off"
                    n_off += 1
                else:
                    o["how"] = "narration"
                    n_nar += 1
        if n_off or n_nar:
            self.notes.append(f"(V.O.): 같은 씬에 그 인물의 다른 대사가 있으면 off({n_off}), 없으면 narration({n_nar})")
        # (N) 해석 기록
        n_head = sum(1 for i in self.heads if re.search(r"[(\[]\s*N\s*[)\]]", self.L[i].f))
        n_spk = sum(1 for o, _ in dlg if re.search(r"\(\s*N\s*\)", o["speaker_raw"] or ""))
        if n_head or n_spk:
            self.notes.append(f"(N): 씬 머리에서는 밤({n_head}번), 화자 꼬리에서는 내레이션({n_spk}번)으로 해석")

    def _overlap(self, out_els: list[dict], dlg: list[tuple[dict, dict]]) -> None:
        g = 0
        prev = None
        for o in out_els:
            if o["type"] != "dialogue":
                if o["type"] not in ("action",) or len(o.get("text", "")) > 40:
                    prev = None
                continue
            if o.pop("_dual", False) or (o.get("_overlap_cue") and len(o.get("_names", [])) >= 2):
                o["overlap"] = g                       # 좌우 두 칸 동시 대사, 한 줄 두 화자, '태우/지웅 (동시에)'
                g += 1
            elif o.get("_overlap_cue") and prev is not None:
                if prev["overlap"] is None:
                    prev["overlap"] = g
                    g += 1
                o["overlap"] = prev["overlap"]
            o.pop("_overlap_cue", None)
            prev = o

    # ------------------------------------------------------------ 인물표
    def _people(self, dlg: list[tuple[dict, dict]]) -> list[dict]:
        surf: dict[str, Counter] = defaultdict(Counter)       # key -> 표기 빈도
        cnt: Counter = Counter()
        for o, _ in dlg:
            for nm in o["_names"]:
                d = T.display_name(nm)
                k = T.name_key(d)
                if not k:
                    continue
                surf[k][d] += 1
                cnt[k] += 1
        # 보수적 별칭: (1) 공백·대소문자만 다름(이미 같은 열쇠) (2) 성 생략 (3) 영어 이름 일부 (4) OCR 오탈자
        parent: dict[str, str] = {k: k for k in cnt}

        def root(k: str) -> str:
            while parent[k] != k:
                k = parent[k]
            return k

        keys = sorted(cnt, key=lambda k: (-cnt[k], k))
        for k in keys:
            if T.is_hangul(k[:1]) and len(k) == 2 and k not in _ROLE_WORDS:
                longs = [a for a in keys if len(a) == 3 and a[1:] == k and a[0] in _SURNAMES]
                if len(longs) == 1:
                    a = longs[0]
                    big, small = (a, k) if cnt[a] >= cnt[k] else (k, a)
                    parent[root(small)] = root(big)
        # 영어 이름 일부('GUILLAM' = 'PETER GUILLAM'): 두 낱말 이름의 첫·끝 낱말이 흔한 역할 낱말이 아니고 하나에만 맞을 때
        multi = [k for k in keys if len(next(iter(surf[k])).split()) == 2 and not T.is_hangul(k[:1])]
        for k in keys:
            if T.is_hangul(k[:1]) or len(next(iter(surf[k])).split()) != 1 or len(k) < 3 or k in _EN_ROLE:
                continue
            hits = [a for a in multi if k in (T.name_key(next(iter(surf[a])).split()[0]),
                                              T.name_key(next(iter(surf[a])).split()[-1]))]
            if len(hits) == 1:
                a = hits[0]
                big, small = (a, k) if cnt[a] >= cnt[k] else (k, a)
                if root(small) != root(big):
                    parent[root(small)] = root(big)
        # 글자 하나 다른 드문 표기(OCR·오타): 번호 붙은 엑스트라, 복수형, 짧은 한글 이름은 묶지 않는다
        # (한글 세 글자 이름은 OCR 문서에서만: '찍벌남'→'쩍벌남'. 글자층 문서의 '옆집여/옆집남'은 다른 사람)
        ocr = Counter(ln.src for ln in self.L).most_common(1)[0][0] == "ocr" if self.L else False
        min_ko = 3 if ocr else 4
        for k in keys:
            if cnt[k] > 2 or root(k) != k or k[-1:].isdigit():
                continue
            for a in keys:
                if a == k or cnt[a] < max(10, 10 * cnt[k]) or root(a) == root(k) or a[-1:].isdigit():
                    continue
                if abs(len(a) - len(k)) > 1 or len(k) < (min_ko if T.is_hangul(k[:1]) else 5):
                    continue
                if a + "S" == k or k + "S" == a:
                    continue
                if _edit1(a, k):
                    parent[k] = root(a)
                    break
        groups: dict[str, list[str]] = defaultdict(list)
        for k in keys:
            groups[root(k)].append(k)
        order = sorted(groups, key=lambda r: -sum(cnt[k] for k in groups[r]))
        people = []
        pid_of: dict[str, str] = {}
        for n, r in enumerate(order, 1):
            pid = f"P{n:02d}"
            allsurf: Counter = Counter()
            for k in groups[r]:
                allsurf.update(surf[k])
                pid_of[k] = pid
            name = allsurf.most_common(1)[0][0]
            aliases = sorted({s for s in allsurf if s != name})
            people.append({"id": pid, "name": name, "aliases": aliases, "count": 0})
        by_id = {p["id"]: p for p in people}
        for o, _ in dlg:
            sp = []
            for nm in o.pop("_names"):
                k = T.name_key(T.display_name(nm))
                pid = pid_of.get(k)
                if pid and pid not in sp:
                    sp.append(pid)
            o["speakers"] = sp
            for pid in sp:
                by_id[pid]["count"] += 1
        return people

    # ------------------------------------------------------------ dropped·front·quality
    def _dropped(self) -> list[dict]:
        out: list[dict] = []
        cur = None
        for ln in self.L:
            why = self.drop.get(ln.i)
            if why is None:
                cur = None
                continue
            if cur and cur["why"] == why and why in ("page_header", "page_footer", "blank") and \
                    self.L[ln.i - 1].i in self.drop and self.L[ln.i - 1].pk == ln.pk:
                cur["lines"].append(ln.n)
            else:
                cur = {"lines": [ln.n], "why": why}
                out.append(cur)
        return out

    def _front_info(self, els: list[dict]) -> dict:
        fl = [L for L in self._front_lines()]
        info = {"title": None, "writer": None, "version_raw": None, "date": None}
        texts = [ln.f for ln in fl]
        hdr = [self.L[i].f for i, w in sorted(self.drop.items()) if w == "page_header"][:3]
        rx_writer = re.compile(r"(?i)^(?:각본\s*/?\s*감독|각본|각색|시나리오|극본|감독|작가|글|written\s+by|screenplay\s+by|"
                               r"teleplay\s+by|story\s+by|script\s+by|by|scénario|un\s+film\s+de)\s*[:：]?\s*(.*)$")
        rx_version = re.compile(r"(?i)(ver(?:sion)?\.?\s*[\d.]+[\w.]*(?:\s*최종)?|제본고|촬영\s?본|촬영\s?대본|최종\s?고|최종\s?본|"
                                r"초\s?고|수정\s?고|\d+\s?고|\d+(?:st|nd|rd|th)\s+draft|(?:first|second|third|final|revised|"
                                r"production|shooting|white|blue|pink|yellow|green|goldenrod|buff|salmon|cherry)\s+"
                                r"(?:draft|script|revisions?|pages)|shooting\s+script|draft|revisions?)")
        rx_date = re.compile(r"((?:19|20)\d\d\s*[.\-/년]\s*\d{1,2}\s*[.\-/월]\s*\d{1,2}\s*일?|(?:19|20)\d\d\s*년\s*\d{1,2}\s*월|"
                             r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
                             r"\s+\d{1,2},?\s+(?:19|20)\d\d|\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|"
                             r"June|July|August|September|October|November|December)\s+(?:19|20)\d\d|"
                             r"\d{1,2}/\d{1,2}/(?:19|20)?\d\d)", re.I)
        for k, t in enumerate(texts):
            if info["title"] is None and T.letters(t) >= 2 and not rx_writer.match(t) and not rx_version.fullmatch(t) \
                    and not re.search(r"(?i)for your consideration|confidential|copyright|©|all rights", t):
                tt = re.sub(r"^(?:제목|TITLE)\s*[:：]\s*", "", t, flags=re.I).strip()
                if len(tt) <= 60:
                    info["title"] = tt
            m = rx_writer.match(t)
            if m and info["writer"] is None:
                w = m.group(1).strip()
                if not w and k + 1 < len(texts):
                    w = texts[k + 1]
                    if k + 2 < len(texts) and len(texts[k + 2]) <= 30 and not rx_writer.match(texts[k + 2]) \
                            and not rx_date.search(texts[k + 2]) and T.letters(texts[k + 2]) >= 3:
                        w = w + ", " + texts[k + 2]
                if w and len(w) <= 80:
                    info["writer"] = w
        for t in texts + hdr:
            if info["version_raw"] is None:
                m = rx_version.search(t)
                if m and len(t) <= 80:
                    info["version_raw"] = m.group(0).strip()
            if info["date"] is None:
                m = rx_date.search(t)
                if m:
                    info["date"] = m.group(0).strip()
        return info

    def _front_lines(self):
        return [self.L[i] for i in self.active[: self.fe]]

    def _quality(self, els: list[dict], dropped: list[dict]) -> dict:
        total = len(self.rows)
        nd = sum(len(d["lines"]) for d in dropped)
        body = max(0, total - nd)
        by = Counter()
        for o in els:
            by[o["type"]] += len(o["lines"])
        nunk = by["unknown"]
        typed = body - nunk
        dl = [o for o in els if o["type"] == "dialogue"]
        nospk = sum(1 for o in dl if not o["speakers"])
        nempty = sum(1 for o in dl if not (o.get("text") or "").strip())
        ninf = sum(1 for o in dl if o.get("speaker_inferred"))
        confs = [ln.conf for ln in self.L if ln.conf is not None]
        tr = typed / body if body else 0.0
        dr = by["dialogue"] / body if body else 0.0
        ur = nunk / body if body else 0.0
        sf = nospk / len(dl) if dl else 0.0
        ef = nempty / len(dl) if dl else 0.0
        score = tr * min(1.0, dr / 0.25) * (1.0 - min(1.0, sf)) * (1.0 - min(1.0, ef))
        if confs:
            score *= min(1.0, statistics.mean(confs) / 0.85)
        notes = list(self.notes)
        notes.append("요소 줄 수: " + ", ".join(f"{k} {v}" for k, v in by.most_common()) + f"; dropped {nd}")
        if confs:
            notes.append(f"OCR 평균 conf {statistics.mean(confs):.3f}")
        return {"lines": total, "typed_ratio": round(tr, 4), "dialogue_ratio": round(dr, 4),
                "unknown_ratio": round(ur, 4), "speakerless_dialogue": nospk, "empty_dialogue": nempty,
                "inferred_speaker": ninf, "score": round(max(0.0, min(1.0, score)), 4), "notes": notes}


def _shot_slug(rest: str) -> bool:
    """'SHOT - PODIUM of AMPHITHEATRE', 'CLOSEUP - BEN', 'OMIT': 번호⇥ 뒤 영어 대문자 장면 설명(50자 이하, 문장 아님)."""
    r = rest.strip()
    if any(T.is_hangul(c) for c in r):
        return False
    letters = [c for c in r if c.isascii() and c.isalpha()]
    if not r or not r[:1].isascii() or not r[:1].isalpha() or len(r) > 50 or r.endswith((".", "!", "?", ",", ":")) \
            or T.match_transition(T.flat(r)) or re.match(r"(?i)CUT\b", r):
        return False
    return len(letters) >= 3 and sum(1 for c in letters if c.isupper()) >= 0.7 * len(letters)


def _edit1(a: str, b: str) -> bool:
    if a == b:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) == 1
    if len(a) < len(b):
        a, b = b, a
    for k in range(len(a)):
        if a[:k] + a[k + 1:] == b:
            return True
    return False


# ================================================================ 실패 대비·바깥 함수

def fallback_script(doc_id: str, rows: list[dict], err: str) -> dict:
    """파싱이 예외로 끝난 문서: 모든 줄을 unknown 요소 하나씩으로."""
    els = [{"id": f"E{k:04d}", "type": "unknown", "scene": None, "text": util.nfc(r.get("text") or ""), "lines": [r["n"]]}
           for k, r in enumerate(rows, 1)]
    return {
        "schema": "sink.script/1", "doc_id": doc_id, "parser": {"name": PARSER_NAME, "version": VERSION},
        "lang": "unknown", "front": {"title": None, "writer": None, "version_raw": None, "date": None},
        "people": [], "scenes": [], "elements": els, "dropped": [],
        "quality": {"lines": len(rows), "typed_ratio": 0.0, "dialogue_ratio": 0.0, "unknown_ratio": 1.0 if rows else 0.0,
                    "speakerless_dialogue": 0, "score": 0.0, "notes": [f"파서 예외로 모든 줄을 unknown 으로 냄: {err}"[:500]]},
    }


def parse_rows(doc_id: str, rows: list[dict]) -> dict:
    """줄 행 목록(머리 행 제외) -> script.json dict. 예외가 나면 모든 줄을 unknown 으로."""
    try:
        return DocParser(doc_id, rows).parse()
    except Exception as e:  # noqa: BLE001 - 문서 하나 실패가 전체를 멈추지 않게
        tb = traceback.format_exc(limit=3).strip().splitlines()[-1]
        return fallback_script(doc_id, rows, f"{type(e).__name__}: {e} | {tb}")


def problems_for(script: dict, doc: dict | None = None) -> list[dict]:
    q = script["quality"]
    doc_id = script["doc_id"]
    film = (doc or {}).get("film_id") or None
    path = f"docs/{doc_id}/script.json"
    if q["lines"] == 0:
        return []
    out = []
    base = {"level": "doc", "doc_id": doc_id, "film_id": film, "item_id": None, "path": path}
    if any(n.startswith("파서 예외") for n in q.get("notes", [])):
        out.append(dict(base, problem="other", score=0, detail=q["notes"][0][:300]))
        return out
    if q["typed_ratio"] < TH_TYPED:
        out.append(dict(base, problem="extract_broken", score=q["typed_ratio"],
                        detail=f"종류를 정하지 못한 줄이 많음: typed_ratio {q['typed_ratio']:.2f} < {TH_TYPED}"))
    if q["dialogue_ratio"] < TH_DIALOGUE:
        out.append(dict(base, problem="parse_low_dialogue", score=q["dialogue_ratio"],
                        detail=f"대사 줄 비율이 낮음: dialogue_ratio {q['dialogue_ratio']:.2f} < {TH_DIALOGUE}"))
    nd = sum(1 for e in script["elements"] if e["type"] == "dialogue")
    if nd and q["speakerless_dialogue"] / nd > TH_SPEAKERLESS:
        out.append(dict(base, problem="speaker_missing", score=round(q["speakerless_dialogue"] / nd, 4),
                        detail=f"화자 없는 대사 {q['speakerless_dialogue']}/{nd} > {TH_SPEAKERLESS:.0%}"))
    ne = q.get("empty_dialogue", 0)
    if nd >= MIN_DIALOGUE_FOR_RATIO and ne / nd >= TH_EMPTY_DIALOGUE:
        out.append(dict(base, problem="parse_low_dialogue", score=round(1 - ne / nd, 4),
                        detail=f"말 없는 대사(이름 줄만 잡힘) {ne}/{nd} >= {TH_EMPTY_DIALOGUE:.0%}: "
                               "말이 지문으로 갔을 수 있음(대사 모양을 잘못 읽음)"))
    ni = q.get("inferred_speaker", 0)
    if nd >= MIN_DIALOGUE_FOR_RATIO and ni / nd > TH_INFERRED:
        out.append(dict(base, problem="speaker_missing", score=round(1 - ni / nd, 4),
                        detail=f"앞 대사에서 이어받은 추정 화자 {ni}/{nd} > {TH_INFERRED:.0%}(speaker_inferred, 틀릴 수 있음)"))
    return out


def lines_path(doc_id: str):
    return paths.DOCS / doc_id / "lines.jsonl"


def parse_doc(doc: dict, force: bool = False) -> tuple[str, dict | None]:
    doc_id = doc["doc_id"]
    lp = lines_path(doc_id)
    out = paths.DOCS / doc_id / "script.json"
    fp = {"version": VERSION, "lines_sha1": util.sha1_file(lp)}
    if not force and out.exists() and util.stage_fp(STAGE, doc_id) == fp:
        return "skip", None
    rows = util.read_jsonl(lp)
    head, rows = (rows[0], rows[1:]) if rows and "schema" in rows[0] else ({}, rows)
    script = parse_rows(doc_id, rows)
    util.write_json(out, script)
    util.set_problems(STAGE, doc_id, problems_for(script, doc))
    util.set_stage_fp(STAGE, doc_id, fp)
    return ("fallback" if script["lang"] == "unknown" and script["quality"]["unknown_ratio"] == 1.0 and rows
            else "ok"), script


def select_docs(args) -> list[dict]:
    from . import catalog

    docs = [d for d in catalog.load() if d.get("status") != "excluded" and lines_path(d["doc_id"]).exists()]
    if getattr(args, "docs", None):
        want = set(args.docs.split(","))
        docs = [d for d in docs if d["doc_id"] in want]
    if getattr(args, "limit", None):
        docs = docs[: args.limit]
    return docs


def run(args) -> int:
    docs = select_docs(args)
    counts = Counter()
    t0 = time.time()
    slow = []
    for k, d in enumerate(docs, 1):
        t = time.time()
        try:
            st, _ = parse_doc(d, force=getattr(args, "force", False))
        except Exception as e:  # noqa: BLE001
            st = "error"
            print(f"  ! {d['doc_id']}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        dt = time.time() - t
        if dt > 1.0:
            slow.append((d["doc_id"], round(dt, 2)))
        counts[st] += 1
        if st not in ("skip",) and (k % 50 == 0 or st != "ok"):
            print(f"[{k}/{len(docs)}] {d['doc_id']} {st}", flush=True)
    print(f"parse: {dict(counts)} ({time.time() - t0:.1f}초)" + (f", 1초 넘은 문서 {slow[:10]}" if slow else ""), flush=True)
    return 1 if counts["error"] else 0
