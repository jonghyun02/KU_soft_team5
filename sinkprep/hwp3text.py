"""HWP 3.0(한글 3.0/96/97) 파일에서 글만 뽑는다. 한컴이 못 여는 HWP3 의 convert 대체 경로.

공개 명세 '한글 3.0 파일 구조'(한글과컴퓨터)를 따랐고, 특수 문자마다 덧붙은 자료의 길이와 hchar 표는
LibreOffice hwpfilter(hwpread.cxx, hpara.cxx, hbox.cxx, hcode.cxx)의 읽는 순서와 맞췄다.

파일 구조
  서명 30 B ("HWP Document File V3.00 \\x1a\\x01\\x02\\x03\\x04\\x05")
  문서 정보 128 B: +96 암호 WORD, +124 압축 BYTE, +126 정보 블록 길이 WORD
  문서 요약 1008 B, 정보 블록(위 길이)
  -- 여기부터 압축 표시면 raw deflate(zlib -15)로 풀어서 --
  글꼴 이름: 7개 언어마다 WORD 개수 + 개수×40 B
  스타일: WORD 개수 + 개수×238 B
  문단 리스트(글자 수 0 인 문단으로 끝남), 그 뒤 추가 정보 블록(읽지 않음)

문단: 문단 정보 43 B(앞 문단 모양 사용 BYTE, 글자 수 WORD, 줄 수 WORD, 글자 모양 포함 BYTE, 기타 BYTE,
  제어 DWORD, 스타일 BYTE, 대표 글자 모양 31 B), 글자 수>0 이고 앞 모양 사용이 0 이면 문단 모양 187 B,
  줄 정보 줄 수×14 B, 글자 모양 포함이면 글자마다 BYTE(0 이면 글자 모양 31 B 가 뒤따름),
  그다음 hchar(2 B) 글자들. 32 미만은 특수 문자로 덧붙은 자료가 있고 글자 수는 _WSIZE 만큼 센다.
  상자(10: 표·글상자·수식·단추)는 칸마다, 그림(11)은 설명(캡션), 숨은 설명(15)·머리말/꼬리말(16)·
  각주(17)는 안에 문단 리스트를 또 가진다.

내놓는 것(extract_paragraphs)
  문단마다 글 한 줄(NFC, 문서 순서). 표·글상자 칸과 상자·그림 설명의 문단도 각각 따로 넣는다
  (표 칸은 칸 좌표 (y, x) 순, 설명은 칸 뒤). 상자를 품은 바깥 문단은 상자 안 문단들보다 앞에 둔다.
  빼는 것: 머리말·꼬리말(쪽마다 되풀이되는 장식), 각주, 숨은 설명(본문 흐름 밖)의 글은 구조만 읽고
  넣지 않는다. 책갈피(6)·날짜(7, 8)·자동 번호·쪽 번호(18~21)·메일 머지(22)·차례/찾아보기 표시(25, 26)
  처럼 한글이 계산해서 보여 주는 값도 넣지 않는다. 다만 개요 번호(28)는 대본에서 장면 번호로 쓰여서
  파일에 적힌 번호로 '2.' '#1.' 처럼 만들어 한글 화면처럼 문단 맨 앞에 둔다(_outline_number). 필드(5)는 덧말(종류 1)의 본문 글만
  넣는다(읽는 법은 뺌; 한컴 PDF 는 본문 뒤에 읽는 법도 찍는다).
  탭(9)은 '\\t', 묶음 빈칸(30)·고정폭 빈칸(31)은 ' ', 하이픈(24)은 '-', 겹침 글자(23)는 그 글자들.
  탭의 확인 코드 자리에 쓰레기 값이 든 문서가 있어(fm54147) 탭은 길이만 믿고 확인하지 않는다.

hchar -> 유니코드(hchar_text)
  0x20~0x7F ASCII, 0x8000 이상 한글(조합형: 완성 음절·낱자, 옛한글은 첫가끝 자모), 0x4000~ 한자
  (KS X 1001 한자 순번 4888자), 0x3400~0x37BF 특수 문자(KS X 1001 기호 행), 0x37C0~ 한컴 기호,
  0x1F00~ 일본 가나, 0x2F00~ 글머리표, 0x3013~ 선 문자(가는 선 상자 그림 글자로).
  0xA0~0x17F·0x2000~0x22FF 는 유니코드 값 그대로, 0x81~0x85 등 낱개는 _SINGLE(한컴 PDF 와 맞대어 정함).
  바꿀 수 없는 글자는 U+FFFD(HWP3 140개 문서에서 940만 자 중 86자).

망가진 곳
  구조가 어긋나면(확인 코드 불일치, 자료 끝 넘음, 깊이 초과, 알 수 없는 코드) 거기서 멈추고 그때까지
  읽은 문단(읽던 문단의 앞부분 포함)을 돌려준다(extract 가 멈춘 까닭도 준다). 다만 끝에 U+FFFD 를 품은
  문단은 어긋난 자료를 글로 읽은 것으로 보고 뺀다. 예외는 HWP3 가 아니거나 암호가 걸린 경우만(ValueError).
  예: fm55645 는 압축 자료가 본문 40% 지점(풀린 자료 242,602 B)부터 망가져(같은 글자 모양 31 B 가 3 B
  빠진 채 복사됨, 그 뒤로 구조가 맞는 문단이 없음) 그 앞 문단까지만 나온다. 한컴도 빈 PDF 를 만든다.
"""
from __future__ import annotations

import zlib
from functools import lru_cache
from pathlib import Path

from . import util
from .detect import HWP3_MAGIC

SIG_LEN, INFO_LEN, SUMMARY_LEN = 30, 128, 1008
FONT_LANGS, FONT_NAME_LEN, STYLE_LEN = 7, 40, 238
PARA_INFO_LEN, PARA_SHAPE_LEN, LINE_INFO_LEN, CHAR_SHAPE_LEN = 43, 187, 14, 31
BOX_INFO_LEN, CELL_INFO_LEN, PICTURE_INFO_LEN, LINE_BOX_LEN = 84, 27, 344, 84
MAX_DEPTH = 16
MAX_COUNT = 10000           # 글꼴·스타일·표 칸 개수가 이보다 크면 구조가 어긋난 것으로 본다
CHUNK = 1 << 16

# 특수 문자(코드 < 32)가 문단 글자 수에서 차지하는 hchar 수(LibreOffice hbox.cxx wsize).
# 덧붙은 가변 길이 자료(상자 내용, 건너뛸 블록 등)는 글자 수에 들어가지 않는다.
_WSIZE = (1, 4, 4, 4, 4, 4, 4, 42, 48, 4, 4, 4, 4, 1, 4, 4,
          4, 4, 4, 4, 4, 4, 12, 5, 3, 3, 123, 4, 32, 4, 2, 2)
_SKIP_BLOCK = {0, 1, 2, 3, 4, 12, 27, 29}     # DWORD 길이 + 확인 코드 + 그 길이만큼 자료
# 확인 코드 앞까지의 고정 길이 자료(코드 자신 뒤부터): 날짜 형식·날짜·번호류·메일 머지·하이픈·
# 차례·찾아보기·개요·빈칸
_FIXED = {7: 80, 8: 92, 18: 4, 19: 4, 20: 4, 21: 4, 22: 20, 24: 2, 25: 2, 26: 242, 30: 0, 31: 0}
_TEXT_OF = {24: "-", 30: " ", 31: " "}
_MUTED = {15, 16, 17}       # 숨은 설명, 머리말/꼬리말, 각주: 글을 넣지 않음

FFFD = "\ufffd"


# ============================================================ hchar -> 유니코드

def _ks(row: int, col: int) -> str | None:
    """KS X 1001(EUC-KR) 두 바이트 -> 글자."""
    try:
        return bytes((row, col)).decode("euc_kr")
    except UnicodeDecodeError:
        return None


# 조합형 5비트 값 -> 첫가끝 자모(옛한글용, hcode.cxx). 0 은 없음.
_CHO = (0x111E, 0, 0x1100, 0x1101, 0x1102, 0x1103, 0x1104, 0x1105,
        0x1106, 0x1107, 0x1108, 0x1109, 0x110A, 0x110B, 0x110C, 0x110D,
        0x110E, 0x110F, 0x1110, 0x1111, 0x1112, 0x1120, 0x1121, 0x1127,
        0x112B, 0x112D, 0x112F, 0x1132, 0x1136, 0x1140, 0x114C, 0x1158)
_JUNG = (0, 0, 0, 0x1161, 0x1162, 0x1163, 0x1164, 0x1165,
         0, 0, 0x1166, 0x1167, 0x1168, 0x1169, 0x116A, 0x116B,
         0, 0x1188, 0x116C, 0x116D, 0x116E, 0x116F, 0x1170, 0x1171,
         0x1191, 0x1194, 0x1172, 0x1173, 0x1174, 0x1175, 0x119E, 0x11A1)
_JONG = (0x11D9, 0, 0x11A8, 0x11A9, 0x11AA, 0x11AB, 0x11AC, 0x11AD,
         0x11AE, 0x11AF, 0x11B0, 0x11B1, 0x11B2, 0x11B3, 0x11B4, 0x11B5,
         0x11B6, 0x11B7, 0x11E6, 0x11B8, 0x11B9, 0x11BA, 0x11BB, 0x11BC,
         0x11BD, 0x11BE, 0x11BF, 0x11C0, 0x11C1, 0x11C2, 0x11EB, 0x11F0)

# 0x37C0~0x37FF 한컴 기호 -> KS X 1001 (hcode.cxx tblhhtg_ks, 0 은 대응 없음). 앞 6개는 '한글과컴퓨터'.
_HNC_TG = (0xC7D1, 0xB1DB, 0xB0FA, 0xC4C4, 0xC7BB, 0xC5CD, 0, 0,
           0xA2B1, 0xA3DF, 0xA2D5, 0xA6B1, 0xA1B8, 0xA1B9, 0xA3DF, 0xA1DA,
           0xA2C6, 0xA2CC, 0xA2CB, 0, 0, 0xA6BE, 0xA6B9, 0xA6C1,
           0xA6C2, 0xA6B4, 0xA6AD, 0xA6AF, 0xA6B0, 0xA6C3, 0xA6C4, 0,
           0xA6AE, 0xA6B0, 0xA2D7, 0xA1E1, 0xA1D6, 0, 0xA6BC, 0xA6B7,
           0xA6B1, 0xA6AE, 0xA6B5, 0xA6B3, 0xA6B2, 0xA6AC, 0xA6B6, 0xA6BA,
           0xA6BF, 0xA6B8, 0xA6BD, 0xA6C5, 0xA6C6, 0xA6C8, 0xA6C7, 0xA6C0,
           0xA6BB, 0, 0, 0, 0xA1E1, 0xA1E1, 0xA1E1, 0xA1E1)


def _hangul(c: int) -> str | None:
    if c == 0xD3C5:                         # 한컴이 쓰는 '한'(아래아 자리) -> 현대 '한'
        return "한"
    try:                                    # 현대 음절·낱자(호환 자모)
        s = c.to_bytes(2, "big").decode("johab")
        if "가" <= s <= "힣" or "ㄱ" <= s <= "ㆎ":
            return s
    except UnicodeDecodeError:
        pass
    cho, jung, jong = (c >> 10) & 31, (c >> 5) & 31, c & 31
    if jung < 2:                            # 한컴 낱자 확장 영역: 표 없음
        return None
    out = []                                # 옛한글: 첫가끝 자모 나열(채움 1·2 는 뺌)
    if cho != 1:
        out.append(_CHO[cho])
    if jung > 2:
        out.append(_JUNG[jung])
    if jong != 1:
        out.append(_JONG[jong])
    if not out or 0 in out:
        return None
    return "".join(map(chr, out))


# 한컴 PDF 글자층과 맞대어 정한 낱개 대응(hcode.cxx 는 0x81~0x84 를 곧은 따옴표로 바꾸지만 한글은
# 굽은 따옴표로 보여 준다). 0x93·0x94 는 cp1252 따옴표 자리로, 한글 화면에는 빈칸으로 보인다.
_SINGLE = {0x81: "“", 0x82: "”", 0x83: "‘", 0x84: "’", 0x85: "·",
           0x93: "“", 0x94: "”", 0x1FC0: "、", 0x1FC1: "。", 0x1FC3: "々",
           0x1FC4: "ー", 0x251A: "♬", 0x2524: "♪", 0x309B: "｢", 0x309D: "｣",
           0x340D: "～"}
_BOX = "┌┬┐├┼┤└┴┘─│"                         # 선 문자 11개씩 7벌(굵기·겹선), 모두 가는 선으로


def _symbol(c: int) -> str | None:
    if c in _SINGLE:
        return _SINGLE[c]
    if 0xA0 <= c < 0x180 or 0x2000 <= c < 0x2300:   # 유니코드 값 그대로 쓰는 영역(라틴 글자, 문장 부호, 수학 기호)
        return chr(c)
    if 0x3013 <= c < 0x3013 + 77:           # 선 문자(표 테두리 글자)
        return _BOX[(c - 0x3013) % 11]
    if 0x2F00 <= c <= 0x2F6F and (c & 0x0F) < 9:    # 글머리표
        base = (0xA1E0, 0xA1DB, 0xA1DE, 0xA1E2, 0xA1E4, 0xA2B7, 0xA2B9)[min((c - 0x2F00) >> 4, 6)]
        k = base + (0 if (c & 0x0F) < 6 else 1)
        return _ks(k >> 8, k & 0xFF)
    if 0x1F00 <= c < 0x1FC0:                # 히라가나·가타카나(KS X 1001 10·11행)
        k = c - 0x1F00
        return _ks(0xAA + k // 0x60, 0xA0 + k % 0x60)
    if 0x3400 <= c < 0x37C0:                # KS X 1001 기호 1~9행, 12행(러시아 글자)
        k = c - 0x3400
        row = 0xA1 + k // 0x60
        return _ks(0xAC if row == 0xAA else row, 0xA0 + k % 0x60)
    if 0x37C0 <= c < 0x3800:
        k = _HNC_TG[c - 0x37C0]
        return _ks(k >> 8, k & 0xFF) if k else None
    return None


@lru_cache(maxsize=None)
def hchar_text(c: int) -> str:
    """hchar 하나(32 이상) -> 유니코드 글자(들). 모르면 U+FFFD."""
    if c < 0x80:
        return chr(c)
    if c >= 0x8000:
        s = _hangul(c)
    elif c >= 0x4000:                       # 한자: KS X 1001 한자 순번(4888자 = 52행×94)
        i = c - 0x4000
        s = _ks(0xCA + i // 94, 0xA1 + i % 94) if i < 4888 else None
    else:
        s = _symbol(c)
    return s or FFFD


def _outline_number(d: bytes) -> str:
    """개요 번호(28) 60 B -> 보이는 번호. 종류 2 B, 모양 1 B, 수준 1 B, 수준별 번호 7×2 B,
    사용자 모양 7×2 B, 앞뒤 꾸밈 글자 7×2×2 B(hbox.cxx Outline::GetUnicode). 숫자 모양만 그대로 내고
    로마 숫자·가나다 같은 사용자 모양도 숫자로 쓴다. 글머리표 모양과 번호 없는 개요는 빈 글."""
    def w(i: int) -> int:
        return int.from_bytes(d[i:i + 2], "little")
    shape, level = d[2], min(d[3], 6)
    if w(0) != 1:
        return ""
    nums = [str(max(1, w(4 + 2 * i))) for i in range(level + 1)]
    if shape in (1, 2):                     # 1.2.  /  1.2
        return ".".join(nums) + ("." if shape == 1 or level == 0 else "")
    if shape in (0, 128):                   # 사용자 모양: 앞 꾸밈 + 번호 + 뒤 꾸밈 (예: '#1.')
        pre, post = w(32 + 4 * level), w(34 + 4 * level)
        body = ".".join(nums) if w(18 + 2 * level) == 12 else nums[-1]
        return (hchar_text(pre) if pre > 31 else "") + body + (hchar_text(post) if post > 31 else "")
    return ""


# ============================================================ 구조 읽기

class _Stop(Exception):
    """구조가 어긋나 더 읽을 수 없음."""


class _Reader:
    def __init__(self, data: bytes):
        self.d = data
        self.i = 0
        self.out: list[str] = []
        self.mute = 0                       # >0 이면 머리말·각주 등 안: 글을 넣지 않음

    # ---------------------------------------------------------------- 바이트
    def take(self, n: int) -> bytes:
        j = self.i + n
        if j > len(self.d):
            raise _Stop(f"eof@{self.i}+{n}")
        b = self.d[self.i:j]
        self.i = j
        return b

    def skip(self, n: int) -> None:
        if self.i + n > len(self.d):
            raise _Stop(f"eof@{self.i}+{n}")
        self.i += n

    def u16(self) -> int:
        return int.from_bytes(self.take(2), "little")

    def u32(self) -> int:
        return int.from_bytes(self.take(4), "little")

    def expect(self, code: int) -> None:
        at = self.i
        if self.u16() != code:
            raise _Stop(f"ctrl{code}_mismatch@{at}")

    # ---------------------------------------------------------------- 머리
    def fonts_styles(self) -> None:
        for _ in range(FONT_LANGS):
            n = self.u16()
            if n > MAX_COUNT:
                raise _Stop(f"font_count{n}@{self.i - 2}")
            self.skip(n * FONT_NAME_LEN)
        n = self.u16()
        if n > MAX_COUNT:
            raise _Stop(f"style_count{n}@{self.i - 2}")
        self.skip(n * STYLE_LEN)

    # ---------------------------------------------------------------- 문단
    def para_list(self, depth: int) -> None:
        if depth > MAX_DEPTH:
            raise _Stop(f"too_deep@{self.i}")
        while self.para(depth):
            pass

    def para(self, depth: int) -> bool:
        info = self.take(PARA_INFO_LEN)
        reuse_shape, has_cshape = info[0], info[5]
        nch = int.from_bytes(info[1:3], "little")
        nline = int.from_bytes(info[3:5], "little")
        if nch and not reuse_shape:
            self.skip(PARA_SHAPE_LEN)
        self.skip(nline * LINE_INFO_LEN)
        if nch == 0:                        # 리스트 끝
            return False
        if has_cshape:
            for _ in range(nch):
                if self.take(1)[0] == 0:
                    self.skip(CHAR_SHAPE_LEN)
        slot = None
        if not self.mute:
            slot = len(self.out)
            self.out.append("")
        chars: list[str] = []
        lead = ""                           # 개요 번호: 한글은 코드 자리와 상관없이 문단 맨 앞에 그린다
        try:
            n = 0
            while n < nch:
                c = self.u16()
                if c == 13:                 # 문단 끝
                    break
                if c > 31:
                    chars.append(hchar_text(c))
                    n += 1
                else:
                    n += _WSIZE[c]
                    lead += self.special(c, chars, depth) or ""
        finally:
            if slot is not None:
                text = "".join(chars)
                if lead:
                    text = lead + ("" if text[:1].isspace() else " ") + text
                self.out[slot] = text
        return True

    def special(self, c: int, chars: list[str], depth: int) -> str | None:
        """특수 문자 하나를 읽는다. 글은 chars 에 넣고, 개요 번호면 그 번호를 돌려준다."""
        if c in _SKIP_BLOCK or c == 6:      # 건너뛸 블록, 책갈피: DWORD 길이 + 확인 코드 + 자료
            size = self.u32()
            self.expect(c)
            self.skip(size)
        elif c == 5:                        # 필드: DWORD + 확인 코드 + 종류 2 B + 28 B + 길이 4개 + 그 자료
            self.skip(4)
            self.expect(c)
            kind = self.take(2)[0]
            self.skip(28)
            lens = [self.u32() for _ in range(4)]
            strs = [self.take(n) for n in lens[:3]]
            self.skip(lens[3])
            if kind == 1:                   # 덧말: 둘째 글이 본문, 셋째가 읽는 법(fm54367 일본어) -> 본문만
                for j in range(0, len(strs[1]) - 1, 2):
                    h = int.from_bytes(strs[1][j:j + 2], "little")
                    if h == 0:
                        break
                    if h > 31:
                        chars.append(hchar_text(h))
        elif c == 9:                        # 탭: 너비·채움 + 확인 코드. 확인 코드 자리에 쓰레기 값을
            self.skip(6)                    # 남기는 문서가 있어(fm54147) 길이만 믿는다(크기 고정이라 안전).
            chars.append("\t")
        elif c == 28:                       # 개요 번호(대본의 장면 번호로 쓰임)
            num = _outline_number(self.take(60))
            self.expect(c)
            return num
        elif c in _FIXED:
            self.skip(_FIXED[c])
            self.expect(c)
            if c in _TEXT_OF:
                chars.append(_TEXT_OF[c])
        elif c == 23:                       # 겹침 글자: hchar 3개
            parts = [self.u16() for _ in range(3)]
            self.expect(c)
            chars.extend(hchar_text(p) for p in parts if p > 31)
        elif c == 10:
            self.box(depth)
        elif c == 11:
            self.skip(4)
            self.expect(c)
            follow = self.u32()
            self.skip(PICTURE_INFO_LEN)
            self.skip(follow)
            self.para_list(depth + 1)       # 설명
        elif c == 14:
            self.skip(4)
            self.expect(c)
            self.skip(LINE_BOX_LEN)
        elif c in _MUTED:
            self.skip(4)
            self.expect(c)
            self.skip({15: 8, 16: 10, 17: 14}[c])
            self.mute += 1
            try:
                self.para_list(depth + 1)
            finally:
                self.mute -= 1
        else:
            raise _Stop(f"ctrl{c}_unknown@{self.i - 2}")

    def box(self, depth: int) -> None:
        """상자(표·글상자·수식·단추): 칸마다 문단 리스트, 그 뒤 설명 문단 리스트."""
        self.skip(4)
        self.expect(10)
        info = self.take(BOX_INFO_LEN)
        ncell = int.from_bytes(info[80:82], "little")
        if not 0 < ncell <= MAX_COUNT:
            raise _Stop(f"box_cells{ncell}@{self.i - BOX_INFO_LEN}")
        pos = []
        for _ in range(ncell):
            cell = self.take(CELL_INFO_LEN)
            pos.append((int.from_bytes(cell[6:8], "little", signed=True),
                        int.from_bytes(cell[4:6], "little", signed=True)))
        base, bounds = len(self.out), []
        for _ in range(ncell):              # 도중에 멈추면 읽은 칸은 파일 순서 그대로 남는다
            start = len(self.out)
            self.para_list(depth + 1)
            bounds.append((start, len(self.out)))
        order = sorted(range(ncell), key=lambda k: pos[k])     # 칸 좌표 (y, x) 순
        self.out[base:] = [p for k in order for p in self.out[bounds[k][0]:bounds[k][1]]]
        self.para_list(depth + 1)           # 설명


def _body(raw: bytes, compressed: bool) -> bytes:
    """압축이면 raw deflate 를 풀 수 있는 만큼 푼다(끝에 남는 자료·망가진 꼬리는 무시)."""
    if not compressed:
        return raw
    d = zlib.decompressobj(-15)
    out = []
    try:
        for i in range(0, len(raw), CHUNK):
            out.append(d.decompress(raw[i:i + CHUNK]))
            if d.eof:
                break
        out.append(d.flush())
    except zlib.error:
        pass
    return b"".join(out)


def extract(path: Path | str) -> tuple[list[str], str | None]:
    """(문단 글 목록, 멈춘 까닭). 끝까지 읽었으면 까닭은 None.

    HWP3 서명이 아니면 ValueError('not_hwp3'), 암호가 걸렸으면 ValueError('hwp3_password').
    """
    data = Path(path).read_bytes()
    if not data.startswith(HWP3_MAGIC):
        raise ValueError("not_hwp3")
    info = data[SIG_LEN:SIG_LEN + INFO_LEN]
    if len(info) < INFO_LEN:
        return [], "truncated_header"
    if int.from_bytes(info[96:98], "little"):
        raise ValueError("hwp3_password")
    start = SIG_LEN + INFO_LEN + SUMMARY_LEN + int.from_bytes(info[126:128], "little")
    r = _Reader(_body(data[start:], bool(info[124])))
    stop = None
    try:
        r.fonts_styles()
        r.para_list(0)
    except _Stop as e:
        stop = str(e)
    except (IndexError, ValueError) as e:   # 방어: 어떤 입력에도 죽지 않는다
        stop = f"{type(e).__name__}:{e}@{r.i}"
    if stop:                                # 어긋난 자리 바로 앞 문단이 깨진 글자를 품으면 쓰레기로 보고 뺀다
        while r.out and FFFD in r.out[-1]:
            r.out.pop()
        stop = f"{stop};read={r.i / max(1, len(r.d)):.2f}"   # 본문 중 읽은 비율
    return [util.nfc(p) for p in r.out], stop


def extract_paragraphs(path: Path | str) -> list[str]:
    """문단 글 목록(NFC, 문서 순서). 자세한 규칙은 모듈 설명. 구조가 망가지면 읽은 데까지."""
    return extract(path)[0]
