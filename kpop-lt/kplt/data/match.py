"""symbol.json 의 줄 지문으로 가사 원문에서 사람이 맞춘 줄·마디를 되살린다 (저자 match.py 의 역할).

줄 지문 = (단어 첫 글자들, 첫 단어, 끝 단어, 공백 뺀 글자 수). 글은 저자 process_en_text / process_kr_text 와
같은 방법으로 정리한다 (소문자, 영숫자·작은따옴표·한글만, 줄바꿈은 공백).

저자 match.py 와 다른 점 (README 에 표시)
- 저자 코드에는 버그가 있다 (`item.split("last")` 가 변수 대신 글자 "last" 로 자름,
  `for candidate, initial in enumerate(zip(...))` 로 첫 글자 비교가 늘 실패). 그래서 같은 규칙을
  버그 없이 다시 짰다: 단어 구간의 첫 글자·첫 단어·끝 단어·글자 수 네 가지가 모두 맞는 구간을 찾는다.
- 같은 지문이 여러 번 나오면 바로 앞 줄 다음 위치부터 찾는다 (후렴 반복).
- 단어 구간으로 못 찾으면 저자처럼 공백을 뺀 글자열에서 첫 단어로 시작하고 끝 단어로 끝나는 같은 길이 구간을 찾는다.
  저자는 영어를 wordninja 로 다시 띄었지만 여기서는 원래 띄어쓰기를 되살린다.
- 저자처럼 한쪽이라도 빈 줄은 쌍에서 뺀다.
- 둥근 따옴표·키릴 글자는 저자처럼 지운 판과 정리한 판(text.normalize_text) 두 가지로 찾는다 (clean 설명).
- 저자가 준 비공식 영어 번역 파일(processed_lyrics)에는 [Verse] 머리줄이 섞여 있어 저자 방식(빈 줄·줄바꿈으로
  자르기)으로는 영어 줄이 한 칸씩 밀린다. 이 파일도 지문으로 줄을 되살린다.
- 한국어와 영어 지문의 줄 수가 다른 마디(480번 곡 한 곳)는 짝을 알 수 없어 뺀다.
"""
from __future__ import annotations

import json
import re
import unicodedata

from ..config import DATA, RAW, VENDOR
from ..text import SECTION_HEADER, normalize_text


def _raw_tokens(text: str) -> list[str]:
    """NFC, [Verse] 머리줄 빼고, 모든 공백(특수 공백 포함)으로 나눈 원래 토큰."""
    text = unicodedata.normalize("NFC", text)
    lines = [l for l in text.splitlines() if not SECTION_HEADER.match(l)]
    return " ".join(lines).split()


def _clean_token(tok: str, normalized: bool) -> str:
    """저자 process_kr_text 규칙 (소문자, 영숫자·한글·곧은 따옴표만).

    [저자와 다름] 영어판에도 한글을 남긴다: symbol.json 영어 지문에 한글 단어가 있다 (예: 008 '빨리빨리').
    저자 규칙은 둥근 따옴표(’)와 키릴 글자를 지운다 (i’m -> im). 지금 Genius 글은 바뀐 곳이 있어
    normalized=True 판(normalize_text: 곧은 따옴표, 라틴 글자)도 함께 써 본다 (MultiMatcher).
    """
    if normalized:
        tok = normalize_text(tok)
    return re.sub(r"[^0-9a-zA-Z가-힣']", "", tok).lower()


def clean(text: str, lang: str, curly_to_straight: bool = False) -> str:
    toks = (_clean_token(t, curly_to_straight) for t in _raw_tokens(text))
    return " ".join(t for t in toks if t)


def _nospace(s: str) -> str:
    return re.sub(r"\s", "", s)


class Matcher:
    def __init__(self, text: str, lang: str, curly_to_straight: bool = False, raw_tokens=None):
        self.lang = lang
        raw = _raw_tokens(text) if raw_tokens is None else raw_tokens
        pairs = [(i, _clean_token(t, curly_to_straight)) for i, t in enumerate(raw)]
        pairs = [(i, t) for i, t in pairs if t]
        self.orig = [i for i, _ in pairs]          # 원래 토큰 번호 (두 판을 잇는 데 씀)
        self.tokens = [t for _, t in pairs]
        self.cleaned = " ".join(self.tokens)
        self.lens = [len(t) for t in self.tokens]
        self.first_idx: dict[str, list[int]] = {}
        for i, t in enumerate(self.tokens):
            self.first_idx.setdefault(t, []).append(i)
        # 공백 없는 글자열과 원래 토큰 위치
        self.ns_chars: list[str] = []
        self.ns_tok: list[int] = []
        for ti, t in enumerate(self.tokens):
            for ch in t:
                self.ns_chars.append(ch)
                self.ns_tok.append(ti)
        self.ns = "".join(self.ns_chars)
        self.pos = 0
        self.last_span = None

    def _span_ok(self, i: int, sym) -> bool:
        initials, first, last, length = sym
        n = len(initials)
        if i + n > len(self.tokens):
            return False
        span = self.tokens[i:i + n]
        return (span[-1] == last and sum(self.lens[i:i + n]) == length
                and "".join(t[0] for t in span) == initials)

    def find(self, sym) -> tuple[str, str]:
        """(되살린 줄, 방법) 방법: exact / fallback / ''(실패)."""
        initials, first, last, length = sym
        self.last_span = None
        if len(initials) <= 2:
            # 저자 match.py 와 같이: 한두 단어 줄은 지문만으로 줄이 정해진다
            line = first if len(initials) == 1 else first + " " + last
            return re.sub(r"\s+", " ", line).strip(), "symbol"
        if re.search(r"\s", first + last):
            # 저자 원문에 특수 공백이 있던 줄: 단어 수가 달라 단어 구간으로는 못 찾으므로 글자열로 찾는다
            return self._find_chars(_nospace(first), _nospace(last), length, None)
        n = len(initials)
        cands = self.first_idx.get(first, [])
        order = [i for i in cands if i >= self.pos] + [i for i in cands if i < self.pos]
        for i in order:
            if self._span_ok(i, sym):
                self.pos = i + n
                self.last_span = (i, i + n)
                return " ".join(self.tokens[i:i + n]), "exact"
        return self._find_chars(first, last, length, n)

    def _find_chars(self, first: str, last: str, length: int, n_words: int | None) -> tuple[str, str]:
        """공백 없는 글자열에서 first 로 시작해 last 로 끝나는 length 글자 구간 (저자 match.py 끝부분)."""
        if not first or not last:
            return "", ""
        start = 0
        hits = []
        while True:
            j = self.ns.find(first, start)
            if j < 0:
                break
            if self.ns[j:j + length].endswith(last) and j + length <= len(self.ns):
                hits.append(j)
            start = j + 1
        if hits:
            after = [j for j in hits if self.ns_tok[j] >= self.pos]
            j = (after or hits)[0]
            t0, t1 = self.ns_tok[j], self.ns_tok[j + length - 1]
            # 원래 띄어쓰기 되살리기: 구간이 토큰 경계와 맞으면 토큰을 잇고, 아니면 글자열 그대로
            k0 = sum(self.lens[:t0])
            k1 = sum(self.lens[:t1 + 1])
            glued = self.lang == "en" and n_words is not None and (t1 + 1 - t0) < n_words
            if k0 == j and k1 == j + length and not glued:
                self.pos = t1 + 1
                self.last_span = (t0, t1 + 1)
                return " ".join(self.tokens[t0:t1 + 1]), "fallback"
            self.pos = t1 + 1
            piece = self.ns[j:j + length]
            if self.lang == "en":
                # 저자처럼 띄어쓰기가 안 맞는 영어 구간은 wordninja 로 다시 띄운다 (붙은 채면 음절 수가 틀림)
                import wordninja
                piece = " ".join(wordninja.split(piece)) or piece
            return piece, "fallback"
        return "", ""


class MultiMatcher:
    """저자 규칙 판(A)과 정리한 판(B)에서 차례로 찾는다 (단어 구간 먼저, 다음 글자열).

    A 에서 찾은 줄도 두 판의 단어 수가 같으면 B 의 같은 단어 구간으로 돌려준다
    (저자 규칙은 키릴 글자를 지워 "thе way" 가 "th way" 가 되므로). 두 판은 원래 토큰 번호로 잇는다.
    """

    def __init__(self, text: str, lang: str):
        raw = _raw_tokens(text)
        self.a = Matcher(text, lang, False, raw)
        self.b = Matcher(text, lang, True, raw)

    def _from_b(self, line, span):
        """A 판 단어 구간을 원래 토큰 번호로 바꿔 B 판의 같은 토큰들을 돌려준다."""
        if span is None:
            return line
        import bisect
        o0, o1 = self.a.orig[span[0]], self.a.orig[span[1] - 1]
        j0 = bisect.bisect_left(self.b.orig, o0)
        j1 = bisect.bisect_right(self.b.orig, o1)
        return " ".join(self.b.tokens[j0:j1]) or line

    def find(self, sym):
        best = ("", "")
        for m in (self.a, self.b):
            line, how = m.find(sym)
            if m is self.a:
                line = self._from_b(line, m.last_span)
            if how in ("exact", "symbol"):
                return line, how
            if how == "fallback" and best[1] == "":
                best = (line, how)
        return best


def recover(text: str, sym_sections, lang: str):
    """한 곡 한 언어. (마디별 줄 목록, 통계)."""
    m = MultiMatcher(text, lang)
    out, stat = [], {"exact": 0, "fallback": 0, "symbol": 0, "": 0}
    for sec in sym_sections:
        lines = []
        for sym in sec:
            line, how = m.find(tuple(sym))
            stat[how] += 1
            lines.append(line)
        out.append(lines)
    return out, stat


def recovery_rate(text: str, sym_sections, lang: str) -> float:
    """글 없이도 정해지는 한두 단어 줄(symbol)은 빼고 센다."""
    _, st = recover(text, sym_sections, lang)
    tot = st["exact"] + st["fallback"] + st[""]
    return (st["exact"] + st["fallback"]) / tot if tot else 0.0


def read_processed(path) -> list[list[str]]:
    """저자 match.py 처럼 빈 줄로 마디, 줄바꿈으로 줄을 나눈다 (정리하지 않음)."""
    sections = path.read_text(encoding="utf-8").split("\n\n")
    return [s.split("\n") for s in sections]


def build_lyrics_json(lyrics_dir=None, out_path=None):
    """저자 match.py 의 lyrics.json 을 만든다. {lid: [[(kr, en), ...], ...]}"""
    lyrics_dir = lyrics_dir or (RAW / "unprocessed_lyrics")
    out_path = out_path or (DATA / "lyrics.json")
    sym = json.load(open(VENDOR / "symbol.json", encoding="utf-8"))
    processed = VENDOR / "processed_lyrics"
    result, report = {}, {}
    for lid, item in sym.items():
        kr_file = lyrics_dir / f"{lid}kr.txt"
        if not kr_file.exists():
            report[lid] = "kr 없음"
            continue
        kr_lists, kr_stat = recover(kr_file.read_text(encoding="utf-8"), item["kr"], "ko")
        if (processed / f"{lid}en.txt").exists():
            en_lists, en_stat = recover((processed / f"{lid}en.txt").read_text(encoding="utf-8"), item["en"], "en")
            en_stat["processed"] = 1
        else:
            en_file = lyrics_dir / f"{lid}en.txt"
            if not en_file.exists():
                report[lid] = "en 없음"
                continue
            en_lists, en_stat = recover(en_file.read_text(encoding="utf-8"), item["en"], "en")
        aligned = [[(k, e) for k, e in zip(ks, es) if len(k) and len(e)] if len(ks) == len(es) else []
                   for ks, es in zip(kr_lists, en_lists)]
        aligned = [s for s in aligned if len(s)]
        result[lid] = aligned
        report[lid] = {"kr": kr_stat, "en": en_stat,
                       "n_sym_lines": sum(len(s) for s in item["kr"]),
                       "n_pairs": sum(len(s) for s in aligned)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    (out_path.parent / "lyrics_match_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                              encoding="utf-8")
    n_sym = sum(sum(len(s) for s in it["kr"]) for it in sym.values())
    n_pairs = sum(sum(len(s) for s in v) for v in result.values())
    print(f"곡 {len(result)}/{len(sym)}, 줄 쌍 {n_pairs}/{n_sym} ({n_pairs / n_sym:.1%})")
    return result


if __name__ == "__main__":
    build_lyrics_json()
