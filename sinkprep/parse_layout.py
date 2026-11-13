"""구조 파싱의 쪽 배치 도구: 쪽 장식(머리말·꼬리말·쪽번호·이어짐 표시) 찾기와 들여쓰기 단계.

고정 기준값 대신 문서 안 분포를 본다:
- 머리말·꼬리말: 여러 쪽의 위·아래 띠에 같은 글(숫자는 # 로)이 되풀이되면 장식. OCR 오탈자는 글자만 남겨 비슷한지로 묶는다.
- 들여쓰기: x 값 봉우리(묶음)를 문서별로 찾고, 글이 가장 많이 시작하는 왼쪽 단계를 지문 여백(M)으로 둔다.
  쪽마다 여백이 조금 밀린 문서(Alien 등)는 쪽별 밀림을 찾아 맞춘다.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from . import parse_text as T


@dataclass
class Ln:
    i: int                      # rows 안 차례
    n: int
    file: int
    page: int | None
    x: float | None
    y: float | None
    raw: str                    # lines.jsonl 글 그대로
    t: str                      # 분석용(특수 공백 정리, 수정 표시 뗌, 탭 유지)
    f: str                      # 탭·공백 하나로
    src: str
    conf: float | None
    pk: tuple = ()              # 쪽 열쇠 (file, page)
    rx: float = 0.0             # 쪽 밀림을 맞춘 x
    ind: float = 0.0            # 여백 M 에서 들여쓴 정도
    cls: str = "M"              # L(여백 왼쪽) M(여백) I(들여씀) R(오른쪽)
    dy: float | None = None     # 같은 쪽 앞 줄과의 y 간격(줄 간격 중앙값 단위)
    pg_first: bool = False
    pg_last: bool = False
    extra: dict = field(default_factory=dict)


def make_lines(rows: list[dict]) -> list[Ln]:
    out: list[Ln] = []
    for i, r in enumerate(rows):
        raw = r.get("text") or ""
        t = T.strip_rev(T.norm(raw)).strip()
        f = T.flat(t)
        page = r.get("page")
        out.append(Ln(i=i, n=r["n"], file=r.get("file") or 0, page=page, x=r.get("x"), y=r.get("y"), raw=raw,
                      t=t, f=f, src=r.get("src") or "", conf=r.get("conf"),
                      pk=(r.get("file") or 0, page if page is not None else -1)))
    by_pk: dict[tuple, list[Ln]] = defaultdict(list)
    for ln in out:
        by_pk[ln.pk].append(ln)
    for pk, ls in by_pk.items():
        if pk[1] == -1:
            continue
        ls[0].pg_first = True
        ls[-1].pg_last = True
    # y 간격: 같은 쪽(쪽 없는 글은 같은 파일) 앞 줄과의 차이를 문서 줄 간격 중앙값으로 나눈다
    gaps = []
    prev = None
    for ln in out:
        if prev is not None and prev.pk == ln.pk and ln.y is not None and prev.y is not None and ln.y > prev.y:
            gaps.append(ln.y - prev.y)
        prev = ln
    med = statistics.median(gaps) if gaps else None
    if med:
        # 줄 간격 중앙값은 '한 줄' 간격이다(빈 줄·문단 간격은 더 큼). 작은 쪽 25% 값으로 한 번 더 다듬는다.
        small = sorted(gaps)[: max(1, len(gaps) // 2)]
        unit = statistics.median(small) or med
        prev = None
        for ln in out:
            if prev is not None and prev.pk == ln.pk and ln.y is not None and prev.y is not None:
                ln.dy = (ln.y - prev.y) / unit if unit else None
            prev = ln
    return out


# ---------------------------------------------------------------- 쪽 장식

_RX_DIGITS = re.compile(r"\d+")
_RX_NONLET = re.compile(r"[^A-Za-z가-힣À-ÿ]")


def _key_exact(s: str) -> str:
    return _RX_DIGITS.sub("#", re.sub(r"\s+", "", s)).casefold()


def _key_letters(s: str) -> str:
    return _RX_NONLET.sub("", s).casefold()


def _is_namey_upper(s: str) -> bool:
    w = s.split()
    return 0 < len(w) <= 3 and s.upper() == s and not any(c.isdigit() for c in s) and any(c.isalpha() for c in s)


def find_decor(L: list[Ln]) -> dict[int, str]:
    """줄 차례 -> dropped 이유. (글 모양으로 정해지는 것 + 쪽 띠 되풀이)"""
    drop: dict[int, str] = {}
    for ln in L:
        f = ln.f
        if not f:
            drop[ln.i] = "blank"
        elif T.RX_REVISION.match(f):
            drop[ln.i] = "revision_mark"
        elif T.RX_CONT_MARKER.match(f):
            drop[ln.i] = "cont_marker"
        elif T.RX_CONTINUED.match(f) and re.search(r"CONT|SUITE", f, re.I):
            drop[ln.i] = "continued"
        elif T.RX_PAGE_NUM.match(f) and (ln.page is not None or f[:1] in "-–—"):
            drop[ln.i] = "page_number"
        elif ln.page is not None and T.RX_BARE_NUM.match(f) and f.isdigit() and (ln.pg_first or ln.pg_last) \
                and ln.x is not None and (ln.x >= 0.38 or ln.pg_last):
            drop[ln.i] = "page_number"
        elif _RX_TIMECODE.match(f) or _RX_SRT_ARROW.match(f):
            drop[ln.i] = "timecode"                            # '01:10:22.00', '00:00:20,000 --> 00:00:25,000'
        elif "\t" in ln.t and re.fullmatch(r"[-–—]\s*\d{1,3}\s*[-–—]", ln.t.split("\t", 1)[0].strip()) \
                and len(ln.t.split("\t", 1)[1]) <= 12 and ln.page is not None:
            drop[ln.i] = "page_number"                         # '- 37 - ⇥ 조명부'
        elif _RX_PAGE_RULE.match(f):
            drop[ln.i] = "page_number"                         # '-------------- page 28' (글로 옮긴 대본의 쪽 나눔 줄)
        elif ln.i and drop.get(ln.i - 1) == "page_number" and _RX_PAGE_RULE.match(L[ln.i - 1].f) \
                and re.fullmatch(r"\d{1,3}\.?", f):
            drop[ln.i] = "page_number"                         # 쪽 나눔 줄 다음 줄의 쪽 번호 '28'
    # 자막 파일(SRT·WebVTT)을 옮긴 문서: 화살표 시각 줄이 20개 넘으면 그 바로 앞 맨 숫자 줄은 자막 번호
    arrows = [ln.i for ln in L if drop.get(ln.i) == "timecode" and _RX_SRT_ARROW.match(ln.f)]
    if len(arrows) >= 20:
        for i in arrows:
            if i and L[i - 1].f.isdigit() and len(L[i - 1].f) <= 5:
                drop[i - 1] = "subtitle_cue"
    _band_repeats(L, drop)
    _digit_repeats(L, drop)
    return drop


_RX_PAGE_RULE = re.compile(r"^[-=_]{10,}\s*(?:page|pg\.?|p\.)\s*\d{0,4}\s*$", re.I)
_RX_SRT_ARROW = re.compile(r"^\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?\s*-->\s*\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?"
                           r"(?:\s+\S.*)?$")
_RX_TIMECODE = re.compile(r"^\d{1,2}:\d{2}:\d{2}(?:[.:,;]\d{1,3})?$")


def _digit_repeats(L: list[Ln], drop: dict[int, str]) -> None:
    """숫자가 든 같은 글(숫자는 # 로)이 쪽의 30% 넘게 되풀이되면 쪽 위치와 상관없이 장식(스캔 두 쪽 한 장 등)."""
    pages = {ln.pk for ln in L if ln.page is not None}
    P = len(pages)
    if P < 6:
        return
    by: dict[str, list[Ln]] = defaultdict(list)
    for ln in L:
        if ln.i in drop or ln.page is None or not any(c.isdigit() for c in ln.f) or len(ln.f) > 60:
            continue
        by[_key_exact(ln.f)].append(ln)
    for k, ls in by.items():
        pg = {ln.pk for ln in ls}
        if len(pg) >= 0.3 * P and len(ls) <= 2 * len(pg) and T.letters(k) >= 1:
            for ln in ls:
                drop[ln.i] = "page_header" if (ln.y or 0) < 0.5 else "page_footer"


def _band_repeats(L: list[Ln], drop: dict[int, str]) -> None:
    pages: dict[tuple, list[Ln]] = defaultdict(list)
    for ln in L:
        if ln.page is not None:
            pages[ln.pk].append(ln)
    P = len(pages)
    if P < 4:
        return
    thr = max(3, int(0.2 * P + 0.5))
    cands: list[tuple[str, Ln]] = []
    for pk, ls in pages.items():
        live = [ln for ln in ls if ln.i not in drop or drop[ln.i] == "page_number"]
        for k, ln in enumerate(live[:3]):
            if ln.y is None or ln.y <= 0.16:
                cands.append(("top", ln))
        for ln in live[-3:]:
            if ln.y is None or ln.y >= 0.80:
                cands.append(("bot", ln))
    ytol = 0.02 if Counter(ln.src for ln in L).most_common(1)[0][0] == "ocr" else 0.012
    # 1) 숫자만 # 로 바꾼 같은 글. 문서 다른 곳(쪽 가운데)에도 자주 나오는 글(인물 이름 등)은 장식이 아니다.
    pages_of: dict[tuple[str, str], set] = defaultdict(set)
    ys_of: dict[tuple[str, str], list] = defaultdict(list)
    for pos, ln in cands:
        k = _key_exact(ln.f)
        if k:
            pages_of[(pos, k)].add(ln.pk)
            ys_of[(pos, k)].append(ln.y if ln.y is not None else 0.0)
    total_of = Counter(_key_exact(ln.f) for ln in L)
    hit: set[int] = set()
    for pos, ln in cands:
        k = _key_exact(ln.f)
        if not k:
            continue
        cnt = len(pages_of.get((pos, k), ()))
        need = thr if not _is_namey_upper(ln.f) else max(thr, int(0.5 * P))
        if cnt < need or total_of[k] > 1.5 * len(ys_of[(pos, k)]) + 2:
            continue
        ys = ys_of[(pos, k)]
        med = statistics.median(ys)
        if statistics.median([abs(y - med) for y in ys]) > ytol:
            continue
        hit.add(ln.i)
    # 2) OCR 오탈자: 글자만 남겨 글자 두 개씩 묶음(bigram)이 많이 겹치면 한 묶음. 글자층이 있는 문서는 1) 로 충분하다.
    if ytol < 0.02:
        rest = []
    else:
        rest = [(pos, ln) for pos, ln in cands if ln.i not in hit and ln.i not in drop and len(_key_letters(ln.f)) >= 4]
    clusters: list[dict] = []
    for pos, ln in rest:
        k = _key_letters(ln.f)
        bg = {k[j:j + 2] for j in range(len(k) - 1)}
        for c in clusters:
            if c["pos"] != pos or abs(len(c["key"]) - len(k)) > max(4, len(k) // 2):
                continue
            inter = len(bg & c["bg"])
            if inter >= 0.6 * max(len(bg), len(c["bg"])):
                c["lines"].append(ln)
                c["pages"].add(ln.pk)
                break
        else:
            clusters.append({"pos": pos, "key": k, "bg": bg, "lines": [ln], "pages": {ln.pk}})
    for c in clusters:
        need = thr if not _is_namey_upper(c["lines"][0].f) else max(thr, int(0.5 * P))
        if len(c["pages"]) < need:
            continue
        ys = [ln.y if ln.y is not None else 0.0 for ln in c["lines"]]
        med = statistics.median(ys)
        if statistics.median([abs(y - med) for y in ys]) > ytol:
            continue
        for ln in c["lines"]:
            hit.add(ln.i)
    for ln in L:
        if ln.i in hit:
            if ln.i in drop and drop[ln.i] == "page_number":
                continue
            top = ln.y is not None and ln.y < 0.5
            drop[ln.i] = "page_header" if top else "page_footer"


# ---------------------------------------------------------------- 들여쓰기 단계

def _tol(L: list[Ln]) -> float:
    srcs = Counter(ln.src for ln in L)
    s = srcs.most_common(1)[0][0] if srcs else ""
    if s == "ocr":
        return 0.014
    if s == "txt":
        return 0.004
    return 0.008


def clusters_x(xs: list[float], tol: float) -> list[tuple[float, int]]:
    xs = sorted(xs)
    out: list[list[float]] = []
    for x in xs:
        if out and x - out[-1][-1] <= tol:
            out[-1].append(x)
        else:
            out.append([x])
    return [(statistics.median(c), len(c)) for c in out]


def analyze_x(L: list[Ln], active: list[int]) -> dict:
    """여백 M 과 쪽 밀림을 정하고 각 줄의 rx·ind·cls 를 채운다. 결과 요약 dict."""
    tol = _tol(L)
    act = [L[i] for i in active if L[i].x is not None]
    if not act:
        return {"M": 0.0, "tol": tol, "levels": []}
    # 여백은 씬 머리를 뺀 줄로 정한다(씬 번호가 왼쪽 여백에 나와 씬 머리만 더 왼쪽에 있는 문서가 많다)
    body = [ln for ln in act if not T.match_heading(ln.f)] or act
    N = len(body)
    M = _margin(clusters_x([ln.x for ln in body], tol), N)
    lev = clusters_x([ln.x for ln in act], tol)
    ocr = Counter(ln.src for ln in act).most_common(1)[0][0] == "ocr"
    centered = None if ocr or tol < 0.008 else _centered_margin([ln.x for ln in body], M, tol)
    if centered is not None:
        M = centered
    # 쪽별 밀림
    by_pk: dict[tuple, list[Ln]] = defaultdict(list)
    for ln in act:
        by_pk[ln.pk].append(ln)
    shifts: dict[tuple, float] = {}
    tol2 = max(tol * 1.5, 0.012)
    if any(pk[1] != -1 for pk in by_pk) and centered is None:
        # 쪽의 가장 왼쪽 묶음(2줄 이상, 쪽 줄의 15% 이상)이 여백 M 에서 조금 밀려 있고, 그 왼쪽에 다른 줄이 없고,
        # 그 묶음에 이름 줄 같은 짧은 줄이 없으면(지문·씬 머리 칸이면) 그 쪽 전체를 밀린 만큼 되돌린다.
        big = [c for c, n in clusters_x([ln.x for ln in body], tol) if n >= max(3, 0.03 * N)]
        for pk, ls in by_pk.items():
            if pk[1] == -1 or len(ls) < 6:
                continue
            pl = [(c, n) for c, n in clusters_x([ln.x for ln in ls], tol) if n >= 2]
            if not pl:
                continue

            def fit(sh: float) -> int:
                """밀림 sh 를 되돌렸을 때 문서 단계(큰 묶음)에 맞는 쪽 줄 수."""
                return sum(1 for ln in ls if any(abs(ln.x - sh - c) <= tol for c in big))
            if ocr:
                # 스캔 쪽은 종이가 밀려 여백이 쪽마다 크게 다를 수 있다: 쪽의 단계들이 문서 단계들에 가장 많이
                # 겹치는 밀림을 고른다(단계 하나만 맞추면 대사 칸을 여백으로 착각할 수 있어 여러 단계를 함께 본다).
                cand = {round(c - d, 4) for c, _ in pl for d in big if abs(c - d) <= 0.25}
                best, bf = 0.0, fit(0.0)
                f0 = bf
                for sh in sorted(cand, key=abs):
                    v = fit(sh)
                    if v > bf:
                        best, bf = sh, v
                if best and bf >= f0 + max(2, 0.15 * len(ls)) and bf >= 0.6 * len(ls):
                    shifts[pk] = best
                continue
            left, ln_n = min(pl)
            s = left - M
            if abs(s) <= tol2 or abs(s) > 0.07 or ln_n < 0.15 * len(ls):
                continue
            if any(ln.x < left - tol2 for ln in ls):
                continue
            col = [ln for ln in ls if abs(ln.x - left) <= tol]
            namey = sum(1 for ln in col if len(ln.f) <= 25 and not T.match_heading(ln.f)
                        and T.name_like(T.split_speaker(ln.f)[0], block=True))
            if namey >= 1:
                continue
            # 대사만 있는 쪽은 가장 왼쪽 묶음이 대사 칸이라 여백이 밀린 것처럼 보인다: 되돌렸을 때 문서 단계에
            # 덜 맞으면 밀지 않는다. 똑같이 맞으면(홀짝 쪽 여백이 번갈아 다른 Alien 은 두 칸 모두 문서 단계) 쪽 어디든
            # 이름 줄 바로 아래 줄이 가장 왼쪽 묶음에 놓일 때(그 묶음이 대사 칸) 밀지 않는다.
            f0, fs = fit(0.0), fit(s)
            if fs < f0:
                continue
            if fs == f0:
                cue_left = False
                for a, b in zip(ls, ls[1:]):
                    if abs(b.x - left) <= tol and abs(a.x - left) > tol and len(a.f) <= 25 \
                            and not T.match_heading(a.f) and (T.name_like(T.split_speaker(a.f)[0], block=True)
                                                              or T.title_name_like(T.split_speaker(a.f)[0])):
                        cue_left = True
                        break
                if cue_left:
                    continue
            shifts[pk] = s
        if shifts:
            lev = clusters_x([ln.x - shifts.get(ln.pk, 0.0) for ln in act], tol)
            M = _margin(clusters_x([ln.x - shifts.get(ln.pk, 0.0) for ln in body], tol), N)
    for ln in L:
        if ln.x is None:
            ln.rx, ln.ind, ln.cls = M, 0.0, "M"
            continue
        ln.rx = ln.x - shifts.get(ln.pk, 0.0)
        ln.ind = ln.rx - M
        if ln.ind < -tol2:
            ln.cls = "L"
        elif ln.ind <= tol2:
            ln.cls = "M"
        elif ln.rx >= 0.58:
            ln.cls = "R"
        else:
            ln.cls = "I"
    return {"M": round(M, 4), "tol": tol, "tol2": tol2, "shifted_pages": len(shifts), "centered": centered is not None,
            "levels": [(round(c, 3), n) for c, n in lev if n >= max(3, 0.01 * N)]}


def _centered_margin(xs: list[float], M: float, tol: float) -> float | None:
    """가운데 정렬 대사 문서(친절한 금자씨, 파이란 등): 대사 줄 x 가 글 길이에 따라 연속으로 퍼져 한 묶음으로 이어지면
    묶음 중앙값이 여백처럼 잡힌다. 여백 M 을 품은 묶음이 0.1 넘게 퍼져 있고 본문의 90% 이상을 덮고, 그 안에 똑같은 x
    (소수 셋째 자리)에 본문 15% 이상이 놓이면 그 x 가 진짜 지문 여백이다. 아니면 None."""
    xs = sorted(xs)
    if len(xs) < 50:
        return None
    groups = [[xs[0]]]
    for x in xs[1:]:
        if x - groups[-1][-1] <= tol:
            groups[-1].append(x)
        else:
            groups.append([x])
    g = next((g for g in groups if g[0] - 1e-9 <= M <= g[-1] + 1e-9), None)
    if g is None or g[-1] - g[0] <= 0.1 or len(g) < 0.9 * len(xs):
        return None
    xm, nm = Counter(round(x, 3) for x in g).most_common(1)[0]
    if nm < 0.15 * len(xs) or xm > 0.45:
        return None
    return xm


def _margin(lev: list[tuple[float, int]], N: int) -> float:
    """지문 여백: 줄이 4% 이상 몰린 단계 중 가장 왼쪽. 바로 오른쪽(0.035 안)에 두 배 넘게 많은 단계가 있으면
    왼쪽은 씬 머리 칸으로 보고 그 단계를 여백으로(없으면 가장 많은 단계)."""
    big = [(c, n) for c, n in sorted(lev) if n >= 0.04 * N and c < 0.55]
    if not big:
        return max(lev, key=lambda cn: cn[1])[0]
    c0, n0 = big[0]
    if len(big) > 1 and big[1][0] - c0 <= 0.035 and big[1][1] >= 2 * n0:
        return big[1][0]
    return c0
