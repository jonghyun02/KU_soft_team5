"""PDF 글자층 깨짐 검사.

check(pdf_path) -> {"broken": bool, "reasons": [...], "metrics": {...}}

글자층이 있어도 그대로 쓰면 안 되는 경우를 잡는다(이 경우 ocr 단계가 쪽 그림을 OCR 한다).
기준은 원본 PDF 112개(글자층 있음)를 모두 재 본 분포로 정했다(notes/ocr_engine.md "글자층 검사").

검사 항목(metrics 이름 / 깨짐 기준):
- ko_token_len   한글 낱말(공백으로 나눈 덩어리) 평균 음절 수. 띄어쓰기가 사라지면 길어진다.   > 5.0
- punct_line     구두점·괄호만 있는 줄 비율(구두점이 따로 줄로 빠짐). `*`만 있는 줄은 뺌.       > 0.10
- single_line    한 글자 줄 비율.                                                              > 0.15
- dup_line       바로 앞 줄과 같은 줄 비율(같은 줄을 두 번 이상 그림).                         > 0.15
- overlap_char   같은 글자가 같은 자리(±1pt)에 겹쳐 그려진 비율.                                > 0.15
- odd_char       이상한 문자(U+2028/2029 줄·문단 구분자, U+A789, U+FFFD, 사용자 영역, 제어문자) 비율. > 0.002
- mojibake       라틴-1 보충 글자(À~ÿ 등) 비율. 한글 글꼴 인코딩이 깨지면 높다.                  > 0.10
- dual_merged    두 칸 동시 대사(화자 둘이 한 줄)가 한 줄로 섞인 곳 수.                           >= 3
- chars_per_page 쪽당 글자 수(공백 제외). 글자층이 사실상 비어 있음.                              < 80
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

VERSION = "0.1.0"

TH = {
    "ko_token_len": 5.0,
    "punct_line": 0.10,
    "single_line": 0.15,
    "dup_line": 0.15,
    "overlap_char": 0.15,
    "odd_char": 0.002,
    "mojibake": 0.10,
    "dual_merged": 3,
    "chars_per_page": 80,
}

_HANGUL = re.compile(r"[가-힣]")
_PUNCT_ONLY = re.compile(r"^[\s.,?!:;()\[\]{}'\"`~\-–—…·/<>*_=+|]+$")
_STAR_ONLY = re.compile(r"^[*\s]+$")
_ODD = re.compile("[  ꞉�-\x00-\x08\x0b\x0c\x0e-\x1f]")
_MOJI = re.compile("[¡-ÿ]")
_LETTER = re.compile(r"[^\W\d_]")
# 두 칸 동시 대사의 화자 줄: 대문자 이름 둘이 공백 3칸 이상으로 떨어져 한 줄에 있음
_CUE = r"[A-Z][A-Z0-9.'’\-]*(?: [A-Z0-9.'’\-()]+)*"
_DUAL = re.compile(rf"^{_CUE} {{3,}}{_CUE}$")
_MIDGAP = re.compile(r"[A-Za-z.,?!'\")] {3,}\S")


def _lines_of(page) -> list[str]:
    return page.get_text("text").split("\n")


def _overlap(page) -> tuple[int, int]:
    """(겹친 글자 수, 전체 글자 수). 같은 글자·같은 자리(1pt 격자)면 겹침."""
    seen: set = set()
    dup = tot = 0
    d = page.get_text("rawdict")
    for b in d.get("blocks", []):
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                for ch in sp.get("chars", []):
                    c = ch.get("c", "")
                    if not c.strip():
                        continue
                    x, y = ch["bbox"][0], ch["bbox"][1]
                    key = (c, round(x), round(y))
                    tot += 1
                    if key in seen:
                        dup += 1
                    else:
                        seen.add(key)
    return dup, tot


def _sample(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    step = n / k
    return sorted({int(i * step) for i in range(k)})


def measure(pdf_path, max_overlap_pages: int = 12) -> dict:
    import pymupdf
    pdf_path = Path(pdf_path)
    with pymupdf.open(pdf_path) as doc:
        n = doc.page_count
        lines: list[str] = []
        for pg in doc:
            lines.extend(_lines_of(pg))
        ov_dup = ov_tot = 0
        for i in _sample(n, max_overlap_pages):
            a, b = _overlap(doc[i])
            ov_dup += a
            ov_tot += b
    text = unicodedata.normalize("NFC", "\n".join(lines))
    lines = text.split("\n")
    ne = [l.strip() for l in lines if l.strip()]
    nonspace = re.sub(r"\s", "", text)
    m: dict = {"pages": n, "lines": len(ne), "chars": len(nonspace)}
    m["chars_per_page"] = round(len(nonspace) / max(1, n), 1)
    # 한글 띄어쓰기
    hangul = len(_HANGUL.findall(text))
    m["hangul"] = hangul
    toks = [t for t in text.split() if _HANGUL.search(t)]
    m["ko_token_len"] = round(sum(len(_HANGUL.findall(t)) for t in toks) / len(toks), 2) if toks else None
    # 줄 모양
    # 여백 수정 표시(`*`, `**`)만 있는 줄은 정상 대본에도 많으므로 줄 모양 비율에서 뺀다.
    body = [l for l in ne if not _STAR_ONLY.match(l)]
    m["star_line"] = round((len(ne) - len(body)) / max(1, len(ne)), 4)
    nl = max(1, len(body))
    m["punct_line"] = round(sum(1 for l in body if _PUNCT_ONLY.match(l)) / nl, 4)
    m["single_line"] = round(sum(1 for l in body if len(l) == 1) / nl, 4)
    m["dup_line"] = round(sum(1 for a, b in zip(ne, ne[1:]) if a == b) / nl, 4)
    m["overlap_char"] = round(ov_dup / ov_tot, 4) if ov_tot else 0.0
    # 이상한 문자
    m["odd_char"] = round(len(_ODD.findall(text)) / max(1, len(nonspace)), 5)
    odd = sorted({f"U+{ord(c):04X}" for c in _ODD.findall(text)})
    m["odd_kinds"] = odd[:10]
    letters = len(_LETTER.findall(text))
    m["mojibake"] = round(len(_MOJI.findall(text)) / max(1, letters), 4)
    # 두 칸 동시 대사가 한 줄로 섞임
    dual = 0
    raw = [l.rstrip() for l in lines if l.strip()]
    for a, b in zip(raw, raw[1:]):
        if _DUAL.match(a.strip()) and _MIDGAP.search(b.strip()):
            dual += 1
    m["dual_merged"] = dual
    return m


def judge(m: dict) -> list[str]:
    r = []
    if m["chars_per_page"] < TH["chars_per_page"]:
        r.append(f"no_text: 쪽당 글자 {m['chars_per_page']} < {TH['chars_per_page']}")
        return r
    if m["hangul"] >= 300 and m["ko_token_len"] is not None and m["ko_token_len"] > TH["ko_token_len"]:
        r.append(f"ko_spacing_lost: 한글 낱말 평균 {m['ko_token_len']}음절 > {TH['ko_token_len']}")
    if m["punct_line"] > TH["punct_line"]:
        r.append(f"punct_lines: 구두점만 있는 줄 {m['punct_line']:.1%}")
    if m["single_line"] > TH["single_line"]:
        r.append(f"single_char_lines: 한 글자 줄 {m['single_line']:.1%}")
    if m["dup_line"] > TH["dup_line"]:
        r.append(f"duplicate_lines: 앞 줄과 같은 줄 {m['dup_line']:.1%}")
    if m["overlap_char"] > TH["overlap_char"]:
        r.append(f"overlapping_glyphs: 같은 자리 겹친 글자 {m['overlap_char']:.1%}")
    if m["odd_char"] > TH["odd_char"]:
        r.append(f"odd_chars: 이상한 문자 {m['odd_char']:.2%} {','.join(m['odd_kinds'])}")
    if m["mojibake"] > TH["mojibake"]:
        r.append(f"mojibake: 라틴-1 보충 글자 {m['mojibake']:.1%}")
    if m["dual_merged"] >= TH["dual_merged"]:
        r.append(f"dual_dialogue_merged: 두 칸 대사가 한 줄로 섞인 곳 {m['dual_merged']}")
    return r


def check(pdf_path) -> dict:
    """글자층 깨짐 검사. 열리지 않으면 broken(reasons=['open_error: …'])."""
    try:
        m = measure(pdf_path)
    except Exception as e:
        return {"broken": True, "reasons": [f"open_error: {e!r}"[:200]], "metrics": {}, "version": VERSION}
    reasons = judge(m)
    return {"broken": bool(reasons), "reasons": reasons, "metrics": m, "version": VERSION}
