"""3절 분석: 표 2 (Semline, Semsec), 그림 3 (줄 sts 밀도), 표 3 (Phodeg, Phovar), 3.1절 영어 줄 비율, 부록 표 8.

- 비교 대상은 공식 번역만: K-pop, 애니메이션(meta 장르 disney), 뮤지컬(musical) (3절 첫 문단).
- K-pop (excluded): 번역되지 않고 그대로 남은 영어 줄을 뺀 값.
  [해석] 한국어판 줄에 한글이 하나도 없거나(영어로만 된 줄, 3.1절 30.2%), 두 줄이 (정리 뒤) 똑같으면 뺀다.
- 곡마다 Semline, Semsec, Phodeg, Phovar 를 구해 장르 안에서 평균한다. [해석] 곡 평균인지 줄 평균인지 논문에 없음.

실행: python -m kplt.analysis
"""
from __future__ import annotations

import json
import re
from collections import Counter
from statistics import mean

import numpy as np

from .config import ANALYSIS_GENRES, DATA, OUT, VENDOR
from .gtranslate import GoogleTranslator
from .metrics import STS
from .phoneme import song_pho
from .text import HANGUL, LATIN_WORD, load_meta


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-z가-힣']", "", s.lower())


def load():
    meta = load_meta(VENDOR / "meta.csv")
    lyrics = json.loads((DATA / "lyrics.json").read_text(encoding="utf-8"))
    return meta, lyrics


def split_lang(song):
    kr = [[k for k, _ in sec] for sec in song]
    en = [[e for _, e in sec] for sec in song]
    return kr, en


def exclude_untranslated(song):
    out = [[(k, e) for k, e in sec if HANGUL.search(k) and _norm(k) != _norm(e)] for sec in song]
    return [s for s in out if s]


def line_lang_stats(meta, lyrics):
    """3.1절: K-pop 한국어 줄 중 영어로만 된 줄, 섞인 줄 비율."""
    c = Counter()
    for lid, song in lyrics.items():
        if meta[lid]["GENRE"] != "k-pop":
            continue
        for sec in song:
            for k, _ in sec:
                h, l = bool(HANGUL.search(k)), bool(LATIN_WORD.search(k))
                c["all"] += 1
                c["english_only"] += (l and not h)
                c["mixed"] += (l and h)
    return {"lines": c["all"], "english_only": c["english_only"] / c["all"], "mixed": c["mixed"] / c["all"]}


def table8(lyrics):
    secs_kr, secs_en, lines_kr, lines_en = [], [], [], []
    for song in lyrics.values():
        for sec in song:
            secs_kr.append("\n".join(k for k, _ in sec))
            secs_en.append("\n".join(e for _, e in sec))
            lines_kr += [k for k, _ in sec]
            lines_en += [e for _, e in sec]
    vocab_en = {w for l in lines_en for w in l.split()}
    try:
        from konlpy.tag import Okt
        okt = Okt()
        vocab_kr = set()
        for l in set(lines_kr):
            ko_only = " ".join(HANGUL.findall(l)) and re.sub(r"[A-Za-z']+", " ", l)
            if ko_only:
                vocab_kr.update(okt.morphs(ko_only))
        kr_note = "KoNLPy Okt"
    except Exception as e:  # noqa: BLE001
        vocab_kr, kr_note = set(), f"KoNLPy 실패: {e}"
    return {"Songs": len(lyrics), "Total sections": len(secs_kr),
            "Unique sections (Korean)": len(set(secs_kr)), "Unique sections (English)": len(set(secs_en)),
            "Total lines": len(lines_kr), "Unique lines (Korean)": len(set(lines_kr)),
            "Unique lines (English)": len(set(lines_en)),
            "Unique vocabulary (Korean)": len(vocab_kr), "Unique vocabulary (English)": len(vocab_en),
            "_korean_vocab_tool": kr_note}


def main():
    meta, lyrics = load()
    sts = STS(translator=GoogleTranslator())
    groups = {"K-pop (included)": [], "K-pop (excluded)": [], "Animation": [], "Theatre": []}
    pho_groups = {"K-pop": [], "Animation": [], "Theatre": []}
    for lid, song in lyrics.items():
        m = meta[lid]
        if m["IS_OFFICIAL"][:1] != "1" or m["GENRE"] not in ANALYSIS_GENRES:
            continue
        g = ANALYSIS_GENRES[m["GENRE"]]
        kr, en = split_lang(song)
        pho_groups[g].append((lid, song_pho(en), song_pho(kr)))
        if g == "K-pop":
            groups["K-pop (included)"].append((lid, sts.song(kr, en)))
            ex = exclude_untranslated(song)
            if ex:
                kr2, en2 = split_lang(ex)
                groups["K-pop (excluded)"].append((lid, sts.song(kr2, en2)))
        else:
            groups[g].append((lid, sts.song(kr, en)))

    table2 = {g: {"n_songs": len(v), "Semline": mean(r["semline"] for _, r in v),
                  "Semsec": mean(r["semsec"] for _, r in v)} for g, v in groups.items() if v}
    table3 = {}
    for g, v in pho_groups.items():
        v = [x for x in v if x[1] and x[2]]
        table3[g] = {"n_songs": len(v),
                     "Phodeg_en": mean(x[1][0] for x in v), "Phodeg_kr": mean(x[2][0] for x in v),
                     "Phovar_en": mean(x[1][1] for x in v), "Phovar_kr": mean(x[2][1] for x in v)}
    res = {"table2": table2, "table3": table3, "line_lang_3_1": line_lang_stats(meta, lyrics),
           "table8": table8(lyrics)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    fig3({g: [s for _, r in v for s in r["line_sts"]] for g, v in groups.items()})
    print(json.dumps(res, ensure_ascii=False, indent=1))


def fig3(line_sts: dict[str, list[float]]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    xs = np.linspace(-0.2, 1.0, 400)
    plt.figure(figsize=(6, 4))
    for g, v in line_sts.items():
        if len(v) > 2:
            plt.plot(xs, gaussian_kde(v)(xs), label=g)
    plt.xlabel("line-wise sts")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "fig3_line_sts_density.png", dpi=150)


if __name__ == "__main__":
    main()
