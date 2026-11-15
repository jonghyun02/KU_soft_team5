"""textlayer.check: 알려진 깨진 글자층 PDF 는 broken, 정상 PDF 는 not broken 으로 고정."""
from __future__ import annotations

import pytest

from sinkprep import paths, textlayer

KO = paths.RAW / "한국영화 시나리오"
FO = paths.RAW / "외국영화 시나리오"

# (파일, 꼭 나와야 하는 깨짐 사유 접두어)
BROKEN = [
    (KO / "더킹 [6535213].pdf", "punct_lines"),
    (KO / "애자 시나리오 [416338].pdf", "overlapping_glyphs"),          # 같은 글자를 여러 번 겹쳐 그림
    (KO / "애자 시나리오 [416338].pdf", "ko_spacing_lost"),
    (KO / "구타유발자들 [54792].pdf", "punct_lines"),
    (KO / "_구타유발자들_ 시나리오 [55216].pdf", "ko_spacing_lost"),
    (KO / "베를린 [3659631].pdf", "ko_spacing_lost"),                  # 띄어쓰기 대부분 사라짐
    (FO / "패스트 라이브즈 대본 (영어) [36187123].pdf", "odd_chars"),   # 띄어쓰기 U+2029, 쌍점 U+A789
    (FO / "Speed (스피드) [56712].pdf", "duplicate_lines"),            # 줄 중복 약 47~50%
    (FO / "이창 Rear Window [56423].pdf", "duplicate_lines"),
    (FO / "현기증 Vertigo [56494].pdf", "overlapping_glyphs"),
    (FO / "Eternal Sunshine (이터널 선샤인) [914411].pdf", "dual_dialogue_merged"),
]

NORMAL = [
    KO / "터널 [5691168].pdf",
    KO / "택시운전사 [6535167].pdf",
    KO / "_가비_ 시나리오 [2405871].pdf",
    KO / "너의 결혼식 [6535184].pdf",
    KO / "변산 [7717269].pdf",
    KO / "_박쥐 Thirst_ 영문 시나리오 [416831].pdf",
    FO / "인셉션 시나리오 pdf 파일 [2453095].pdf",
    FO / "카사블랑카 (Casablanca) [56023].pdf",
    FO / "The Bucket List (버킷리스트_ 죽기전에 꼭 하고싶은 것들) [56932].pdf",   # 여백 * 줄이 많아도 정상
    FO / "본 슈프리머시 Bourne Supremacy [56594].pdf",
    FO / "추락의 해부 대본 (영어) [36187196].pdf",                                # 프랑스어 악센트 글자가 있어도 정상
]


def _need(p):
    if not p.exists():
        pytest.skip(f"원본 없음: {p.name}")


@pytest.mark.parametrize("pdf,reason", BROKEN, ids=[f"{p.stem[:18]}-{r}" for p, r in BROKEN])
def test_broken(pdf, reason):
    _need(pdf)
    r = textlayer.check(pdf)
    assert r["broken"] is True, r
    assert any(x.startswith(reason) for x in r["reasons"]), r["reasons"]


@pytest.mark.parametrize("pdf", NORMAL, ids=[p.stem[:24] for p in NORMAL])
def test_normal(pdf):
    _need(pdf)
    r = textlayer.check(pdf)
    assert r["broken"] is False, r["reasons"]
    assert r["reasons"] == []


def test_result_shape():
    pdf = KO / "베를린 [3659631].pdf"
    _need(pdf)
    r = textlayer.check(pdf)
    assert set(r) >= {"broken", "reasons", "metrics"}
    m = r["metrics"]
    for k in ("ko_token_len", "punct_line", "single_line", "dup_line", "overlap_char",
              "odd_char", "mojibake", "dual_merged", "chars_per_page", "pages"):
        assert k in m
    assert m["pages"] == 108
    assert m["ko_token_len"] > textlayer.TH["ko_token_len"]


def test_open_error(tmp_path):
    bad = tmp_path / "not_a.pdf"
    bad.write_bytes(b"this is not a pdf")
    r = textlayer.check(bad)
    assert r["broken"] is True
    assert r["reasons"][0].startswith("open_error")


def test_synthetic_overlap_and_dup(tmp_path):
    """같은 줄을 같은 자리에 두 번 그린 PDF -> 줄 중복·겹친 글자."""
    import pymupdf
    doc = pymupdf.open()
    for _ in range(3):
        pg = doc.new_page()
        y = 72
        for i in range(30):
            for _rep in range(2):
                pg.insert_text((72, y), f"Line number {i} of the test script.", fontsize=11)
            y += 14
    p = tmp_path / "dup.pdf"
    doc.save(p)
    r = textlayer.check(p)
    assert r["broken"]
    assert r["metrics"]["overlap_char"] > 0.4
    assert any(x.startswith("overlapping_glyphs") for x in r["reasons"])


def test_synthetic_clean(tmp_path):
    import pymupdf
    doc = pymupdf.open()
    for n in range(3):
        pg = doc.new_page()
        y = 72
        for i in range(40):
            pg.insert_text((72, y), f"Page {n} line {i}: Gordon looks into the sky at the signal.", fontsize=11)
            y += 14
    p = tmp_path / "clean.pdf"
    doc.save(p)
    r = textlayer.check(p)
    assert r["broken"] is False, r["reasons"]


def test_synthetic_empty_text(tmp_path):
    import pymupdf
    doc = pymupdf.open()
    for _ in range(2):
        doc.new_page()
    p = tmp_path / "empty.pdf"
    doc.save(p)
    r = textlayer.check(p)
    assert r["broken"] and r["reasons"][0].startswith("no_text")
