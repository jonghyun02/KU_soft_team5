"""vowels: PanPhon 모음 거리표, 비용 행렬, 캐시."""
from __future__ import annotations

import csv
import json

import numpy as np
import panphon
import pytest

from pstts import vowels
from pstts.phonetics import EN_VOWELS, KO_VOWEL_JAMO, KO_VOWELS
from pstts.vowels import (TABLE_CSV, TABLE_JSON, VowelTable, build_vowel_table, cost_matrix, load_or_build,
                          load_table, nearest_english, panphon_vector, table_symbols)


@pytest.fixture(scope="module")
def table() -> VowelTable:
    return build_vowel_table(table_symbols())


@pytest.fixture(scope="module")
def ft() -> panphon.FeatureTable:
    return panphon.FeatureTable()


@pytest.mark.parametrize("sym", sorted(EN_VOWELS | KO_VOWELS))
def test_every_vowel_is_one_panphon_segment(sym: str, ft: panphon.FeatureTable) -> None:
    segs = ft.word_to_vector_list(sym, numeric=True)
    assert len(segs) == 1
    assert len(segs[0]) == 24
    assert panphon_vector(sym) == [int(v) for v in segs[0]]


def test_multi_segment_symbol_rejected() -> None:
    with pytest.raises(ValueError):
        build_vowel_table(["a", "aɪ"])
    with pytest.raises(ValueError):
        panphon_vector("ɚ")          # PanPhon 이 모르는 기호(분절 0개)


def test_table_shape_and_normalization(table: VowelTable) -> None:
    m = table.matrix
    k = len(table.symbols)
    assert set(table.symbols) == EN_VOWELS | KO_VOWELS
    assert m.shape == (k, k)
    assert np.allclose(m, m.T)
    assert np.all(np.diag(m) == 0)
    assert m.min() >= 0 and m.max() == pytest.approx(1.0)
    # 비대각 최댓값으로 나눈 값이라 원래 거리의 비율이 그대로다
    raw = np.array([table.vectors[s] for s in table.symbols], dtype=float)
    d = np.linalg.norm(raw[:, None] - raw[None], axis=-1)
    assert np.allclose(m, d / d.max())


def test_korean_u_nearest_english_u(table: VowelTable) -> None:
    order = nearest_english(table)["u"]
    assert order[0][0] == "u"
    assert order[0][1] < order[1][1]          # 논문 그림 5(c): ㅜ 는 영어 u 와 가장 가깝다


def test_same_symbol_zero_distance_across_languages(table: VowelTable) -> None:
    for k in KO_VOWELS & EN_VOWELS:
        assert nearest_english(table)[k][0] == (k, 0.0)


def test_cost_matrix_values(table: VowelTable) -> None:
    src = ["a", None, "i", None]
    tgt = [None, "a", "u"]
    c = cost_matrix(src, tgt, table, null_cost=0.7)
    assert c.shape == (4, 3)
    assert c[1, 0] == 0.0 and c[3, 0] == 0.0          # null-null
    assert c[0, 0] == 0.7 and c[1, 1] == 0.7 and c[2, 0] == 0.7   # null-모음
    assert c[0, 1] == 0.0
    assert c[2, 2] == pytest.approx(table.distance("i", "u"))
    assert c[0, 2] == pytest.approx(table.distance("a", "u"))


def test_cost_matrix_matches_loop(table: VowelTable) -> None:
    rng = np.random.default_rng(0)
    pool = [None, *table.symbols]
    src = [pool[i] for i in rng.integers(0, len(pool), 57)]
    tgt = [pool[i] for i in rng.integers(0, len(pool), 43)]

    def d(a: str | None, b: str | None) -> float:
        if a is None and b is None:
            return 0.0
        if a is None or b is None:
            return 1.0
        return table.distance(a, b)

    ref = np.array([[d(a, b) for b in tgt] for a in src])
    assert np.array_equal(cost_matrix(src, tgt, table, 1.0), ref)


def test_cost_matrix_unknown_symbol(table: VowelTable) -> None:
    with pytest.raises(KeyError):
        cost_matrix(["a", "ɤ"], ["a"], table, 1.0)


def test_cost_matrix_empty(table: VowelTable) -> None:
    assert cost_matrix([], ["a"], table, 1.0).shape == (0, 1)


def test_load_or_build_writes_and_reads(tmp_path, monkeypatch, table: VowelTable) -> None:
    path = tmp_path / "vt.json"
    monkeypatch.setattr(vowels, "_CACHE", {})
    built = load_or_build(path)
    assert path.exists() and (tmp_path / TABLE_CSV.name).exists()
    assert built.symbols == table.symbols
    monkeypatch.setattr(vowels, "_CACHE", {})
    again = load_or_build(path)
    assert again.symbols == table.symbols
    assert np.allclose(again.matrix, table.matrix, atol=1e-6)

    # 기호가 바뀐 낡은 캐시는 다시 만든다
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["symbols"] = ["a", "i"]
    doc["matrix"] = [[0, 1], [1, 0]]
    path.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(vowels, "_CACHE", {})
    rebuilt = load_or_build(path)
    assert rebuilt.symbols == table.symbols
    assert json.loads(path.read_text(encoding="utf-8"))["symbols"] == table.symbols


def test_shipped_cache_matches_fresh_build(table: VowelTable) -> None:
    cached = load_table(TABLE_JSON)
    assert cached.symbols == table.symbols
    assert cached.method == table.method
    assert np.allclose(cached.matrix, table.matrix, atol=1e-6)


def test_shipped_csv_layout(table: VowelTable) -> None:
    with TABLE_CSV.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    en = [s for s in table.symbols if s in EN_VOWELS]
    ko = [s for s in table.symbols if s in KO_VOWELS]
    assert rows[0] == ["ko", "jamo", *en]
    assert [r[0] for r in rows[1:]] == ko
    assert [r[1] for r in rows[1:]] == [KO_VOWEL_JAMO[k] for k in ko]
    for r in rows[1:]:
        for e, v in zip(en, r[2:]):
            assert float(v) == pytest.approx(table.distance(r[0], e), abs=1e-4)
