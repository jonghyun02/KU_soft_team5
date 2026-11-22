"""paper.vowel_embed: 식 (8) KMeans 5개 중심 평균, 식 (9) 유클리드 거리, 문장별 모음 평균, null 비용, 표 쓰기·읽기."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import KMeans

from pstts.paper import g2p
from pstts.paper.vowel_embed import (KO_U, EmbedVowelTable, build_vowel_table, centroid_mean, check_u, file_sha1,
                                     load_vowel_table, nearest_english, save_vowel_table, sentence_vowel_means,
                                     table_from_doc)

B = g2p.BLANK
KO_A, KO_I = "ㅏ", "ㅣ"          # 호환 모음 자모 ㅏ, ㅣ


# ---------------------------------------------------------------------------- 식 (8)
def test_centroid_mean_is_mean_of_five_kmeans_centres() -> None:
    rng = np.random.default_rng(3)
    centres = rng.normal(0, 10, size=(5, 4))
    x = np.concatenate([c + rng.normal(0, 0.1, size=(40, 4)) for c in centres])
    mu, n = centroid_mean(x, k=5, seed=0)
    km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(x)
    assert n == 5
    np.testing.assert_allclose(mu, km.cluster_centers_.mean(axis=0), rtol=0, atol=1e-12)
    # 묶음이 뚜렷하면 다섯 중심은 참 중심이고 그 평균은 참 중심 평균에 가깝다(식 8, N = 5)
    np.testing.assert_allclose(mu, centres.mean(axis=0), atol=0.05)


def test_centroid_mean_differs_from_plain_mean_when_clusters_are_unbalanced() -> None:
    """식 (8) 은 벡터 평균이 아니라 중심 평균이다. 한 묶음에 벡터가 몰려도 중심마다 같은 무게."""
    rng = np.random.default_rng(1)
    centres = np.array([[0.0, 0], [10, 0], [0, 10], [10, 10], [20, 20]])
    sizes = [200, 5, 5, 5, 5]
    x = np.concatenate([c + rng.normal(0, 0.01, size=(s, 2)) for c, s in zip(centres, sizes)])
    mu, _ = centroid_mean(x, 5, 0)
    np.testing.assert_allclose(mu, centres.mean(axis=0), atol=0.01)
    assert np.linalg.norm(x.mean(axis=0) - mu) > 3


def test_centroid_mean_fewer_vectors_than_k() -> None:
    x = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 8.0]])
    mu, n = centroid_mean(x, k=5, seed=0)
    assert n == 3
    np.testing.assert_allclose(mu, x.mean(axis=0))


@pytest.mark.parametrize("bad", [np.zeros((0, 3)), np.zeros(3)])
def test_centroid_mean_rejects_empty(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        centroid_mean(bad)


def test_sentence_vowel_means_averages_positions_of_same_vowel() -> None:
    toks = [B, "ㄱ", B, KO_A, B, "ㄴ", B, KO_A, B, KO_I, B]
    means = np.arange(2 * len(toks), dtype=float).reshape(2, len(toks))
    out = sentence_vowel_means(toks, means, "ko")
    assert set(out) == {KO_A, KO_I}
    np.testing.assert_allclose(out[KO_A], means[:, [3, 7]].mean(axis=1))
    np.testing.assert_allclose(out[KO_I], means[:, 9])
    with pytest.raises(ValueError):
        sentence_vowel_means(toks, means[:, :-1], "ko")


# ---------------------------------------------------------------------------- 만들기(가짜 프런트엔드)
class FakeFront:
    """모음 토큰마다 정해 둔 벡터(+ 문장별 작은 흔들림)를 m_p 로 주는 가짜 프런트엔드."""

    checkpoint_path = "fake.pth"
    checkpoint_sha1 = "f" * 40

    def __init__(self, base: dict[str, np.ndarray], vocab: set[str]) -> None:
        self.base = base
        self.vocab = vocab
        self.calls = 0

    def tokens(self, text: str, lang: str) -> list[str]:
        return g2p.with_blanks(text.split("|"))

    def unknown_tokens(self, toks: list[str]) -> list[str]:
        return [t for t in dict.fromkeys(toks) if t != B and t not in self.vocab]

    def vocab_vowels(self, lang: str) -> list[str]:
        return sorted(t for t in self.vocab if g2p.is_vowel(t, lang))

    def encoder_means(self, toks: list[str], lang: str) -> np.ndarray:
        self.calls += 1
        rng = np.random.default_rng(self.calls)
        cols = [self.base.get(t, np.zeros(3)) + rng.normal(0, 0.01, 3) for t in toks]
        return np.stack(cols, axis=1)


def _fake() -> FakeFront:
    base = {KO_U: np.array([0.0, 0, 0]), KO_A: np.array([5.0, 0, 0]), "u": np.array([0.3, 0, 0]),
            "a": np.array([5.2, 0, 0]), "i": np.array([0.0, 9, 0])}
    vocab = {KO_U, KO_A, "ㄱ", "u", "a", "i", "k", "ㅣ"}        # ㅣ 는 목록에 있지만 문장에 안 나온다
    return FakeFront(base, vocab)


def test_build_vowel_table_equation_9_and_metadata() -> None:
    front = _fake()
    ko = [f"ㄱ|{KO_U}|ㄱ|{KO_A}", f"{KO_U}|{KO_A}|{KO_A}", f"{KO_U}", "", "Z|" + KO_A]
    en = ["k|u|a", "u|i", "a|k|i"]
    doc = build_vowel_table(front, ko, en, k=5, seed=0)
    assert doc["ko"] == [KO_A, KO_U] and doc["en"] == ["i", "a", "u"]
    mu = {s: np.asarray(v) for s, v in doc["mu"].items()}
    m = np.asarray(doc["matrix"])
    for i, k in enumerate(doc["ko"]):
        for j, e in enumerate(doc["en"]):
            assert m[i, j] == pytest.approx(np.linalg.norm(mu[k] - mu[e]))          # 식 (9)
    assert doc["max_distance"] == pytest.approx(m.max())
    assert doc["n_sentences_per_vowel"] == {KO_A: 2, KO_U: 3, "i": 2, "a": 2, "u": 2}
    assert doc["n_clusters"][KO_U] == 3 and doc["k"] == 5 and doc["seed"] == 0
    assert doc["sentences"]["ko"]["used"] == 3
    assert doc["sentences"]["ko"]["skipped_empty"] == 1
    assert doc["sentences"]["ko"]["skipped_unknown_tokens"] == 1
    assert doc["sentences"]["ko"]["unknown_tokens"] == {"Z": 1}
    assert doc["checkpoint"] == {"path": "fake.pth", "sha1": "f" * 40}
    assert doc["vocab_vowels_without_data"] == {"ko": ["ㅣ"], "en": []}
    assert doc["nearest"][KO_U][0] == "u" and doc["nearest"][KO_A][0] == "a"
    assert doc["check_u"]["is_u"] is True and doc["check_u"]["nearest_en"] == "u"
    json.dumps(doc, ensure_ascii=False)


def test_build_vowel_table_needs_vowels_in_both_languages() -> None:
    with pytest.raises(ValueError):
        build_vowel_table(_fake(), ["ㄱ"], ["k|u"])


def test_check_u_without_u_is_undecided() -> None:
    doc = {"ko": [KO_A], "en": ["a"], "matrix": [[1.0]]}
    assert check_u(doc)["is_u"] is None
    doc = {"ko": [KO_A, KO_U], "en": ["a", "u"], "matrix": [[0.1, 2.0], [0.5, 0.9]]}
    out = check_u(doc)
    assert out["nearest_en"] == "a" and out["is_u"] is False and out["distance_to_u"] == pytest.approx(0.9)
    assert nearest_english(doc) == {KO_A: ["a", 0.1], KO_U: ["a", 0.5]}


# ---------------------------------------------------------------------------- 국소 비용
def _table() -> EmbedVowelTable:
    return EmbedVowelTable(ko=[KO_A, KO_U], en=["a", "u", "i"], matrix=np.array([[0.5, 2.0, 3.0], [2.5, 0.4, 1.0]]))


def test_cost_null_rules_and_orientation() -> None:
    t = _table()
    src = ["a", None, "u", "i"]         # 영어 원문
    tgt = [KO_U, None, KO_A]            # 한국어 후보
    c = t.cost(src, tgt)
    assert c.shape == (4, 3)
    assert c[1, 1] == 0.0                                       # null 끼리 0
    assert c[1, 0] == c[1, 2] == c[0, 1] == c[2, 1] == c[3, 1] == 3.0     # null 과 모음 사이 = 표의 최대 거리
    assert c[0, 0] == t.distance(KO_U, "a") == 2.5              # 모음끼리 = 식 (9) 거리
    assert c[2, 0] == 0.4 and c[0, 2] == 0.5 and c[3, 2] == 3.0
    assert t.max_distance == 3.0
    assert t.symbols("en") == ["a", "u", "i"] and t.symbols("ko") == [KO_A, KO_U]


def test_cost_unknown_label_is_an_error() -> None:
    with pytest.raises(KeyError):
        _table().cost(["o"], [KO_A])
    with pytest.raises(KeyError):
        _table().cost(["a"], ["a"])


@pytest.mark.parametrize("matrix", [np.zeros((2, 2)), np.array([[0.1, -1, 0], [0, 0, 0]]),
                                    np.array([[np.nan, 0, 0], [0, 0, 0]])])
def test_table_validation(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError):
        EmbedVowelTable(ko=[KO_A, KO_U], en=["a", "u", "i"], matrix=matrix)


# ---------------------------------------------------------------------------- 쓰기, 읽기
def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    doc = build_vowel_table(_fake(), [f"{KO_U}|{KO_A}", f"{KO_A}"], ["u|a", "i"], k=5, seed=0)
    doc["selection"] = {"ko": {"picked": 2}}
    jp, cp = save_vowel_table(doc, tmp_path / "vowel_table.json")
    assert cp == tmp_path / "vowel_table.csv"
    back = json.loads(jp.read_text(encoding="utf-8"))
    assert back["ko"] == doc["ko"] and back["en"] == doc["en"] and back["selection"] == {"ko": {"picked": 2}}
    np.testing.assert_allclose(back["matrix"], doc["matrix"], atol=1e-6)
    table = load_vowel_table(jp)
    np.testing.assert_allclose(table.matrix, np.asarray(doc["matrix"]), atol=1e-6)
    assert table.meta["checkpoint"]["sha1"] == "f" * 40
    assert table.meta["file_sha1"] == file_sha1(jp) and table.meta["check_u"]["is_u"] is True
    with cp.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["ko", *doc["en"]]
    assert [r[0] for r in rows[1:]] == doc["ko"]
    assert float(rows[1][1]) == pytest.approx(doc["matrix"][0][0], abs=1e-6)
    t2 = table_from_doc(back)
    assert t2.ko == table.ko and "mu" not in t2.meta
