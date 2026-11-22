"""논문 판 PS 시험: 누적 반올림 늘리기, tslearn DTW(반경 None = 반경 1 띠)와 손 계산 DP, PS(식 11), PS-Comet(식 20).

Frontend 는 frontend.py 를 불러오지 않고 같은 꼴의 가짜(FakeFront)를 쓴다.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from tslearn.metrics import GLOBAL_CONSTRAINT_CODE, compute_mask, dtw_path_from_metric, sakoe_chiba_mask

from pstts.paper.ps import (PsCandidate, dtw, expand, frame_counts, frames_of, measure_line, ps_comet, select_line,
                            select_ps)

B = "<b>"


# ---------------------------------------------------------------- 손 계산 참조(tslearn 0.10.0 _masks.py 와 같은 규칙)

def band_mask(sz1: int, sz2: int, radius: int) -> np.ndarray:
    """tslearn _sakoe_chiba_mask_generic 을 그대로 옮긴 것: 길이 차이만큼 띠를 넓힌다."""
    mask = np.zeros((sz1, sz2), dtype=bool)
    if sz1 > sz2:
        width = sz1 - sz2 + radius
        for i in range(sz2):
            mask[max(0, i - radius):min(sz1, i + width) + 1, i] = True
    else:
        width = sz2 - sz1 + radius
        for i in range(sz1):
            mask[i, max(0, i - radius):min(sz2, i + width) + 1] = True
    return mask


def dp_reference(cost: np.ndarray, radius: int) -> float:
    """띠 안에서 대각, 아래, 오른쪽으로 국소 비용을 더하는 DP(가중치 1, 제곱근 없음)."""
    n, m = cost.shape
    mask = band_mask(n, m, radius)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(n):
        for j in range(m):
            if mask[i, j]:
                acc[i + 1, j + 1] = cost[i, j] + min(acc[i, j + 1], acc[i + 1, j], acc[i, j])
    return float(acc[n, m])


def precomputed(cost: np.ndarray):
    """라벨 대신 번호 열을 받아 미리 만든 비용 행렬에서 꺼내는 cost_fn."""
    return lambda s, t: cost[np.ix_(list(s), list(t))]


# ---------------------------------------------------------------- 늘리기

def test_frame_counts_cumulative_rounding() -> None:
    # 프레임 0.25초. 누적 [0.5, 0.5, 0.875, 0.9375] / 0.25 = [2, 2, 3.5, 3.75] -> 반올림 [2, 2, 4, 4]
    counts = frame_counts([0.5, 0.0, 0.375, 0.0625], 0.25)
    assert counts.tolist() == [2, 0, 2, 0] and counts.sum() == 4
    # 0.5 는 올림(파이썬 round 의 짝수 쪽 반올림이 아님): 2.5 프레임 -> 3
    assert frame_counts([0.625], 0.25).tolist() == [3]
    # 토큰마다 0.4 프레임: 하나씩 반올림하면 0 이지만 누적이라 합이 반올림(4.0) = 4
    small = frame_counts([0.1] * 10, 0.25)
    assert small.sum() == 4 and set(small.tolist()) <= {0, 1}
    assert frame_counts([], 0.25).tolist() == []


@pytest.mark.parametrize("seed", range(10))
def test_frame_counts_total_and_drift(seed: int) -> None:
    rng = np.random.default_rng(seed)
    durs = rng.random(int(rng.integers(1, 60))) * 0.2
    durs[rng.random(durs.size) < 0.2] = 0.0                            # 길이 0 토큰도 섞는다
    frame = 256 / 22050
    counts = frame_counts(durs, frame)
    cum = np.cumsum(durs) / frame
    assert counts.sum() == math.floor(cum[-1] + 0.5)
    assert np.all(counts >= 0)
    assert np.all(np.abs(np.cumsum(counts) - cum) <= 0.5 + 1e-9)       # 어긋남이 쌓이지 않는다


def test_expand_repeats_labels_and_drops_zero_frames() -> None:
    labels = [None, "a", None, "i", None]
    out = expand(labels, [0.25, 0.5, 0.0, 0.25, 0.1], 0.25)
    # 누적 [1, 3, 3, 4, 4.4] 프레임 -> [1, 2, 0, 1, 0]
    assert out == [None, "a", "a", "i"]


@pytest.mark.parametrize("durs, frame", [([-0.1], 0.25), ([float("nan")], 0.25), ([0.1], 0.0), ([0.1], -1.0),
                                         ([[0.1]], 0.25)])
def test_frame_counts_bad_input(durs, frame) -> None:
    with pytest.raises(ValueError):
        frame_counts(durs, frame)


def test_expand_length_mismatch() -> None:
    with pytest.raises(ValueError, match="수가 다릅니다"):
        expand(["a"], [0.1, 0.2], 0.25)


# ---------------------------------------------------------------- DTW(식 10)

def test_reference_band_is_tslearn_default() -> None:
    # tslearn 은 반경 None 을 반경 1 로 바꾼다(_compute_mask_generic). 참조 띠가 tslearn 띠와 같은지 확인
    code = GLOBAL_CONSTRAINT_CODE["sakoe_chiba"]
    for n, m in [(5, 3), (3, 5), (7, 7), (1, 9), (12, 4), (2, 1), (1, 1)]:
        expected = band_mask(n, m, 1)
        assert np.array_equal(np.asarray(compute_mask(n, m, code, None)), expected)
        assert np.array_equal(np.asarray(sakoe_chiba_mask(n, m, 1)), expected)


@pytest.mark.parametrize("seed", range(24))
def test_dtw_none_radius_matches_reference_dp(seed: int) -> None:
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(1, 45))
    m = int(rng.integers(1, 45))
    while m == n:                                       # 길이가 다른 행렬만
        m = int(rng.integers(1, 45))
    cost = rng.random((n, m)) * float(rng.choice([1.0, 3.0]))
    got = dtw(list(range(n)), list(range(m)), precomputed(cost), None)
    want = dp_reference(cost, 1)
    assert got == pytest.approx(want, rel=1e-12, abs=1e-12)
    _, tslearn_default = dtw_path_from_metric(cost, metric="precomputed", global_constraint="sakoe_chiba")
    assert got == float(tslearn_default)
    assert dtw(list(range(n)), list(range(m)), precomputed(cost), 3) == pytest.approx(dp_reference(cost, 3))


def test_default_band_is_radius_one_on_unequal_lengths() -> None:
    # 8 x 6 행렬. 비용 0 인 길은 (0, 2) 를 지나는데 반경 1 띠(열 2 는 행 1~5)에는 없고 반경 2 띠에는 있다
    cost = np.ones((8, 6))
    for i, j in [(0, 0), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5)]:
        cost[i, j] = 0.0
    rows, cols = list(range(8)), list(range(6))
    assert not band_mask(8, 6, 1)[0, 2] and band_mask(8, 6, 2)[0, 2]
    default = dtw(rows, cols, precomputed(cost), None)
    assert default == dp_reference(cost, 1) > 0.0
    assert dtw(rows, cols, precomputed(cost), 2) == dp_reference(cost, 2) == 0.0


def test_dtw_is_sum_without_square_root() -> None:
    cost = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert dtw([0, 1], [0, 1], precomputed(cost), None) == 5.0          # 대각 1 + 4, 제곱근(2.236)이 아님
    # 길이가 다르면 tslearn 띠가 넓어져 끝까지 길이 있다
    tall = np.array([[1.0], [2.0], [4.0]])
    assert dtw([0, 1, 2], [0], precomputed(tall), None) == 7.0


def test_dtw_labels_and_empty_frames() -> None:
    def cost(s, t):
        return np.array([[0.0 if a == b else 1.0 for b in t] for a in s])
    assert dtw(["a", "a", None], ["a", None], cost, None) == 0.0
    assert dtw(["a", "i"], ["u"], cost, None) == 2.0
    assert dtw([], ["a"], cost, None) is None and dtw(["a"], [], cost, None) is None
    with pytest.raises(ValueError, match="꼴"):
        dtw(["a"], ["a"], lambda s, t: np.zeros((2, 2)), None)


# ---------------------------------------------------------------- PS(식 11)

def test_select_ps_argmin_first_wins_ties() -> None:
    assert select_ps([3.0, None, 1.0, 1.0, 2.0]) == 2
    assert select_ps([1.0 + 1e-13, 1.0]) == 0                          # 부동소수 잡음 안의 차이는 같은 값
    assert select_ps([2.0, 1.0]) == 1
    assert select_ps([None, None]) is None and select_ps([]) is None
    with pytest.raises(ValueError):
        select_ps([float("nan")])


# ---------------------------------------------------------------- PS-Comet(식 20)

def test_ps_comet_normalizes_inverts_and_scores() -> None:
    inv, score, best = ps_comet([10.0, 20.0, None, 30.0], [0.1, 0.5, 0.9, 0.2], 1.6, 0.4)
    assert inv == [1.0, 0.5, None, 0.0]
    assert score == pytest.approx([1.6 * 1.0 + 0.4 * 0.1, 1.6 * 0.5 + 0.4 * 0.5, None, 0.4 * 0.2])
    assert score[2] is None and best == 0


def test_ps_comet_comet_can_win() -> None:
    # DTW 차이가 작고 COMET 차이가 크면 COMET 쪽이 이긴다: 1.6 x 0.9 + 0.4 x 1.0 > 1.6 x 1.0 + 0.4 x 0.2
    inv, score, best = ps_comet([10.0, 10.1, 11.0], [0.2, 1.0, 0.0], 1.6, 0.4)
    assert inv == pytest.approx([1.0, 0.9, 0.0])
    assert best == 1 and select_ps([10.0, 10.1, 11.0]) == 0


def test_ps_comet_all_equal_dtw_gives_inv_one() -> None:
    inv, score, best = ps_comet([7.5, 7.5, 7.5], [0.3, 0.8, 0.8], 1.6, 0.4)
    assert inv == [1.0, 1.0, 1.0]
    assert score == pytest.approx([1.6 + 0.12, 1.6 + 0.32, 1.6 + 0.32])
    assert best == 1                                                   # 같은 점수면 먼저 나온 것
    inv, score, best = ps_comet([4.0, None], [0.5, None], 1.6, 0.4)     # 하나뿐이면 max == min
    assert inv == [1.0, None] and best == 0


def test_ps_comet_excludes_none_and_checks_input() -> None:
    assert ps_comet([None, None], [0.1, 0.2], 1.6, 0.4) == ([None, None], [None, None], None)
    assert ps_comet([], [], 1.6, 0.4) == ([], [], None)
    with pytest.raises(ValueError, match="COMET"):
        ps_comet([1.0, 2.0], [0.1, None], 1.6, 0.4)
    with pytest.raises(ValueError, match="수가 다릅니다"):
        ps_comet([1.0], [0.1, 0.2], 1.6, 0.4)


# ---------------------------------------------------------------- 줄 하나(가짜 Frontend)

class FakeFront:
    """frontend.Frontend 와 같은 꼴의 가짜. 토큰 = 글자(사이사이 빈 토큰), 프레임 0.25초.

    길이: 빈 토큰 0.25초, 모음 글자 0.5초, 공백 0, 그 밖의 글자 0.25초. 라벨: 모음 글자는 그 글자, 나머지 None.
    비용: 같은 모음 0, 다른 모음 1, 모음과 null 1, null 끼리 0.
    """
    name = "fake"
    frame_sec = 0.25
    VOWELS = set("aeiou")

    def __init__(self) -> None:
        self.duration_calls: list[tuple[str, int]] = []

    def tokens(self, text: str, lang: str) -> list[str]:
        out = [B]
        for ch in text:
            out += [ch, B]
        return out

    def durations(self, seqs: list[list[str]], lang: str) -> list[np.ndarray]:
        self.duration_calls.append((lang, len(seqs)))
        return [np.array([0.25 if t == B else 0.5 if t in self.VOWELS else 0.0 if t == " " else 0.25 for t in s])
                for s in seqs]

    def labels(self, toks: list[str], lang: str) -> list[str | None]:
        return [t if t in self.VOWELS else None for t in toks]

    def cost(self, src, tgt) -> np.ndarray:
        def c(a, b):
            if a is None and b is None:
                return 0.0
            return 0.0 if a == b else 1.0
        return np.array([[c(a, b) for b in tgt] for a in src], dtype=float).reshape(len(src), len(tgt))

    def info(self) -> dict:
        return {"name": self.name}

    def close(self) -> None:
        return None


def test_frames_of_uses_token_durations() -> None:
    front = FakeFront()
    (frames, dur), = frames_of(front, ["ba"], "ko")
    # 토큰 [b, 'b', b, 'a', b] -> 길이 [0.25, 0.25, 0.25, 0.5, 0.25] -> 프레임 [1, 1, 1, 2, 1]
    assert frames == [None, None, None, "a", "a", None] and dur == 1.5


def test_measure_line_and_select_line() -> None:
    front = FakeFront()
    texts = ["ma", "mi", "", "mama"]
    cands, n_src = measure_line(front, "ma", texts, None)
    assert n_src == 6 and front.duration_calls == [("en", 1), ("ko", 4)]      # 후보 길이는 한 번에
    assert [c.n_frames for c in cands] == [6, 6, 1, 11]                     # 빈 글도 빈 토큰 하나(1프레임)
    assert cands[0].dtw == 0.0 and cands[1].dtw == 2.0
    assert cands[2].dtw == pytest.approx(dp_reference(front.cost(
        [None, None, None, "a", "a", None], [None]), 1))
    for c, comet in zip(cands, [0.1, 0.9, 0.5, 0.2]):
        c.comet = comet
    choice = select_line(cands, 1.6, 0.4)
    assert choice.ps == 0 and choice.excluded == []
    assert cands[0].dtw_inv == 1.0 and cands[0].score == pytest.approx(1.6 + 0.04)
    assert choice.ps_comet == int(np.argmax([c.score for c in cands]))


def test_zero_frame_candidate_is_excluded() -> None:
    class Silent(FakeFront):
        """빈 글은 빈 토큰까지 길이 0(프레임 0개)."""

        def durations(self, seqs, lang):
            base = super().durations(seqs, lang)
            return [np.zeros(len(s)) if all(t == B for t in s) else d for s, d in zip(seqs, base)]
    front = Silent()
    cands, n_src = measure_line(front, "ma", ["ma", "", "mo"], None)
    assert cands[1].n_frames == 0 and cands[1].dtw is None and cands[1].dur_sec == 0.0
    for c, comet in zip(cands, [0.0, 1.0, 0.0]):
        c.comet = comet
    choice = select_line(cands, 1.6, 0.4)
    assert choice.excluded == [1] and cands[1].dtw_inv is None and cands[1].score is None
    assert choice.ps == 0 and choice.ps_comet == 0


def test_source_without_frames_excludes_everything() -> None:
    class Mute(FakeFront):
        """영어 원문은 모든 토큰 길이 0."""

        def durations(self, seqs, lang):
            base = super().durations(seqs, lang)
            return [np.zeros(len(s)) for s in seqs] if lang == "en" else base
    cands, n_src = measure_line(Mute(), "ma", ["ma", "mi"], None)
    assert n_src == 0 and [c.dtw for c in cands] == [None, None]
    for c in cands:
        c.comet = 0.5
    choice = select_line(cands, 1.6, 0.4)
    assert (choice.ps, choice.ps_comet, choice.excluded) == (None, None, [0, 1])


def test_candidate_record_fields() -> None:
    c = PsCandidate(text="가")
    assert (c.dur_sec, c.n_frames, c.dtw, c.dtw_inv, c.comet, c.score) == (None, 0, None, None, None, None)
