"""dtw: tslearn 모음 DTW 와 넘파이 참조 DP 비교."""
from __future__ import annotations

import numpy as np
import pytest
from tslearn.metrics import dtw_path_from_metric, sakoe_chiba_mask

from pstts.config import Config
from pstts.dtw import dtw_cost, ps_distance
from pstts.phonetics import english_syllables, korean_syllables
from pstts.timing import frame_labels
from pstts.vowels import load_or_build

CFG = Config()


def mask_ref(n: int, m: int, radius: int) -> np.ndarray:
    """tslearn 0.10.0 _sakoe_chiba_mask_generic 과 같은 규칙: 길이 차이만큼 띠를 넓힌다."""
    mask = np.zeros((n, m), dtype=bool)
    if n > m:
        width = n - m + radius
        for j in range(m):
            mask[max(0, j - radius):min(n, j + width) + 1, j] = True
    else:
        width = m - n + radius
        for i in range(n):
            mask[i, max(0, i - radius):min(m, i + width) + 1] = True
    return mask


def dtw_ref(cost: np.ndarray, radius: int) -> float:
    """국소 비용을 경로 따라 더하는 DTW(대각·아래·오른쪽, 가중치 1, 제곱근 없음)."""
    n, m = cost.shape
    mask = mask_ref(n, m, radius)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(n):
        for j in range(m):
            if mask[i, j]:
                acc[i + 1, j + 1] = cost[i, j] + min(acc[i, j + 1], acc[i + 1, j], acc[i, j])
    return float(acc[n, m])


def test_hand_computed() -> None:
    d = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert dtw_cost(d, 1) == 5.0                       # 대각 1 + 4. 제곱근(2.236...)이 아니다
    assert dtw_cost(np.array([[1.0], [2.0], [4.0]]), 0) == 7.0     # 한 줄로만 갈 수 있다
    e = np.arange(16, dtype=float).reshape(4, 4)
    assert dtw_cost(e, 0) == np.trace(e)              # 반경 0, 같은 길이 -> 대각선만


@pytest.mark.parametrize("seed", range(24))
def test_matches_numpy_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n, m = int(rng.integers(1, 41)), int(rng.integers(1, 41))
    if seed % 6 == 0:
        m = n                                          # 같은 길이도 섞는다
    cost = rng.random((n, m)) * rng.choice([1.0, 5.0])
    for radius in (0, 1, 2, 5):
        assert dtw_cost(cost, radius) == pytest.approx(dtw_ref(cost, radius), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("n, m, radius", [(5, 3, 1), (3, 5, 1), (7, 7, 0), (7, 7, 2), (1, 9, 1), (12, 4, 3)])
def test_mask_rule_matches_tslearn(n: int, m: int, radius: int) -> None:
    assert np.array_equal(np.asarray(sakoe_chiba_mask(n, m, radius)), mask_ref(n, m, radius))


def test_default_radius_is_one() -> None:
    # 대각선에서 3칸 비켜 간 길만 비용 0. 반경 1 이면 그 길로 못 가고, 반경 4 면 간다.
    cost = np.ones((8, 8))
    for i, j in [(0, 0), (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 7), (6, 7), (7, 7)]:
        cost[i, j] = 0.0
    _, default = dtw_path_from_metric(cost, metric="precomputed", global_constraint="sakoe_chiba")
    assert float(default) == dtw_cost(cost, 1) == dtw_ref(cost, 1)
    assert dtw_cost(cost, 4) == 0.0 < float(default)


def test_transpose_symmetric() -> None:
    rng = np.random.default_rng(11)
    cost = rng.random((23, 9))
    assert dtw_cost(cost, 2) == pytest.approx(dtw_cost(cost.T, 2))


def test_bad_input() -> None:
    with pytest.raises(ValueError):
        dtw_cost(np.zeros((0, 3)), 1)
    with pytest.raises(ValueError):
        dtw_cost(np.zeros(3), 1)
    with pytest.raises(ValueError):
        dtw_cost(np.zeros((2, 2)), -1)


def test_ps_distance_basic() -> None:
    table = load_or_build()
    a = ["a"] * 10
    assert CFG.sakoe_chiba_radius == CFG.iso_tol_frames == 26
    assert ps_distance(a, a, table, CFG) == 0.0
    # 모든 칸이 같은 비용이면 대각선(10칸)이 가장 싸다: 10 x d(a, u)
    assert ps_distance(a, ["u"] * 10, table, CFG) == pytest.approx(10 * table.distance("a", "u"))
    # 모음이 아닌 칸과 모음 사이는 null_cost
    assert ps_distance([None] * 4, ["a"] * 4, table, Config(null_cost=0.5)) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        ps_distance([], a, table, CFG)


BAND_CASES = {
    "I'm not mad at you. I'm just tired of waiting.": [
        "화난 게 아니야. 기다리다 지친 거야.", "나 화 안 났어. 기다리다 지쳤을 뿐이야.", "너한테 화난 거 아냐. 그냥 기다리다 지쳤어.",
        "화난 거 아니야, 기다리는 게 지겨울 뿐이야.", "안 화났어. 그냥 기다리다 지쳤어."],
    "I know you're scared. But if we stay here, we'll never find out what happened to her.": [
        "겁나는 거 알아요. 근데 여기 있으면 그녀 일은 영영 몰라요.", "무서운 거 알아요. 그래도 여기 있으면 그 여자 일을 절대 못 알아내요.",
        "겁먹은 거 알아요. 하지만 여기 있으면 무슨 일인지 영원히 몰라요."],
    "Excuse me, could you tell me where the nearest subway station is?": [
        "저기요, 가까운 지하철역이 어디인지 아세요?", "실례지만 제일 가까운 지하철역이 어디예요?", "죄송한데 가까운 지하철역 좀 알려 주시겠어요?"],
}


def _stretch_ratios(cfg: Config) -> list[float]:
    """글마다 (원문과 프레임 수가 같을 때 DTW) / (12프레임 넘게 늘이거나 줄였을 때 DTW 평균)."""
    table = load_or_build()
    ratios = []
    for en, kos in BAND_CASES.items():
        src = frame_labels(english_syllables(en), CFG.en_rate, CFG)
        n = len(src)
        for ko in kos:
            syls = korean_syllables(ko)
            by_offset = {}
            for d in range(-24, 25, 4):
                tgt = frame_labels(syls, len(syls) / ((n + d) * CFG.frame_sec), CFG)
                by_offset[len(tgt) - n] = ps_distance(src, tgt, table, cfg)
            at_equal = by_offset[min(by_offset, key=abs)]
            far = [v for off, v in by_offset.items() if abs(off) >= 12]
            ratios.append(at_equal / (sum(far) / len(far)))
    return ratios


def test_band_does_not_reward_length_mismatch() -> None:
    # tslearn 은 길이가 다르면 띠를 |n - m| 만큼 넓힌다. 반경 1 이면 같은 글이라도 원문과 프레임 수가 같을 때
    # DTW 가 가장 커서(길이가 맞는 후보가 진다) 2026-11-22 검토에서 기본 반경을 26프레임으로 바꿨다.
    assert all(r < 1.10 for r in _stretch_ratios(CFG)), "기본 반경에서 길이가 맞을 때 DTW 가 솟으면 안 된다"
    assert all(r > 1.15 for r in _stretch_ratios(Config(sakoe_chiba_radius=1)))    # 시험이 그 쏠림을 잡는지


def test_ps_distance_real_text_prefers_similar_vowels() -> None:
    table = load_or_build()
    src = frame_labels(english_syllables("Mama, papa, banana."), CFG.en_rate, CFG)
    close = frame_labels(korean_syllables("마마 파파 바나나"), CFG.en_rate, CFG)
    far = frame_labels(korean_syllables("우우 부부 구구구"), CFG.en_rate, CFG)
    d_close = ps_distance(src, close, table, CFG)
    d_far = ps_distance(src, far, table, CFG)
    assert 0 <= d_close < d_far
    assert ps_distance(close, src, table, CFG) == pytest.approx(ps_distance(src, close, table, CFG))
