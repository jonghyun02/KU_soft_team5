"""모음 DTW(논문 식 10). 공개 라이브러리 [44] = tslearn 으로 잰다.

설치된 tslearn 0.10.0 소스(tslearn/metrics/_dtw.py, _masks.py)에서 확인한 것:
- dtw_path_from_metric(..., metric="precomputed") 는 (경로, acc_cost_mat[-1, -1]) 를 돌려준다.
  누적 행렬은 _njit_acc_matrix_from_dist_matrix 가 cum[i+1, j+1] = D[i, j] + min(위, 왼쪽, 대각) 로 만든다.
  곧 경로 위 국소 비용의 합이고 제곱근을 씌우지 않는다(제곱근은 dtw()·dtw_path() 에만 있다).
- sakoe_chiba_radius 를 None 으로 주면 반경 1 을 쓴다(_compute_mask_generic).
- 길이가 다르면(sz1 > sz2) 띠 너비 = sz1 - sz2 + radius 이고, 열 j 마다 행 max(0, j - radius) ~
  min(sz1, j + 너비) 를 연다(sz1 <= sz2 면 행·열을 바꿔 같은 규칙). 그래서 끝 (sz1-1, sz2-1) 까지 늘 길이 있다.
  곧 길이가 어긋난 만큼 휠 자리가 넓어진다. 반경이 작으면(tslearn 기본 1) 원문과 프레임 수가 같은 후보만 대각선에
  묶여 DTW 가 커지고, 길이가 어긋난 후보가 이긴다. 그래서 Config.sakoe_chiba_radius 기본값은 식 (6) 의 허용과 같은
  26프레임이다(근거와 잰 값은 config.py).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import Config
from .vowels import VowelTable, cost_matrix


def dtw_cost(cost: np.ndarray, radius: int | None) -> float:
    """국소 비용 행렬의 DTW 누적 비용(Sakoe-Chiba 반경 radius).

    radius 가 None 이면 None 그대로 tslearn 에 넘긴다(tslearn 기본값, 반경 1. 논문 판이 쓴다).
    """
    from tslearn.metrics import dtw_path_from_metric

    mat = np.ascontiguousarray(cost, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] == 0 or mat.shape[1] == 0:
        raise ValueError(f"DTW 비용 행렬이 비었거나 2차원이 아닙니다: {mat.shape}")
    if radius is not None and radius < 0:
        raise ValueError(f"Sakoe-Chiba 반경은 0 이상이어야 합니다: {radius}")
    _, acc = dtw_path_from_metric(mat, metric="precomputed", global_constraint="sakoe_chiba",
                                  sakoe_chiba_radius=None if radius is None else int(radius))
    return float(acc)


def ps_distance(src_labels: Sequence[str | None], tgt_labels: Sequence[str | None],
                table: VowelTable, cfg: Config) -> float:
    """원문 프레임 열과 후보 프레임 열 사이 모음 DTW 거리(작을수록 입 모양이 비슷)."""
    if not src_labels or not tgt_labels:
        raise ValueError("프레임 열이 비어서 모음 DTW 를 잴 수 없습니다.")
    cost = cost_matrix(src_labels, tgt_labels, table, cfg.null_cost)
    return dtw_cost(cost, cfg.sakoe_chiba_radius)
