"""논문 판 PS(3.2절, 식 9~11)와 PS-Comet(3.3절, 식 20).

1. 후보마다 3.1절의 음소화와 토큰화로 T_b,i 를 만들고, 모음표(Fig. 5c)에 없는 토큰은 null(None)로 바꾼다
   (front.labels). 빈 토큰, 공백, 자음, 문장부호도 null 이다.
2. TTS 길이 예측기(front.durations)로 토큰마다 길이 d_i 를 얻고 그만큼 되풀이해 T'_b,i 를 만든다(expand).
   T'_b,i = {b*d0, t1*d1, ..., b*d2m}. 원문도 같게 S'_b 를 만든다.
3. 국소 비용 = 모음 거리(식 9, front.cost)로 S'_b 와 T'_b,i 사이 DTW 를 잰다(식 10). Sakoe-Chiba 제약,
   공개 라이브러리 [44] tslearn. 누적 비용 = 경로 위 국소 비용의 합(제곱근 없음, pstts.dtw.dtw_cost).
   반경은 논문에 없어 tslearn 기본값(None = 반경 1)을 None 그대로 넘긴다(해석).
4. PS = DTW 가 가장 작은 후보(식 11). 같으면 먼저 나온 것.
5. PS-Comet: DTW 를 후보들 안에서 [0,1] 로 최소-최대 정규화하고 뒤집는다: inv = 1 - (d - min) / (max - min).
   max == min 이면 모두 inv = 1(해석). 점수 = alpha x inv + beta x COMET(S, T_i)(식 20, alpha 1.6, beta 0.4).
   COMET 은 그대로 쓴다. 점수가 가장 큰 후보, 같으면 먼저 나온 것.
프레임이 0개인 후보(빈 글 등)는 DTW 를 None 으로 두고 PS, PS-Comet 의 정규화와 고르기에서 뺀다.
식 (20) 은 정규화한 DTW 를 '뒤집는다' 고 적었다(본문). 그래서 식에 적힌 d_DTW 자리에 inv 를 쓴다.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..dtw import dtw_cost

LANG_EN = "en"
LANG_KO = "ko"
# 값이 이만큼 안으로 같으면(상대 오차) 같은 값으로 본다: 정규화 max == min, 고르기에서 먼저 나온 후보 우선
_SAME = 1e-9

Label = str | None
CostFn = Callable[[Sequence[Label], Sequence[Label]], np.ndarray]


class PsFrontend(Protocol):
    """PS 가 쓰는 Frontend(frontend.py) 부분. 구조로만 맞춘다."""
    frame_sec: float
    def tokens(self, text: str, lang: str) -> list[str]: ...
    def durations(self, seqs: list[list[str]], lang: str) -> list[np.ndarray]: ...
    def labels(self, toks: list[str], lang: str) -> list[str | None]: ...
    def cost(self, src_labels: Sequence[str | None], tgt_labels: Sequence[str | None]) -> np.ndarray: ...


@dataclass
class PsCandidate:
    """PS 후보 하나와 그 값."""
    text: str
    dur_sec: float | None = None    # TTS 길이(초, 빈 토큰 포함 토큰 열의 합)
    n_frames: int = 0               # 늘린 프레임 수(T'_b,i 길이)
    dtw: float | None = None        # 식 (10). 프레임이 0개면 None
    dtw_inv: float | None = None    # 정규화하고 뒤집은 DTW(클수록 입 모양이 비슷)
    comet: float | None = None      # COMET(S, T_i), 그대로
    score: float | None = None      # 식 (20)


@dataclass
class PsChoice:
    """한 줄의 고르기 결과(candidates 안의 번호)."""
    ps: int | None                  # 식 (11)
    ps_comet: int | None            # 식 (20)
    excluded: list[int] = field(default_factory=list)   # 프레임이 0개라 뺀 후보


# ---------------------------------------------------------------- 늘리기

def frame_counts(durs_sec: Sequence[float] | np.ndarray, frame_sec: float) -> np.ndarray:
    """토큰별 프레임 수. 누적 시간을 프레임으로 나눠 반올림(0.5 는 올림)한 경계의 차이라서
    반올림 어긋남이 쌓이지 않고, 합 = 반올림(전체 길이 / frame_sec) 이다. 0 프레임 토큰도 있다."""
    if not (math.isfinite(float(frame_sec)) and frame_sec > 0):
        raise ValueError(f"프레임 길이는 0 보다 커야 합니다: {frame_sec}")
    d = np.asarray(durs_sec, dtype=np.float64)
    if d.ndim != 1:
        raise ValueError(f"토큰 길이는 1차원이어야 합니다: {d.shape}")
    if d.size == 0:
        return np.zeros(0, dtype=np.int64)
    if not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError("토큰 길이에 음수나 NaN 이 있습니다.")
    edges = np.floor(np.cumsum(d) / float(frame_sec) + 0.5).astype(np.int64)
    return np.diff(edges, prepend=np.int64(0))


def expand(labels: Sequence[Label], durs_sec: Sequence[float] | np.ndarray, frame_sec: float) -> list[Label]:
    """라벨을 토큰별 프레임 수만큼 되풀이한다(T'_b = {b*d0, t1*d1, ...}). 0 프레임 토큰은 사라진다."""
    counts = frame_counts(durs_sec, frame_sec)
    if len(labels) != len(counts):
        raise ValueError(f"라벨 {len(labels)}개와 길이 {len(counts)}개의 수가 다릅니다.")
    out: list[Label] = []
    for lab, n in zip(labels, counts):
        out.extend([lab] * int(n))
    return out


def frames_of(front: PsFrontend, texts: Sequence[str], lang: str) -> list[tuple[list[Label], float]]:
    """글마다 (늘린 프레임 라벨 열, TTS 길이 초). 토큰 -> 길이(한 번에) -> 라벨 -> 늘리기."""
    if not texts:
        return []
    seqs = [list(front.tokens(t, lang)) for t in texts]
    durs = front.durations(seqs, lang)
    if len(durs) != len(seqs):
        raise ValueError(f"TTS 길이가 {len(seqs)}개 글 가운데 {len(durs)}개만 왔습니다({lang}).")
    out = []
    for toks, d in zip(seqs, durs):
        a = np.asarray(d, dtype=np.float64)
        if a.shape != (len(toks),):
            raise ValueError(f"TTS 길이 꼴 {a.shape} 가 토큰 {len(toks)}개와 맞지 않습니다({lang}).")
        labels = list(front.labels(toks, lang))
        out.append((expand(labels, a, front.frame_sec), float(a.sum())))
    return out


# ---------------------------------------------------------------- DTW 와 고르기

def dtw(src_frames: Sequence[Label], tgt_frames: Sequence[Label], cost_fn: CostFn, radius: int | None) -> float | None:
    """식 (10): 국소 비용 행렬(cost_fn)의 Sakoe-Chiba DTW 누적 비용. 어느 쪽이든 프레임이 0개면 None.

    tslearn dtw_path_from_metric(cost, metric='precomputed', global_constraint='sakoe_chiba',
    sakoe_chiba_radius=radius) 를 부른다. radius 가 None 이면 None 그대로 넘긴다(tslearn 기본, 반경 1).
    """
    if len(src_frames) == 0 or len(tgt_frames) == 0:
        return None
    cost = np.asarray(cost_fn(src_frames, tgt_frames), dtype=np.float64)
    if cost.shape != (len(src_frames), len(tgt_frames)):
        raise ValueError(f"국소 비용 행렬 꼴 {cost.shape} 가 ({len(src_frames)}, {len(tgt_frames)}) 이 아닙니다.")
    return dtw_cost(cost, radius)


def _beats(a: float, b: float) -> bool:
    """a 가 b 보다 확실히 큰지(같은 값으로 볼 차이 안이면 거짓)."""
    return a > b + _SAME * max(1.0, abs(a), abs(b))


def select_ps(dtws: Sequence[float | None]) -> int | None:
    """식 (11): DTW 가 가장 작은 후보의 번호. None 은 뺀다. 같으면 먼저 나온 것. 고를 후보가 없으면 None."""
    best: int | None = None
    for i, d in enumerate(dtws):
        if d is None:
            continue
        if not math.isfinite(float(d)):
            raise ValueError(f"{i}번 후보의 DTW 가 수가 아닙니다: {d}")
        if best is None or _beats(float(dtws[best]), float(d)):
            best = i
    return best


def ps_comet(dtws: Sequence[float | None], comets: Sequence[float | None], alpha: float, beta: float
             ) -> tuple[list[float | None], list[float | None], int | None]:
    """식 (20): (inv, score, 고른 번호). DTW 가 None 인 후보는 정규화와 고르기에서 빼고 inv, score 도 None 이다.

    inv = 1 - (d - min) / (max - min) (max == min 이면 1), score = alpha x inv + beta x COMET. 가장 큰 점수,
    같으면 먼저 나온 것.
    """
    if len(dtws) != len(comets):
        raise ValueError(f"DTW {len(dtws)}개와 COMET {len(comets)}개의 수가 다릅니다.")
    n = len(dtws)
    inv: list[float | None] = [None] * n
    score: list[float | None] = [None] * n
    valid = [i for i, d in enumerate(dtws) if d is not None]
    for i in valid:
        if not math.isfinite(float(dtws[i])):
            raise ValueError(f"{i}번 후보의 DTW 가 수가 아닙니다: {dtws[i]}")
        if comets[i] is None or not math.isfinite(float(comets[i])):
            raise ValueError(f"{i}번 후보의 COMET 점수가 없습니다: {comets[i]}")
    if not valid:
        return inv, score, None
    lo = min(float(dtws[i]) for i in valid)
    hi = max(float(dtws[i]) for i in valid)
    same = not _beats(hi, lo)
    best: int | None = None
    for i in valid:
        inv[i] = 1.0 if same else 1.0 - (float(dtws[i]) - lo) / (hi - lo)
        score[i] = alpha * inv[i] + beta * float(comets[i])
        if best is None or _beats(score[i], score[best]):
            best = i
    return inv, score, best


# ---------------------------------------------------------------- 줄 하나

def measure_line(front: PsFrontend, source_en: str, texts: Sequence[str], radius: int | None
                 ) -> tuple[list[PsCandidate], int]:
    """원문 S'_b 와 후보마다 T'_b,i 를 만들고 DTW(식 10)를 잰다. (후보 기록, 원문 프레임 수).

    원문 프레임이 0개면 모든 후보의 DTW 가 None 이다.
    """
    src_frames, _ = frames_of(front, [source_en], LANG_EN)[0]
    out = []
    for text, (frames, dur) in zip(texts, frames_of(front, list(texts), LANG_KO)):
        out.append(PsCandidate(text=text, dur_sec=dur, n_frames=len(frames),
                               dtw=dtw(src_frames, frames, front.cost, radius)))
    return out, len(src_frames)


def select_line(cands: Sequence[PsCandidate], alpha: float, beta: float) -> PsChoice:
    """PS(식 11)와 PS-Comet(식 20)을 고르고 후보의 dtw_inv, score 를 채운다. COMET 은 미리 채워 둔다."""
    dtws = [c.dtw for c in cands]
    inv, score, pc = ps_comet(dtws, [c.comet for c in cands], alpha, beta)
    for c, a, s in zip(cands, inv, score):
        c.dtw_inv, c.score = a, s
    return PsChoice(ps=select_ps(dtws), ps_comet=pc, excluded=[i for i, d in enumerate(dtws) if d is None])
