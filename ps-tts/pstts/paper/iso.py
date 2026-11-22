"""논문 판 ISO(3.1절, 식 2~7): 길이와 뜻 조건으로 번역문을 고르고, 못 채우면 바꿔 쓴 후보를 최대 60번 검사한다.

단위는 초로 통일한다. 아래 값은 논문 식과 같다(프레임 단위 환산이 지워짐).
- dur_S = sum(durations(S_b)), dur_T = sum(durations(T_b)): 빈 토큰 b 를 끼운 토큰 열의 TTS 길이(초).
- 식 (2) K' = seconds / frame_sec_sa. seconds 는 입력 칸(쉼을 뺀 원문 말 길이, 초).
  칸이 없으면 K' = dur_S / frame_sec_sa (해석: 원문 화자가 TTS 와 같은 빠르기로 말한다고 둠. 그러면 R = 1).
- 식 (3) R(S_b) = K' / duration_TTS(S_b) = (K' x frame_sec_sa) / dur_S.
- 식 (4) K(T_b) = R(S_b) x duration_TTS(T_b) = R x dur_T / frame_sec_sa (SA 프레임).
- 식 (5), (7) C_sm = cos(LaBSE(S), LaBSE(T)) >= 0.75. 식은 S_b, T_b 에 매기는 꼴이지만 뜻을 재는 것이라
  문장 글에 매긴다(해석).
- 식 (6) K' - 26 <= K(T_b) <= K' + 26. 정확히 26 은 통과한다. 본문의 'within 30 frames (~350ms)' 는 식과 어긋나
  식을 따른다.

흐름(여러 줄을 함께 돌림):
1. 초벌 T0 를 검사한다. 통과면 ISO = T0 (status 'draft', iterations 0).
2. 아니면 차수마다 남은 줄에 min(iso_batch, iso_max - 검사한 수) 개를 묻는다. 바꿔 쓰는 바탕은 늘 T0 이고
   (논문: 번역문을 바꿔 씀), 이미 나온 후보(T0 포함)는 avoid 로 넘긴다. 받은 순서대로 검사하다가 두 조건을 처음
   채운 후보에서 멈춘다(status 'paraphrase'). '한 번' = 바꿔 쓴 후보 하나를 검사하는 것이고 초벌은 세지 않는다.
3. iso_max(60)개를 검사해도 없으면 초벌과 검사한 후보 전부에서 LaBSE 가 가장 높은 후보를 고른다. 같으면 먼저 나온 것
   (status 'fallback'). 3.1절 끝 'The candidate with the highest cosine similarity is selected'.
4. 모델 2 가 두 차수 연달아 새 후보를 하나도 주지 않으면 그 줄은 멈추고 3 으로 간다(해석, message 에 적음).
   모델 2 를 쓸 수 없게 되면(LLMUnavailable) 그 차수에 받은 후보까지 검사하고 남은 줄 모두 3 으로 간다.
5. 원문의 TTS 길이가 0 이라 R 을 잴 수 없거나 초벌이 비면 그 줄은 status 'error'(target None) 이다.
   초벌의 TTS 길이가 0 이면 길이 비율을 정할 수 없어 묻지 않고 3 으로 간다.
LaBSE 는 차수마다 모든 줄의 새 후보를 한 번에, TTS 길이도 언어마다 한 번에 잰다.
바꿔 쓰기와 LaBSE 는 주입받는다(pipeline 이 prompts.iso_generate 와 LabseScorer.similarity 를 넘김).
"""
from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .config import PaperConfig
from .prompts import STAGE_ISO, GenOutcome, IsoRequest, keep_new

LANG_EN = "en"
LANG_KO = "ko"
# 식 (6) 경계에서 부동소수 오차를 견디는 몫(프레임). 1e-9 프레임 = 약 1e-11 초라 판정에는 영향이 없다.
_EPS_FRAMES = 1e-9
# LaBSE 가 이만큼 안으로 같으면 같은 값으로 보고 먼저 나온 후보를 고른다.
_TIE = 1e-9

STATUS_DRAFT = "draft"
STATUS_PARAPHRASE = "paraphrase"
STATUS_FALLBACK = "fallback"
STATUS_ERROR = "error"


class DurationFrontend(Protocol):
    """ISO 가 쓰는 Frontend(frontend.py) 부분. 구조로만 맞춘다."""
    def tokens(self, text: str, lang: str) -> list[str]: ...
    def durations(self, seqs: list[list[str]], lang: str) -> list[np.ndarray]: ...


Paraphrase = Callable[[list[IsoRequest]], GenOutcome]
Similarity = Callable[[Sequence[str], Sequence[str]], np.ndarray]


@dataclass
class IsoInput:
    """ISO 에 넣는 대사 한 줄."""
    id: str
    source_en: str
    draft_ko: str                   # 초벌 T0(모델 1 번역문이나 입력의 draft_ko)
    seconds: float | None = None    # 쉼을 뺀 원문 말 길이(초). K' = seconds / frame_sec_sa


@dataclass
class IsoCandidate:
    """검사한(또는 받은) 후보 하나."""
    text: str
    origin: str                     # 'draft' | 'llm'
    round: int                      # 0 = 초벌, 1부터 = 바꿔 쓰기 차수
    iteration: int | None           # 0 = 초벌, 1~60 = 검사한 차례. 검사 전에 멈춰 안 본 후보는 None
    checked: bool                   # 두 조건 검사 차례에 들었는지(같은 차수에서 앞 후보가 통과하면 뒤는 False)
    dur_sec: float                  # TTS 길이(초, 빈 토큰 포함 토큰 열의 합)
    k_frames: float                 # 식 (4) K(T_b), SA 프레임
    ok_len: bool                    # 식 (6)
    labse: float                    # 식 (5) C_sm
    ok_sem: bool                    # 식 (7)
    passed: bool                    # 두 조건을 다 채움


@dataclass
class IsoLine:
    """줄 하나의 ISO 결과."""
    id: str
    source_en: str
    draft_ko: str
    seconds: float | None = None
    status: str = ""                # 'draft' | 'paraphrase' | 'fallback' | 'error'
    target: str | None = None       # ISO 문장(error 면 None)
    selected: int | None = None     # candidates 안의 번호
    iterations: int = 0             # 검사한 바꿔 쓰기 후보 수(초벌 제외)
    k_prime: float | None = None    # 식 (2) K'(SA 프레임)
    k_prime_basis: str = ""         # 'seconds'(입력 칸) | 'tts'(원문 TTS 길이, 해석)
    r: float | None = None          # 식 (3) R(S_b)
    src_dur_sec: float | None = None
    length_ratio: float | None = None   # 바꿔 쓰기 프롬프트의 K' / K(T0)(소수 둘째 자리)
    rounds: int = 0                 # 모델 2 에 물은 차수
    candidates: list[IsoCandidate] = field(default_factory=list)
    message: str = ""

    @property
    def chosen(self) -> IsoCandidate | None:
        return None if self.selected is None else self.candidates[self.selected]


@dataclass
class IsoRun:
    """ISO 단계 전체 결과와 통계."""
    lines: list[IsoLine]
    rounds: list[dict] = field(default_factory=list)    # 차수별 통계
    calls: int = 0                  # 모델 2 를 실제로 부른 횟수(재시도 포함, 캐시 제외)
    cache_hits: int = 0
    errors: list[str] = field(default_factory=list)
    stopped: str = ""               # 모델 2 를 쓸 수 없어 멈췄으면 그 까닭
    rows: list[dict] = field(default_factory=list)      # candidates.jsonl 재생 줄(stage 'iso')
    seconds: dict[str, float] = field(default_factory=lambda: {"tts": 0.0, "labse": 0.0, "llm": 0.0})


# ---------------------------------------------------------------- 식 (2)~(7)

def k_prime_of(seconds: float | None, src_dur_sec: float, frame_sec: float) -> tuple[float, str]:
    """식 (2) K'(SA 프레임)와 근거. seconds 가 있으면 seconds / frame_sec('seconds'),
    없으면 원문 TTS 길이 / frame_sec('tts', 해석)."""
    if not frame_sec > 0:
        raise ValueError(f"SA 프레임 길이는 0 보다 커야 합니다: {frame_sec}")
    if seconds is not None:
        return float(seconds) / frame_sec, "seconds"
    return float(src_dur_sec) / frame_sec, "tts"


def speaking_rate(k_prime: float, src_dur_sec: float, frame_sec: float) -> float:
    """식 (3) R(S_b) = K' / duration_TTS(S_b) 를 초로 적은 값 (K' x frame_sec) / dur_S."""
    if not src_dur_sec > 0:
        raise ValueError("원문의 TTS 길이가 0 이라 말 빠르기 R(식 3)을 잴 수 없습니다.")
    return float(k_prime) * frame_sec / float(src_dur_sec)


def k_frames(r: float, tgt_dur_sec: float, frame_sec: float) -> float:
    """식 (4) K(T_b) = R(S_b) x duration_TTS(T_b) 를 SA 프레임으로: R x dur_T / frame_sec."""
    return float(r) * float(tgt_dur_sec) / frame_sec


def ok_len(k: float, k_prime: float, tol_frames: float) -> bool:
    """식 (6) K' - tol <= K(T_b) <= K' + tol. 정확히 tol 만큼 벗어나면 통과(부동소수 오차 1e-9 프레임 허용)."""
    return abs(float(k) - float(k_prime)) <= float(tol_frames) + _EPS_FRAMES


def ok_sem(labse: float, labse_min: float) -> bool:
    """식 (7) C_sm(S_b, T_b) >= labse_min(0.75)."""
    return float(labse) >= float(labse_min)


def fallback_index(cands: Sequence[IsoCandidate]) -> int | None:
    """3.1절 끝의 대안: 검사한 후보(초벌 포함) 가운데 LaBSE 가 가장 높은 후보의 번호. 같으면 먼저 나온 것."""
    best: int | None = None
    for i, c in enumerate(cands):
        if not c.checked:
            continue
        if best is None or c.labse > cands[best].labse + _TIE:
            best = i
    return best


# ---------------------------------------------------------------- 재기

def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _totals(durs: Sequence[np.ndarray], seqs: Sequence[Sequence[str]], lang: str) -> list[float]:
    """토큰별 길이(초) 배열을 확인하고 합한다."""
    if len(durs) != len(seqs):
        raise ValueError(f"TTS 길이가 {len(seqs)}개 글 가운데 {len(durs)}개만 왔습니다({lang}).")
    out = []
    for d, s in zip(durs, seqs):
        a = np.asarray(d, dtype=np.float64)
        if a.shape != (len(s),):
            raise ValueError(f"TTS 길이 꼴 {a.shape} 가 토큰 {len(s)}개와 맞지 않습니다({lang}).")
        if not np.all(np.isfinite(a)) or np.any(a < 0):
            raise ValueError(f"TTS 길이에 음수나 NaN 이 있습니다({lang}).")
        out.append(float(a.sum()))
    return out


def tts_seconds(front: DurationFrontend, texts: Sequence[str], lang: str) -> list[float]:
    """글마다 빈 토큰을 끼운 토큰 열의 TTS 길이 합(초). 길이 예측은 한 번에 부른다."""
    if not texts:
        return []
    seqs = [list(front.tokens(t, lang)) for t in texts]
    return _totals(front.durations(seqs, lang), seqs, lang)


def labse_scores(similarity: Similarity, srcs: Sequence[str], tgts: Sequence[str]) -> np.ndarray:
    """짝마다 LaBSE 코사인(한 번에). 꼴과 값을 확인한다."""
    if not tgts:
        return np.zeros(0, dtype=np.float64)
    sims = np.asarray(similarity(list(srcs), list(tgts)), dtype=np.float64)
    if sims.shape != (len(tgts),):
        raise ValueError(f"LaBSE 점수 꼴 {sims.shape} 가 후보 {len(tgts)}개와 맞지 않습니다.")
    if not np.all(np.isfinite(sims)):
        raise ValueError("LaBSE 점수에 NaN 이나 무한대가 있습니다.")
    return sims


def _assess(line: IsoLine, text: str, origin: str, rnd: int, dur: float, labse: float,
            cfg: PaperConfig) -> IsoCandidate:
    k = k_frames(line.r, dur, cfg.frame_sec_sa)
    a = ok_len(k, line.k_prime, cfg.iso_tol_frames)
    b = ok_sem(labse, cfg.labse_min)
    return IsoCandidate(text=text, origin=origin, round=rnd, iteration=None, checked=False, dur_sec=float(dur),
                        k_frames=k, ok_len=a, labse=float(labse), ok_sem=b, passed=a and b)


def _cap_message(line: IsoLine, cfg: PaperConfig) -> str:
    return (f"바꿔 쓴 후보 {line.iterations}개(최대 {cfg.iso_max}번)를 검사해도 두 조건(식 6, 7)을 다 채운 후보가 없어 "
            "LaBSE 가 가장 높은 후보를 고름(3.1절 끝)")


def _fall_back(line: IsoLine, message: str) -> None:
    idx = fallback_index(line.candidates)
    if idx is None:
        raise ValueError(f"{line.id}: LaBSE 를 잰 후보가 없어 대안을 고를 수 없습니다.")
    line.status, line.selected, line.target = STATUS_FALLBACK, idx, line.candidates[idx].text
    line.message = message


def check_inputs(lines: Sequence[IsoInput], cfg: PaperConfig) -> None:
    """입력과 설정을 확인한다. 틀리면 한국어 ValueError(모델을 부르기 전에 멈춤)."""
    if not lines:
        raise ValueError("대사가 하나도 없습니다.")
    seen: set[str] = set()
    for x in lines:
        if x.id in seen:
            raise ValueError(f"같은 id 가 두 번 있습니다: {x.id}")
        seen.add(x.id)
        if not x.source_en.strip():
            raise ValueError(f"{x.id}: 영어 대사가 비었습니다.")
        if x.seconds is not None and not (math.isfinite(float(x.seconds)) and float(x.seconds) > 0):
            raise ValueError(f"{x.id}: seconds 는 0 보다 큰 수(초)여야 합니다: {x.seconds}")
    if not cfg.frame_sec_sa > 0:
        raise ValueError(f"frame_sec_sa 는 0 보다 커야 합니다: {cfg.frame_sec_sa}")
    if cfg.iso_tol_frames < 0:
        raise ValueError(f"iso_tol_frames 는 0 이상이어야 합니다: {cfg.iso_tol_frames}")
    if cfg.iso_batch < 1 or cfg.iso_max < 0:
        raise ValueError(f"iso_batch 는 1 이상, iso_max 는 0 이상이어야 합니다: {cfg.iso_batch}, {cfg.iso_max}")
    if not -1.0 <= cfg.labse_min <= 1.0:
        raise ValueError(f"labse_min 은 -1 과 1 사이여야 합니다: {cfg.labse_min}")


# ---------------------------------------------------------------- 흐름

def run_iso(lines: Sequence[IsoInput], front: DurationFrontend, cfg: PaperConfig, *,
            paraphrase: Paraphrase, similarity: Similarity,
            log: Callable[[str], None] | None = None) -> IsoRun:
    """여러 줄의 ISO 를 함께 돌린다(모듈 설명의 흐름).

    paraphrase(reqs) 는 다듬고 겹침을 뺀 후보를 GenOutcome 으로 돌려준다(prompts.iso_generate).
    similarity(srcs, tgts) 는 짝마다 LaBSE 코사인 배열이다(LabseScorer.similarity).
    """
    log = log or _log
    check_inputs(lines, cfg)
    run = IsoRun(lines=[IsoLine(id=str(x.id), source_en=x.source_en, draft_ko=" ".join((x.draft_ko or "").split()),
                                seconds=None if x.seconds is None else float(x.seconds)) for x in lines])
    f = cfg.frame_sec_sa

    # 식 (2), (3): 원문 TTS 길이, K', R
    t0 = time.perf_counter()
    for line, dur in zip(run.lines, tts_seconds(front, [x.source_en for x in run.lines], LANG_EN)):
        line.src_dur_sec = dur
        line.k_prime, line.k_prime_basis = k_prime_of(line.seconds, dur, f)
        if not dur > 0:
            line.status, line.message = STATUS_ERROR, "원문의 TTS 길이가 0 이라 말 빠르기 R(식 3)을 잴 수 없음"
        elif not line.draft_ko:
            line.status, line.message = STATUS_ERROR, "초벌(T0)이 비어 ISO 를 할 수 없음"
        else:
            line.r = speaking_rate(line.k_prime, dur, f)
    ready = [line for line in run.lines if not line.status]
    draft_durs = tts_seconds(front, [line.draft_ko for line in ready], LANG_KO)
    run.seconds["tts"] += time.perf_counter() - t0

    # 초벌 T0 검사(식 4~7)
    t0 = time.perf_counter()
    sims = labse_scores(similarity, [line.source_en for line in ready], [line.draft_ko for line in ready])
    run.seconds["labse"] += time.perf_counter() - t0
    for line, dur, sim in zip(ready, draft_durs, sims):
        cand = _assess(line, line.draft_ko, "draft", 0, dur, float(sim), cfg)
        cand.iteration, cand.checked = 0, True
        line.candidates.append(cand)
        if cand.passed:
            line.status, line.selected, line.target = STATUS_DRAFT, 0, line.draft_ko
    active = [line for line in ready if not line.status]
    for line in active:
        k0 = line.candidates[0].k_frames
        if k0 > 0:
            # 소수 둘째 자리. 0.005 보다 작은 비율이 0 으로 사라지지 않게 그 자릿수의 가장 작은 값(0.01)에서 멈춘다
            line.length_ratio = max(0.01, round(line.k_prime / k0, 2))
        else:
            _fall_back(line, "초벌의 TTS 길이가 0 이라 길이 비율(K'/K(T0))을 정할 수 없어 바꿔 쓰기를 묻지 않고 "
                             "LaBSE 가 가장 높은 후보를 고름")
    active = [line for line in active if not line.status]
    n_error = sum(1 for line in run.lines if line.status == STATUS_ERROR)
    n_draft = sum(1 for line in ready if line.status == STATUS_DRAFT)
    log(f"[ISO] 초벌 {len(ready)}줄 검사: 통과 {n_draft}줄, 바꿔 쓰기 필요 {len(active)}줄"
        + (f", 오류 {n_error}줄" if n_error else ""))

    empty_streak = {line.id: 0 for line in active}
    k = 0
    while True:
        for line in active:
            if line.iterations >= cfg.iso_max:          # iso_max 가 0 이면 묻지 않고 곧바로 대안
                _fall_back(line, _cap_message(line, cfg))
        active = [line for line in active if not line.status]
        if not active:
            break
        k += 1
        reqs = [IsoRequest(id=line.id, en=line.source_en, ko=line.draft_ko, length_ratio=float(line.length_ratio),
                           n_new=min(cfg.iso_batch, cfg.iso_max - line.iterations), round=k,
                           avoid=[c.text for c in line.candidates]) for line in active]
        log(f"[ISO] {k}차: {len(reqs)}줄에 바꿔 쓰기 요청(줄마다 최대 {max(r.n_new for r in reqs)}개)")
        t0 = time.perf_counter()
        res = paraphrase(reqs)
        t_llm = time.perf_counter() - t0
        run.seconds["llm"] += t_llm
        run.calls += int(res.calls)
        run.cache_hits += int(res.cache_hits)
        run.errors.extend(f"ISO {k}차: {e}" for e in res.errors)

        fresh: list[list[str]] = []
        for line, rq in zip(active, reqs):
            line.rounds = k
            got = keep_new(rq.avoid, res.candidates.get(line.id, []), rq.n_new)
            fresh.append(got)
            if got:
                run.rows.append({"id": line.id, "stage": STAGE_ISO, "round": k, "candidates": list(got)})
        pairs = [(line, text) for line, got in zip(active, fresh) for text in got]
        t0 = time.perf_counter()
        durs = tts_seconds(front, [text for _, text in pairs], LANG_KO)
        t_tts = time.perf_counter() - t0
        t0 = time.perf_counter()
        sims = labse_scores(similarity, [line.source_en for line, _ in pairs], [text for _, text in pairs])
        t_lab = time.perf_counter() - t0
        run.seconds["tts"] += t_tts
        run.seconds["labse"] += t_lab

        pos = n_checked = n_passed = 0
        for line, got in zip(active, fresh):
            for j, text in enumerate(got):
                cand = _assess(line, text, "llm", k, durs[pos + j], float(sims[pos + j]), cfg)
                if not line.status:             # 받은 순서대로 검사하다 처음 통과한 후보에서 멈춤
                    line.iterations += 1
                    cand.iteration, cand.checked = line.iterations, True
                    n_checked += 1
                    if cand.passed:
                        line.status, line.selected, line.target = STATUS_PARAPHRASE, len(line.candidates), text
                        n_passed += 1
                line.candidates.append(cand)
            pos += len(got)
            if line.status:
                continue
            empty_streak[line.id] = 0 if got else empty_streak[line.id] + 1
            if line.iterations >= cfg.iso_max:
                _fall_back(line, _cap_message(line, cfg))
            elif empty_streak[line.id] >= 2:
                _fall_back(line, "모델 2 가 두 차수 연달아 새 후보를 주지 않아 멈추고 LaBSE 가 가장 높은 후보를 고름")
        if res.stopped:
            run.stopped = res.stopped
            for line in active:
                if not line.status:
                    _fall_back(line, "모델 2 를 쓸 수 없어 바꿔 쓰기를 멈추고 LaBSE 가 가장 높은 후보를 고름: "
                                     + res.stopped)
        run.rounds.append({"round": k, "lines": len(reqs), "requested": sum(r.n_new for r in reqs),
                           "new": len(pairs), "checked": n_checked, "passed": n_passed, "calls": int(res.calls),
                           "cache_hits": int(res.cache_hits), "llm_seconds": round(t_llm, 2),
                           "labse_seconds": round(t_lab, 2), "tts_seconds": round(t_tts, 2)})
        log(f"[ISO] {k}차: 새 후보 {len(pairs)}개, 검사 {n_checked}개, 통과 {n_passed}줄, 호출 {res.calls}번, "
            f"캐시 {res.cache_hits}번, 모델 2 {t_llm:.1f}초 / LaBSE {t_lab:.1f}초")
        if res.stopped:
            log(f"[ISO] 모델 2 를 쓸 수 없어 멈춤: {res.stopped}")
            break
        active = [line for line in active if not line.status]

    counts = {s: sum(1 for line in run.lines if line.status == s)
              for s in (STATUS_DRAFT, STATUS_PARAPHRASE, STATUS_FALLBACK, STATUS_ERROR)}
    log(f"[ISO] 끝: 초벌 {counts[STATUS_DRAFT]}줄, 바꿔 쓰기 {counts[STATUS_PARAPHRASE]}줄, "
        f"대안 {counts[STATUS_FALLBACK]}줄, 오류 {counts[STATUS_ERROR]}줄. "
        f"모델 2 호출 {run.calls}번(캐시 {run.cache_hits}번)")
    for key in run.seconds:
        run.seconds[key] = round(run.seconds[key], 2)
    return run
