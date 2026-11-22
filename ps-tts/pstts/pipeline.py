"""전체 흐름: 길이 목표 -> 모델 1 초벌 -> 모델 2 후보(차수 반복) -> 거르기 -> PS-Comet 선택.

1. 줄마다 길이 목표(ruler.target_for).
2. 초벌이 없는 줄만 모델 1 로 한 번에 번역하고 곧바로 모델을 내린다(GPU 메모리 비움).
3. 후보 풀 = [초벌(차수 0)]. LaBSE 로 재고 거른다(원문에 숫자가 있으면 숫자를 말하는 꼴로 푼 원문과도 재서 큰 값).
4. 차수 1..max_rounds: 통과 후보가 min_pass 개보다 적고 후보가 max_candidates 개보다 적은 줄만 모델 2 에 묻는다.
   avoid = 지금까지 나온 후보, feedback = 탈락 후보 몇 개와 그 까닭. 새 후보는 다듬고 겹침을 뺀 뒤
   LaBSE(여러 줄 한 번에)로 재고 거른다. 차수가 끝나면 말투 미지정 줄은 모델 2 후보 다수의 말투로 다시 거른다.
5. LaBSE 를 내리고 COMET 을 올려 고를 묶음(여러 줄 한 번에)을 잰다. 입이 보이는 줄은 모음 DTW 도 잰다.
   묶음은 통과 후보, 없으면 논문 식 6·7(길이, LaBSE)만 채운 후보, 그것도 없으면 LaBSE 가 가장 높은 후보다.
   앞의 둘은 PS-Comet 으로 고른다(select.selection_pool). --comet-ref draft 면 초벌은 참조라서 묶음에서 뺀다.
모델은 하나씩 올리고 내린다: 모델 1 -> LaBSE -> COMET. 진행 상황은 stderr 에 한국어 한 줄씩 찍는다.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import numpy as np

from .config import Config
from .llm import GenReply, GenRequest, LLMUnavailable, make_backend
from .records import REGISTERS, Candidate, LineItem, LineResult, Target
from .ruler import target_for
from .select import (POOL_ISO, POOL_LABSE, POOL_NONE, POOL_PASS, add_dtw, apply_line_register, assess, ps_comet,
                     selection_pool)
from .textnorm import clean_candidate, dedupe_key, speak_english_numbers
from .vowels import VowelTable, load_or_build

LLM_BACKENDS = ("claude", "openai", "file")
COMET_REFS = ("none", "draft")
# 차수마다 모델 2 에 넘기는 탈락 후보 수. 프롬프트를 짧게 두려고 여기서 정했다.
# 까닭의 종류(길이, 뜻, 말투)를 보여 주는 데는 이만큼이면 된다.
FEEDBACK_MAX = 6


class Drafter(Protocol):
    def translate(self, sentences: Sequence[str]) -> list[str]: ...
    def close(self) -> None: ...


class Similarity(Protocol):
    def similarity(self, srcs: Sequence[str], tgts: Sequence[str]) -> np.ndarray: ...
    def close(self) -> None: ...


class Scorer(Protocol):
    def score(self, srcs: Sequence[str], mts: Sequence[str], refs: Sequence[str] | None = None) -> np.ndarray: ...
    def close(self) -> None: ...


class Generator(Protocol):
    def generate(self, reqs: list[GenRequest]) -> GenReply: ...


@dataclass
class RunInfo:
    """실행 기록(run.json 과 report.md 에 씀)."""
    started_at: str = ""
    finished_at: str = ""
    device: str = ""
    seconds: dict[str, float] = field(default_factory=dict)     # 단계별 걸린 시간
    rounds: list[dict] = field(default_factory=list)            # 차수별 요청 줄 수·새 후보·호출·시간
    llm_backend: str = ""
    llm_calls: int = 0              # 실제로 부른 횟수(재시도 포함, 캐시 적중 제외)
    llm_cache_hits: int = 0         # 캐시로 답한 조각 수
    llm_errors: list[str] = field(default_factory=list)
    llm_stopped: str = ""           # 모델 2 를 쓸 수 없어 멈췄으면 그 까닭
    model1: dict = field(default_factory=dict)
    comet_checkpoint: str | None = None
    status_counts: dict[str, int] = field(default_factory=dict)


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def check_config(cfg: Config) -> None:
    """실행 전에 설정값을 확인한다. 틀리면 한국어 ValueError."""
    problems = []
    if cfg.llm not in LLM_BACKENDS:
        problems.append(f"llm 은 {', '.join(LLM_BACKENDS)} 중 하나여야 합니다: {cfg.llm}")
    if cfg.comet_ref not in COMET_REFS:
        problems.append(f"comet_ref 는 {', '.join(COMET_REFS)} 중 하나여야 합니다: {cfg.comet_ref}")
    for name, low in (("n_per_round", 1), ("max_rounds", 0), ("min_pass", 1), ("max_candidates", 1),
                      ("llm_workers", 1), ("llm_chunk", 1), ("nmt_beam", 1), ("cpu_threads", 1),
                      ("comet_batch", 1)):
        if int(getattr(cfg, name)) < low:
            problems.append(f"{name} 은 {low} 이상이어야 합니다: {getattr(cfg, name)}")
    if cfg.alpha < 0 or cfg.beta < 0:
        problems.append(f"alpha, beta 는 0 이상이어야 합니다: {cfg.alpha}, {cfg.beta}")
    if not -1.0 <= cfg.labse_min <= 1.0:
        problems.append(f"labse_min 은 -1 과 1 사이여야 합니다: {cfg.labse_min}")
    if cfg.coef <= 0 or cfg.en_rate <= 0:
        problems.append(f"coef 와 en_rate 는 0 보다 커야 합니다: {cfg.coef}, {cfg.en_rate}")
    if cfg.tol_syllables is not None and cfg.tol_syllables < 0:
        problems.append(f"허용 오차는 0 이상이어야 합니다: {cfg.tol_syllables}")
    if problems:
        raise ValueError(" / ".join(problems))


def check_items(items: Sequence[LineItem]) -> None:
    """대사 목록을 확인한다(빈 목록, 겹친 id, 빈 영어 대사, 모르는 말투)."""
    if not items:
        raise ValueError("대사가 하나도 없습니다.")
    seen: set[str] = set()
    for it in items:
        if it.id in seen:
            raise ValueError(f"같은 id 가 두 번 있습니다: {it.id}")
        seen.add(it.id)
        if not it.source_en.strip():
            raise ValueError(f"{it.id}: 영어 대사가 비었습니다.")
        if it.register not in REGISTERS:
            raise ValueError(f"{it.id}: 알 수 없는 말투 '{it.register}' ({', '.join(REGISTERS)} 중 하나)")


def _default_drafter(cfg: Config, device: str) -> Drafter:
    from .nmt import Model1Drafter
    return Model1Drafter(cfg.nmt_dir, cfg.nmt_ckpt, device=device, beam=cfg.nmt_beam, cpu_threads=cfg.cpu_threads)


def _default_labse(cfg: Config, device: str) -> Similarity:
    from .scorers import LabseScorer
    return LabseScorer(cfg.labse_model, device=device, cpu_threads=cfg.cpu_threads)


def _default_comet(cfg: Config, device: str) -> Scorer:
    from .scorers import CometScorer
    return CometScorer(cfg.comet_model, device=device, batch_size=cfg.comet_batch, cpu_threads=cfg.cpu_threads)


def _span(values: Iterable[int]) -> str:
    """값 범위를 적는다. 하나뿐이면 그 값(9), 여럿이면 최소~최대(9~24)."""
    vals = list(values)
    lo, hi = min(vals), max(vals)
    return f"{lo}" if lo == hi else f"{lo}~{hi}"


def _passing(pool: Sequence[Candidate]) -> int:
    return sum(1 for c in pool if c.passed)


def _needs_more(pool: Sequence[Candidate], cfg: Config) -> bool:
    return _passing(pool) < cfg.min_pass and len(pool) < cfg.max_candidates


def feedback_for(pool: Sequence[Candidate], limit: int = FEEDBACK_MAX) -> list[dict]:
    """모델 2 에 돌려줄 탈락 후보와 까닭. 최근 차수의 탈락 후보부터 limit 개."""
    failed = [c for c in pool if not c.passed]
    failed.sort(key=lambda c: -c.round)          # 안정 정렬이라 같은 차수 안에서는 나온 순서를 지킨다
    return [{"text": c.text, "syllables": c.syllables, "reason": "; ".join(c.reasons)} for c in failed[:limit]]


def fresh_candidates(pool: Sequence[Candidate], texts: Sequence[str], origin: str, rnd: int,
                     max_candidates: int) -> list[Candidate]:
    """새로 받은 글을 다듬고, 풀에 있거나 서로 겹치는(공백·문장부호만 다른) 것을 빼 후보로 만든다."""
    keys = {dedupe_key(c.text) for c in pool}
    room = max(0, max_candidates - len(pool))
    out: list[Candidate] = []
    for raw in texts:
        if len(out) >= room:
            break
        text = clean_candidate(raw)
        key = dedupe_key(text)
        if not text or not key or key in keys:
            continue
        keys.add(key)
        out.append(Candidate(text=text, origin=origin, round=rnd))
    return out


def labse_scores(labse: Similarity, srcs: Sequence[str], tgts: Sequence[str]) -> np.ndarray:
    """짝마다 LaBSE 코사인. 원문에 숫자가 있으면 숫자를 말하는 꼴로 푼 원문과도 재서 큰 값을 쓴다.

    모델 2 는 숫자를 한글 소리로 쓰라고 받는데(서른다섯 살), LaBSE 는 숫자 원문('35 years')과 한글 수 사이를
    낮게 본다(검토에서 Call 911! 대 구일일에 전화해! 가 0.693). 논문 식 (5) 도 G2P 를 거친 열 S_b, T_b 사이
    코사인으로 적혀 있다(G2P 는 숫자를 소리로 푼다). 숫자가 없는 원문은 한 번만 잰다. LaBSE 호출은 한 번이다.
    """
    srcs, tgts = list(srcs), list(tgts)
    spoken = [speak_english_numbers(s) for s in srcs]
    extra = [k for k, (s, a) in enumerate(zip(srcs, spoken)) if a != s]
    sims = np.asarray(labse.similarity(srcs + [spoken[k] for k in extra], tgts + [tgts[k] for k in extra]),
                      dtype=np.float64)
    if sims.shape != (len(tgts) + len(extra),):
        raise ValueError(f"LaBSE 점수 {sims.shape} 가 후보 {len(tgts) + len(extra)}개와 맞지 않습니다.")
    out = sims[:len(tgts)].copy()
    for j, k in enumerate(extra):
        out[k] = max(out[k], sims[len(tgts) + j])
    return out


def _score_and_assess(labse: Similarity, batch: list[tuple[int, list[Candidate]]], items: Sequence[LineItem],
                      targets: Sequence[Target], cfg: Config) -> int:
    """여러 줄의 새 후보를 LaBSE 한 번으로 재고 거른다. 잰 후보 수를 돌려준다."""
    srcs: list[str] = []
    tgts: list[str] = []
    for i, cands in batch:
        srcs.extend(items[i].source_en for _ in cands)
        tgts.extend(c.text for c in cands)
    if not tgts:
        return 0
    sims = labse_scores(labse, srcs, tgts)
    pos = 0
    for i, cands in batch:
        assess(cands, items[i], targets[i], sims[pos:pos + len(cands)], cfg)
        pos += len(cands)
    return len(tgts)


def _choose(kind: str, group: list[Candidate], item: LineItem, lip_ok: bool, target: Target,
            cfg: Config) -> tuple[Candidate | None, str, str]:
    """고를 묶음에서 한 후보를 고른다. (고른 후보, 상태, 앞에 붙일 메모)."""
    if kind == POOL_NONE:
        return None, "error", "후보가 하나도 없음"
    if kind == POOL_LABSE:
        return group[0], "fallback", ("통과 후보 없음: 길이와 뜻 조건(논문 식 6, 7)을 함께 채운 후보도 없어 "
                                      "LaBSE 가 가장 높은 후보를 고름(논문 3.1절 끝의 규칙)")
    if len(group) == 1 and group[0].comet is None:          # 초벌을 COMET 참조로 쓰는데 초벌만 남은 줄
        selected, how, unusual = group[0], "초벌만 남아 점수 없이 초벌을 고름(COMET 참조가 초벌)", True
    elif item.lip and not lip_ok:
        selected = ps_comet(group, cfg, False, target)
        how, unusual = "영어 음절이 없어 입 모양 비교 없이 COMET 으로 고름", True
    else:
        selected = ps_comet(group, cfg, lip_ok, target)
        how, unusual = ("PS-Comet" if lip_ok else "COMET") + " 으로 고름", False
    if kind == POOL_PASS:
        return selected, "ok", how if unusual else ""
    failed = "·".join(name for name, flag in (("말투", selected.ok_register), ("숫자", selected.ok_numbers))
                      if flag is False)
    return selected, "fallback", (f"통과 후보 없음: 길이와 뜻 조건(논문 식 6, 7)을 채운 후보 가운데 {how}"
                                  + (f"({failed} 조건은 못 채움)" if failed else ""))


def run_lines(items: Sequence[LineItem], cfg: Config, *, backend: Generator | None = None,
              make_drafter: Callable[[], Drafter] | None = None,
              make_labse: Callable[[], Similarity] | None = None,
              make_comet: Callable[[], Scorer] | None = None,
              vowel_table: VowelTable | None = None,
              log: Callable[[str], None] | None = None) -> tuple[list[LineResult], RunInfo]:
    """대사 목록을 처리해 줄마다 LineResult 와 실행 기록을 돌려준다.

    backend, make_* 를 주면 그것을 쓴다(시험용 가짜). 안 주면 cfg 대로 진짜 모델을 만든다.
    """
    log = log or _stderr
    check_config(cfg)
    check_items(items)
    items = list(items)
    n = len(items)
    info = RunInfo(started_at=datetime.now().isoformat(timespec="seconds"))
    t_all = time.perf_counter()

    real_models = make_drafter is None or make_labse is None or make_comet is None
    if real_models:
        from .scorers import resolve_device
        device = resolve_device(cfg.device, cfg.cpu_threads)     # cuda 를 못 쓰면 여기서 멈춘다
    else:
        device = cfg.device
    info.device = device
    if cfg.max_rounds > 0 and backend is None:
        backend = make_backend(cfg)          # 후보 파일이 없는 것 같은 설정 오류를 모델 올리기 전에 알린다
    info.llm_backend = str(getattr(backend, "name", cfg.llm)) if cfg.max_rounds > 0 else "-"
    if cfg.comet_ref == "draft" and not cfg.use_draft:
        missing = [it.id for it in items if not (it.draft_ko or "").strip()]
        if missing:
            raise ValueError(f"--comet-ref draft 는 초벌이 있어야 합니다. 초벌 없는 줄: {', '.join(missing[:10])}")

    # 1. 길이 목표
    targets = [target_for(it, cfg) for it in items]
    log(f"[준비] 대사 {n}줄, 장치 {device}, 목표 음절 {_span(t.n for t in targets)}"
        f"(허용 ±{_span(t.tol for t in targets)}), 모델 2 {info.llm_backend}")

    # 2. 모델 1 초벌
    drafts = [" ".join((it.draft_ko or "").split()) for it in items]
    need = [i for i in range(n) if not drafts[i]] if cfg.use_draft else []
    if need:
        t0 = time.perf_counter()
        drafter = (make_drafter or (lambda: _default_drafter(cfg, device)))()
        try:
            info.model1 = {k: v for k, v in (getattr(drafter, "info", None) or {}).items() if k != "config"}
            ckpt = getattr(drafter, "ckpt", None)
            if ckpt is not None:
                info.model1["ckpt"] = str(ckpt)
            out = drafter.translate([items[i].source_en for i in need])
        finally:
            drafter.close()
        if len(out) != len(need):
            raise ValueError(f"모델 1 이 {len(need)}줄 중 {len(out)}줄만 돌려줬습니다.")
        for i, d in zip(need, out):
            drafts[i] = " ".join((d or "").split())
        info.seconds["model1"] = round(time.perf_counter() - t0, 2)
        log(f"[모델 1] 초벌 {len(need)}줄 번역 {info.seconds['model1']:.1f}초(올리기 포함)")
    if cfg.comet_ref == "draft":
        missing = [items[i].id for i in range(n) if not drafts[i]]
        if missing:
            raise ValueError(f"--comet-ref draft 인데 초벌이 빈 줄이 있습니다: {', '.join(missing[:10])}")

    pools: list[list[Candidate]] = []
    for i, it in enumerate(items):
        origin = "given" if (it.draft_ko or "").strip() else "model1"
        pools.append([Candidate(text=drafts[i], origin=origin, round=0)] if drafts[i] else [])

    # 3~4. LaBSE 거르기와 모델 2 차수
    rounds_used = [0] * n
    notes: list[list[str]] = [[] for _ in range(n)]
    origin_new = "file" if getattr(backend, "name", "") == "file" else "llm"
    t_labse = 0.0
    labse = (make_labse or (lambda: _default_labse(cfg, device)))()
    try:
        t0 = time.perf_counter()
        scored = _score_and_assess(labse, [(i, pools[i]) for i in range(n) if pools[i]], items, targets, cfg)
        t_labse += time.perf_counter() - t0
        if scored:
            log(f"[LaBSE] 초벌 {scored}개 거르기, 통과 {sum(_passing(p) for p in pools)}개 "
                f"{time.perf_counter() - t0:.1f}초(올리기 포함)")
        for k in range(1, cfg.max_rounds + 1):
            todo = [i for i in range(n) if _needs_more(pools[i], cfg)]
            if not todo:
                break
            reqs = [GenRequest(
                id=items[i].id, source_en=items[i].source_en, prev_en=items[i].prev_en,
                relationship=items[i].relationship, register=items[i].register,
                n_target=targets[i].n, tol=targets[i].tol, draft_ko=drafts[i] or None,
                meaning_notes=items[i].meaning_notes,
                n_new=min(cfg.n_per_round, cfg.max_candidates - len(pools[i])), round=k,
                avoid=[c.text for c in pools[i]], feedback=feedback_for(pools[i]),
            ) for i in todo]
            log(f"[모델 2] {k}차: {len(todo)}줄에 후보 요청(줄마다 최대 {max(r.n_new for r in reqs)}개)")
            t0 = time.perf_counter()
            try:
                reply = backend.generate(reqs)
            except LLMUnavailable as e:
                reply = e.partial if e.partial is not None else GenReply()
                info.llm_stopped = str(e)
            t_llm = time.perf_counter() - t0
            calls, hits = int(getattr(reply, "calls", 0)), int(getattr(reply, "cache_hits", 0))
            info.llm_calls += calls
            info.llm_cache_hits += hits
            info.llm_errors.extend(getattr(reply, "errors", []))
            batch = []
            for i in todo:
                rounds_used[i] = k
                fresh = fresh_candidates(pools[i], reply.get(items[i].id, []), origin_new, k, cfg.max_candidates)
                if not fresh:
                    notes[i].append(f"{k}차 새 후보 없음")
                batch.append((i, fresh))
            t0 = time.perf_counter()
            new = _score_and_assess(labse, batch, items, targets, cfg)
            t_lab = time.perf_counter() - t0
            t_labse += t_lab
            new_pass = sum(_passing(c) for _, c in batch)
            for i, fresh in batch:
                pools[i].extend(fresh)
            done = sum(1 for p in pools if _passing(p) >= cfg.min_pass)
            info.rounds.append({"round": k, "lines": len(todo), "requested": sum(r.n_new for r in reqs),
                                "new": new, "new_pass": new_pass, "calls": calls, "cache_hits": hits,
                                "llm_seconds": round(t_llm, 2), "labse_seconds": round(t_lab, 2)})
            log(f"[모델 2] {k}차: 새 후보 {new}개(통과 {new_pass}개), 호출 {calls}번, 캐시 {hits}번, "
                f"{t_llm:.1f}초 / LaBSE {t_lab:.1f}초. 통과 {cfg.min_pass}개 채운 줄 {done}/{n}")
            if info.llm_stopped:
                log(f"[모델 2] 쓸 수 없어 멈춤: {info.llm_stopped}")
                for i in range(n):
                    if _needs_more(pools[i], cfg):
                        notes[i].append("모델 2 를 쓸 수 없어 후보 모으기를 멈춤")
                break
    finally:
        labse.close()
    info.seconds["labse"] = round(t_labse, 2)
    info.seconds["llm"] = round(sum(r["llm_seconds"] for r in info.rounds), 2)

    # 말투 미지정 줄: 모델 2 후보 다수의 말투로 다시 거른다(차수가 다 끝난 뒤 한 번)
    for i, it in enumerate(items):
        reg = apply_line_register(it, pools[i], cfg)
        if reg is not None:
            notes[i].append(f"말투 미지정: 모델 2 후보 다수를 따라 {'반말' if reg == '반말' else '높임말'}로 봄")

    # 5. 고를 묶음: 통과 후보, 없으면 식 6·7 을 채운 후보, 그것도 없으면 LaBSE 가 가장 높은 후보(select.selection_pool)
    draft_ref = cfg.comet_ref == "draft"
    plans = [selection_pool(pool, draft_is_reference=draft_ref) for pool in pools]

    # COMET(고를 묶음의 후보). 초벌을 참조로 쓰면 초벌 자신은 재지 않는다(자기 자신이 참조라 늘 유리함)
    t0 = time.perf_counter()
    to_score = [(i, c) for i, (_, group) in enumerate(plans) for c in group if not (draft_ref and c.round == 0)]
    if to_score:
        comet = (make_comet or (lambda: _default_comet(cfg, device)))()
        try:
            refs = [drafts[i] for i, _ in to_score] if draft_ref else None
            scores = np.asarray(comet.score([items[i].source_en for i, _ in to_score],
                                            [c.text for _, c in to_score], refs), dtype=np.float64)
            if scores.shape != (len(to_score),):
                raise ValueError(f"COMET 점수 {scores.shape} 가 후보 {len(to_score)}개와 맞지 않습니다.")
            for (_, c), s in zip(to_score, scores):
                c.comet = float(s)
            info.comet_checkpoint = getattr(comet, "checkpoint_path", None)
        finally:
            comet.close()
        info.seconds["comet"] = round(time.perf_counter() - t0, 2)
        log(f"[COMET] 후보 {len(to_score)}개 점수 {info.seconds['comet']:.1f}초(올리기 포함)"
            + (". 초벌은 참조라서 재지 않고 고르기에서 뺌" if draft_ref else ""))

    # 입이 보이는 줄의 모음 DTW(모든 후보. 고르기에는 고를 묶음만 쓴다)
    t0 = time.perf_counter()
    lip_ok = [False] * n
    n_dtw = 0
    for i, it in enumerate(items):
        if it.lip and pools[i]:
            vowel_table = vowel_table if vowel_table is not None else load_or_build()
            lip_ok[i] = add_dtw(it, targets[i], pools[i], cfg, vowel_table)
            if lip_ok[i]:
                n_dtw += len(pools[i])
    if n_dtw:
        info.seconds["dtw"] = round(time.perf_counter() - t0, 2)
        log(f"[DTW] 입 모양 비교 {sum(lip_ok)}줄, 후보 {n_dtw}개 {info.seconds['dtw']:.1f}초")

    # 고르기
    results: list[LineResult] = []
    for i, it in enumerate(items):
        kind, group = plans[i]
        selected, status, note = _choose(kind, group, it, lip_ok[i], targets[i], cfg)
        if note:
            notes[i].insert(0, note)
        results.append(LineResult(item=it, target=targets[i], draft_ko=drafts[i], candidates=pools[i],
                                  selected=selected, status=status, rounds_used=rounds_used[i],
                                  message="; ".join(dict.fromkeys(notes[i]))))
    info.status_counts = dict(Counter(r.status for r in results))
    info.seconds["total"] = round(time.perf_counter() - t_all, 2)
    info.finished_at = datetime.now().isoformat(timespec="seconds")
    counts = info.status_counts
    log(f"[선택] 고름 {counts.get('ok', 0)}줄, 대안 {counts.get('fallback', 0)}줄, 실패 {counts.get('error', 0)}줄. "
        f"모델 2 호출 {info.llm_calls}번(캐시 {info.llm_cache_hits}번). 전체 {info.seconds['total']:.1f}초")
    return results, info
