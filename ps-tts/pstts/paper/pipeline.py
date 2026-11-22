"""논문 그대로 판 전체 흐름(명세 3.9).

1. 입력은 LineItem 이다(파일은 paper.io.read_lines 로 읽음). 쓰는 칸은 id, source_en, seconds(K'), draft_ko(T0) 뿐이다.
   register, lip, target_syllables, relationship, prev_en, meaning_notes 에 값이 있으면
   '논문 판은 이 칸을 쓰지 않습니다' 를 한 줄 알린다. 이 칸들의 값이 기존 판 꼴에 맞지 않아도 멈추지 않는다.
2. 장치를 확인한다(pstts.scorers.resolve_device. cuda 를 못 쓰면 멈추고 조용히 cpu 로 바꾸지 않는다).
   프런트엔드 준비물(근사의 모음표와 모음 대응, yourtts 의 g2p.json 과 모음표)도 여기서 먼저 확인한다(preflight_frontend).
   초벌이 없는 줄만 모델 1 로 한꺼번에 번역하고 곧바로 내린다.
3. TTS 프런트엔드를 올린다(approx 또는 yourtts).
4. LaBSE 를 올리고 ISO(iso.run_iso, 3.1절)를 돈 다음 내린다.
5. ISO 문장마다 PS 후보 ps_n(60)개를 모은다(prompts.ps_generate, 3.2절). 프런트엔드로 토큰별 길이를 재고 늘린 뒤
   모음 DTW(식 10)를 잰다(ps.measure_line).
6. COMET 을 올려 PS 후보 전부를 재고 내린다.
7. PS(식 11)와 PS-Comet(식 20)을 고른다(ps.select_line). 최종 대사는 PS-Comet 이 고른 후보다.
8. 단계별 시간과 단계별 모델 2 호출 수, 캐시 적중 수를 기록한다. 진행은 stderr 에 한국어 한 줄씩 찍는다.

Your-TTS 프런트엔드는 모델 글자 목록에 없는 글자에서 멈추므로(yourtts.UnknownTokenError) 여기서 VocabFilter 로 감싼다.
coqui-tts 0.28 의 TTSTokenizer.encode 가 그 글자를 버리고 빈 토큰을 끼우는 것과 같은 열을 만든다(해석, README_paper).
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import numpy as np

from ..llm import make_backend
from ..records import LineItem
from . import g2p
from .config import PaperConfig
from .iso import (STATUS_DRAFT, STATUS_ERROR, STATUS_FALLBACK, STATUS_PARAPHRASE, IsoInput, IsoLine, check_inputs,
                  run_iso)
from .prompts import GenOutcome, PaperBackend, PsRequest, iso_generate, ps_generate
from .ps import PsCandidate, PsChoice, measure_line, select_line

LLM_BACKENDS = ("claude", "openai", "file")
TTS_KINDS = ("approx", "yourtts")
TTS_KO = {"approx": "근사(approx)", "yourtts": "Your-TTS(yourtts)"}
STATUS_OK = "ok"
ISO_STATUSES = (STATUS_DRAFT, STATUS_PARAPHRASE, STATUS_FALLBACK, STATUS_ERROR)

# 기존 판 읽기가 값을 검사하는 칸 가운데 논문 판이 쓰지 않는 것. 논문 판 읽기(io.read_lines)는 이 칸 값이 기존 판 꼴에
# 맞지 않아도 멈추지 않고, 적힌 글 그대로 LineItem.extra 에 둔다(LineItem 칸은 기본값).
LENIENT_COLUMNS = ("register", "lip", "target_syllables")


def raw_unused(item: LineItem, name: str) -> str:
    """기존 판 꼴에 맞지 않아 extra 에 그대로 둔 값(LENIENT_COLUMNS 칸만). 없으면 빈 글."""
    if name not in LENIENT_COLUMNS:
        return ""
    value = item.extra.get(name)
    return "" if value is None else str(value)


# 논문 판이 쓰지 않는 입력 칸과 '값이 있다' 의 뜻. lip 은 빈칸이 on 이라 off 만 값으로 본다.
# 기존 판 꼴에 맞지 않아 extra 에 둔 값(raw_unused)도 값이다.
UNUSED_FIELDS: tuple[tuple[str, Callable[[LineItem], bool]], ...] = (
    ("register", lambda it: it.register != "미지정" or bool(raw_unused(it, "register"))),
    ("lip", lambda it: not it.lip or bool(raw_unused(it, "lip"))),
    ("target_syllables", lambda it: it.target_syllables is not None or bool(raw_unused(it, "target_syllables"))),
    ("relationship", lambda it: bool(it.relationship)),
    ("prev_en", lambda it: bool(it.prev_en)),
    ("meaning_notes", lambda it: bool(it.meaning_notes)),
)


class Drafter(Protocol):
    def translate(self, sentences: Sequence[str]) -> list[str]: ...
    def close(self) -> None: ...


class Similarity(Protocol):
    def similarity(self, srcs: Sequence[str], tgts: Sequence[str]) -> np.ndarray: ...
    def close(self) -> None: ...


class Scorer(Protocol):
    def score(self, srcs: Sequence[str], mts: Sequence[str], refs: Sequence[str] | None = None) -> np.ndarray: ...
    def close(self) -> None: ...


@dataclass
class PaperLine:
    """줄 하나의 결과: ISO, PS 후보와 값, PS, PS-Comet 고르기."""
    item: LineItem
    iso: IsoLine
    ps_candidates: list[PsCandidate] = field(default_factory=list)
    ps_rounds: list[int] = field(default_factory=list)      # PS 후보마다 받은 차수(ps_candidates 와 같은 순서)
    n_src_frames: int = 0                                   # 늘린 원문 프레임 수(S'_b 길이)
    choice: PsChoice | None = None
    status: str = ""                                        # 'ok'(PS-Comet 이 고름) | 'error'
    message: str = ""

    @property
    def ps(self) -> PsCandidate | None:
        """식 (11) PS 가 고른 후보."""
        c = self.choice
        return None if c is None or c.ps is None else self.ps_candidates[c.ps]

    @property
    def final(self) -> PsCandidate | None:
        """식 (20) PS-Comet 이 고른 후보(최종 대사)."""
        c = self.choice
        return None if c is None or c.ps_comet is None else self.ps_candidates[c.ps_comet]

    @property
    def final_ko(self) -> str:
        f = self.final
        return f.text if f is not None else ""


@dataclass
class PaperRunInfo:
    """실행 기록(run.json 과 report.md 에 씀)."""
    started_at: str = ""
    finished_at: str = ""
    device: str = ""
    seconds: dict[str, float] = field(default_factory=dict)     # 단계별 걸린 시간
    llm_backend: str = ""
    llm: dict[str, dict[str, int]] = field(default_factory=dict)  # 단계('iso', 'ps')별 호출 수와 캐시 적중 수
    llm_errors: list[str] = field(default_factory=list)
    llm_stopped: str = ""                                       # 모델 2 를 쓸 수 없어 멈췄으면 그 까닭
    iso_rounds: list[dict] = field(default_factory=list)
    ps_rounds: list[dict] = field(default_factory=list)
    model1: dict = field(default_factory=dict)
    comet_checkpoint: str | None = None
    frontend: dict = field(default_factory=dict)                # 프런트엔드 info()
    unused_columns: dict[str, int] = field(default_factory=dict)  # 논문 판이 쓰지 않은 칸과 값이 있던 줄 수
    unknown_tokens: list[list[str]] = field(default_factory=list)  # g2p 가 분류하지 못한 (언어, 토큰)
    rows: list[dict] = field(default_factory=list)              # candidates.jsonl 재생 줄
    status_counts: dict[str, int] = field(default_factory=dict)
    iso_counts: dict[str, int] = field(default_factory=dict)

    @property
    def llm_calls(self) -> int:
        return sum(v.get("calls", 0) for v in self.llm.values())

    @property
    def llm_cache_hits(self) -> int:
        return sum(v.get("cache_hits", 0) for v in self.llm.values())


# ---------------------------------------------------------------- Your-TTS 글자 목록 거르기

class VocabFilter:
    """Your-TTS 프런트엔드 감싸개: 모델 글자 목록에 없는 글자는 버리고 빈 토큰을 다시 끼운다.

    coqui-tts 0.28 TTSTokenizer.encode 는 목록에 없는 글자를 경고와 함께 버리고(not_found_characters), text_to_ids 가
    그 뒤에 빈 토큰을 끼운다. 학습한 모델에게 그 글을 읽힐 때와 같은 토큰 열이라 토큰별 길이와 라벨이 어긋나지 않는다.
    버린 글자는 글마다 한 번씩 세어 info() 에 남긴다. 나머지 일은 감싼 프런트엔드에 넘긴다.
    """

    RULE = "모델 글자 목록에 없는 글자는 Coqui TTSTokenizer.encode 처럼 버리고 빈 토큰을 다시 끼움"

    def __init__(self, front: Any) -> None:
        self.front = front
        self.name = front.name
        self.frame_sec = front.frame_sec
        self.dropped: dict[str, Counter[str]] = {"en": Counter(), "ko": Counter()}
        self.dropped_texts: dict[str, int] = {"en": 0, "ko": 0}
        self._seen: set[tuple[str, str]] = set()

    def tokens(self, text: str, lang: str) -> list[str]:
        code = g2p.check_lang(lang)
        toks = self.front.tokens(text, code)
        bad = set(self.front.unknown_tokens(toks))
        if not bad:
            return toks
        kept = [t for t in toks[1::2] if t not in bad]           # toks = with_blanks(음소) 라 홀수 자리가 음소다
        if (code, text) not in self._seen:
            self._seen.add((code, text))
            self.dropped_texts[code] += 1
            self.dropped[code].update(t for t in toks[1::2] if t in bad)
        return g2p.with_blanks(kept)

    def durations(self, seqs: list[list[str]], lang: str) -> list[np.ndarray]:
        return self.front.durations(seqs, lang)

    def labels(self, toks: list[str], lang: str) -> list[str | None]:
        return self.front.labels(toks, lang)

    def cost(self, src_labels: Sequence[str | None], tgt_labels: Sequence[str | None]) -> np.ndarray:
        return self.front.cost(src_labels, tgt_labels)

    def info(self) -> dict:
        out = dict(self.front.info())
        out["unknown_token_rule"] = self.RULE
        out["dropped_texts"] = dict(self.dropped_texts)
        out["dropped_tokens"] = {k: dict(v.most_common()) for k, v in self.dropped.items()}
        return out

    def close(self) -> None:
        self.front.close()


# ---------------------------------------------------------------- 준비

def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def check_config(cfg: PaperConfig) -> None:
    """실행 전에 설정값을 확인한다(모델을 올리기 전에 멈춤). 틀리면 한국어 ValueError."""
    problems = []
    if cfg.llm not in LLM_BACKENDS:
        problems.append(f"llm 은 {', '.join(LLM_BACKENDS)} 중 하나여야 합니다: {cfg.llm}")
    if str(cfg.tts).strip().lower() not in TTS_KINDS:
        problems.append(f"tts 는 {', '.join(TTS_KINDS)} 중 하나여야 합니다: {cfg.tts}")
    elif str(cfg.tts).strip().lower() == "yourtts" and not cfg.tts_dir:
        problems.append("--tts yourtts 에는 체크포인트 폴더(--tts-dir)가 필요합니다.")
    for name, low in (("iso_batch", 1), ("iso_max", 0), ("ps_n", 1), ("ps_chunk", 1), ("ps_extra_rounds", 0),
                      ("llm_workers", 1), ("llm_chunk", 1), ("nmt_beam", 1), ("cpu_threads", 1),
                      ("comet_batch", 1), ("iso_tol_frames", 0)):
        if getattr(cfg, name) < low:
            problems.append(f"{name} 은 {low} 이상이어야 합니다: {getattr(cfg, name)}")
    if cfg.sakoe_chiba_radius is not None and cfg.sakoe_chiba_radius < 0:
        problems.append(f"sakoe_chiba_radius 는 0 이상이거나 없음(tslearn 기본)이어야 합니다: {cfg.sakoe_chiba_radius}")
    if cfg.alpha < 0 or cfg.beta < 0:
        problems.append(f"alpha, beta 는 0 이상이어야 합니다: {cfg.alpha}, {cfg.beta}")
    if not -1.0 <= cfg.labse_min <= 1.0:
        problems.append(f"labse_min 은 -1 과 1 사이여야 합니다: {cfg.labse_min}")
    if not cfg.frame_sec_sa > 0:
        problems.append(f"frame_sec_sa 는 0 보다 커야 합니다: {cfg.frame_sec_sa}")
    try:
        g2p.check_en_voice(cfg.g2p_en_voice)
    except ValueError as e:
        problems.append(str(e))
    if problems:
        raise ValueError(" / ".join(problems))


def unused_columns(items: Sequence[LineItem]) -> dict[str, int]:
    """논문 판이 쓰지 않는 칸마다 값이 있는 줄 수(0 인 칸은 뺌)."""
    out = {name: sum(1 for it in items if has(it)) for name, has in UNUSED_FIELDS}
    return {k: v for k, v in out.items() if v}


def _default_drafter(cfg: PaperConfig, device: str) -> Drafter:
    from ..nmt import Model1Drafter
    return Model1Drafter(cfg.nmt_dir, cfg.nmt_ckpt, device=device, beam=cfg.nmt_beam, cpu_threads=cfg.cpu_threads)


def _default_labse(cfg: PaperConfig, device: str) -> Similarity:
    from ..scorers import LabseScorer
    return LabseScorer(cfg.labse_model, device=device, cpu_threads=cfg.cpu_threads)


def _default_comet(cfg: PaperConfig, device: str) -> Scorer:
    from ..scorers import CometScorer
    return CometScorer(cfg.comet_model, device=device, batch_size=cfg.comet_batch, cpu_threads=cfg.cpu_threads)


def open_frontend(cfg: PaperConfig, device: str) -> Any:
    """cfg.tts 프런트엔드. yourtts 면 VocabFilter 로 감싼다."""
    from .frontend import make_frontend
    front = make_frontend(cfg, device)
    return VocabFilter(front) if hasattr(front, "unknown_tokens") else front


def preflight_frontend(cfg: PaperConfig) -> None:
    """모델 1 을 올리기 전에 프런트엔드가 쓸 가벼운 준비물을 확인한다. 틀리면 모델 1, LaBSE, 모델 2 를 부르기 전에 멈춘다.

    approx: 모음표(캐시 JSON)와 모음 대응. yourtts: 체크포인트 폴더의 config.json, 체크포인트, g2p.json, 모음표(yourtts.check_folder).
    모델은 올리지 않는다. 프런트엔드를 만들 때 같은 확인을 한 번 더 한다.
    """
    if str(cfg.tts).strip().lower() == "yourtts":
        from .yourtts import check_folder
        check_folder(cfg)
        return
    from ..vowels import load_or_build
    from . import frontend
    load_or_build()
    frontend.en_label_map()
    frontend.ko_label_map()


def _round_list(rows: Sequence[dict], line_id: str) -> list[int]:
    """재생 줄에서 한 줄의 후보마다 받은 차수(후보가 붙은 순서와 같음)."""
    out: list[int] = []
    for row in rows:
        if row["id"] == line_id:
            out.extend([int(row["round"])] * len(row["candidates"]))
    return out


def _span(values: Sequence[float], digits: int = 1) -> str:
    lo, hi = min(values), max(values)
    return f"{lo:.{digits}f}" if lo == hi else f"{lo:.{digits}f}~{hi:.{digits}f}"


# ---------------------------------------------------------------- 흐름

def run_paper(items: Sequence[LineItem], cfg: PaperConfig, *, backend: PaperBackend | None = None,
              make_drafter: Callable[[], Drafter] | None = None,
              make_front: Callable[[], Any] | None = None,
              make_labse: Callable[[], Similarity] | None = None,
              make_comet: Callable[[], Scorer] | None = None,
              log: Callable[[str], None] | None = None) -> tuple[list[PaperLine], PaperRunInfo]:
    """대사 목록을 논문 절차대로 처리해 줄마다 PaperLine 과 실행 기록을 돌려준다.

    backend, make_* 를 주면 그것을 쓴다(시험용 가짜). 안 주면 cfg 대로 진짜 모델을 만든다.
    """
    log = log or _stderr
    check_config(cfg)
    items = list(items)
    inputs = [IsoInput(id=it.id, source_en=it.source_en, draft_ko="", seconds=it.seconds) for it in items]
    check_inputs(inputs, cfg)                       # 빈 목록, 겹친 id, 빈 영어 대사, seconds 를 모델 올리기 전에 확인
    n = len(items)
    tts = str(cfg.tts).strip().lower()
    info = PaperRunInfo(started_at=datetime.now().isoformat(timespec="seconds"))
    t_all = time.perf_counter()

    real = make_drafter is None or make_labse is None or make_comet is None or make_front is None
    if real:
        from ..scorers import resolve_device
        device = resolve_device(cfg.device, cfg.cpu_threads)      # cuda 를 못 쓰면 여기서 멈춘다
    else:
        device = cfg.device
    info.device = device
    if backend is None:
        backend = make_backend(cfg.base_config())     # 후보 파일이 없는 것 같은 설정 오류를 모델 올리기 전에 알린다
    if make_front is None:
        preflight_frontend(cfg)                       # 프런트엔드 준비물(모음표, g2p.json 등)도 모델 1 앞에서 확인
    info.llm_backend = str(getattr(backend, "name", cfg.llm))
    info.unused_columns = unused_columns(items)
    log(f"[준비] 대사 {n}줄, 장치 {device}, TTS {TTS_KO.get(tts, tts)}, 모델 2 {info.llm_backend}")
    if info.unused_columns:
        cols = ", ".join(f"{k} {v}줄" for k, v in info.unused_columns.items())
        log(f"[준비] 논문 판은 이 칸을 쓰지 않습니다: {cols}")

    # 2. 모델 1 초벌(논문의 NMT 번역문 T0)
    drafts = [" ".join((it.draft_ko or "").split()) for it in items]
    need = [i for i in range(n) if not drafts[i]]
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
    for x, d in zip(inputs, drafts):
        x.draft_ko = d

    # 3. TTS 프런트엔드
    t0 = time.perf_counter()
    front = (make_front or (lambda: open_frontend(cfg, device)))()
    lines: list[PaperLine] = []
    ps_out = GenOutcome(candidates={})
    try:
        info.seconds["frontend"] = round(time.perf_counter() - t0, 2)
        log(f"[TTS] {TTS_KO.get(tts, tts)} 프런트엔드 준비 {info.seconds['frontend']:.1f}초")

        # 4. ISO(3.1절): LaBSE 를 올리고 돌린 다음 내린다
        labse = (make_labse or (lambda: _default_labse(cfg, device)))()
        try:
            iso_run = run_iso(inputs, front, cfg, paraphrase=lambda reqs: iso_generate(backend, reqs),
                              similarity=labse.similarity, log=log)
        finally:
            labse.close()
        info.seconds["iso_tts"] = iso_run.seconds["tts"]
        info.seconds["iso_labse"] = iso_run.seconds["labse"]
        info.seconds["iso_llm"] = iso_run.seconds["llm"]
        info.llm["iso"] = {"calls": iso_run.calls, "cache_hits": iso_run.cache_hits}
        info.llm_errors.extend(iso_run.errors)
        info.iso_rounds = iso_run.rounds
        info.rows.extend(iso_run.rows)
        lines = [PaperLine(item=it, iso=line) for it, line in zip(items, iso_run.lines)]

        # 5. PS 후보(3.2절): ISO 문장마다 ps_n 개
        ready = [p for p in lines if p.iso.target]
        if iso_run.stopped:
            info.llm_stopped = iso_run.stopped
        elif ready:
            ps_out = ps_generate(backend, [PsRequest(id=p.item.id, ko=str(p.iso.target), n_new=cfg.ps_n)
                                           for p in ready],
                                 extra_rounds=cfg.ps_extra_rounds, chunk=cfg.ps_chunk, log=log)
            info.llm_stopped = ps_out.stopped
        info.llm["ps"] = {"calls": ps_out.calls, "cache_hits": ps_out.cache_hits}
        info.llm_errors.extend(ps_out.errors)
        info.ps_rounds = ps_out.rounds
        info.seconds["ps_llm"] = round(sum(r["llm_seconds"] for r in ps_out.rounds), 2)
        info.rows.extend(ps_out.rows)

        # 토큰별 길이로 늘리고 모음 DTW(식 10)
        t0 = time.perf_counter()
        n_meas = 0
        for p in ready:
            texts = ps_out.candidates.get(p.item.id, [])
            if not texts:
                continue
            p.ps_candidates, p.n_src_frames = measure_line(front, p.item.source_en, texts, cfg.sakoe_chiba_radius)
            p.ps_rounds = _round_list(ps_out.rows, p.item.id)
            if len(p.ps_rounds) != len(p.ps_candidates):
                raise ValueError(f"{p.item.id}: PS 후보 {len(p.ps_candidates)}개와 재생 줄의 후보 수가 다릅니다.")
            n_meas += len(texts)
        info.seconds["ps_dtw"] = round(time.perf_counter() - t0, 2)
        if n_meas:
            dtws = [c.dtw for p in lines for c in p.ps_candidates if c.dtw is not None]
            log(f"[PS] 후보 {n_meas}개 길이 재기, 늘리기, 모음 DTW {info.seconds['ps_dtw']:.1f}초"
                + (f". DTW {_span(dtws)}" if dtws else ""))
        info.frontend = front.info()
    finally:
        front.close()

    # 6. COMET(3.3절): PS 후보 전부를 한 번에
    to_score = [(p, c) for p in lines for c in p.ps_candidates]
    if to_score:
        t0 = time.perf_counter()
        comet = (make_comet or (lambda: _default_comet(cfg, device)))()
        try:
            scores = np.asarray(comet.score([p.item.source_en for p, _ in to_score], [c.text for _, c in to_score]),
                                dtype=np.float64)
            if scores.shape != (len(to_score),):
                raise ValueError(f"COMET 점수 {scores.shape} 가 후보 {len(to_score)}개와 맞지 않습니다.")
            for (_, c), s in zip(to_score, scores):
                c.comet = float(s)
            info.comet_checkpoint = getattr(comet, "checkpoint_path", None)
        finally:
            comet.close()
        info.seconds["comet"] = round(time.perf_counter() - t0, 2)
        log(f"[COMET] 후보 {len(to_score)}개 점수 {info.seconds['comet']:.1f}초(올리기 포함)")

    # 7. PS(식 11), PS-Comet(식 20)
    for p in lines:
        notes = [p.iso.message] if p.iso.message else []
        if p.iso.status == STATUS_ERROR:
            p.status = STATUS_ERROR
        elif not p.ps_candidates:
            p.status = STATUS_ERROR
            notes.append("PS 후보를 하나도 받지 못해 PS, PS-Comet 을 고르지 못함(ISO 문장은 iso_ko)"
                         + (f": {info.llm_stopped}" if info.llm_stopped else ""))
        else:
            p.choice = select_line(p.ps_candidates, cfg.alpha, cfg.beta)
            if len(p.ps_candidates) < cfg.ps_n:
                notes.append(f"PS 후보 {len(p.ps_candidates)}개(요청 {cfg.ps_n}개)")
            if p.choice.excluded:
                notes.append(f"프레임이 0개라 PS 고르기에서 뺀 후보 {len(p.choice.excluded)}개")
            if p.choice.ps_comet is None:
                p.status = STATUS_ERROR
                notes.append("모든 PS 후보의 프레임이 0개라 고르지 못함")
            else:
                p.status = STATUS_OK
        p.message = "; ".join(dict.fromkeys(notes))

    info.unknown_tokens = [list(t) for t in sorted(g2p.UNKNOWN_TOKENS)]
    info.status_counts = dict(Counter(p.status for p in lines))
    info.iso_counts = {s: sum(1 for p in lines if p.iso.status == s) for s in ISO_STATUSES}
    info.seconds["total"] = round(time.perf_counter() - t_all, 2)
    info.finished_at = datetime.now().isoformat(timespec="seconds")
    counts = info.status_counts
    log(f"[선택] PS-Comet 고름 {counts.get(STATUS_OK, 0)}줄, 실패 {counts.get(STATUS_ERROR, 0)}줄. "
        f"모델 2 호출 ISO {info.llm['iso']['calls']}번(캐시 {info.llm['iso']['cache_hits']}번), "
        f"PS {info.llm['ps']['calls']}번(캐시 {info.llm['ps']['cache_hits']}번). 전체 {info.seconds['total']:.1f}초")
    return lines, info
