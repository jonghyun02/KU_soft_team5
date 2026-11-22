"""논문 그대로 판 입력 읽기와 결과 파일(명세 3.10).

입력(read_lines): 기존 판 pstts.io 와 같은 파일(.csv, .tsv, .txt)과 같은 규칙(머리글, 따옴표, 행 번호, UTF-8, 빈 id 매기기)
으로 읽는다. 다른 점은 하나다. 논문 판이 쓰지 않는 register, lip, target_syllables 칸은 값이 기존 판 꼴에 맞지 않아도
멈추지 않는다. 맞는 값은 LineItem 칸에, 맞지 않는 값은 적힌 글 그대로 LineItem.extra 에 두고 결과 csv 와 알림에 쓴다.
쓰는 칸(id, source_en, seconds, draft_ko)은 기존 판과 똑같이 검사한다.

--out 폴더(기본 out\\paper)에 다섯 파일을 쓴다. 다섯 파일은 기존 판 pstts.io._publish 로 한꺼번에 바뀐다(하나라도
잠겨 있으면 모두 예전 그대로 두고 한국어 OSError).
- results.csv (utf-8-sig): 줄마다 RESULT_COLUMNS 와 입력의 나머지 칸.
- results.jsonl: 줄마다 results.csv 의 값 전부, ISO 후보 전부, PS 후보 전부(text, dur_sec, n_frames, dtw, dtw_inv,
  comet, score), 고른 번호.
- report.md: 사람이 읽는 보고(설정, 프런트엔드, 줄마다 ISO, PS, PS-Comet 고른 대사와 점수, PS-Comet 상위 3개).
- candidates.jsonl: 재생용 모델 2 후보 {"id", "stage": "iso"|"ps", "round", "candidates"}. --llm file --llm-file 로 주면
  같은 결정이 그대로 나온다.
- run.json: 설정 전체, 판(패키지, 프롬프트), 단계별 시간, 단계별 모델 2 호출과 캐시, 프런트엔드 정보,
  모델 1 과 COMET 체크포인트.
"""
from __future__ import annotations

import csv
import json
import platform
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from io import StringIO
from pathlib import Path
from typing import Any

from .. import __version__
from ..io import INPUT_SUFFIXES, _check_ids, _clean, _fmt, _input_value, _jsonl, _md, _not_utf8, _publish, _read_txt
from ..io import item_from_row as main_item_from_row
from ..records import LineItem
from ..textnorm import ko_syllable_count
from .config import PaperConfig
from .iso import IsoCandidate
from .pipeline import LENIENT_COLUMNS, STATUS_OK, PaperLine, PaperRunInfo, raw_unused
from .prompts import PROMPT_VERSION_PAPER
from .ps import PsCandidate

RESULT_COLUMNS = ("id", "source_en", "k_prime", "k_prime_basis", "draft_ko", "iso_ko", "iso_status", "iso_iterations",
                  "iso_k", "iso_labse", "ps_ko", "ps_dtw", "final_ko", "ps_comet_score", "ps_comet_dtw",
                  "ps_comet_comet", "n_ps_candidates", "message")
# 문장 하나로 돌릴 때 결과 csv 에 붙이는 입력 칸
SINGLE_COLUMNS = ("source_en", "seconds")
OUTPUT_NAMES = ("results.jsonl", "results.csv", "report.md", "candidates.jsonl", "run.json")
STATUS_KO = {"ok": "고름", "error": "실패"}
ISO_STATUS_KO = {"draft": "초벌 통과", "paraphrase": "바꿔 쓰기", "fallback": "대안(LaBSE 최고)", "error": "오류"}
STAGE_KO = {"model1": "모델 1", "frontend": "TTS 준비", "iso_tts": "ISO TTS 길이", "iso_labse": "ISO LaBSE",
            "iso_llm": "ISO 모델 2", "ps_llm": "PS 모델 2", "ps_dtw": "PS 길이와 DTW", "comet": "COMET"}
PACKAGES = ("torch", "transformers", "sentence-transformers", "unbabel-comet", "tslearn", "panphon", "espeakng-loader",
            "numpy", "scikit-learn", "phonemizer", "g2pK", "python-mecab-ko", "python-mecab-ko-dic", "jamo", "nltk",
            "coqui-tts", "coqui-tts-trainer")


# ---------------------------------------------------------------- 입력 읽기

def _main_accepts(name: str, value: str) -> bool:
    """기존 판 item_from_row 가 그 칸 값을 받아들이는지(기존 판 검사를 그대로 빌려 씀)."""
    try:
        main_item_from_row({"source_en": "x", name: value}, "", "")
    except ValueError:
        return False
    return True


def item_from_row(row: dict[str, str | None], where: str, auto_id: str) -> LineItem:
    """입력 한 행을 LineItem 으로. 기존 판 pstts.io.item_from_row 와 같고, LENIENT_COLUMNS 칸 값이 기존 판 꼴에 맞지 않으면
    멈추지 않고 그 칸을 기본값으로 둔 채 적힌 글을 extra 에 둔다."""
    keep = dict(row)
    raw: dict[str, str] = {}
    for name in LENIENT_COLUMNS:
        value = _clean(row.get(name))
        if value and not _main_accepts(name, value):
            raw[name] = value
            keep[name] = ""
    item = main_item_from_row(keep, where, auto_id)
    item.extra.update(raw)
    return item


def _read_table(path: Path, delimiter: str) -> tuple[list[LineItem], list[str]]:
    """csv(쉼표)·tsv(탭) 표. 규칙은 기존 판 pstts.io._read_table 과 같고 행 읽기만 이 모듈의 item_from_row 다."""
    tsv = delimiter == "\t"
    items: list[LineItem] = []
    auto: list[int] = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = (csv.DictReader(f, delimiter=delimiter, quoting=csv.QUOTE_NONE) if tsv
                      else csv.DictReader(f, delimiter=delimiter, strict=True))
            try:
                header = [h.strip() if h else "" for h in (reader.fieldnames or [])]
                if "source_en" not in header:
                    raise ValueError(f"{path.name}: 머리글에 source_en 칸이 없습니다. "
                                     f"머리글: {','.join(header) or '(없음)'}")
                dupes = sorted(h for h, k in Counter(header).items() if k > 1)
                if dupes or "" in header:
                    raise ValueError(f"{path.name}: 머리글에 겹치거나 빈 칸 이름이 있습니다: {dupes or ['(빈 이름)']}")
                reader.fieldnames = header
                for row in reader:
                    where = f"{path.name} {reader.line_num}행"
                    if None in row:
                        raise ValueError(f"{where}: 칸 수가 머리글({len(header)}칸)보다 많습니다.")
                    if not any(_clean(v) for v in row.values()):
                        continue
                    item = item_from_row(row, where, "")
                    if not item.id:
                        auto.append(len(items))
                    items.append(item)
            except csv.Error as e:
                raise ValueError(f"{path.name} {reader.line_num + 1}행: 따옴표가 맞지 않습니다({e}). 칸 안의 "
                                 "큰따옴표는 두 번(\"\") 적거나 칸 전체를 큰따옴표로 감싸세요.") from None
    except UnicodeDecodeError:
        raise _not_utf8(path) from None
    _check_ids([it for it in items if it.id], path.name)
    used = {it.id for it in items if it.id}
    for k in auto:
        no = k + 1
        while f"L{no:04d}" in used:
            no += 1
        items[k].id = f"L{no:04d}"
        used.add(items[k].id)
    return items, header


def read_lines(path: str | Path) -> tuple[list[LineItem], list[str]]:
    """입력 파일을 읽어 (대사 목록, 입력 칸 이름 순서)를 돌려준다(모듈 머리말의 논문 판 읽기)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"입력 파일이 없습니다: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        items, columns = _read_table(path, ",")
    elif suffix == ".tsv":
        items, columns = _read_table(path, "\t")
    elif suffix == ".txt":
        items, columns = _read_txt(path)
    else:
        raise ValueError(f"입력은 {', '.join(INPUT_SUFFIXES)} 파일만 됩니다: {path.name}")
    if not items:
        raise ValueError(f"{path.name} 에 대사가 하나도 없습니다.")
    return items, columns


def input_value(item: LineItem, column: str) -> str:
    """결과 csv 에 다시 적는 입력 칸 값. 기존 판 꼴에 맞지 않아 extra 에 둔 값은 적힌 글 그대로."""
    raw = raw_unused(item, column)
    return raw if raw else _input_value(item, column)


# ---------------------------------------------------------------- 값 꺼내기

def _r(x: float | None, digits: int) -> float | None:
    return None if x is None else round(float(x), digits)


def syllables(text: str) -> int:
    """화면에 적는 한글 음절 수(기존 판과 같은 셈: 숫자는 한자어로 읽음)."""
    return ko_syllable_count(text)


def iso_candidate_dict(c: IsoCandidate) -> dict[str, Any]:
    return {"text": c.text, "origin": c.origin, "round": c.round, "iteration": c.iteration, "checked": c.checked,
            "dur_sec": _r(c.dur_sec, 6), "k_frames": _r(c.k_frames, 4), "ok_len": c.ok_len, "labse": _r(c.labse, 6),
            "ok_sem": c.ok_sem, "passed": c.passed}


def ps_candidate_dict(c: PsCandidate, rnd: int | None) -> dict[str, Any]:
    return {"text": c.text, "round": rnd, "dur_sec": _r(c.dur_sec, 6), "n_frames": c.n_frames,
            "dtw": _r(c.dtw, 6), "dtw_inv": _r(c.dtw_inv, 6), "comet": _r(c.comet, 6), "score": _r(c.score, 6)}


def flat_values(p: PaperLine) -> dict[str, Any]:
    """results.csv 한 줄의 값(RESULT_COLUMNS 순서, 숫자는 수 그대로)."""
    iso, chosen, ps, fin = p.iso, p.iso.chosen, p.ps, p.final
    return {
        "id": p.item.id, "source_en": p.item.source_en,
        "k_prime": _r(iso.k_prime, 4), "k_prime_basis": iso.k_prime_basis,
        "draft_ko": iso.draft_ko,
        "iso_ko": iso.target or "", "iso_status": iso.status, "iso_iterations": iso.iterations,
        "iso_k": _r(chosen.k_frames, 4) if chosen else None, "iso_labse": _r(chosen.labse, 6) if chosen else None,
        "ps_ko": ps.text if ps else "", "ps_dtw": _r(ps.dtw, 6) if ps else None,
        "final_ko": fin.text if fin else "",
        "ps_comet_score": _r(fin.score, 6) if fin else None, "ps_comet_dtw": _r(fin.dtw, 6) if fin else None,
        "ps_comet_comet": _r(fin.comet, 6) if fin else None,
        "n_ps_candidates": len(p.ps_candidates), "message": p.message,
    }


def result_dict(p: PaperLine) -> dict[str, Any]:
    """results.jsonl 한 줄."""
    it, iso, ch = p.item, p.iso, p.choice
    rounds = p.ps_rounds if len(p.ps_rounds) == len(p.ps_candidates) else [None] * len(p.ps_candidates)
    out = flat_values(p)
    out.update({
        "status": p.status, "seconds_input": it.seconds,
        "draft_origin": "given" if (it.draft_ko or "").strip() else "model1",
        "r": _r(iso.r, 6), "src_dur_sec": _r(iso.src_dur_sec, 6), "length_ratio": iso.length_ratio,
        "iso": {"status": iso.status, "selected_index": iso.selected, "iterations": iso.iterations,
                "rounds": iso.rounds, "message": iso.message,
                "candidates": [iso_candidate_dict(c) for c in iso.candidates]},
        "ps": {"index": ch.ps if ch else None, "n_src_frames": p.n_src_frames,
               "excluded": list(ch.excluded) if ch else [],
               "candidates": [ps_candidate_dict(c, r) for c, r in zip(p.ps_candidates, rounds)]},
        "ps_comet": {"index": ch.ps_comet if ch else None,
                     "dtw_inv": _r(p.final.dtw_inv, 6) if p.final else None},
        "input_unused": {"register": raw_unused(it, "register") or it.register,
                         "lip": raw_unused(it, "lip") or it.lip,
                         "target_syllables": raw_unused(it, "target_syllables") or it.target_syllables,
                         "relationship": it.relationship, "prev_en": it.prev_en, "meaning_notes": it.meaning_notes},
        "extra": {k: v for k, v in it.extra.items() if not raw_unused(it, k)},
    })
    return out


def _csv_value(name: str, value: Any) -> Any:
    digits = {"k_prime": 2, "iso_k": 2, "iso_labse": 4, "ps_dtw": 3, "ps_comet_score": 4, "ps_comet_dtw": 3,
              "ps_comet_comet": 4}
    if name in digits:
        return _fmt(value, digits[name])
    return "" if value is None else value


def csv_rows(lines: Sequence[PaperLine], input_columns: Sequence[str]) -> tuple[list[str], list[list[Any]]]:
    rest = [c for c in input_columns if c not in RESULT_COLUMNS]
    header = list(RESULT_COLUMNS) + rest
    rows = []
    for p in lines:
        flat = flat_values(p)
        rows.append([_csv_value(k, flat[k]) for k in RESULT_COLUMNS] + [input_value(p.item, c) for c in rest])
    return header, rows


def versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"pstts": __version__, "paper_prompt": PROMPT_VERSION_PAPER,
                                  "python": platform.python_version(), "platform": platform.platform()}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def run_dict(lines: Sequence[PaperLine], info: PaperRunInfo, cfg: PaperConfig, source: str) -> dict[str, Any]:
    n = len(lines)
    total = info.seconds.get("total")
    return {
        "version": "paper", "input": source, "n_lines": n, "started_at": info.started_at,
        "finished_at": info.finished_at, "device": info.device, "seconds": info.seconds,
        "seconds_per_line": round(total / n, 2) if total is not None and n else None,
        "status_counts": info.status_counts, "iso_counts": info.iso_counts,
        "llm": {"backend": info.llm_backend, "model": cfg.llm_model, "effort": cfg.llm_effort,
                "calls": info.llm_calls, "cache_hits": info.llm_cache_hits, "by_stage": info.llm,
                "errors": info.llm_errors, "stopped": info.llm_stopped},
        "iso_rounds": info.iso_rounds, "ps_rounds": info.ps_rounds,
        "model1": info.model1, "comet_checkpoint": info.comet_checkpoint, "frontend": info.frontend,
        "unused_columns": info.unused_columns, "unknown_tokens": info.unknown_tokens,
        "config": asdict(cfg), "ko_rate": cfg.ko_rate, "versions": versions(),
    }


# ---------------------------------------------------------------- report.md

def _iso_details(c: IsoCandidate) -> str:
    state = "통과" if c.passed else "탈락(" + ", ".join(
        name for name, ok in (("길이 식 6", c.ok_len), ("뜻 식 7", c.ok_sem)) if not ok) + ")"
    return f"{syllables(c.text)}음절, K {c.k_frames:.1f}, LaBSE {c.labse:.3f}, {state}"


def _ps_details(c: PsCandidate, digits: int = 3) -> str:
    parts = [f"{syllables(c.text)}음절", f"{c.n_frames}프레임"]
    for name, value, d in (("DTW", c.dtw, 1), ("inv", c.dtw_inv, 3), ("COMET", c.comet, digits),
                           ("점수", c.score, digits)):
        if value is not None:
            parts.append(f"{name} {value:.{d}f}")
    return ", ".join(parts)


def _top(p: PaperLine, k: int = 3) -> list[tuple[int, PsCandidate]]:
    """PS-Comet 점수 상위 k 개(같으면 먼저 나온 것)."""
    scored = [(i, c) for i, c in enumerate(p.ps_candidates) if c.score is not None]
    scored.sort(key=lambda ic: (-float(ic[1].score), ic[0]))
    return scored[:k]


def _round_counts(rounds: Sequence[int]) -> str:
    counts = Counter(rounds)
    return ", ".join(f"{k}차 {v}" for k, v in sorted(counts.items()))


def k_prime_text(p: PaperLine) -> str:
    """K' 와 그 근거를 한 줄로(식 2, 3)."""
    iso = p.iso
    if iso.k_prime is None:
        return ""
    src = f"원문 TTS 길이 {iso.src_dur_sec:.3f}초" if iso.src_dur_sec is not None else "원문 TTS 길이 모름"
    rr = f", R {iso.r:.3f}" if iso.r is not None else ""
    if iso.k_prime_basis == "seconds" and p.item.seconds is not None:
        return f"{iso.k_prime:.1f}프레임(seconds 칸 {p.item.seconds:g}초, {src}{rr})"
    return f"{iso.k_prime:.1f}프레임(seconds 칸이 없어 {src}로 정함{rr})"


def _g2p_text(info: dict[str, Any]) -> str:
    en = (info.get("g2p") or {}).get("en") or {}
    if not en:
        return ""
    stress = "강세표 남김" if en.get("with_stress") else "강세표 없음"
    return f". G2P 영어 phonemizer {en.get('voice')}({stress}), 한국어 g2pK 호환 자모"


def frontend_text(info: dict[str, Any]) -> str:
    """프런트엔드 한 줄 설명."""
    name = info.get("name", "")
    if name == "approx":
        w = info.get("weights", {})
        table = info.get("vowel_table", {})
        return (f"근사(approx): 문장 길이 = 음절 수 / 속도(영어 {info.get('en_rate')}, "
                f"한국어 {info.get('ko_rate', 0):.4g}"
                f" 음절/초), 토큰 무게 모음 {w.get('vowel')}, 한국어 글라이드 모음 {w.get('ko_glide_vowel')}, "
                f"자음 {w.get('consonant')}, 글라이드 {w.get('glide')}, 빈 토큰·공백·문장부호·소리 없는 초성 ㅇ 0. "
                f"모음표 PanPhon {len(table.get('symbols', []))}기호(sha1 {str(table.get('sha1', ''))[:12]}), null 비용 "
                f"{info.get('null_cost')}{_g2p_text(info)}")
    if name == "yourtts":
        table = info.get("vowel_table") or {}
        spk = info.get("speaker") or {}
        dropped = info.get("dropped_texts") or {}
        return (f"Your-TTS(yourtts): 체크포인트 {info.get('checkpoint')} "
                f"(sha1 {str(info.get('checkpoint_sha1'))[:12]}), "
                f"화자 {spk.get('name', '-')}({spk.get('kind', '-')}), 잡음 {info.get('noise_scale_dp')}, "
                f"모음표 {table.get('path', '-')}, 글자 목록에 없어 버린 글자가 든 글 영어 {dropped.get('en', 0)}개, "
                f"한국어 {dropped.get('ko', 0)}개{_g2p_text(info)}")
    return str(name)


def report_markdown(lines: Sequence[PaperLine], info: PaperRunInfo, cfg: PaperConfig, source: str) -> str:
    n = len(lines)
    total = info.seconds.get("total", 0.0)
    counts, ic = info.status_counts, info.iso_counts
    out = ["# PS-TTS 논문 그대로 판 결과", ""]
    out.append(f"- 입력: {_md(source)}, 대사 {n}줄")
    out.append(f"- 시작 {info.started_at}, 걸린 시간 {total:.1f}초(줄당 {total / max(n, 1):.1f}초), 장치 {info.device}")
    out.append(f"- 결과: PS-Comet 고름 {counts.get('ok', 0)}줄, 실패 {counts.get('error', 0)}줄. ISO: 초벌 통과 "
               f"{ic.get('draft', 0)}줄, 바꿔 쓰기 {ic.get('paraphrase', 0)}줄, 대안 {ic.get('fallback', 0)}줄, "
               f"오류 {ic.get('error', 0)}줄")
    stages = ", ".join(f"{STAGE_KO.get(k, k)} {v:.1f}초" for k, v in info.seconds.items() if k != "total")
    if stages:
        out.append(f"- 단계별 시간: {stages}")
    llm_line = f"- 모델 2: {info.llm_backend}"
    if info.llm_backend.startswith("claude"):
        llm_line += f", {cfg.llm_model}(effort {cfg.llm_effort})"
    elif info.llm_backend.startswith("openai"):
        llm_line += f", {cfg.llm_model}({cfg.llm_base_url})"
    iso_llm, ps_llm = info.llm.get("iso", {}), info.llm.get("ps", {})
    llm_line += (f". ISO 호출 {iso_llm.get('calls', 0)}번(캐시 {iso_llm.get('cache_hits', 0)}번), "
                 f"PS 호출 {ps_llm.get('calls', 0)}번(캐시 {ps_llm.get('cache_hits', 0)}번)")
    out.append(llm_line)
    for rd in info.iso_rounds:
        out.append(f"  - ISO {rd['round']}차: {rd['lines']}줄에 {rd['requested']}개 요청, 새 후보 {rd['new']}개, "
                   f"검사 {rd['checked']}개, 통과 {rd['passed']}줄, 호출 {rd['calls']}번, 캐시 {rd['cache_hits']}번, "
                   f"{rd['llm_seconds']:.1f}초")
    for rd in info.ps_rounds:
        out.append(f"  - PS {rd['round']}차: {rd['lines']}줄에 {rd['requested']}개 요청, 새 후보 {rd['new']}개, "
                   f"호출 {rd['calls']}번, 캐시 {rd['cache_hits']}번, {rd['llm_seconds']:.1f}초")
    if info.llm_stopped:
        out.append(f"- 모델 2 를 쓸 수 없어 멈춤: {_md(info.llm_stopped)}")
    for err in info.llm_errors:
        out.append(f"- 모델 2 오류: {_md(err)}")
    out.append(f"- TTS 프런트엔드: {_md(frontend_text(info.frontend))}")
    radius = "없음(tslearn 기본, 반경 1)" if cfg.sakoe_chiba_radius is None else f"{cfg.sakoe_chiba_radius}프레임"
    out.append(f"- 설정: 식 6 K' ±{cfg.iso_tol_frames}프레임(SA 프레임 {cfg.frame_sec_sa * 1000:.2f}ms), 식 7 LaBSE "
               f"{cfg.labse_min:g} 이상, ISO 한 번에 {cfg.iso_batch}개 최대 {cfg.iso_max}번, PS 후보 {cfg.ps_n}개"
               f"(모자라면 {cfg.ps_extra_rounds}번 더 물음), DTW Sakoe-Chiba 반경 {radius}, 식 20 alpha {cfg.alpha:g}, "
               f"beta {cfg.beta:g}, COMET {cfg.comet_model}")
    if info.unused_columns:
        cols = ", ".join(f"{k} {v}줄" for k, v in info.unused_columns.items())
        out.append(f"- 논문 판이 쓰지 않은 입력 칸: {cols}")
    if info.unknown_tokens:
        out.append(f"- G2P 가 분류하지 못한 토큰(0길이로 둠): {_md(str(info.unknown_tokens))}")

    out += ["", "| id | K' | ISO | PS(식 11) | PS-Comet(식 20, 최종) | 상태 |", "|---|---|---|---|---|---|"]
    for p in lines:
        kp = f"{p.iso.k_prime:.1f}" if p.iso.k_prime is not None else ""
        ps, fin = p.ps, p.final
        out.append(f"| {_md(p.item.id)} | {kp} | {_md(p.iso.target or '')} | {_md(ps.text) if ps else ''} | "
                   f"{_md(fin.text) if fin else ''} | {STATUS_KO.get(p.status, p.status)} |")

    for p in lines:
        iso, it = p.iso, p.item
        out += ["", f"## {_md(it.id)} ({STATUS_KO.get(p.status, p.status)})", ""]
        out.append(f"- 영어: {_md(it.source_en)}")
        if iso.k_prime is not None:
            out.append(f"- K': {k_prime_text(p)}")
        draft = iso.candidates[0] if iso.candidates and iso.candidates[0].origin == "draft" else None
        origin = "입력 초벌" if (it.draft_ko or "").strip() else "모델 1 초벌"
        if draft is not None:
            out.append(f"- 초벌 T0({origin}): {_md(draft.text)} ({_iso_details(draft)})")
        else:
            out.append(f"- 초벌 T0({origin}): {_md(iso.draft_ko) or '(없음)'}")
        chosen = iso.chosen
        if chosen is not None:
            how = ISO_STATUS_KO.get(iso.status, iso.status)
            out.append(f"- ISO({how}, 바꿔 쓴 후보 {iso.iterations}개 검사, 모델 2 {iso.rounds}차): "
                       f"**{_md(chosen.text)}** ({_iso_details(chosen)})")
            if iso.length_ratio is not None:
                out.append(f"  - 바꿔 쓰기 요청의 length_ratio {iso.length_ratio:g}, 받은 후보 "
                           f"{max(0, len(iso.candidates) - 1)}개")
        ps, fin = p.ps, p.final
        if p.ps_candidates:
            dtws = [c.dtw for c in p.ps_candidates if c.dtw is not None]
            span = f", DTW {min(dtws):.1f}~{max(dtws):.1f}" if dtws else ""
            out.append(f"- PS 후보 {len(p.ps_candidates)}개({_round_counts(p.ps_rounds) or '-'}), 원문 "
                       f"{p.n_src_frames}프레임{span}")
        if ps is not None:
            out.append(f"- PS(식 11, DTW 최소): **{_md(ps.text)}** ({_ps_details(ps)})")
        if fin is not None:
            out.append(f"- PS-Comet(식 20, 최종): **{_md(fin.text)}** ({_ps_details(fin)})")
            out.append(f"  - 점수 {fin.score:.3f} = {cfg.alpha:g} x {fin.dtw_inv:.3f} + {cfg.beta:g} x {fin.comet:.3f}")
            out.append("- PS-Comet 상위 3개:")
            out.extend(f"  {k}. {_md(c.text)} ({_ps_details(c)})" for k, (_, c) in enumerate(_top(p), 1))
        if p.message:
            out.append(f"- 메모: {_md(p.message)}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- 파일 쓰기

def write_outputs(out_dir: str | Path, lines: Sequence[PaperLine], info: PaperRunInfo, cfg: PaperConfig,
                  input_columns: Sequence[str], source: str) -> dict[str, Path]:
    """다섯 가지 결과 파일을 out_dir 에 쓰고 경로를 돌려준다. 같은 이름 파일은 덮어쓴다(한꺼번에 바뀜)."""
    out = Path(out_dir)
    paths = {name: out / name for name in OUTPUT_NAMES}
    header, rows = csv_rows(lines, input_columns)
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    run = run_dict(lines, info, cfg, source)
    run["outputs"] = {k: str(v) for k, v in paths.items()}
    _publish(out, {
        "results.jsonl": (_jsonl(result_dict(p) for p in lines), "utf-8"),
        "results.csv": (buf.getvalue(), "utf-8-sig"),
        "report.md": (report_markdown(lines, info, cfg, source), "utf-8"),
        "candidates.jsonl": (_jsonl(info.rows), "utf-8"),
        "run.json": (json.dumps(run, ensure_ascii=False, indent=1, default=str) + "\n", "utf-8"),
    })
    return paths


def print_choice(p: PaperLine, cfg: PaperConfig, stream: Any = None) -> None:
    """문장 하나 모드: ISO, PS, PS-Comet 고른 대사와 PS-Comet 상위 3개를 화면에 찍는다."""
    stream = stream or sys.stdout
    iso = p.iso
    print(f"영어: {p.item.source_en}", file=stream)
    if iso.k_prime is not None:
        print(f"K': {k_prime_text(p)}", file=stream)
    if iso.draft_ko:
        print(f"초벌 T0: {iso.draft_ko}", file=stream)
    chosen = iso.chosen
    if chosen is not None:
        print(f"ISO({ISO_STATUS_KO.get(iso.status, iso.status)}, 검사 {iso.iterations}번): {chosen.text} "
              f"({_iso_details(chosen)})", file=stream)
    if p.ps is not None:
        print(f"PS(식 11): {p.ps.text} ({_ps_details(p.ps)})", file=stream)
    fin = p.final
    if fin is None:
        print(f"고른 대사 없음: {p.message}", file=stream)
        return
    print(f"PS-Comet(식 20, 최종): {fin.text} ({_ps_details(fin)})", file=stream)
    print(f"PS-Comet 상위 3개(후보 {len(p.ps_candidates)}개 중):", file=stream)
    for k, (_, c) in enumerate(_top(p), 1):
        print(f"  {k}. {c.text} ({_ps_details(c)})", file=stream)
    if p.message:
        print(f"메모: {p.message}", file=stream)


def summary_line(lines: Sequence[PaperLine], info: PaperRunInfo) -> str:
    """끝에 찍는 한 줄 요약."""
    counts, ic = info.status_counts, info.iso_counts
    total = info.seconds.get("total", 0.0)
    iso_llm, ps_llm = info.llm.get("iso", {}), info.llm.get("ps", {})
    return (f"PS-Comet 고름 {counts.get(STATUS_OK, 0)}줄, 실패 {counts.get('error', 0)}줄(ISO 초벌 통과 "
            f"{ic.get('draft', 0)}, 바꿔 쓰기 {ic.get('paraphrase', 0)}, 대안 {ic.get('fallback', 0)}, 오류 "
            f"{ic.get('error', 0)}). 걸린 시간 {total:.1f}초(줄당 {total / max(len(lines), 1):.1f}초). 모델 2 호출 "
            f"ISO {iso_llm.get('calls', 0)}번(캐시 {iso_llm.get('cache_hits', 0)}번), PS {ps_llm.get('calls', 0)}번"
            f"(캐시 {ps_llm.get('cache_hits', 0)}번)")

