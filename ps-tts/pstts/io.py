"""입력 읽기와 결과 쓰기.

읽기
- .csv: 대사 양식 머리글(TEMPLATE_HEADER, 'C:/Users/jongh/Desktop/Study/메모/더빙 실습 코드/대사 양식.csv' 와 같음).
  utf-8-sig. source_en 만 꼭 있어야 한다. seconds 칸이 있으면 쓴다. lip 은 on/off(빈칸 on),
  register 는 REGISTERS 중 하나(빈칸 미지정). 잘못된 값과 맞지 않는 따옴표는 행 번호와 함께 오류를 낸다.
  UTF-8 이 아니면(엑셀의 cp949 CSV) 'CSV UTF-8' 로 저장하라는 오류를 낸다.
  id 가 빈 행은 다 읽은 뒤 L0001 꼴로 매기고, 직접 적은 id 와 겹치면 다음 번호를 쓴다.
  LineItem 에 없는 칸(split, work_id, length_request …)은 extra 에 담아 결과에 그대로 붙인다.
- .tsv: 같은 머리글 이름, 탭으로 나눔. 따옴표는 글자 그대로 둔다(QUOTE_NONE).
- .txt: 한 줄에 영어 대사 하나. id 는 L0001 부터.

쓰기(out 폴더)
- results.jsonl: 줄마다 모든 후보와 점수.
- results.csv: 줄마다 고른 대사, 그 말투(final_register)와 점수(utf-8-sig) + 입력의 나머지 칸.
다섯 파일은 한꺼번에 바뀐다. 하나라도 잠겨 있으면 모두 예전 그대로 두고 한국어 OSError 를 낸다(_publish).
- report.md: 사람이 읽는 보고(조건, 초벌, 고른 대사, 다음 후보 3개, 탈락 까닭 요약).
- candidates.jsonl: 모델 2 후보를 FileBackend 꼴로(--llm file --llm-file 로 같은 후보를 다시 쓸 수 있음).
- run.json: 설정, 판(버전), 걸린 시간, 모델 2 호출 수.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from io import StringIO
from pathlib import Path
from typing import Any

from . import __version__
from .config import Config
from .llm import PROMPT_VERSION
from .pipeline import RunInfo
from .records import REGISTERS, Candidate, LineItem, LineResult
from .select import ranked

TEMPLATE_HEADER = ("id", "split", "work_id", "scene_id", "source_en", "prev_en", "relationship", "register",
                   "length_request", "target_syllables", "lip", "draft_ko", "draft_model", "approved_ko",
                   "meaning_notes")
# LineItem 이 직접 쓰는 칸. 나머지 칸은 extra 로 간다.
ITEM_COLUMNS = ("id", "source_en", "prev_en", "relationship", "register", "target_syllables", "seconds", "lip",
                "draft_ko", "meaning_notes")
# 문장 하나로 돌릴 때 결과 csv 에 붙이는 입력 칸
SINGLE_COLUMNS = ("source_en", "prev_en", "relationship", "register", "target_syllables", "seconds", "lip",
                  "meaning_notes")
RESULT_COLUMNS = ("id", "source_en", "register", "target", "tol", "draft_ko", "final_ko", "final_syllables",
                  "final_register", "labse", "comet", "dtw", "score", "status", "n_candidates", "n_pass", "message")
STATUS_KO = {"ok": "고름", "fallback": "대안(통과 후보 없음)", "error": "실패"}
ORIGIN_KO = {"model1": "모델 1 초벌", "given": "입력 초벌", "llm": "모델 2", "file": "후보 파일"}
STAGE_KO = {"model1": "모델 1", "labse": "LaBSE", "llm": "모델 2", "comet": "COMET", "dtw": "모음 DTW"}
INPUT_SUFFIXES = (".csv", ".tsv", ".txt")
PACKAGES = ("torch", "transformers", "sentence-transformers", "unbabel-comet", "tslearn", "panphon",
            "espeakng-loader", "numpy")


# ---------------------------------------------------------------- 읽기

def _clean(value: str | None) -> str:
    return " ".join((value or "").split())


def _parse_int(text: str, where: str, name: str) -> int:
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"{where}: {name} '{text}' 는 1 이상의 정수여야 합니다.") from None
    if not number.is_integer() or number < 1:
        raise ValueError(f"{where}: {name} '{text}' 는 1 이상의 정수여야 합니다.")
    return int(number)


def _parse_seconds(text: str, where: str) -> float:
    try:
        sec = float(text)
    except ValueError:
        raise ValueError(f"{where}: seconds '{text}' 는 0 보다 큰 수(초)여야 합니다.") from None
    if not sec > 0 or sec == float("inf"):
        raise ValueError(f"{where}: seconds '{text}' 는 0 보다 큰 수(초)여야 합니다.")
    return sec


def item_from_row(row: dict[str, str | None], where: str, auto_id: str) -> LineItem:
    """입력 한 행(칸 이름 -> 값)을 LineItem 으로. where 는 오류 문구의 위치(예: 'lines.csv 3행')."""
    source = _clean(row.get("source_en"))
    if not source:
        raise ValueError(f"{where}: source_en(영어 대사)이 비었습니다.")
    reg = _clean(row.get("register")) or "미지정"
    if reg not in REGISTERS:
        raise ValueError(f"{where}: register '{reg}' 는 {', '.join(REGISTERS)} 중 하나여야 합니다(빈칸은 미지정).")
    ts = _clean(row.get("target_syllables"))
    sec = _clean(row.get("seconds"))
    lip = _clean(row.get("lip")).lower() or "on"
    if lip not in ("on", "off"):
        raise ValueError(f"{where}: lip '{lip}' 는 on 또는 off 여야 합니다(빈칸은 on).")
    extra = {k: (v if v is not None else "") for k, v in row.items() if k not in ITEM_COLUMNS}
    return LineItem(
        id=_clean(row.get("id")) or auto_id,
        source_en=source,
        prev_en=_clean(row.get("prev_en")),
        relationship=_clean(row.get("relationship")),
        register=reg,
        target_syllables=_parse_int(ts, where, "target_syllables") if ts else None,
        seconds=_parse_seconds(sec, where) if sec else None,
        lip=lip == "on",
        draft_ko=_clean(row.get("draft_ko")) or None,
        meaning_notes=_clean(row.get("meaning_notes")),
        extra=extra,
    )


def _not_utf8(path: Path) -> ValueError:
    return ValueError(f"{path.name}: UTF-8 로 읽을 수 없습니다. 엑셀에서 저장했다면 '다른 이름으로 저장' 에서 "
                      "'CSV UTF-8(쉼표로 분리)' 로 다시 저장하세요(그냥 'CSV(쉼표로 분리)' 는 한국어 Windows 에서 cp949 로 저장됨).")


def _read_table(path: Path, delimiter: str) -> tuple[list[LineItem], list[str]]:
    """csv(쉼표)·tsv(탭) 표를 읽는다.

    tsv 는 따옴표를 글자 그대로 둔다(TSV 에는 따옴표 규칙이 없어서, 따옴표로 시작하는 대사가 다음 행을 삼키지 않게).
    csv 는 따옴표가 맞지 않으면 행 번호와 함께 멈춘다(strict). id 가 빈 행은 다 읽은 뒤 L0001 꼴로 매기되
    다른 행이 직접 적은 id 와 겹치면 다음 번호를 쓴다.
    """
    tsv = delimiter == "\t"
    items: list[LineItem] = []
    auto: list[int] = []                                  # id 칸이 빈 행(items 안의 위치)
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
                        continue                          # 빈 행은 건너뜀
                    item = item_from_row(row, where, "")
                    if not item.id:
                        auto.append(len(items))
                    items.append(item)
            except csv.Error as e:
                # DictReader.line_num 은 마지막으로 다 읽은 행의 끝 줄이라, 망가진 행은 그다음 줄에서 시작한다
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


def _read_txt(path: Path) -> tuple[list[LineItem], list[str]]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        raise _not_utf8(path) from None
    items: list[LineItem] = []
    for line in lines:
        text = _clean(line)
        if text:
            items.append(LineItem(id=f"L{len(items) + 1:04d}", source_en=text))
    return items, ["source_en"]


def _check_ids(items: Sequence[LineItem], name: str) -> None:
    seen: set[str] = set()
    for it in items:
        if it.id in seen:
            raise ValueError(f"{name}: 같은 id 가 두 번 있습니다: {it.id}")
        seen.add(it.id)


def read_lines(path: str | Path) -> tuple[list[LineItem], list[str]]:
    """입력 파일을 읽어 (대사 목록, 입력 칸 이름 순서)를 돌려준다."""
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


# ---------------------------------------------------------------- 결과 꼴

def _num(x: float | None, digits: int) -> float | None:
    return None if x is None else round(float(x), digits)


def candidate_dict(c: Candidate) -> dict[str, Any]:
    return {
        "text": c.text, "origin": c.origin, "round": c.round, "syllables": c.syllables, "passed": c.passed,
        "ok_length": c.ok_length, "ok_meaning": c.ok_meaning, "ok_register": c.ok_register, "ok_numbers": c.ok_numbers,
        "register_found": c.register_found, "labse": c.labse, "comet": c.comet, "dtw": c.dtw,
        "dtw_inv": c.dtw_inv, "score": c.score, "reasons": list(c.reasons),
    }


def _n_pass(result: LineResult) -> int:
    return sum(1 for c in result.candidates if c.passed)


def result_dict(r: LineResult) -> dict[str, Any]:
    it, t = r.item, r.target
    sel_index = next((k for k, c in enumerate(r.candidates) if c is r.selected), None)
    return {
        "id": it.id, "source_en": it.source_en, "prev_en": it.prev_en, "relationship": it.relationship,
        "register": it.register, "lip": it.lip, "meaning_notes": it.meaning_notes,
        "target_syllables_input": it.target_syllables, "seconds_input": it.seconds,
        "target": {**asdict(t), "low": t.low, "high": t.high},
        "draft_ko": r.draft_ko, "status": r.status, "message": r.message, "rounds_used": r.rounds_used,
        "final_ko": r.selected.text if r.selected else "", "selected_index": sel_index,
        "n_candidates": len(r.candidates), "n_pass": _n_pass(r),
        "candidates": [candidate_dict(c) for c in r.candidates], "extra": dict(it.extra),
    }


def _input_value(item: LineItem, column: str) -> str:
    """결과 csv 에 다시 적는 입력 칸 값(입력 꼴 그대로)."""
    if column == "target_syllables":
        return "" if item.target_syllables is None else str(item.target_syllables)
    if column == "seconds":
        return "" if item.seconds is None else f"{item.seconds:g}"
    if column == "lip":
        return "on" if item.lip else "off"
    if column in ("prev_en", "relationship", "meaning_notes", "source_en", "register", "id"):
        return str(getattr(item, column))
    if column == "draft_ko":
        return item.draft_ko or ""
    value = item.extra.get(column, "")
    return "" if value is None else str(value)


def csv_rows(results: Sequence[LineResult], input_columns: Sequence[str]) -> tuple[list[str], list[list[Any]]]:
    rest = [c for c in input_columns if c not in RESULT_COLUMNS]
    header = list(RESULT_COLUMNS) + rest
    rows = []
    for r in results:
        s = r.selected
        rows.append([
            r.item.id, r.item.source_en, r.item.register, r.target.n, r.target.tol, r.draft_ko,
            s.text if s else "", s.syllables if s else "", "·".join(s.register_found or []) if s else "",
            _fmt(s.labse if s else None, 4),
            _fmt(s.comet if s else None, 4), _fmt(s.dtw if s else None, 3), _fmt(s.score if s else None, 4),
            r.status, len(r.candidates), _n_pass(r), r.message,
            *(_input_value(r.item, c) for c in rest),
        ])
    return header, rows


def _fmt(x: float | None, digits: int) -> str:
    return "" if x is None else f"{x:.{digits}f}"


def candidates_rows(results: Sequence[LineResult]) -> list[dict[str, Any]]:
    """모델 2 후보를 FileBackend 꼴 {"id", "round", "candidates"} 로(차수 1 이상만)."""
    rows = []
    for r in results:
        by_round: dict[int, list[str]] = {}
        for c in r.candidates:
            if c.round >= 1:
                by_round.setdefault(c.round, []).append(c.text)
        rows.extend({"id": r.item.id, "round": k, "candidates": texts} for k, texts in sorted(by_round.items()))
    return rows


def versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"pstts": __version__, "prompt": PROMPT_VERSION,
                                  "python": platform.python_version(), "platform": platform.platform()}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def run_dict(results: Sequence[LineResult], info: RunInfo, cfg: Config, source: str) -> dict[str, Any]:
    n = len(results)
    total = info.seconds.get("total")
    return {
        "input": source, "n_lines": n, "started_at": info.started_at, "finished_at": info.finished_at,
        "device": info.device, "seconds": info.seconds,
        "seconds_per_line": round(total / n, 2) if total is not None and n else None,
        "status_counts": info.status_counts,
        "llm": {"backend": info.llm_backend, "model": cfg.llm_model, "effort": cfg.llm_effort,
                "calls": info.llm_calls, "cache_hits": info.llm_cache_hits, "errors": info.llm_errors,
                "stopped": info.llm_stopped},
        "rounds": info.rounds, "model1": info.model1, "comet_checkpoint": info.comet_checkpoint,
        "config": asdict(cfg), "ko_rate": cfg.ko_rate, "tolerance": cfg.tolerance(), "versions": versions(),
    }


# ---------------------------------------------------------------- report.md

def _md(text: str) -> str:
    """표·목록 안에서 깨지지 않게 | 와 줄바꿈을 바꾼다."""
    return " ".join(str(text).split()).replace("|", "\\|")


def _basis_text(r: LineResult, cfg: Config) -> str:
    t = r.target
    if t.basis == "explicit":
        return "사람이 정함"
    if t.basis == "seconds":
        return f"칸 {r.item.seconds:g}초 x 한국어 {cfg.ko_rate:.2f}음절/초"
    return f"영어 {t.en_syllables}음절 x coef {cfg.coef:g}"


def _reason_kind(reason: str) -> str:
    for kind in ("길이", "뜻", "말투", "숫자", "한국어 아님"):
        if reason.startswith(kind):
            return kind
    return "기타"


def _cand_details(c: Candidate, origin: bool = True) -> str:
    """후보 하나의 음절·말투·점수·출처·통과 여부를 한 줄로. 말투 '반말' 에는 -다 로 끝나는 해라체 서술도 든다."""
    parts = [f"{c.syllables}음절" if c.syllables is not None else "음절 모름"]
    if c.register_found:
        parts.append("·".join(c.register_found))
    for name, value, digits in (("LaBSE", c.labse, 3), ("COMET", c.comet, 3), ("DTW", c.dtw, 1),
                                ("점수", c.score, 3)):
        if value is not None:
            parts.append(f"{name} {value:.{digits}f}")
    if origin:
        parts.append(ORIGIN_KO.get(c.origin, c.origin) + (f" {c.round}차" if c.round else ""))
    state = "통과" if c.passed else "탈락: " + "; ".join(c.reasons)
    return f"{', '.join(parts)}, {state}"


def _cand_line(c: Candidate, origin: bool = True) -> str:
    return f"{_md(c.text)} ({_cand_details(c, origin)})"


def _round_counts(r: LineResult) -> str:
    counts = Counter(c.round for c in r.candidates)
    return ", ".join(("초벌" if k == 0 else f"{k}차") + f" {v}" for k, v in sorted(counts.items()))


def report_markdown(results: Sequence[LineResult], info: RunInfo, cfg: Config, source: str) -> str:
    n = len(results)
    total = info.seconds.get("total", 0.0)
    counts = info.status_counts
    out = ["# PS-TTS 텍스트판 결과", ""]
    out.append(f"- 입력: {_md(source)}, 대사 {n}줄")
    out.append(f"- 시작 {info.started_at}, 걸린 시간 {total:.1f}초(줄당 {total / max(n, 1):.1f}초), 장치 {info.device}")
    out.append(f"- 결과: 고름 {counts.get('ok', 0)}줄, 대안(통과 후보 없음) {counts.get('fallback', 0)}줄, "
               f"실패 {counts.get('error', 0)}줄")
    stages = ", ".join(f"{STAGE_KO.get(k, k)} {v:.1f}초" for k, v in info.seconds.items() if k != "total")
    if stages:
        out.append(f"- 단계별 시간: {stages}")
    llm_line = f"- 모델 2: {info.llm_backend}"
    if info.llm_backend.startswith("claude"):
        llm_line += f", {cfg.llm_model}(effort {cfg.llm_effort})"
    elif info.llm_backend.startswith("openai"):
        llm_line += f", {cfg.llm_model}({cfg.llm_base_url})"
    llm_line += f", 호출 {info.llm_calls}번, 캐시 적중 {info.llm_cache_hits}번"
    out.append(llm_line)
    for rd in info.rounds:
        out.append(f"  - {rd['round']}차: {rd['lines']}줄 요청 {rd['requested']}개, 새 후보 {rd['new']}개"
                   f"(통과 {rd['new_pass']}개), 호출 {rd['calls']}번, 캐시 {rd['cache_hits']}번, "
                   f"{rd['llm_seconds']:.1f}초")
    if info.llm_stopped:
        out.append(f"- 모델 2 를 쓸 수 없어 멈춤: {_md(info.llm_stopped)}")
    for err in info.llm_errors:
        out.append(f"- 모델 2 오류: {_md(err)}")
    out.append(f"- 설정: coef {cfg.coef:g}, 허용 ±{cfg.tolerance()}음절, LaBSE {cfg.labse_min:g} 이상, "
               f"말투 확인 {'켬' if cfg.check_register else '끔'}, 숫자 확인 {'켬' if cfg.check_numbers else '끔'}, "
               f"alpha {cfg.alpha:g}, beta {cfg.beta:g}, DTW 반경 {cfg.sakoe_chiba_radius}프레임, "
               f"COMET {cfg.comet_model}, 차수 최대 {cfg.max_rounds}, 차수당 {cfg.n_per_round}개, "
               f"통과 {cfg.min_pass}개 모이면 멈춤, 줄당 최대 {cfg.max_candidates}개")
    out += ["", "| id | 말투 | 목표 | 고른 대사 | 음절 | 상태 |", "|---|---|---|---|---|---|"]
    for r in results:
        s = r.selected
        out.append(f"| {_md(r.item.id)} | {r.item.register} | {r.target.n}±{r.target.tol} | "
                   f"{_md(s.text) if s else ''} | {s.syllables if s else ''} | {STATUS_KO.get(r.status, r.status)} |")
    for r in results:
        out += ["", f"## {_md(r.item.id)} ({STATUS_KO.get(r.status, r.status)})", ""]
        it, t = r.item, r.target
        out.append(f"- 영어: {_md(it.source_en)}")
        if it.prev_en:
            out.append(f"- 앞 대사: {_md(it.prev_en)}")
        cond = [f"말투 {it.register}"]
        if it.relationship:
            cond.append(f"관계 {_md(it.relationship)}")
        cond.append(f"목표 {t.n}음절(허용 {t.low}~{t.high}, {_basis_text(r, cfg)})")
        cond.append("입 모양 비교 " + ("켬" if it.lip else "끔"))
        out.append(f"- 조건: {', '.join(cond)}")
        if it.meaning_notes:
            out.append(f"- 꼭 살릴 뜻: {_md(it.meaning_notes)}")
        draft = next((c for c in r.candidates if c.round == 0), None)
        if draft is not None:
            out.append(f"- 초벌({ORIGIN_KO.get(draft.origin, draft.origin)}): {_cand_line(draft, origin=False)}")
        else:
            out.append("- 초벌: 없음")
        if r.selected is not None:
            out.append(f"- 고른 대사: **{_md(r.selected.text)}**")
            out.append(f"  - {_cand_details(r.selected)}")
        nxt = [c for c in ranked(r.candidates, t) if c is not r.selected][:3]
        if nxt:
            out.append("- 다음 후보:")
            out.extend(f"  {k}. {_cand_line(c)}" for k, c in enumerate(nxt, 1))
        kinds = Counter(_reason_kind(reason) for c in r.candidates for reason in c.reasons)
        why = ", ".join(f"{k} {v}개" for k, v in kinds.most_common()) or "없음"
        out.append(f"- 후보 {len(r.candidates)}개({_round_counts(r) or '없음'}), 통과 {_n_pass(r)}개. "
                   f"탈락 까닭: {why}")
        if r.message:
            out.append(f"- 메모: {_md(r.message)}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- 파일 쓰기

def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _publish(out: Path, files: dict[str, tuple[str, str]]) -> None:
    """결과 파일을 한꺼번에 바꿔 끼운다. 하나라도 못 바꾸면 모두 예전 그대로 두고 한국어 OSError 를 낸다.

    1) 모든 내용을 임시 파일(.tmp)에 쓴다. 2) 있던 결과 파일을 모두 잠시 다른 이름(.old)으로 옮긴다. 엑셀 등이 열고
    있으면 여기서 막히고, 그때까지 옮긴 파일을 되돌린다. 3) 임시 파일을 제자리로 옮기고 .old 를 지운다.
    """
    out.mkdir(parents=True, exist_ok=True)
    tag = str(os.getpid())
    tmps: dict[str, Path] = {}
    moved: list[tuple[Path, Path]] = []                 # (결과 파일, 잠시 옮겨 둔 파일)
    try:
        for name, (text, encoding) in files.items():
            tmp = out / f"{name}.{tag}.tmp"
            tmp.write_text(text, encoding=encoding, newline="")
            tmps[name] = tmp
        for name in files:
            target = out / name
            if not target.exists():
                continue
            old = out / f"{name}.{tag}.old"
            try:
                os.replace(target, old)
            except PermissionError as e:
                raise OSError(f"{target} 를 바꿀 수 없습니다. 엑셀 같은 다른 프로그램이 열고 있으면 닫고 다시 돌리세요. "
                              f"결과 파일은 하나도 바꾸지 않았습니다.") from e
            moved.append((target, old))
        for name in files:
            os.replace(tmps.pop(name), out / name)
    except BaseException:
        for target, old in reversed(moved):            # 옮겨 둔 예전 파일을 되돌린다
            try:
                os.replace(old, target)
            except OSError:
                pass
        moved.clear()
        raise
    finally:
        for tmp in tmps.values():
            _unlink_quietly(tmp)
        for _, old in moved:
            _unlink_quietly(old)


def _jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def write_outputs(out_dir: str | Path, results: Sequence[LineResult], info: RunInfo, cfg: Config,
                  input_columns: Sequence[str], source: str) -> dict[str, Path]:
    """다섯 가지 결과 파일을 out_dir 에 쓰고 경로를 돌려준다. 같은 이름 파일은 덮어쓴다.

    다섯 파일은 한꺼번에 바뀐다. 하나라도 잠겨 있으면(엑셀에서 연 results.csv 등) 모두 예전 그대로 남는다.
    """
    out = Path(out_dir)
    paths = {name: out / name for name in ("results.jsonl", "results.csv", "report.md", "candidates.jsonl",
                                           "run.json")}
    header, rows = csv_rows(results, input_columns)
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    run = run_dict(results, info, cfg, source)
    run["outputs"] = {k: str(v) for k, v in paths.items()}
    _publish(out, {
        "results.jsonl": (_jsonl(result_dict(r) for r in results), "utf-8"),
        "results.csv": (buf.getvalue(), "utf-8-sig"),
        "report.md": (report_markdown(results, info, cfg, source), "utf-8"),
        "candidates.jsonl": (_jsonl(candidates_rows(results)), "utf-8"),
        "run.json": (json.dumps(run, ensure_ascii=False, indent=1, default=str) + "\n", "utf-8"),
    })
    return paths


def print_choice(result: LineResult, stream: Any = None) -> None:
    """문장 하나 모드: 고른 대사와 다음 후보 3개를 화면에 찍는다."""
    stream = stream or sys.stdout
    r = result
    print(f"영어: {r.item.source_en}", file=stream)
    print(f"목표: {r.target.n}음절(허용 {r.target.low}~{r.target.high}), 말투 {r.item.register}, "
          f"상태 {STATUS_KO.get(r.status, r.status)}", file=stream)
    if r.draft_ko:
        print(f"초벌: {r.draft_ko}", file=stream)
    if r.selected is None:
        print(f"고른 대사 없음: {r.message}", file=stream)
        return
    print(f"고른 대사: {r.selected.text}", file=stream)
    print(f"  {_cand_details(r.selected)}", file=stream)
    nxt = [c for c in ranked(r.candidates, r.target) if c is not r.selected][:3]
    if nxt:
        print("다음 후보:", file=stream)
        for k, c in enumerate(nxt, 1):
            print(f"  {k}. {_cand_line(c)}", file=stream)
    if r.message:
        print(f"메모: {r.message}", file=stream)
