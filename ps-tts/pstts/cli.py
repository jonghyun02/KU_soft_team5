"""명령줄.

  run.cmd "I didn't order two tickets." --register 해요체
  run.cmd --input examples\\lines.csv --out out\\demo

문장 하나 모드는 고른 대사와 다음 후보 3개를 화면에 찍고 out 폴더에도 저장한다.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import traceback
from pathlib import Path
from typing import Any

from .config import Config
from .io import SINGLE_COLUMNS, STATUS_KO, item_from_row, print_choice, read_lines, write_outputs
from .pipeline import COMET_REFS, LLM_BACKENDS, run_lines
from .records import REGISTERS, LineItem
from .ruler import LENGTH_BASES

# 문장 하나로 돌릴 때만 쓰는 대사 옵션(파일 입력에서는 칸으로 적는다)
LINE_OPTIONS = ("register", "prev", "relationship", "target_syllables", "seconds", "no_lip", "meaning_notes")
# 명령줄 옵션 이름 -> Config 칸 이름(같은 이름은 적지 않음)
CONFIG_NAMES = {"tol": "tol_syllables", "beam": "nmt_beam", "out": "out_dir"}
CONFIG_OPTIONS = ("llm", "llm_model", "llm_effort", "llm_base_url", "llm_file", "llm_workers", "llm_chunk",
                  "llm_api_key_env", "n_per_round", "max_rounds", "min_pass", "max_candidates", "alpha", "beta",
                  "labse_min", "tol", "coef", "length_basis", "comet_model", "comet_ref", "labse_model",
                  "nmt_dir", "nmt_ckpt", "beam", "device", "cpu_threads", "out")


def build_parser() -> argparse.ArgumentParser:
    cfg = Config()
    ap = argparse.ArgumentParser(
        prog="pstts",
        description="영어 대사를 길이·뜻·말투·입 모양에 맞는 한국어 더빙 대사로 옮긴다(PS-TTS 텍스트판).",
        epilog='예: run.cmd "I didn\'t order two tickets." --register 해요체 / '
               "run.cmd --input examples\\lines.csv --out out\\demo")
    ap.add_argument("text", nargs="?", help="영어 대사 하나(문장 하나 모드)")
    ap.add_argument("--input", type=Path, help="대사 파일(.csv, .tsv, .txt)")
    ap.add_argument("--out", help=f"결과 폴더(기본 {cfg.out_dir}). 같은 이름 파일은 덮어쓴다")

    line = ap.add_argument_group("대사 옵션(문장 하나 모드에서만)")
    line.add_argument("--register", choices=REGISTERS, help="말투(기본 미지정)")
    line.add_argument("--prev", help="바로 앞 대사(영어)")
    line.add_argument("--relationship", help="화자 관계·장면")
    line.add_argument("--target-syllables", type=int, help="목표 한글 음절 수(주면 그대로 씀)")
    line.add_argument("--seconds", type=float, help="대사 칸 길이(초, 쉼 제외). --length-basis seconds 일 때 씀")
    line.add_argument("--no-lip", action="store_true", help="입이 안 보이는 장면(COMET 만으로 고름)")
    line.add_argument("--meaning-notes", help="꼭 살릴 뜻(사실·부정·숫자·강조)")

    llm = ap.add_argument_group("모델 2(후보 생성)")
    llm.add_argument("--llm", choices=LLM_BACKENDS, help=f"후보 생성 백엔드(기본 {cfg.llm})")
    llm.add_argument("--llm-model", help=f"모델 ID(기본 {cfg.llm_model})")
    llm.add_argument("--llm-effort", help=f"claude --effort(기본 {cfg.llm_effort})")
    llm.add_argument("--llm-base-url", help=f"--llm openai 의 서버 주소(기본 {cfg.llm_base_url})")
    llm.add_argument("--llm-api-key-env", help="--llm openai 에서 API 키가 든 환경 변수 이름")
    llm.add_argument("--llm-file", help="--llm file 의 후보 JSONL(결과 폴더의 candidates.jsonl 꼴)")
    llm.add_argument("--llm-workers", type=int, help=f"동시 호출 수(기본 {cfg.llm_workers})")
    llm.add_argument("--llm-chunk", type=int, help=f"한 번 부를 때 넣는 대사 수(기본 {cfg.llm_chunk})")
    llm.add_argument("--no-llm-cache", action="store_true", help="LLM 응답 캐시(cache/llm)를 읽지도 쓰지도 않음")
    llm.add_argument("--n-per-round", type=int, help=f"차수마다 줄당 새 후보 수(기본 {cfg.n_per_round})")
    llm.add_argument("--max-rounds", type=int, help=f"최대 차수(기본 {cfg.max_rounds}, 0 이면 초벌만)")
    llm.add_argument("--min-pass", type=int, help=f"통과 후보가 이만큼 모이면 멈춤(기본 {cfg.min_pass})")
    llm.add_argument("--max-candidates", type=int, help=f"줄당 최대 후보 수(기본 {cfg.max_candidates})")

    sel = ap.add_argument_group("거르기·고르기")
    sel.add_argument("--alpha", type=float, help=f"PS-Comet 의 DTW 무게(기본 {cfg.alpha})")
    sel.add_argument("--beta", type=float, help=f"PS-Comet 의 COMET 무게(기본 {cfg.beta})")
    sel.add_argument("--labse-min", type=float, help=f"뜻 조건 LaBSE 최솟값(기본 {cfg.labse_min})")
    sel.add_argument("--tol", type=int, help=f"허용 오차 ±음절(기본 {cfg.tolerance()})")
    sel.add_argument("--coef", type=float, help=f"영어 1음절당 한국어 음절 수(기본 {cfg.coef})")
    sel.add_argument("--length-basis", choices=LENGTH_BASES, help=f"길이 기준(기본 {cfg.length_basis})")
    sel.add_argument("--no-register-check", action="store_true", help="말투 조건으로 거르지 않음")
    sel.add_argument("--number-check", action="store_true",
                     help="영어 대사의 수가 후보에 없으면 떨어뜨림(논문에 없음, 기본 끔. 반·과장·굳은 말을 떨어뜨릴 수 있음)")

    mod = ap.add_argument_group("모델")
    mod.add_argument("--comet-model", help=f"COMET 모델(기본 {cfg.comet_model})")
    mod.add_argument("--comet-ref", choices=COMET_REFS, help="참조형 COMET 이면 draft 로 초벌을 참조로 씀(기본 none)")
    mod.add_argument("--labse-model", help=f"LaBSE 모델(기본 {cfg.labse_model})")
    mod.add_argument("--nmt-dir", help=f"모델 1 폴더(기본 {cfg.nmt_dir})")
    mod.add_argument("--nmt-ckpt", help=f"모델 1 체크포인트(기본 {cfg.nmt_ckpt}, 폴더 기준 상대 경로 가능)")
    mod.add_argument("--beam", type=int, help=f"모델 1 beam 크기(기본 {cfg.nmt_beam})")
    mod.add_argument("--no-draft", action="store_true", help="모델 1 초벌을 만들지 않음(입력의 draft_ko 는 씀)")
    mod.add_argument("--device", help="cuda 또는 cpu(기본 cuda). cuda 를 못 쓰면 멈추니 CPU 는 직접 적는다")
    mod.add_argument("--cpu-threads", type=int, help=f"CPU 로 돌릴 때 torch 스레드 수(기본 {cfg.cpu_threads})")
    return ap


def config_from_args(args: argparse.Namespace) -> Config:
    """명령줄 옵션을 Config 에 덮어쓴다. 주지 않은 옵션은 기본값 그대로."""
    changes: dict[str, Any] = {}
    for name in CONFIG_OPTIONS:
        value = getattr(args, name)
        if value is not None:
            changes[CONFIG_NAMES.get(name, name)] = value
    if args.no_llm_cache:
        changes["llm_cache"] = False
    if args.no_draft:
        changes["use_draft"] = False
    if args.no_register_check:
        changes["check_register"] = False
    if args.number_check:
        changes["check_numbers"] = True
    return dataclasses.replace(Config(), **changes)


def single_item(args: argparse.Namespace) -> LineItem:
    """명령줄 문장 하나를 LineItem 으로(입력 파일과 같은 검사를 거친다)."""
    row = {
        "source_en": args.text, "register": args.register or "", "prev_en": args.prev or "",
        "relationship": args.relationship or "", "meaning_notes": args.meaning_notes or "",
        "target_syllables": "" if args.target_syllables is None else str(args.target_syllables),
        "seconds": "" if args.seconds is None else str(args.seconds),
        "lip": "off" if args.no_lip else "on",
    }
    return item_from_row(row, "명령줄", "L0001")


def _reconfigure_output() -> None:
    """한글·IPA 가 cp949 콘솔에서 깨지지 않게 표준 출력을 utf-8 로."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _reconfigure_output()
    ap = build_parser()
    args = ap.parse_args(argv)
    if (args.text is None) == (args.input is None):
        ap.error("영어 대사 하나나 --input 파일 가운데 하나만 주세요.")
    if args.input is not None:
        used = [f"--{name.replace('_', '-')}" for name in LINE_OPTIONS if getattr(args, name) not in (None, False)]
        if used:
            ap.error(f"{', '.join(used)} 는 문장 하나 모드에서만 씁니다. 파일에서는 그 칸에 적으세요.")
    cfg = config_from_args(args)
    out_dir = Path(cfg.out_dir)
    try:
        if args.input is not None:
            items, columns = read_lines(args.input)
            source = str(args.input)
        else:
            items, columns = [single_item(args)], list(SINGLE_COLUMNS)
            source = "명령줄 문장 1개"
        results, info = run_lines(items, cfg)
        paths = write_outputs(out_dir, results, info, cfg, columns, source)
    except (ValueError, FileNotFoundError) as e:
        print(f"오류: {e}", file=sys.stderr)
        return 2
    except (RuntimeError, OSError) as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("멈춤(Ctrl+C). 이미 받은 모델 2 답은 cache\\llm 에 남아 있어 다시 돌리면 그대로 쓴다.", file=sys.stderr)
        return 130
    except Exception as e:      # 라이브러리 안의 예상 못 한 오류도 추적 대신 한국어 한 줄로 알린다
        where = traceback.extract_tb(e.__traceback__)[-1]
        print(f"오류: 예상하지 못한 문제로 멈췄습니다. {type(e).__name__}: {e} "
              f"({Path(where.filename).name} {where.lineno}행)", file=sys.stderr)
        return 1

    if args.input is None:
        print_choice(results[0])
    else:
        for r in results:
            goal = f"목표 {r.target.n}±{r.target.tol}"
            if r.selected is None:
                print(f"{r.item.id} [{STATUS_KO.get(r.status, r.status)}] (없음) ({goal})")
            else:
                print(f"{r.item.id} [{STATUS_KO.get(r.status, r.status)}] {r.selected.text} "
                      f"({r.selected.syllables}음절, {goal})")
    counts = info.status_counts
    total = info.seconds.get("total", 0.0)
    print(f"고름 {counts.get('ok', 0)}줄, 대안 {counts.get('fallback', 0)}줄, 실패 {counts.get('error', 0)}줄. "
          f"걸린 시간 {total:.1f}초(줄당 {total / len(results):.1f}초). "
          f"모델 2 호출 {info.llm_calls}번, 캐시 적중 {info.llm_cache_hits}번")
    print(f"저장: {out_dir.resolve()} ({', '.join(p.name for p in paths.values())})")
    if info.llm_stopped:
        print(f"주의: 모델 2 를 쓸 수 없어 후보 모으기를 멈췄습니다. {info.llm_stopped}", file=sys.stderr)
        return 1
    return 1 if counts.get("error") else 0
