"""논문 그대로 판 명령줄(명세 3.11).

  run_paper.cmd "We have to leave before the storm hits." [--seconds 2.1]
  run_paper.cmd --input examples\\lines.csv --out out\\paper_demo

문장 하나 모드는 ISO, PS, PS-Comet 이 고른 대사와 PS-Comet 상위 3개를 화면에 찍고 out 폴더에도 저장한다.
끝 코드는 기존 판(run.cmd)과 같다: 0 모두 고름, 1 실패한 줄이 있거나 모델 2 를 쓸 수 없었음(또는 실행 중 오류),
2 입력이나 설정 오류, 130 Ctrl+C.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import traceback
from pathlib import Path
from typing import Any

from .config import PaperConfig
from .io import (SINGLE_COLUMNS, STATUS_KO, ISO_STATUS_KO, item_from_row, print_choice, read_lines, summary_line,
                 write_outputs)
from .pipeline import LLM_BACKENDS, STATUS_ERROR, TTS_KINDS, run_paper

# 명령줄 옵션 이름 -> PaperConfig 칸 이름(같은 이름은 적지 않음)
CONFIG_NAMES = {"beam": "nmt_beam", "out": "out_dir"}
CONFIG_OPTIONS = ("device", "cpu_threads", "llm", "llm_model", "llm_effort", "llm_file", "llm_base_url",
                  "llm_api_key_env", "llm_workers", "llm_chunk", "iso_batch", "iso_max", "ps_n", "ps_chunk",
                  "ps_extra_rounds", "alpha", "beta", "sakoe_chiba_radius", "labse_min", "iso_tol_frames", "tts",
                  "tts_dir", "tts_checkpoint", "tts_speaker", "tts_lang_en", "tts_lang_ko", "vowel_table",
                  "g2p_en_voice", "g2p_en_stress", "comet_model", "labse_model", "nmt_dir", "nmt_ckpt", "beam", "out")


def build_parser() -> argparse.ArgumentParser:
    cfg = PaperConfig()
    ap = argparse.ArgumentParser(
        prog="pstts.paper",
        description="PS-TTS 논문(arXiv 2604.09111) 3장 절차 그대로: ISO(길이와 뜻, 최대 60번 바꿔 쓰기), "
                    "PS(후보 60개, 모음 DTW), PS-Comet 으로 영어 대사의 한국어 더빙 대사를 고른다.",
        epilog='예: run_paper.cmd "We have to leave before the storm hits." --seconds 2.1 / '
               "run_paper.cmd --input examples\\lines.csv --out out\\paper_demo")
    ap.add_argument("text", nargs="?", help="영어 대사 하나(문장 하나 모드)")
    ap.add_argument("--input", type=Path,
                    help="대사 파일(.csv, .tsv, .txt). 쓰는 칸은 id, source_en, seconds, draft_ko")
    ap.add_argument("--out", help=f"결과 폴더(기본 {cfg.out_dir}). 같은 이름 파일은 덮어쓴다")
    ap.add_argument("--seconds", type=float,
                    help="문장 하나 모드: 쉼을 뺀 원문 말 길이(초). K' = seconds / SA 프레임(식 2). "
                         "없으면 원문 TTS 길이로 둔다(R = 1)")

    llm = ap.add_argument_group("모델 2(논문의 GPT-4o 자리)")
    llm.add_argument("--llm", choices=LLM_BACKENDS, help=f"후보 생성 백엔드(기본 {cfg.llm})")
    llm.add_argument("--llm-model", help=f"모델 ID(기본 {cfg.llm_model}). GPT-4o 는 --llm openai --llm-model gpt-4o")
    llm.add_argument("--llm-effort", help=f"claude --effort(기본 {cfg.llm_effort})")
    llm.add_argument("--llm-base-url", help=f"--llm openai 의 서버 주소(기본 {cfg.llm_base_url}). "
                                            "OpenAI 는 https://api.openai.com")
    llm.add_argument("--llm-api-key-env", help="--llm openai 에서 API 키가 든 환경 변수 이름(예: OPENAI_API_KEY)")
    llm.add_argument("--llm-file", help="--llm file 의 후보 JSONL(논문 판 결과 폴더의 candidates.jsonl). "
                                        "--llm 을 주지 않으면 --llm file 로 돈다")
    llm.add_argument("--llm-workers", type=int, help=f"동시 호출 수(기본 {cfg.llm_workers})")
    llm.add_argument("--llm-chunk", type=int,
                     help=f"ISO 바꿔 쓰기에서 한 번 부를 때 넣는 대사 수(기본 {cfg.llm_chunk})")
    llm.add_argument("--no-llm-cache", action="store_true", help="LLM 응답 캐시(cache/llm)를 읽지도 쓰지도 않음")

    sel = ap.add_argument_group("ISO, PS, PS-Comet")
    sel.add_argument("--iso-batch", type=int,
                     help=f"ISO 한 번 부를 때 줄마다 받는 바꿔 쓰기 후보 수(기본 {cfg.iso_batch})")
    sel.add_argument("--iso-max", type=int, help=f"ISO 바꿔 쓴 후보 검사 최대 수(기본 {cfg.iso_max})")
    sel.add_argument("--iso-tol-frames", type=int, help=f"식 6 허용(SA 프레임, 기본 {cfg.iso_tol_frames})")
    sel.add_argument("--labse-min", type=float, help=f"식 7 LaBSE 최솟값(기본 {cfg.labse_min})")
    sel.add_argument("--ps-n", type=int, help=f"PS 후보 수(기본 {cfg.ps_n})")
    sel.add_argument("--ps-chunk", type=int, help=f"PS 후보 생성에서 한 번 부를 때 넣는 대사 수(기본 {cfg.ps_chunk})")
    sel.add_argument("--ps-extra-rounds", type=int,
                     help=f"PS 후보가 모자랄 때 더 묻는 횟수(기본 {cfg.ps_extra_rounds})")
    sel.add_argument("--sakoe-chiba-radius", type=int,
                     help="DTW Sakoe-Chiba 반경(프레임). 주지 않으면 tslearn 기본(반경 1)")
    sel.add_argument("--alpha", type=float, help=f"식 20 DTW 무게(기본 {cfg.alpha})")
    sel.add_argument("--beta", type=float, help=f"식 20 COMET 무게(기본 {cfg.beta})")

    tts = ap.add_argument_group("TTS 프런트엔드(음소 길이, 모음표)")
    tts.add_argument("--tts", choices=TTS_KINDS, help=f"approx(근사) 또는 yourtts(학습한 체크포인트). 기본 {cfg.tts}")
    tts.add_argument("--tts-dir",
                     help="--tts yourtts 의 체크포인트 폴더(config.json, .pth, d-vector, vowel_table.json)")
    tts.add_argument("--tts-checkpoint", help="체크포인트 파일 이름(없으면 best_model*.pth 나 model_file.pth)")
    tts.add_argument("--tts-speaker", help="d-vector 키나 화자 이름(없으면 이름순 첫 화자의 평균)")
    tts.add_argument("--tts-lang-en", help=f"모델의 영어 언어 이름(기본 {cfg.tts_lang_en})")
    tts.add_argument("--tts-lang-ko", help=f"모델의 한국어 언어 이름(기본 {cfg.tts_lang_ko})")
    tts.add_argument("--vowel-table", help="식 8, 9 모음표 JSON(없으면 --tts-dir 의 vowel_table.json)")
    tts.add_argument("--g2p-en-voice",
                     help=f"영어 G2P espeak 목소리(기본 {cfg.g2p_en_voice}: 논문 Fig. 5c 의 영어 모음과 같음. "
                          "Fig. 4 꼴은 en-us)")
    tts.add_argument("--g2p-en-stress", action="store_true", default=None,
                     help="영어 강세표를 토큰에 남긴다(기본 끔: 논문 Fig. 4 에 강세표 없음)")

    mod = ap.add_argument_group("모델")
    mod.add_argument("--comet-model", help=f"COMET 모델(기본 {cfg.comet_model})")
    mod.add_argument("--labse-model", help=f"LaBSE 모델(기본 {cfg.labse_model})")
    mod.add_argument("--nmt-dir", help=f"모델 1 폴더(기본 {cfg.nmt_dir})")
    mod.add_argument("--nmt-ckpt", help=f"모델 1 체크포인트(기본 {cfg.nmt_ckpt}, 폴더 기준 상대 경로 가능)")
    mod.add_argument("--beam", type=int, help=f"모델 1 beam 크기(기본 {cfg.nmt_beam})")
    mod.add_argument("--device", help="cuda 또는 cpu(기본 cuda). cuda 를 못 쓰면 멈추니 CPU 는 직접 적는다")
    mod.add_argument("--cpu-threads", type=int, help=f"CPU 로 돌릴 때 torch 스레드 수(기본 {cfg.cpu_threads})")
    return ap


def config_from_args(args: argparse.Namespace) -> PaperConfig:
    """명령줄 옵션을 PaperConfig 에 덮어쓴다. 주지 않은 옵션은 기본값 그대로."""
    changes: dict[str, Any] = {}
    for name in CONFIG_OPTIONS:
        value = getattr(args, name)
        if value is not None:
            changes[CONFIG_NAMES.get(name, name)] = value
    if args.no_llm_cache:
        changes["llm_cache"] = False
    return dataclasses.replace(PaperConfig(), **changes)


def _reconfigure_output() -> None:
    """한글, IPA 가 cp949 콘솔에서 깨지지 않게 표준 출력을 utf-8 로."""
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
    if args.input is not None and args.seconds is not None:
        ap.error("--seconds 는 문장 하나 모드에서만 씁니다. 파일에서는 seconds 칸에 적으세요.")
    if args.llm_file is not None:
        # 재생 명령에서 --llm file 을 빠뜨려 구독 claude 를 부르는 일이 없게 한다
        if args.llm is None:
            args.llm = "file"
            print("알림: --llm-file 을 주어 --llm file 로 돈다(모델 2 를 부르지 않음).", file=sys.stderr)
        elif args.llm != "file":
            ap.error(f"--llm-file 은 --llm file 과 함께 씁니다(지금 --llm {args.llm}).")
    cfg = config_from_args(args)
    out_dir = Path(cfg.out_dir)
    try:
        if args.input is not None:
            items, columns = read_lines(args.input)
            source = str(args.input)
        else:
            row = {"source_en": args.text, "seconds": "" if args.seconds is None else str(args.seconds)}
            items, columns = [item_from_row(row, "명령줄", "L0001")], list(SINGLE_COLUMNS)
            source = "명령줄 문장 1개"
        lines, info = run_paper(items, cfg)
        paths = write_outputs(out_dir, lines, info, cfg, columns, source)
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
        print_choice(lines[0], cfg)
    else:
        for p in lines:
            iso = ISO_STATUS_KO.get(p.iso.status, p.iso.status)
            if p.final is None:
                print(f"{p.item.id} [{STATUS_KO.get(p.status, p.status)}] (없음) (ISO {iso}) {p.message}")
            else:
                print(f"{p.item.id} [{STATUS_KO.get(p.status, p.status)}] {p.final.text} "
                      f"(ISO {iso}, PS-Comet 점수 {p.final.score:.3f}, DTW {p.final.dtw:.1f})")
    print(summary_line(lines, info))
    print(f"저장: {out_dir.resolve()} ({', '.join(p.name for p in paths.values())})")
    if info.llm_stopped:
        print(f"주의: 모델 2 를 쓸 수 없어 후보 모으기를 멈췄습니다. {info.llm_stopped}", file=sys.stderr)
        return 1
    return 1 if any(p.status == STATUS_ERROR for p in lines) else 0
