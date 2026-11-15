"""CLI: python -I -m sinkprep <명령> [--docs id,id] [--limit N] [--force]"""
from __future__ import annotations

import argparse
import importlib
import sys

# 명령 -> (모듈, 설명). 각 모듈은 run(args) 를 가진다.
COMMANDS = {
    "catalog": ("sinkprep.catalog", "원본 목록·압축 해제·형식 판별 -> catalog.csv"),
    "convert": ("sinkprep.convert", "한컴 자동화로 HWP/doc/docx/rtf -> cache/pdf"),
    "ocr": ("sinkprep.ocr", "스캔·깨진 글자층 PDF OCR -> cache/ocr"),
    "lines": ("sinkprep.lines", "형식 통일 줄 -> docs/{doc_id}/lines.jsonl"),
    "parse": ("sinkprep.parse", "구조 파싱 -> docs/{doc_id}/script.json"),
    "llm": ("sinkprep.llm_parse", "규칙 파서가 무너진 문서를 claude -p 로 다시 파싱(나아질 때만 script.json 교체)"),
    "review": ("sinkprep.review", "확인 필요 목록 -> qa/needs_review.csv"),
    "validate": ("sinkprep.validate", "스키마·의미 규칙 검사"),
    "all": ("sinkprep.pipeline", "catalog→convert→ocr→lines→parse→llm→asr→align→final→review→validate "
                                 "(llm·asr 는 캐시만: 받아쓰기는 asr 명령으로 따로)"),
    # 영상 쪽 단계(all 에는 들지 않는다: 영상이 있는 영화만 따로 돌린다)
    "video": ("sinkprep.video", "영상 등록: video <영상 경로> --docs <doc_id> -> videos/{film_id}.*, catalog 영상 칸"),
    "asr": ("sinkprep.asr", "받아쓰기 -> films/{film_id}/work/asr.json"),
    # 맞추기·최종본(all 에 든다. work/asr.json 이 있는 영화만 다루고, all 은 받아쓰기 모델을 부르지 않는다: pipeline.py)
    "align": ("sinkprep.align", "대본과 받아쓰기 맞추기 -> films/{film_id}/work/align.json, catalog match"),
    "final": ("sinkprep.final", "최종 영상 기준 대본 -> films/{film_id}/final.json (검사 통과 뒤 catalog status=done)"),
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sinkprep", description="영상 기준 대본(sink) 형식 통일 전처리")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, (_, desc) in COMMANDS.items():
        p = sub.add_parser(name, help=desc, description=desc)
        p.add_argument("--docs", default=None, help="쉼표로 나눈 doc_id 목록(없으면 전체)")
        p.add_argument("--limit", type=int, default=None, help="앞에서부터 N개 문서만")
        p.add_argument("--force", action="store_true", help="지문이 같아도 다시 만들기")
        p.add_argument("--workers", type=int, default=None, help="병렬 작업 수")
        if name in ("llm", "all"):
            p.add_argument("--images", action="store_true",
                           help="(llm) OCR 줄이 있는 조각에 원본 쪽 그림도 함께 준다")
        if name == "all":
            p.add_argument("--llm", action="store_true",
                           help="llm 단계에서 캐시에 없는 claude 호출도 한다(기본은 캐시만 써서 구독을 쓰지 않음). "
                                "align 의 LLM 짝 판정은 all 에서 늘 캐시만 쓴다")
        if name == "align":
            p.add_argument("--llm", action="store_true",
                           help="LLM 짝 판정에서 캐시에 없는 창을 claude 로 묻는다(기본은 캐시에 있는 판정만 써서 구독을 쓰지 않음)")
        if name == "video":
            p.add_argument("--lang", default=None,
                           help="영상 말소리 언어(ko, en, mixed ...). 없으면 한국영화 게시판 문서는 ko, 아니면 unknown")
        if name == "asr":
            # 기본값은 여기 두지 않는다(sinkprep.asr.DEFAULT_* 한 곳). 주지 않으면 그 영화를 앞서 받아쓴 값을 이어 쓴다.
            p.add_argument("--passes", default=None,
                           help="받아쓰기 판 목록: id=모델[,id=모델] (모델: large-v3-turbo(=turbo), large-v3). "
                                "없으면 앞서 받아쓴 값, 처음이면 plain=large-v3-turbo")
            p.add_argument("--threads", type=int, default=None, help="CPU 스레드 수(기본 6)")
            p.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None,
                           help="auto 는 여유 VRAM 이 5000MiB 이상일 때만 cuda, 아니면 cpu(int8). "
                                "없으면 앞서 받아쓴 값, 처음이면 auto")
            p.add_argument("--rescue", action=argparse.BooleanOptionalAction, default=None,
                           help="말 구간 찾기가 놓친 대사 되찾기: 소리 전체를 덮는 창 두 벌을 plain 의 모델로 더 받아써 "
                                "pass rescue 를 더한다(--no-rescue 는 뺀다). 없으면 앞서 받아쓴 값, 처음이면 하지 않음")
        p.add_argument("paths", nargs="*", help="(validate) 검사할 파일·폴더, (video) 영상 파일 하나")
    return ap


def main(argv: list[str] | None = None) -> int:
    # -I 는 PYTHONIOENCODING 을 무시하므로 직접 UTF-8 로 맞춘다(파이프·파일로 보낼 때 한글 깨짐 방지).
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    mod = importlib.import_module(COMMANDS[args.cmd][0])
    rc = mod.run(args)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
