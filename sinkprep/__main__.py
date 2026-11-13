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
    "review": ("sinkprep.review", "확인 필요 목록 -> qa/needs_review.csv"),
    "validate": ("sinkprep.validate", "스키마·의미 규칙 검사"),
    "all": ("sinkprep.pipeline", "catalog→convert→ocr→lines→parse→review→validate"),
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
        p.add_argument("paths", nargs="*", help="(validate) 검사할 파일·폴더")
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
