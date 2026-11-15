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
    "video": ("sinkprep.video", "영상 등록: video <영상 경로> --docs <doc_id> -> videos/{film_id}.*, catalog 영상 칸 "
                                "(등록 해제: video --unregister --docs <doc_id>)"),
    "asr": ("sinkprep.asr", "받아쓰기 -> films/{film_id}/work/asr.json"),
    # 맞추기·최종본(all 에 든다. work/asr.json 이 있는 영화만 다루고, all 은 받아쓰기 모델을 부르지 않는다: pipeline.py)
    "align": ("sinkprep.align", "대본과 받아쓰기 맞추기 -> films/{film_id}/work/align.json, catalog match"),
    "final": ("sinkprep.final", "최종 영상 기준 대본 -> films/{film_id}/final.json (검사 통과 뒤 catalog status=done)"),
    # 영화 한 편을 한 번에(all 에는 들지 않는다): video → asr → align → final → 검사 → 미리보기 자막
    "film": ("sinkprep.film", "대본과 영상 한 쌍을 한 번에: film <대본 파일|doc_id|제목> <영상 파일> -> 등록·받아쓰기·"
                              "맞추기·최종본·검사·films/{film_id}/preview.srt (sink-film.cmd 가 이 명령을 부른다)"),
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sinkprep", description="영상 기준 대본(sink) 형식 통일 전처리")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, (_, desc) in COMMANDS.items():
        p = sub.add_parser(name, help=desc, description=desc)
        p.add_argument("--docs", default=None, help="쉼표로 나눈 doc_id 목록(없으면 전체)")
        p.add_argument("--limit", type=int, default=None, help="앞에서부터 N개 문서만")
        p.add_argument("--force", action="store_true",
                       help="(film) 그 영화에 등록된 다른 영상을 이 영상으로 바꾼다" if name == "film"
                       else "지문이 같아도 다시 만들기")
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
                           help="영상 말소리 언어(ko, en, mixed ...). 없으면 한국영화 게시판 문서는 ko, 아니면 unknown. "
                                "지역 꼬리(ko-KR)는 기본 언어(ko)로 줄여 적는다")
            p.add_argument("--unregister", action="store_true",
                           help="영상 경로 없이 --docs 와 함께: 그 영화의 영상 등록을 지운다(catalog 영상 칸·match, "
                                "status aligned·done -> parsed, asr·align·final 문제 기록, 다른 하드링크가 있는 "
                                "videos/{film_id}.*; --force 면 링크가 하나뿐인 사본도). films/ 결과와 캐시는 남긴다")
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
        if name == "film":
            # 기본값은 sinkprep.film 한 곳에 둔다(passes·lang). 주지 않은 값은 None 으로 넘긴다.
            p.add_argument("--lang", default=None, help="영상 말소리 언어(기본 ko)")
            p.add_argument("--passes", default=None,
                           help="받아쓰기 판 목록: id=모델[,id=모델] (기본 plain=large-v3-turbo,lv3=large-v3)")
            p.add_argument("--threads", type=int, default=None, help="CPU 스레드 수(기본 6)")
            p.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None,
                           help="처음 받아쓰는 영화는 주지 않으면 GPU(cuda)로만 시작한다: GPU 를 못 쓰면 받아쓰지 않고 "
                                "멈춘다. 몇 시간 걸려도 CPU 로 돌리려면 --device cpu")
            p.add_argument("--rescue", action=argparse.BooleanOptionalAction, default=None,
                           help="말 구간 찾기가 놓친 대사 되찾기(기본 켬, --no-rescue 는 뺀다)")
            p.add_argument("--llm", action=argparse.BooleanOptionalAction, default=None,
                           help="맞추기의 LLM 짝 판정에서 claude 를 부른다(기본 켬, --no-llm 은 캐시에 있는 판정만)")
            p.add_argument("--open", action="store_true", help="끝나면 결과 폴더를 탐색기로 연다")
        p.add_argument("paths", nargs="*", help="(validate) 검사할 파일·폴더, (video) 영상 파일 하나, "
                                                "(film) 대본과 영상(순서 무관)")
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
