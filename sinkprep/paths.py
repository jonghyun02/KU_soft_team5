"""경로 상수. 모든 모듈은 여기 경로만 쓴다.

RAW(원본)는 읽기 전용이다. 이 패키지는 RAW 아래에 아무것도 쓰지 않는다.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "filmmakers_scripts"          # 명세의 raw/
EXTRACTED = ROOT / "extracted"             # 압축 해제 결과: extracted/{post_id}/
CACHE = ROOT / "cache"                     # 중간 산출물
PDF_CACHE = CACHE / "pdf"                  # 한컴 변환 PDF: cache/pdf/{doc_id}/{k}.pdf
OCR_CACHE = CACHE / "ocr"                  # OCR 결과: cache/ocr/{doc_id}/{k}.json
OCR_SUPERSEDED = CACHE / "ocr_superseded"  # 글자층 복원으로 충분해져 쓰지 않는 예전 OCR 결과(지우지 않고 보관)
PROBLEMS = CACHE / "problems"              # 단계별 문제: cache/problems/{stage}/{doc_id}.json
STATE = CACHE / "state"                    # 단계별 입력 지문: cache/state/{stage}/{doc_id}.json
AUDIO = CACHE / "audio"                    # 영상에서 꺼낸 소리와 시각 대응표: cache/audio/{video_sha1}.wav, .timemap.json
ASR_CACHE = CACHE / "asr"                  # 받아쓰기 묶음 캐시: cache/asr/{video_sha1}/vad.json, {모델 꼬리표}/{a}-{b}.json
VIDEOS = ROOT / "videos"                   # 등록한 영상(원본의 하드링크): videos/{film_id}.mp4
DOCS = ROOT / "docs"                       # docs/{doc_id}/lines.jsonl, script.json
FILMS = ROOT / "films"                     # films/{film_id}/final.json, work/
QA = ROOT / "qa"
CATALOG = ROOT / "catalog.csv"
NEEDS_REVIEW = QA / "needs_review.csv"
SCHEMA = ROOT / "schema"
EXAMPLES = ROOT / "examples"

BANDIZIP = Path(r"C:\Program Files\Bandizip\bz.exe")
HWP_EXE = Path(r"C:\Program Files (x86)\Hnc\Office 2024\HOffice130\Bin\Hwp.exe")
HWP_SECURITY_DLL = Path(r"C:\Program Files (x86)\Hnc\securitymoduel\FilePathCheckerModuleExample.dll")


def rel(p: Path) -> str:
    """ROOT 기준 상대 경로(슬래시)."""
    return Path(p).resolve().relative_to(ROOT).as_posix()


def absp(relpath: str) -> Path:
    return ROOT / relpath


def assert_not_raw(p: Path) -> None:
    """쓰기 전에 원본 폴더가 아닌지 확인."""
    rp = Path(p).resolve()
    if rp == RAW or RAW in rp.parents:
        raise PermissionError(f"원본 폴더에는 쓰지 않습니다: {rp}")
