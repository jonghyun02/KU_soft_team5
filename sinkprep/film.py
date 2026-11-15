"""영화 한 편을 한 번에(film): 대본과 영상 두 파일로 영상 등록 → 받아쓰기 → 맞추기 → 최종본 → 검사 → 미리보기 자막.

python -I -m sinkprep film <대본> <영상> [--lang ko] [--passes plain=large-v3-turbo,lv3=large-v3] [--no-rescue]
                           [--threads 6] [--device auto|cpu|cuda] [--no-llm] [--workers 3] [--force] [--open]
sink-film.cmd <대본> <영상> [옵션]          (저장소 맨 위에 있는 감싼 명령. 인자는 같다)

인자 둘은 순서가 바뀌어도 된다. ffprobe(media.probe)로 영상 줄이 있는 파일이 영상이다. 확장자는 보조로만 쓴다: 대본
확장자(.hwp, .pdf, .txt …)는 영상으로 보지 않고(ffprobe 는 평문을 글자 그림 영상으로 읽는다), ffprobe 가 읽지 못한 파일은
영상 확장자(.mp4, .mkv …)일 때만 영상으로 본다. 대본 쪽 인자는
 - 파일: sha1 이 catalog 의 파일 sha1 칸(문서마다 '|' 로 이은 목록)에 있는 문서. 제외된 중복 문서(duplicate_file_of)면 가리키는
   원래 문서. catalog 에 없는 파일은 멈춘다: 이름(확장자 뺀 것)과 제목(title_norm)이 같은 대본이 있으면 그 doc_id 를 주라고
   알리고(내용이 다른 대본으로 조용히 바꿔 돌리지 않는다), 없으면 통일해 둔 문서에 없는 대본은 아직 넣을 수 없다고 알린다.
 - fm 숫자 꼴이면 doc_id, 그 밖은 제목(제외 문서 빼고 title_norm 이 같은 것). 여럿이면 목록을 찍고 멈춘다.
찾은 문서가 그 영화의 대표 대본(is_primary)이 아니면 대표 대본으로 돈다(asr·align·final 은 영화마다 대표 대본만 맞춘다).
대표 대본이 한국어(catalog script_lang ko)가 아니거나 번역 대본이면 미리 멈춘다(final 이 한국어 영화만 끝까지 만든다).

단계마다 그 모듈의 run(args) 를 명령을 친 것과 같은 Namespace 로 부른다. 끝난 단계는 각 단계의 지문으로 건너뛰므로,
멈춘 뒤 같은 명령을 다시 치면 이어서 한다.
 - video: --lang(기본 ko: 대본이 한국어인 영화만 받으므로). 같은 영상이 이미 등록돼 있으면 video 가 skip 한다.
   그 영화에 다른 영상이 이미 등록돼 있으면 시작하지 않는다(--force 면 이 영상으로 바꾼다. --force 는 video 에만 넘긴다).
 - asr: 기본 passes plain=large-v3-turbo,lv3=large-v3 와 rescue. 장치: 이 영상을 처음 받아쓰고(asr 지문이 없거나 다른 영상의
   것) --device 를 주지 않았으면 GPU 를 먼저 본다. cuda 를 쓸 수 없으면(여유 VRAM 모자람, CUDA 장치·라이브러리 없음)
   받아쓰기를 시작하지 않고 까닭과 방법을 알린다(CPU 로 몇 시간 조용히 돌지 않게). 쓸 수 있으면 --device cuda 로 넘긴다
   (auto 는 cuda 를 열다 실패하면 말없이 cpu 로 가고, 지문에 auto 가 남으면 뒤 실행이 cpu 판으로 갈아 탈 수 있다).
   이미 받아쓴 영화는 --device 를 넘기지 않아 앞선 값을 이어 쓴다.
 - align: LLM 짝 판정에서 캐시에 없는 창도 claude 로 묻는다(align --llm. --no-llm 이면 캐시만). claude 를 쓸 수 없으면
   llm.call 이 LLMUnavailable 을 내고 align_llm.ask 가 그 창을 규칙 결과로 둔 채 끝까지 가므로, 그때 알림 한 줄을 찍는다.
 - final → 검사(films/{film_id} 의 final.json·work/align.json·work/asr.json 을 validate.check_file 로) → 미리보기 자막
   (tools/final_to_srt.py, absent 포함, films/{film_id}/preview.srt).
단계가 0 으로 끝나도 결과 파일이 이 영상·이 대본의 것인지 본다(대상이 없으면 단계는 0 으로 끝난다). 끝에 요약을 찍는다.
--open 이면 결과 폴더를 탐색기로 연다.
종료 코드: 0 성공, 1 단계 실패, 2 인자·대본 찾기·장치 문제로 시작하지 못함.
catalog·films 를 바꾸는 실행이라 sinkprep all 에는 들지 않는다.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from . import paths, util

STAGE = "film"
DEFAULT_PASSES = "plain=large-v3-turbo,lv3=large-v3"
DEFAULT_LANG = "ko"
TOOLS = Path(__file__).resolve().parent.parent / "tools"     # 저장소의 tools 폴더(시험이 paths.ROOT 를 바꿔도 그대로)
VIDEO_EXTS = frozenset((".mp4", ".m4v", ".mkv", ".avi", ".mov", ".wmv", ".asf", ".ts", ".m2ts", ".mts", ".mpg",
                        ".mpeg", ".vob", ".webm", ".flv", ".3gp", ".ogv", ".divx", ".rm", ".rmvb"))
SCRIPT_EXTS = frozenset((".hwp", ".hwpx", ".doc", ".docx", ".rtf", ".pdf", ".txt", ".xls", ".xlsx", ".zip", ".alz",
                         ".egg", ".7z", ".rar"))
NOT_VIDEO_CODECS = frozenset(("ansi", "bintext", "xbin", "idf", "png", "bmp", "gif", "tiff", "webp", "jpegls"))
CUDA_DLLS = (("cuBLAS", "cublas64_*.dll"), ("cuDNN", "cudnn_ops64_*.dll"))   # ctranslate2 cuda 가 찾는 DLL(asr-gpu)
RE_DOC = re.compile(r"^fm[0-9]+(_[0-9]+)?$")
PROBLEM_STAGES = ("asr", "align", "final")
CHECKED = ("final.json", "work/align.json", "work/asr.json")
SHOW_ERRORS = 20
MODULES = {"video": "sinkprep.video", "asr": "sinkprep.asr", "align": "sinkprep.align", "final": "sinkprep.final"}
USAGE = ("쓰는 법: sink-film.cmd <대본 파일|doc_id|제목> <영상 파일> [--lang ko] [--passes id=모델,...] [--no-rescue] "
         "[--threads N] [--device auto|cpu|cuda] [--no-llm] [--workers N] [--force] [--open]\n"
         "        (python -I -m sinkprep film 도 같다. 인자 둘은 순서가 바뀌어도 된다)")


class Stop(Exception):
    """시작하지 못함(인자·대본 찾기·장치): 종료 코드 2."""


def _say(msg: str = "") -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ 인자 가리기
def is_video(path: Path) -> bool:
    """영상 파일인가: ffprobe 로 영상 줄이 있으면 영상. 확장자는 보조(맨 위 설명)."""
    from . import media

    ext = path.suffix.lower()
    try:
        info = media.probe(path)
    except media.MediaError:
        return ext in VIDEO_EXTS          # ffprobe 가 못 읽거나 없음: 영상 확장자면 영상으로 두고 video 단계가 까닭을 알린다
    v = info.get("video")
    if v is None or ext in SCRIPT_EXTS:
        return False
    return (v.get("codec") or "") not in NOT_VIDEO_CODECS


def _looks_like_path(s: str) -> bool:
    return "/" in s or "\\" in s or Path(s).suffix.lower() in (VIDEO_EXTS | SCRIPT_EXTS)


def split_inputs(items: list[str], docs: str | None = None) -> tuple[str, Path]:
    """(대본 쪽 인자, 영상 경로). 순서는 상관없다. --docs 를 주면 그것이 대본 쪽 인자다. 가릴 수 없으면 Stop."""
    items = [str(x) for x in items if str(x).strip()]
    if docs:
        if len(items) != 1 or "," in docs:
            raise Stop("--docs 를 주면 doc_id 하나와 영상 파일 하나만 준다\n" + USAGE)
        items = [docs, items[0]]
    if len(items) != 2:
        raise Stop(f"인자가 {len(items)}개다. 대본과 영상 두 개를 준다\n" + USAGE)
    kinds = []
    for x in items:
        p = Path(x)
        if p.is_file():
            kinds.append("video" if is_video(p) else "script")
        elif p.exists():
            raise Stop(f"파일이 아니라 폴더다: {x}")
        elif _looks_like_path(x):
            raise Stop(f"파일을 찾을 수 없다: {x}")
        else:
            kinds.append("script")                     # doc_id 나 제목
    if kinds.count("video") != 1:
        what = "둘 다 영상이다" if kinds.count("video") == 2 else "영상 줄이 있는 파일이 없다(ffprobe)"
        raise Stop(f"대본 하나와 영상 하나를 가릴 수 없다: {what}: "
                   + ", ".join(f"{x} ({'영상' if k == 'video' else '영상 아님'})" for x, k in zip(items, kinds)))
    vi = kinds.index("video")
    return items[1 - vi], Path(os.path.abspath(items[vi]))


# ------------------------------------------------------------------ 대본 찾기
def _title_keys(text: str) -> set[str]:
    """제목 견주기 열쇠: catalog 가 같은 영화를 묶는 norm_key. '써니 시나리오' 처럼 군더더기가 붙은 이름도 다듬어 본다."""
    from . import catalog

    keys = {catalog.norm_key(text)}
    try:
        keys.add(catalog.norm_key(catalog.parse_title(text, "ko_film")["title_norm"]))
    except Exception:  # noqa: BLE001 - 제목 다듬기는 보조
        pass
    keys.discard("")
    return keys


def _same_title(rows: list[dict], keys: set[str]) -> list[dict]:
    from . import catalog

    return [r for r in rows if r.get("status") != "excluded" and catalog.norm_key(r.get("title_norm") or "") in keys]


def _original(row: dict, by_id: dict[str, dict]) -> dict | None:
    """제외된 중복 문서(note duplicate_file_of:X)가 가리키는 제외되지 않은 원래 문서. 없으면 None."""
    seen = set()
    r = row
    while r is not None and r.get("status") == "excluded" and r["doc_id"] not in seen:
        seen.add(r["doc_id"])
        note = r.get("note") or ""
        if not note.startswith("duplicate_file_of:"):
            return None
        r = by_id.get(note.split(":", 1)[1].strip())
    return r if r is not None and r.get("status") != "excluded" else None


def _doc_list(rows: list[dict]) -> str:
    return "\n".join(f"    {r['doc_id']}  {r.get('film_id') or '-'}  {r.get('title_norm') or ''}"
                     f"  판본 {r.get('version_label') or '?'}  언어 {r.get('script_lang') or '?'}"
                     + ("  (대표)" if r.get("is_primary") == "true" else "") for r in rows)


def find_doc(spec: str, rows: list[dict]) -> tuple[dict, list[str]]:
    """대본 쪽 인자 -> (catalog 행, 알릴 말). 찾지 못하면 Stop."""
    by_id = {r["doc_id"]: r for r in rows}
    notes: list[str] = []
    p = Path(spec)
    if p.is_file():
        sha = util.sha1_file(p)
        hits = [r for r in rows if sha in (r.get("sha1_list") or [x for x in (r.get("sha1") or "").split("|") if x])]
        live = [r for r in hits if r.get("status") != "excluded"]
        if hits and not live:
            for r in hits:
                o = _original(r, by_id)
                if o is not None and o not in live:
                    live.append(o)
                    notes.append(f"대본 파일 {p.name} 은 제외된 중복 문서 {r['doc_id']}({r.get('note')}): "
                                 f"원래 문서 {o['doc_id']} 로 돈다")
            if not live:
                raise Stop(f"대본 파일 {p.name} 은 catalog 에서 제외된 문서다: "
                           + ", ".join(f"{r['doc_id']}({r.get('note') or '까닭 없음'})" for r in hits))
        if live:
            if len({r.get("film_id") for r in live}) > 1:
                raise Stop(f"대본 파일 {p.name} 과 같은 파일이 여러 영화의 문서에 있다. doc_id 를 하나 골라 준다:\n"
                           + _doc_list(live))
            live.sort(key=lambda r: (r.get("is_primary") != "true", r["doc_id"]))
            notes.insert(0, f"대본 파일 {p.name}: sha1 {sha[:8]} = catalog {live[0]['doc_id']}")
            return live[0], notes
        same = _same_title(rows, _title_keys(p.stem))
        if same:
            raise Stop(f"대본 파일 {p.name}(sha1 {sha[:8]})은 통일해 둔 대본과 내용이 다르다(catalog 에 같은 파일이 없음). "
                       f"같은 제목의 대본:\n{_doc_list(same)}\n"
                       f"  그 대본으로 돌리려면 파일 대신 그 번호를 준다: sink-film.cmd {same[0]['doc_id']} <영상>")
        n = sum(1 for r in rows if r.get("status") != "excluded")
        raise Stop(f"대본 파일 {p.name}(sha1 {sha[:8]})은 catalog 에 없다. 통일해 둔 {n}문서에 없는 대본은 아직 넣을 수 없다")
    if RE_DOC.match(spec):
        r = by_id.get(spec)
        if r is None:
            raise Stop(f"catalog 에 없는 doc_id: {spec}")
        if r.get("status") == "excluded":
            o = _original(r, by_id)
            if o is None:
                raise Stop(f"제외된 문서: {spec} ({r.get('note') or '까닭 없음'})")
            notes.append(f"{spec} 는 제외된 중복 문서({r.get('note')}): 원래 문서 {o['doc_id']} 로 돈다")
            return o, notes
        return r, notes
    same = _same_title(rows, _title_keys(spec))
    if not same:
        raise Stop(f"제목이 {spec!r} 인 대본이 catalog 에 없다(제외 문서 뺌). 대본 파일 경로나 doc_id(fm…)를 준다")
    if len(same) > 1:
        raise Stop(f"제목이 {spec!r} 인 대본이 {len(same)}개다. 하나를 doc_id 로 준다(같은 영화면 대표 대본으로 돈다):\n"
                   + _doc_list(same))
    return same[0], notes


def primary_of(row: dict, rows: list[dict]) -> tuple[dict, list[str]]:
    """그 문서의 영화의 대표 대본 행과 알릴 말. 대표를 하나로 정할 수 없으면 Stop."""
    film = row.get("film_id") or ""
    if not film:
        raise Stop(f"{row['doc_id']}: film_id 가 없는 문서")
    prim = [r for r in rows if r.get("film_id") == film and r.get("status") != "excluded" and r.get("is_primary") == "true"]
    if len(prim) != 1:
        raise Stop(f"{film} 의 대표 대본(is_primary)이 {len(prim)}개다: python -I -m sinkprep review 로 대표를 정한 뒤 다시")
    notes = []
    if prim[0]["doc_id"] != row["doc_id"]:
        notes.append(f"{row['doc_id']} 는 {film} 의 대표 대본이 아니다: 대표 대본 {prim[0]['doc_id']} 로 돈다"
                     "(asr·align·final 은 영화마다 대표 대본만 맞춘다)")
    return prim[0], notes


def check_script(prim: dict) -> None:
    """영상 쪽 단계가 끝까지 갈 수 있는 대본인가(한국어, 번역 아님, script.json 있음). 아니면 Stop."""
    doc = prim["doc_id"]
    lang = (prim.get("script_lang") or "").strip() or "빈칸"
    if str(prim.get("is_translation") or "").lower() == "true" or lang != "ko":
        what = "번역 대본" if str(prim.get("is_translation") or "").lower() == "true" else f"대본 언어 {lang}"
        raise Stop(f"{doc} 는 {what}이다(catalog script_lang={lang}, is_translation={prim.get('is_translation')}). "
                   "영상 쪽은 지금 한국어 대본과 한국어 영화만 최종본까지 간다(번역은 범위 밖)")
    if not (paths.DOCS / doc / "script.json").is_file():
        raise Stop(f"docs/{doc}/script.json 이 없다: 대본 쪽 처리(python -I -m sinkprep all)를 먼저 한다")


# ------------------------------------------------------------------ 장치
def _cuda_count() -> int:
    try:
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001
        return -1 if importlib.util.find_spec("ctranslate2") is None else 0


def _has_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


def cuda_libs_missing() -> list[str]:
    """cuda 받아쓰기에 필요한 DLL(cuBLAS·cuDNN) 가운데 찾지 못한 것. ctranslate2 폴더, site-packages/nvidia/*/bin
    (asr.py 가 DLL 찾는 길에 더하는 곳), PATH 를 본다. 윈도가 아니면 보지 않는다."""
    if os.name != "nt":
        return []
    dirs: list[Path] = []
    spec = importlib.util.find_spec("ctranslate2")
    if spec is not None and spec.origin:
        base = Path(spec.origin).resolve().parent
        dirs.append(base)
        dirs.extend(sorted((base.parent / "nvidia").glob("*/bin")))
    dirs.extend(Path(d) for d in os.environ.get("PATH", "").split(os.pathsep) if d.strip())

    def has(pat: str) -> bool:
        for d in dirs:
            try:
                if any(d.glob(pat)):
                    return True
            except (OSError, ValueError):
                continue
        return False
    return [name for name, pat in CUDA_DLLS if not has(pat)]


def gpu_problem() -> str | None:
    """받아쓰기를 cuda 로 할 수 없는 까닭. 할 수 있으면 None. asr.pick_device("auto") 가 고르는 장치를 따른다."""
    from . import asr

    try:
        dev, _ = asr.pick_device("auto")
    except Exception as e:  # noqa: BLE001
        return f"장치를 고를 수 없음({type(e).__name__}: {e})"
    if dev == "cuda":
        miss = cuda_libs_missing()
        return f"CUDA 라이브러리({', '.join(miss)})를 찾을 수 없다" if miss else None
    n = _cuda_count()
    if n < 0:
        return "받아쓰기 도구(faster-whisper, ctranslate2)가 설치되지 않았다"
    if n == 0:
        return "CUDA 장치를 찾지 못했다(그래픽 카드 드라이버 또는 CUDA 라이브러리 없음)"
    if not _has_nvidia_smi():
        return "nvidia-smi 가 없어 여유 VRAM 을 잴 수 없다"
    return (f"GPU 여유 VRAM 이 모자란다({asr.free_vram_mib()}MiB, {asr.CUDA_MIN_FREE_MIB}MiB 이상 필요: "
            "다른 프로그램이 GPU 를 쓰는 중)")


def choose_device(args, film_id: str, sha1: str) -> tuple[str | None, str]:
    """asr 에 넘길 --device 값(None 이면 넘기지 않음)과 알릴 말. 처음 받아쓰는데 cuda 를 쓸 수 없으면 Stop."""
    want = getattr(args, "device", None)
    if want:
        return want, f"받아쓰기 장치 {want}(--device)"
    fp = util.stage_fp("asr", film_id) or {}
    if fp and fp.get("video_sha1", sha1) == sha1:
        return None, f"받아쓰기 장치: 앞서 받아쓴 값({fp.get('device') or '?'})을 이어 씀"
    why = gpu_problem()
    if why:
        raise Stop(f"받아쓰기를 GPU 로 할 수 없어 시작하지 않는다: {why}.\n"
                   "  방법 1: GPU 를 쓰는 프로그램(게임, 영상 편집, 다른 받아쓰기 등)을 끄고 같은 명령을 다시 친다\n"
                   "  방법 2: CUDA 라이브러리를 깐다: .venv\\Scripts\\python -m pip install -e .[asr,asr-gpu]\n"
                   "  방법 3: 몇 시간 걸려도 좋으면 CPU 로: 같은 명령 끝에 --device cpu "
                   "(추격자 2시간 영화에 약 3시간. 그래도 cuda 로 해 보려면 --device cuda)")
    return "cuda", "받아쓰기 장치 cuda(처음 받아쓰는 영상, GPU 확인함)"


# ------------------------------------------------------------------ 단계
def _ns(cmd: str, **kw) -> argparse.Namespace:
    base = {"cmd": cmd, "docs": None, "limit": None, "force": False, "workers": None, "paths": []}
    base.update(kw)
    return argparse.Namespace(**base)


def _call(name: str, ns: argparse.Namespace) -> int:
    """그 단계 모듈의 run(ns). 밖으로 나온 예외도 그 단계의 실패(1)로."""
    mod = importlib.import_module(MODULES[name])
    try:
        rc = mod.run(ns)
    except Exception as e:  # noqa: BLE001
        print(f"  ! {name}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        rc = 1
    return int(rc or 0)


def _row(doc_id: str) -> dict | None:
    from . import catalog

    return next((r for r in catalog.load() if r["doc_id"] == doc_id), None)


def _after_video(doc: str, sha1: str) -> str | None:
    from . import media

    r = _row(doc)
    if r is None:
        return f"catalog 에서 {doc} 행이 사라졌다"
    if (r.get("video_sha1") or "") != sha1:
        return f"catalog 의 video_sha1({(r.get('video_sha1') or '빈칸')[:8]})이 이 영상({sha1[:8]})이 아니다"
    if not r.get("video_path") or not media.resolve_video(r["video_path"]).is_file():
        return f"등록한 영상 파일이 없다: {r.get('video_path') or '(video_path 빈칸)'}"
    return None


def _after_asr(film_id: str, sha1: str) -> str | None:
    p = paths.FILMS / film_id / "work" / "asr.json"
    if not p.is_file():
        return f"films/{film_id}/work/asr.json 이 없다"
    if util.read_json(p).get("video_sha1") != sha1:
        return f"films/{film_id}/work/asr.json 이 이 영상의 받아쓰기가 아니다"
    return None


def _after_align(film_id: str, doc: str) -> str | None:
    work = paths.FILMS / film_id / "work"
    p = work / "align.json"
    if not p.is_file():
        return f"films/{film_id}/work/align.json 이 없다"
    d = util.read_json(p)
    if d.get("script_doc") != doc:
        return f"align.json 의 script_doc {d.get('script_doc')!r} 가 대표 대본 {doc} 가 아니다"
    if d.get("asr_sha1") != util.sha1_file(work / "asr.json"):
        return "align.json 이 지금 asr.json 으로 맞춘 것이 아니다"
    return None


def _after_final(film_id: str, doc: str, sha1: str) -> str | None:
    p = paths.FILMS / film_id / "final.json"
    if not p.is_file():
        return f"films/{film_id}/final.json 이 없다"
    film = util.read_json(p).get("film") or {}
    if (film.get("script") or {}).get("doc_id") != doc or (film.get("video") or {}).get("sha1") != sha1:
        return "final.json 이 이 대본·이 영상의 것이 아니다"
    return None


def check_outputs(film_id: str) -> int:
    """final.json·align.json·asr.json 을 validate.check_file 로 검사해 찍는다. 실패가 있으면 1."""
    from . import validate

    bad = 0
    for rel in CHECKED:
        p = paths.FILMS / film_id / rel
        try:
            kind, errs = validate.check_file(p)
        except Exception as e:  # noqa: BLE001 - 검사기 자체 오류도 실패로
            kind, errs = "error", [f"[crash] 검사 중 예외: {type(e).__name__}: {e}"]
        if errs:
            bad += 1
            _say(f"FAIL films/{film_id}/{rel} ({kind}): 오류 {len(errs)}개")
            for e in errs[:SHOW_ERRORS]:
                _say(f"  {e}")
            if len(errs) > SHOW_ERRORS:
                _say(f"  … {len(errs) - SHOW_ERRORS}개 더")
        else:
            _say(f"ok   films/{film_id}/{rel} ({kind})")
    return 1 if bad else 0


def srt_tool():
    """tools/final_to_srt.py 모듈(tools 는 패키지가 아니라 파일 경로로 불러온다. 이미 불러온 것이 있으면 그것)."""
    mod = sys.modules.get("final_to_srt")
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location("final_to_srt", TOOLS / "final_to_srt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["final_to_srt"] = mod
    return mod


def write_preview(film_id: str) -> int:
    """films/{film_id}/preview.srt (absent 포함). final_to_srt 의 main 을 그 명령과 같은 인자로 부른다."""
    return int(srt_tool().main([str(paths.FILMS / film_id / "final.json"), "--absent"]) or 0)


# ------------------------------------------------------------------ 요약
def film_problems(film_id: str) -> Counter:
    """그 영화의 영상 쪽 문제 기록(asr·align·final) 수를 종류별로. review 처럼 같은 줄(같은 항목·종류·글)은 한 번만."""
    seen, out = set(), Counter()
    for stage in PROBLEM_STAGES:
        p = paths.PROBLEMS / stage / f"{film_id}.json"
        if not p.is_file():
            continue
        for x in util.read_json(p) or []:
            k = (x.get("level"), x.get("doc_id") or "", x.get("film_id") or "", x.get("item_id") or "",
                 x.get("problem"), (util.nfc(str(x.get("detail") or "")) or "").replace("\n", " "))
            if k not in seen:
                seen.add(k)
                out[x.get("problem") or "other"] += 1
    return out


def _hms(sec: float) -> str:
    s = int(round(sec))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def _took(sec: float) -> str:
    s = int(round(sec))
    return f"{s // 3600}시간 {s // 60 % 60}분 {s % 60}초" if s >= 3600 else f"{s // 60}분 {s % 60}초"


def summary(doc: str, film_id: str, seconds: float) -> list[str]:
    from . import media

    row = _row(doc) or {}
    fdir = paths.FILMS / film_id
    fin = util.read_json(fdir / "final.json")
    asr = util.read_json(fdir / "work" / "asr.json")
    q = fin.get("quality") or {}
    film = fin.get("film") or {}
    try:
        dur = float(row.get("video_duration") or (film.get("video") or {}).get("duration"))
    except (TypeError, ValueError):
        dur = None
    words = " · ".join(f"{p.get('id')} {len(p.get('words') or []):,}" for p in asr.get("passes") or [])
    match = (film.get("script") or {}).get("match", q.get("match"))
    rate = f", 일치율 {match:.3f}" if isinstance(match, (int, float)) else ""
    probs = film_problems(film_id)
    kinds = ", ".join(f"{k} {v}" for k, v in sorted(probs.items(), key=lambda kv: (-kv[1], kv[0])))
    video_copy = media.resolve_video(row["video_path"]) if row.get("video_path") else None
    out = [f"{film.get('title') or row.get('title_norm') or doc} ({doc}, {film_id})",
           f"  영상 {_hms(dur)} ({dur:.3f}초)" if dur is not None else "  영상 길이 모름",
           f"  받아쓰기 낱말: {words or '없음'}",
           f"  대사: same {q.get('same', 0):,} · changed {q.get('changed', 0):,} · "
           f"absent {q.get('absent_spoken', 0):,}(말 있는 대사) · added {q.get('added', 0):,}{rate}",
           f"  확인 필요 {sum(probs.values())}건" + (f" ({kinds})" if kinds else "")
           + ": python -I -m sinkprep review 를 돌리면 qa/needs_review.csv 에 들어간다",
           "  결과 파일:",
           f"    최종본     {fdir / 'final.json'}",
           f"    미리보기   {fdir / 'preview.srt'}",
           f"    영상 사본  {video_copy if video_copy is not None else '(없음)'}",
           "  보는 법: 플레이어(팟플레이어, VLC 등)에서 영상을 열고 preview.srt 를 자막으로 불러온다.",
           "           [+] 는 대본에 없는 말, [없음] 은 영상에서 못 찾은 대본 대사다(이름 뒤 ? 는 추정 화자).",
           f"  걸린 시간 {_took(seconds)}"]
    return out


def open_folder(p: Path) -> None:
    try:
        os.startfile(str(p))  # type: ignore[attr-defined]
    except (AttributeError, OSError) as e:
        _say(f"  폴더를 열 수 없다({e}): {p}")


def again(args) -> str:
    """같은 명령(다시 치면 끝난 단계는 건너뛴다)."""
    def q(s) -> str:
        s = str(s)
        return f'"{s}"' if (not s or re.search(r"[\s&()^%!,;=]", s) or not s.isascii()) else s

    parts = ["sink-film.cmd"] + [q(p) for p in getattr(args, "paths", None) or []]
    if getattr(args, "docs", None):
        parts += ["--docs", q(args.docs)]
    for opt in ("lang", "passes", "threads", "device", "workers"):
        v = getattr(args, opt, None)
        if v is not None:
            parts += [f"--{opt}", q(v)]
    if getattr(args, "rescue", None) is False:
        parts.append("--no-rescue")
    if getattr(args, "llm", None) is False:
        parts.append("--no-llm")
    for flag in ("force", "open"):
        if getattr(args, flag, False):
            parts.append(f"--{flag}")
    return " ".join(parts)


def _fail(name: str, why: str, args, hints: list[str] | None = None) -> int:
    lines = [f"film: {name} 단계에서 멈춤: {why}"]
    lines += [f"  {h}" for h in hints or []]
    lines += ["  다시 하기: 같은 명령을 다시 치면 끝난 단계는 건너뛰고 이어서 한다:", f"    {again(args)}"]
    print("\n".join(lines), file=sys.stderr, flush=True)
    return 1


# ------------------------------------------------------------------ 실행
def prepare(args) -> dict:
    """인자를 풀고 시작해도 되는지 본다(아무것도 고치지 않는다. 영상 sha1 캐시만 적는다). 안 되면 Stop."""
    from . import catalog, media

    spec, video = split_inputs(list(getattr(args, "paths", None) or []), getattr(args, "docs", None))
    rows = catalog.load()
    row, notes = find_doc(spec, rows)
    prim, more = primary_of(row, rows)
    for n in notes + more:
        _say(f"  {n}")
    check_script(prim)
    film_id, doc = prim["film_id"], prim["doc_id"]
    _say(f"  대본 {doc} ({prim.get('title_norm') or ''}, 영화 {film_id}), 영상 {video}")
    _say("  영상 sha1 확인(처음 보는 영상이면 크기에 따라 몇십 초)…")
    try:
        sha1 = media.file_sha1(video)
    except OSError as e:
        raise Stop(f"영상 파일을 읽을 수 없다({type(e).__name__}: {e}): {video}") from e
    force = bool(getattr(args, "force", False))
    other = next((r for r in rows if r.get("film_id") == film_id and r.get("status") != "excluded"
                  and r.get("video_sha1") and r["video_sha1"] != sha1), None)
    if other is not None:
        if not force:
            raise Stop(f"{film_id} 에는 다른 영상({other.get('video_path')}, sha1 {other['video_sha1'][:8]})이 이미 "
                       f"등록돼 있다(이 영상은 sha1 {sha1[:8]}). 이 영상으로 바꾸려면 --force 를 붙여 다시 친다"
                       "(받아쓰기부터 새로 한다)")
        _say(f"  --force: {film_id} 의 영상을 이 영상으로 바꾼다(옛 영상 sha1 {other['video_sha1'][:8]})")
    device, dnote = choose_device(args, film_id, sha1)
    _say(f"  {dnote}")
    used = device or (util.stage_fp("asr", film_id) or {}).get("device")     # asr 가 실제로 쓸 장치 인자
    return {"doc": doc, "film_id": film_id, "video": video, "sha1": sha1, "device": device, "used_device": used,
            "lang": getattr(args, "lang", None) or DEFAULT_LANG, "force": force}


def run(args) -> int:
    t0 = time.time()
    _say("== film: 준비 ==")
    try:
        plan = prepare(args)
    except Stop as e:
        print(f"film: 시작하지 않음: {e}", file=sys.stderr, flush=True)
        return 2
    doc, film_id, sha1 = plan["doc"], plan["film_id"], plan["sha1"]
    passes = getattr(args, "passes", None) or DEFAULT_PASSES
    rescue = getattr(args, "rescue", None)
    rescue = True if rescue is None else bool(rescue)
    llm_on = getattr(args, "llm", None) is not False
    asr_hints = ([] if plan["used_device"] != "cuda" else
                 ["cuda 로 모델을 열지 못했으면(AsrUnavailable): GPU 를 쓰는 프로그램을 끄고 다시 치거나, "
                  ".venv\\Scripts\\python -m pip install -e .[asr,asr-gpu], 또는 --device cpu(몇 시간)"])
    steps = [
        ("video", _ns("video", docs=doc, paths=[str(plan["video"])], lang=plan["lang"], force=plan["force"],
                      unregister=False),
         lambda: _after_video(doc, sha1), []),
        ("asr", _ns("asr", docs=doc, passes=passes, threads=getattr(args, "threads", None), device=plan["device"],
                    rescue=rescue),
         lambda: _after_asr(film_id, sha1), asr_hints),
        ("align", _ns("align", docs=doc, workers=getattr(args, "workers", None), llm=llm_on),
         lambda: _after_align(film_id, doc), []),
        ("final", _ns("final", docs=doc), lambda: _after_final(film_id, doc, sha1), []),
    ]
    name = "video"
    try:
        for name, ns, after, hints in steps:
            _say(f"== film: {name} ==")
            rc = _call(name, ns)
            if rc != 0:
                return _fail(name, f"종료 코드 {rc}(까닭은 위의 '!' 줄)", args, hints)
            try:
                why = after()
            except Exception as e:  # noqa: BLE001 - 결과 파일을 읽을 수 없음
                why = f"결과 파일을 읽을 수 없다({type(e).__name__}: {e})"
            if why:
                return _fail(name, f"단계는 끝났지만 {why}", args, hints)
            if name == "align" and llm_on:
                from . import llm
                stopped = llm.stopped()
                if stopped:
                    _say(f"  film: claude 를 부를 수 없어({stopped}) LLM 짝 판정은 캐시에 있는 것만 썼다. "
                         "claude 가 되면 같은 명령을 다시 치면 이 영화만 다시 맞춘다")
                elif llm.calls_made():
                    _say(f"  film: claude 를 {llm.calls_made()}번 불렀다. 다른 PC 와 결과를 맞추려면 "
                         ".venv\\Scripts\\python -I tools\\llm_cache_zip.py pack 뒤 data/llm_cache.zip 을 함께 커밋")
        name = "validate"
        _say("== film: validate ==")
        if check_outputs(film_id):
            return _fail(name, "결과 파일 검사 실패(위 FAIL 줄)", args)
        name = "preview"
        _say("== film: preview ==")
        try:
            rc = write_preview(film_id)
        except Exception as e:  # noqa: BLE001
            print(f"  ! preview: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            rc = 1
        if rc:
            return _fail(name, f"미리보기 자막을 만들지 못함(종료 코드 {rc})", args)
    except KeyboardInterrupt:
        print(f"\nfilm: {name} 단계에서 중단됨(Ctrl+C). 같은 명령을 다시 치면 끝난 단계는 건너뛰고 이어서 한다:\n"
              f"    {again(args)}", file=sys.stderr, flush=True)
        return 1
    _say("== film: 끝 ==")
    try:
        lines = summary(doc, film_id, time.time() - t0)
    except Exception as e:  # noqa: BLE001 - 단계는 모두 끝났다: 요약만 못 만든 것
        lines = [f"  요약을 만들 수 없다({type(e).__name__}: {e}): films/{film_id}/ 를 직접 본다"]
    for line in lines:
        _say(line)
    if getattr(args, "open", False):
        open_folder(paths.FILMS / film_id)
    return 0
