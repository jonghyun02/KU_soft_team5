"""받아쓰기(asr): 영상 소리 -> films/{film_id}/work/asr.json (명세 4.5, schema/asr.schema.json).

python -I -m sinkprep asr [--docs id] [--passes plain=large-v3-turbo] [--rescue|--no-rescue] [--threads 6]
                          [--device auto|cpu|cuda] [--force]

대상은 catalog 에서 영상이 등록된(video_path) 영화의 대표 문서(is_primary)다. 열쇠는 film_id.
--passes·--device·--rescue 를 주지 않으면 그 영화를 앞서 받아쓴 값(단계 지문)을 이어 쓴다. 처음이면 기본값.
sinkprep all 안에서는 모델을 열지 않고 캐시만으로 조립한다(ALL_CACHE_ONLY): 받아쓰기는 이 명령으로 따로 돌린다.
같은 영화를 두 프로세스가 함께 받아쓰지 못한다(cache/asr/{sha1}/.lock).

흐름
1. 영상 sha1 확인 -> 소리(cache/audio/{sha1}.wav)와 시각 대응표(media.TimeMap).
2. 말 구간(VAD, faster-whisper 에 든 Silero)을 영화 전체에서 찾아 cache/asr/{sha1}/vad.json 에 둔다.
   전체 소리를 메모리에 올리는 것은 이때뿐이고, 배열을 지운 뒤에 모델을 올린다.
3. 말 구간을 25초 이하 묶음으로 묶는다(앞 묶음과 틈 1초 미만이면 붙임). 묶음 앞뒤로 0.1초(이웃과의 틈의 절반까지)를
   덧대어 이웃과 겹치지 않게 자른다.
4. pass 마다 묶음을 하나씩 따로 받아쓴다(vad_filter·앞 글 조건 없음). 묶음 결과는 곧바로
   cache/asr/{sha1}/{모델 꼬리표}/{a:010d}-{b:010d}.json 에 쓴다. 중간에 죽어도 다시 돌리면 끝난 묶음은 건너뛴다.
5. --rescue: 말 구간 찾기가 놓친 대사를 되찾는다. 소리 전체를 빈틈없이 덮는 약 25초 창 두 벌(경계가 서로 엇갈림,
   경계는 조용한 곳)을 plain 의 모델로 같은 방식으로 받아써 cache/asr/{sha1}/{모델 꼬리표}/full/ 에 둔다.
   벌 A 의 낱말 가운데 plain 묶음이 덮지 않은 시간대에 있고(plain 낱말과 같은 소리도 뺀다), 같은 시각에 벌 B 도 같은 글을
   낸 조각만 pass "rescue" 로 넣는다(헛소리는 창이 달라지면 같은 시각에 같은 글로 다시 나오지 않는다). 두 벌 자체는
   asr.json 에 넣지 않는다. 규칙은 rescue_fragments 에 있다: plain 이 낸 소리 빼기, 조각 나누기(세그먼트·쉼 0.6초),
   글자 닮음 0.6, 닮은 낱말의 시간 겹침, 되풀이 버리기, 버릇 글귀(여러 창에서 똑같이 나오는 글) 버리기.
   말 구간 찾기를 창마다 새로 돌려(창 단위 VAD) 조각을 거르는 것은 넣지 않았다: 진짜 대사 조각 다섯에 하나가 거기서도
   말로 잡히지 않는다(비명·속삭임·전화 목소리).
6. 조립: 낱말 시각(소리 시각)을 영상 시각으로 옮겨 소수 3자리로, 묶음 차례·묶음 안 낱말 차례 그대로 i = 0.. 을 붙인다.
   묶음 첫 낱말은 시작이 잘라 낸 조각의 맨 앞에 붙어 나오므로(Whisper 정렬의 성질) 묶음 시작 + 0.05초로 당긴다.
   글자를 못 푼 낱말(U+FFFD)은 버린다. 묶음 캐시는 원래 값 그대로다.

asr.json 에는 스키마가 허락한 칸만 쓴다. 모델·옵션은 passes[].model 문자열에, 세그먼트 통계(avg_logprob, temperature 등)는
묶음 캐시에만 있다(film_records, pass_word_stats 로 읽는다).
같은 캐시로 다시 조립하면 바이트가 같다(시각·난수·걸린 시간은 넣지 않음). 캐시를 지우고 처음부터 다시 받아쓰면 같다는
보장은 없다: 확신이 낮은 묶음은 faster-whisper 가 온도를 올려 무작위로 다시 뽑는다(묶음 열에 하나꼴, 씨앗 없음).

다른 모듈이 불러 쓰는 것은 film_habits 하나다(align 이 rescue 줄의 버릇 글귀를 거를 때). encode_clip·force_text(대본
글을 소리에 teacher forcing), pass_word_stats·film_rescue_fragments(캐시에서 낱말별 세그먼트 통계·rescue 조각 판정을
다시 냄)는 운영 차례에서 부르지 않고 시험과 조사용으로 열어 둔 것이다.
시험은 set_backend(가짜) 로 모델 없이 돈다.
"""
from __future__ import annotations

import bisect
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from . import media, paths, util

STAGE = "asr"
VERSION = "0.3.0"            # 조립 방식이 바뀌면 올린다(단계 지문). 0.2.0: 첫 낱말 보정, U+FFFD 낱말 버림, rescue pass
#                              0.3.0: rescue 에서 plain 낱말과 같은 소리를 뺌(RESCUE_DUP_*), 시간 겹침을 닮은 낱말
#                                     글자 기준으로(RESCUE_MIN_COLOC), 버릇 글귀를 버림(RESCUE_HABIT_*), 버린 까닭 통계
DECODE_VERSION = "1"         # 받아쓰기 호출·낱말 처리 방식이 바뀌면 올린다(묶음 캐시 열쇠)
VAD_VERSION = "1"            # 말 구간 찾기 방식이 바뀌면 올린다(vad.json 열쇠)
WINDOW_VERSION = "1"         # 전체 덮기 창을 나누는 방식이 바뀌면 올린다(windows.json 열쇠)
SCHEMA = "sink.asr/1"

SR = media.SR
VAD_OPTIONS = {"threshold": 0.35, "min_speech_duration_ms": 100, "min_silence_duration_ms": 300,
               "speech_pad_ms": 150, "max_speech_duration_s": 25.0}
MAX_CHUNK_S = 25.0           # 묶음 길이 상한(초)
JOIN_GAP_S = 1.0             # 앞 묶음과 틈이 이보다 짧으면 붙임
CLIP_PAD_S = 0.1             # 묶음 앞뒤 덧댐(이웃 묶음과의 틈의 절반까지)
# 온도는 주지 않는다: faster-whisper 기본값(0.0 → 0.2 … 1.0 되돌이, 씨앗 없음)이 쓰인다. 이 표는 묶음 캐시 열쇠에
# 그대로 들어가므로 칸을 더하거나 바꾸면 DECODE_VERSION 을 올리지 않아도 캐시가 전부 빗나간다.
DECODE = {"beam_size": 5, "word_timestamps": True, "vad_filter": False, "condition_on_previous_text": False}

# 조립(캐시는 그대로 두고 asr.json 을 만들 때 적용)
FIRST_WORD_GLUE_S = 0.02     # 묶음 첫 낱말의 시작이 잘라 낸 조각의 시작에서 이 안이면 '붙었다'고 본다
FIRST_WORD_LEAD_S = 0.05     # 붙은 첫 낱말의 시작을 (덧대기 전) 묶음 시작 + 이 값으로 당긴다
FIRST_WORD_MIN_S = 0.1       # 당긴 뒤에도 낱말 길이를 이만큼은 남긴다(원래 더 짧으면 그대로)
BAD_CHAR = "�"          # 글자를 못 푼 자리(이 글자가 든 낱말은 asr.json 에 넣지 않는다)
HOLE_MIN_S = 1.0             # 말 구간 안에서 낱말 없이 이만큼 넘게 빈 자리를 센다
HOLE_WARN_SHARE = 0.1        # 그런 자리의 합이 말 구간 합의 이 비율을 넘으면 문제로 남긴다

# rescue pass(말 구간 찾기가 놓친 대사 되찾기)
BASE_ID = "plain"            # rescue 가 모델을 빌리고 '덮은 시간대'를 재는 pass
RESCUE_ID = "rescue"
RESCUE_CONTEXT = "full_window_stable"
WINDOW_DIR = "full"          # 전체 덮기 창 기록을 두는 하위 폴더(묶음 캐시와 같은 꼴·같은 열쇠 규칙)
WINDOW_S = 25.0              # 창 길이(기준 경계 사이)
WINDOW_SEARCH_S = 5.0        # 기준 경계 앞 이 안에서 가장 조용한 곳을 경계로
ENV_STEP_S = 0.1             # 소리 크기 포락선의 틀 길이
RESCUE_GAP_S = 0.6           # 벌 A 낱말을 조각내는 쉼(앞 낱말 끝에서 이만큼 넘게 비면 새 조각)
RESCUE_MARGIN_S = 1.0        # 조각 앞뒤로 이만큼 넓힌 시간대의 벌 B 낱말과 견준다
RESCUE_MIN_SIM = 0.6         # 글자 닮음(rapidfuzz ratio, 0~1)이 이 이상이면 안정적인 조각
RESCUE_MIN_COLOC = 0.5       # 닮은 벌 B 낱말들의 글자 가운데 이 몫 이상이 그 조각과 시간이 겹치는 낱말의 것이어야 한다
RESCUE_DUP_SHARE = 0.5       # plain 낱말과 겹친 시간이 둘 가운데 짧은 쪽 길이의 이 몫 이상이면 plain 이 이미 낸 소리
RESCUE_DUP_GAP_S = 0.1       # plain 낱말과 글이 같고 사이가 이 안이면(맞닿거나 조금이라도 겹침) 역시 같은 소리
RESCUE_MAX_REPEAT = 4        # 같은 낱말(또는 한 낱말 안 같은 글자)이 이보다 많이 잇달으면 되풀이 헛소리
RESCUE_HABIT_CHARS = 7       # 버릇 글귀: 글자 수(문장부호·공백 뺀 것)가 이 이상인 조각 글이
RESCUE_HABIT_WINDOWS = 3     # 벌 A 의 서로 다른 창 이만큼에서 똑같이 나오면 모델이 말 없는 소리에 버릇처럼 내는 글로 본다
RESCUE_DROPS = ("b_silent", "b_differs", "b_elsewhere", "same_start", "habit")   # 조각을 버린 까닭(rescue_fragments)

# sinkprep all(pipeline.py)이 읽는 약속: args.cmd == "all" 이면 받아쓰기 모델을 열지 않고(소리 추출·말 구간 찾기도 안 함)
# 캐시만으로 조립하며, 옵션은 그 영화를 앞서 받아쓴 값만 쓴다. 캐시에 없는 것이 있으면 asr.json 을 그대로 두고 알린다.
ALL_CACHE_ONLY = True

DEFAULT_PASSES = "plain=large-v3-turbo"   # 속도·메모리를 잰 모델(CPU int8 묶음당 약 5초, 1.8GB). large-v3 는 재지 않았다
DEFAULT_THREADS = 6
DEFAULT_DEVICE = "auto"
MODEL_ALIASES = {"turbo": "large-v3-turbo"}
CUDA_MIN_FREE_MIB = 5000     # auto: 여유 VRAM 이 이만큼 있어야 cuda 를 시도
LOW_CONF = 0.5               # 낱말 평균 확률이 이보다 낮으면 asr_low_conf
MAX_CLIP_S = 30.0            # Whisper 창 길이
MAX_TOKENS = 448             # Whisper 디코더 길이 한도
PUNCT_PRE = "\"'“¿([{-"
PUNCT_APP = "\"'.。,，!！?？:：”)]}、"
SENTENCE_END = ".。!！?？"
_PASS_ID = re.compile(r"^[a-z][a-z0-9_]*$")
INSTALL_HINT = r"faster-whisper 가 없습니다. 설치: .venv\Scripts\python -m pip install -e .[asr]"


class AsrUnavailable(RuntimeError):
    """받아쓰기 도구(faster-whisper)가 없거나 모델을 열 수 없음."""


class NotCached(RuntimeError):
    """캐시만 쓰는 실행(sinkprep all)인데 받아써야 할 것이 남아 있음."""


# ------------------------------------------------------------------ 작은 도구
def _sha1_json(obj: Any) -> str:
    return util.sha1_bytes(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _dur(sec: float) -> str:
    sec = int(round(max(sec, 0.0)))
    if sec < 60:
        return f"{sec}초"
    m = sec // 60
    return f"{m}분" if m < 60 else f"{m // 60}시간 {m % 60}분"


def parse_passes(spec: str) -> list[tuple[str, str]]:
    """"plain=large-v3,ctx=turbo" -> [("plain", "large-v3"), ("ctx", "large-v3-turbo")]. 꼴이 틀리면 ValueError."""
    out: list[tuple[str, str]] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        pid, sep, name = part.partition("=")
        pid, name = pid.strip(), name.strip()
        if not sep or not name or not _PASS_ID.match(pid):
            raise ValueError(f"--passes 꼴이 틀림: {part!r} (id=모델[,id=모델], id 는 소문자로 시작하는 영숫자·밑줄)")
        if pid in {p for p, _ in out}:
            raise ValueError(f"--passes 에 같은 id 가 두 번: {pid}")
        out.append((pid, MODEL_ALIASES.get(name, name)))
    if not out:
        raise ValueError("--passes 가 비어 있음")
    return out


def whisper_lang(video_lang: str | None) -> str:
    """catalog video_lang -> Whisper 언어 코드("ko-KR" -> "ko"). 언어를 모르면 ValueError."""
    lang = (video_lang or "").strip()
    if not lang or lang in ("unknown", "mixed"):
        raise ValueError(f"video_lang 이 {lang or '빈칸'}: 받아쓸 언어를 정해야 함 "
                         "(python -I -m sinkprep video <영상> --docs <doc_id> --lang ko)")
    return lang.split("-")[0].lower()


# ------------------------------------------------------------------ 묶음 규칙
def chunks(regions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """말 구간(표본 번호) -> 받아쓰기 묶음. 25초 넘는 구간은 25초마다 자르고, 앞 묶음과 틈이 1초 미만이면서
    합쳐 25초 이하면 앞 묶음에 붙인다(CBoE 규칙)."""
    cap = int(MAX_CHUNK_S * SR)
    out: list[list[int]] = []
    for a, b in regions:
        while b - a > cap:
            out.append([a, a + cap])
            a += cap
        if out and a - out[-1][1] < JOIN_GAP_S * SR and b - out[-1][0] <= cap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def pads(groups: list[tuple[int, int]], n_total: int) -> list[tuple[int, int]]:
    """묶음마다 실제로 잘라 받아쓸 범위(표본 번호). 앞뒤로 0.1초, 이웃 묶음이 가까우면 틈의 절반까지만 덧대어
    이웃과 겹치지 않는다."""
    pad = int(CLIP_PAD_S * SR)
    out = []
    for k, (a, b) in enumerate(groups):
        before = min(pad, max(a - groups[k - 1][1], 0) // 2) if k else min(pad, max(a, 0))
        after = min(pad, max(groups[k + 1][0] - b, 0) // 2) if k + 1 < len(groups) else min(pad, max(n_total - b, 0))
        out.append((a - before, b + after))
    return out


def _clean_regions(regions, n_total: int) -> list[tuple[int, int]]:
    out = []
    for a, b in sorted((int(a), int(b)) for a, b in regions):
        a, b = max(a, 0), min(b, n_total)
        if out and a < out[-1][1]:
            a = out[-1][1]
        if b > a:
            out.append((a, b))
    return out


# ------------------------------------------------------------------ 모델
class ModelHandle:
    """열린(또는 처음 쓸 때 여는) 받아쓰기 모델.

    model: faster_whisper.WhisperModel(가짜 백엔드는 아무 값). tag: 묶음 캐시 폴더 이름.
    describe(): passes[].model 문자열의 앞부분. device: "cpu"|"cuda". compute_type: "int8"|"float16".
    """

    def __init__(self, name: str, repo: str, snapshot: str, device: str, compute_type: str, *,
                 threads: int = DEFAULT_THREADS, path: str | None = None, versions: dict | None = None,
                 model: Any = None):
        self.name = name
        self.repo = repo
        self.snapshot = snapshot
        self.device = device
        self.compute_type = compute_type
        self.threads = threads
        self.path = path
        self.versions = dict(versions or {})
        self._model = model
        self._toks: dict[str, Any] = {}

    @property
    def model(self):
        if self._model is None and self.path:
            self._model = _open(self)
        return self._model

    @property
    def opened(self) -> bool:
        return self._model is not None

    @property
    def tag(self) -> str:
        raw = f"{self.name}@{self.snapshot[:8]}-{self.device}-{self.compute_type}"
        return re.sub(r"[^A-Za-z0-9._@-]", "_", raw)

    def describe(self) -> str:
        fw = self.versions.get("faster_whisper", "?")
        ct = self.versions.get("ctranslate2", "?")
        return f"faster-whisper {fw} | {self.repo}@{self.snapshot[:8]} | ct2 {ct} {self.device} {self.compute_type}"

    def key(self) -> dict:
        """묶음 캐시 열쇠에 드는 모델 쪽 값."""
        return {"repo": self.repo, "snapshot": self.snapshot, "device": self.device,
                "compute_type": self.compute_type, "versions": self.versions}

    def tokenizer(self, lang: str):
        """faster_whisper.tokenizer.Tokenizer(받아쓰기, 그 언어)."""
        tok = self._toks.get(lang)
        if tok is None:
            from faster_whisper.tokenizer import Tokenizer
            m = self.model
            tok = Tokenizer(m.hf_tokenizer, m.model.is_multilingual, task="transcribe", language=lang)
            self._toks[lang] = tok
        return tok

    def close(self) -> None:
        """모델을 메모리에서 내린다."""
        self._model = None
        self._toks.clear()
        gc.collect()


def _fw():
    try:
        import ctranslate2
        import faster_whisper
    except ImportError as e:
        raise AsrUnavailable(INSTALL_HINT) from e
    return faster_whisper, ctranslate2


def free_vram_mib() -> int:
    """첫 GPU 의 여유 VRAM(MiB). nvidia-smi 가 없거나 실패하면 0."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return 0
    try:
        r = subprocess.run([exe, "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20)
        total, used = (int(x) for x in r.stdout.strip().splitlines()[0].split(","))
        return max(total - used, 0)
    except Exception:  # noqa: BLE001 - 못 재면 GPU 를 쓰지 않는다
        return 0


def pick_device(device: str = DEFAULT_DEVICE) -> tuple[str, str]:
    """(device, compute_type). auto 는 CUDA 장치가 있고 여유 VRAM 이 5000MiB 이상일 때만 cuda."""
    if device == "cpu":
        return "cpu", "int8"
    if device == "cuda":
        return "cuda", "float16"
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
    except Exception:  # noqa: BLE001
        n = 0
    if n > 0 and free_vram_mib() >= CUDA_MIN_FREE_MIB:
        return "cuda", "float16"
    return "cpu", "int8"


def _cuda_dlls() -> int:
    """site-packages/nvidia/*/bin(pip 로 깐 cuBLAS·cuDNN)을 DLL 찾는 길에 더한다. 더한 폴더 수(없으면 0)."""
    try:
        import ctranslate2
    except ImportError:
        return 0
    n = 0
    for d in sorted((Path(ctranslate2.__file__).resolve().parent.parent / "nvidia").glob("*/bin")):
        if not d.is_dir():
            continue
        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(d))
        except OSError:
            continue
        if str(d) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        n += 1
    return n


def _repo_snapshot(path: str, name: str) -> tuple[str, str]:
    """HF 캐시 경로(models--조직--이름/snapshots/해시) -> ("조직/이름", 해시). 그냥 폴더면 (폴더 이름, model.bin 크기)."""
    p = Path(path)
    if p.parent.name == "snapshots" and p.parent.parent.name.startswith("models--"):
        return p.parent.parent.name[len("models--"):].replace("--", "/"), p.name
    try:
        size = (p / "model.bin").stat().st_size
    except OSError:
        size = 0
    return f"local:{p.name or name}", f"size{size}"


def _avail_gb() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except Exception:  # noqa: BLE001
        return None


def _open(h: ModelHandle):
    fw, _ = _fw()
    if h.device == "cuda":
        _cuda_dlls()
    else:
        avail = _avail_gb()
        try:
            need = (Path(h.path) / "model.bin").stat().st_size * 1.4 / 2 ** 30
        except OSError:
            need = 0.0
        note = "" if avail is None else f" (여유 메모리 {avail:.1f}GB, 필요 약 {need:.1f}GB)"
        print(f"  모델 여는 중: {h.describe()}{note}", flush=True)
        if avail is not None and need and avail < need:
            print(f"  ! 여유 메모리가 모자랄 수 있음: 페이지 파일로 넘어가면 몹시 느려진다"
                  f"(더 작은 모델: --passes plain=large-v3-turbo)", flush=True)
    return fw.WhisperModel(h.path, device=h.device, compute_type=h.compute_type,
                           cpu_threads=h.threads, num_workers=1)


def resolve_model(name: str) -> tuple[str, str, str, str, dict]:
    """모델 이름 -> (이름, 폴더 경로, 저장소, 스냅숏, 라이브러리 버전). 모델을 열지 않는다.

    로컬 캐시를 먼저 찾고(네트워크 접속 없음) 없을 때만 내려받는다. faster-whisper 가 없으면 AsrUnavailable.
    """
    fw, ct2 = _fw()
    name = MODEL_ALIASES.get(name, name)
    if Path(name).is_dir():
        path = str(Path(name))
    else:
        from faster_whisper.utils import download_model
        try:
            path = download_model(name, local_files_only=True)
        except ValueError:
            raise
        except Exception:  # noqa: BLE001 - 로컬에 없음
            print(f"  모델 {name} 이 로컬에 없어 내려받음", flush=True)
            path = download_model(name)
    repo, snap = _repo_snapshot(path, name)
    return name, str(path), repo, snap, {"faster_whisper": fw.__version__, "ctranslate2": ct2.__version__}


def load_model(name: str, *, device: str = DEFAULT_DEVICE, threads: int = DEFAULT_THREADS,
               lazy: bool = False, probe: bool = True) -> ModelHandle:
    """받아쓰기 모델을 연다. name: "large-v3", "large-v3-turbo"(= "turbo"), 그 밖에 faster-whisper 가 아는 이름·폴더.

    로컬 캐시를 먼저 찾고(네트워크 접속 없음) 없을 때만 내려받는다.
    device auto: CUDA 장치가 있고 여유 VRAM 이 5000MiB 이상이면 cuda(float16)를 열어 보고 실패하면 cpu(int8).
    lazy 면 cpu 모델은 handle.model 을 처음 읽을 때 연다(묶음이 모두 캐시에 있으면 열지 않음).
    probe 가 False 면 cuda 도 열어 보지 않는다(캐시만 읽을 때: 꼬리표·열쇠만 필요하다).
    faster-whisper 가 없으면 AsrUnavailable.
    """
    name, path, repo, snap, versions = resolve_model(name)
    dev, ctype = pick_device(device)
    h = ModelHandle(name, repo, snap, dev, ctype, threads=threads, path=path, versions=versions)
    if dev == "cuda" and probe:
        try:
            m = h.model
            import numpy as np
            from faster_whisper.audio import pad_or_trim
            m.encode(pad_or_trim(m.feature_extractor(np.zeros(SR, dtype=np.float32))[:, :100]))   # DLL 을 실제로 불러 본다
        except Exception as e:  # noqa: BLE001
            if device == "cuda":
                raise AsrUnavailable(f"cuda 로 모델을 열 수 없음: {type(e).__name__}: {e}") from e
            print(f"  ! cuda 로 모델을 열 수 없어 cpu(int8)로 감: {type(e).__name__}: {str(e)[:200]}", flush=True)
            h.close()
            h.device, h.compute_type = "cpu", "int8"
    if not lazy:
        h.model  # noqa: B018 - 지금 연다
    return h


# ------------------------------------------------------------------ 말 구간·받아쓰기(실제)
def speech_regions(audio) -> list[tuple[int, int]]:
    """말 구간 [(시작 표본, 끝 표본)]. faster-whisper 에 든 Silero VAD(VAD_OPTIONS)."""
    _fw()
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    ts = get_speech_timestamps(audio, VadOptions(**VAD_OPTIONS), sampling_rate=SR)
    return [(int(t["start"]), int(t["end"])) for t in ts]


def _prob(p: Any) -> float:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return 0.0
    return round(min(max(p, 0.0), 1.0), 3) if p == p else 0.0


def _num(x: Any, nd: int = 4) -> float | None:
    """JSON 에 넣을 수(없거나 NaN·무한대면 None)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if x == x and abs(x) != float("inf") else None


def transcribe_clip(handle: ModelHandle, clip, lang: str, *, hotwords: str | None = None,
                    initial_prompt: str | None = None) -> tuple[list[dict], list[dict]]:
    """소리 조각(float32, 16 kHz, 30초 이하 권장) 하나를 받아쓴다.

    words: [{"w","s","e","p"}] 조각 기준 초(소수 3자리). 공백을 떼고 빈 낱말은 버리고 NFC, 시각은 조각 안으로 자름.
    segs: [{"start","end","text","avg_logprob","no_speech_prob","compression_ratio","temperature"}].
      temperature 가 0 보다 크면 첫 시도(온도 0)가 문턱에 걸려 무작위로 다시 뽑은 결과다(1.0 이면 여섯 온도 모두 걸림).
      no_speech_prob 은 large-v3-turbo 에서 늘 0 에 가까워(추격자 1,115개 모두 5e-5 미만) 소수 10자리로 적는다.
    hotwords·initial_prompt 가 없으면 대본 없이 받아쓰기(plain).
    묶음 첫 낱말의 시작은 조각의 0초에 붙어 나온다(정렬 경로가 0 에서 시작): 조립 때 record_words 가 당긴다.
    """
    it, _ = handle.model.transcribe(clip, language=lang, hotwords=hotwords or None,
                                    initial_prompt=initial_prompt or None, **DECODE)
    dur = len(clip) / SR
    words: list[dict] = []
    segs: list[dict] = []
    for seg in it:                                   # 실제 계산은 여기서 일어난다
        segs.append({"start": round(float(seg.start), 3), "end": round(float(seg.end), 3),
                     "text": util.nfc((seg.text or "").strip()),
                     "avg_logprob": _num(seg.avg_logprob), "no_speech_prob": _num(seg.no_speech_prob, 10),
                     "compression_ratio": _num(seg.compression_ratio), "temperature": _num(seg.temperature)})
        for x in seg.words or []:
            txt = util.nfc(x.word.strip())
            if not txt:
                continue
            s = min(max(float(x.start), 0.0), dur)
            e = min(max(float(x.end), s), dur)
            words.append({"w": txt, "s": round(s, 3), "e": round(e, 3), "p": _prob(x.probability)})
    return words, segs


def encode_clip(handle: ModelHandle, clip) -> tuple[Any, int]:
    """소리 조각(30초 이하) -> (인코더 출력, 내용 프레임 수). 같은 출력으로 force_text 를 여러 번 부를 수 있다."""
    if len(clip) == 0:
        raise ValueError("빈 소리 조각")
    if len(clip) > MAX_CLIP_S * SR:
        raise ValueError(f"소리 조각이 {len(clip) / SR:.1f}초: {MAX_CLIP_S:.0f}초 창을 넘음(나눠서 넣을 것)")
    from faster_whisper.audio import pad_or_trim
    m = handle.model
    feats = m.feature_extractor(clip)
    n = min(feats.shape[-1] - 1, m.feature_extractor.nb_max_frames)
    return m.encode(pad_or_trim(feats[:, :n])), int(n)


def _is_punct(piece: str) -> bool:
    s = piece.strip()
    return bool(s) and "�" not in s and not any(c.isalnum() for c in s)


def _force_result(tok, ids: list[int], probs, pairs, n_frames: int, tokens_per_second: float = 50.0) -> dict:
    """teacher forcing 결과(토큰 확률, (토큰 번호, 시간 번호) 짝)를 토큰 점수와 낱말 시각으로 묶는다.

    faster-whisper 의 find_alignment·add_word_timestamps 와 같은 차례: 낱말로 묶기 -> 문장 끝 옆 긴 낱말 줄이기 ->
    문장부호를 이웃 낱말에 붙이기(시각·확률은 낱말 것) -> 조각 첫 낱말이 쉼을 삼켰으면 줄이기.
    """
    import numpy as np
    n = len(ids)
    p = np.asarray(list(probs), dtype=np.float64)[:n]
    if len(p) < n:
        p = np.pad(p, (0, n - len(p)))
    logp = np.log(np.clip(p, 1e-10, 1.0))
    punct = [_is_punct(tok.decode([t])) for t in ids]
    keep = [not x for x in punct]
    n_np = sum(keep)
    sum_np = float(logp[keep].sum()) if n_np else 0.0

    pairs = np.asarray(list(pairs), dtype=np.int64).reshape(-1, 2)
    dur = n_frames / (tokens_per_second * 2)
    words: list[dict] = []
    word_texts, word_tokens = tok.split_to_word_tokens(ids + [tok.eot])
    if len(pairs) and len(word_tokens) > 1:
        ti, tt = pairs[:, 0], pairs[:, 1]
        bounds = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
        jumps = np.pad(np.diff(ti), (1, 0), constant_values=1).astype(bool)
        jt = tt[jumps] / tokens_per_second
        if len(jt) <= bounds[-1]:                           # 끝 토큰 줄이 없으면 조각 끝으로 채움
            jt = np.pad(jt, (0, int(bounds[-1]) + 1 - len(jt)), constant_values=dur)
        al = [{"word": w, "t0": int(i), "t1": int(j), "start": float(jt[i]), "end": float(jt[j]),
               "p": float(p[i:j].mean()) if j > i else 0.0}
              for w, i, j in zip(word_texts[:-1], bounds[:-1], bounds[1:])]
        durs = [a["end"] - a["start"] for a in al if a["end"] - a["start"] != 0]
        med = min(0.7, float(np.median(durs))) if durs else 0.0
        mx = med * 2
        if durs:
            for i in range(1, len(al)):
                if al[i]["end"] - al[i]["start"] > mx:
                    if al[i]["word"] in SENTENCE_END:
                        al[i]["end"] = al[i]["start"] + mx
                    elif al[i - 1]["word"] in SENTENCE_END:
                        al[i]["start"] = al[i]["end"] - mx
        _merge_punct(al)
        al = [a for a in al if a["word"].strip()]
        if al and al[0]["end"] > med * 4 and (
                al[0]["end"] - al[0]["start"] > mx or (len(al) > 1 and al[1]["end"] - al[0]["start"] > mx * 2)):
            if len(al) > 1 and al[1]["end"] - al[1]["start"] > mx:
                al[0]["end"] = al[1]["start"] = max(al[1]["end"] / 2, al[1]["end"] - mx)
            al[0]["start"] = max(0.0, al[0]["end"] - mx)
        for a in al:
            s = min(max(a["start"], 0.0), dur)
            e = min(max(a["end"], s), dur)
            words.append({"w": util.nfc(a["word"].strip()), "s": round(s, 3), "e": round(e, 3),
                          "p": _prob(a["p"]), "t0": a["t0"], "t1": a["t1"]})
    return {"logp_sum": float(logp.sum()), "logp_mean": float(logp.mean()) if n else 0.0,
            "logp_sum_nopunct": sum_np, "logp_mean_nopunct": (sum_np / n_np) if n_np else None,
            "n_tokens": n, "n_tokens_nopunct": n_np,
            "token_logp": [float(x) for x in logp], "tokens": list(ids), "punct": punct, "words": words}


def _merge_punct(al: list[dict]) -> None:
    """앞에 붙는 문장부호는 뒤 낱말에, 뒤에 붙는 문장부호는 앞 낱말에 글자와 토큰 범위만 붙인다(시각·확률은 그대로)."""
    i, j = len(al) - 2, len(al) - 1
    while i >= 0:
        prev, nxt = al[i], al[j]
        if prev["word"].startswith(" ") and prev["word"].strip() in PUNCT_PRE:
            nxt["word"] = prev["word"] + nxt["word"]
            nxt["t0"] = prev["t0"]
            prev["word"] = ""
        else:
            j = i
        i -= 1
    i, j = 0, 1
    while j < len(al):
        prev, nxt = al[i], al[j]
        if not prev["word"].endswith(" ") and nxt["word"] in PUNCT_APP:
            prev["word"] = prev["word"] + nxt["word"]
            prev["t1"] = nxt["t1"]
            nxt["word"] = ""
        else:
            i = j
        j += 1


def force_text(handle: ModelHandle, enc, n_frames: int, text: str, lang: str) -> dict:
    """후보 글자를 그 소리에 teacher forcing 한다(encode_clip 의 출력을 받음). 디코더만 한 번 돌아 싸다.

    돌려주는 값:
      logp_sum, logp_mean          글 토큰 로그 확률의 합·평균(끝 토큰 EOT 는 점수에 없다: 끝이 잘린 글은 벌점이 없음)
      logp_sum_nopunct, logp_mean_nopunct   문장부호 토큰을 뺀 합·평균(뺄 토큰뿐이면 평균은 None)
      n_tokens, n_tokens_nopunct, token_logp[], tokens[], punct[]
      words: [{"w","s","e","p","t0","t1"}]  조각 기준 초. t0:t1 은 그 낱말의 token_logp 범위
    글이 비었거나 448 토큰 한도를 넘으면 ValueError.
    """
    text = util.nfc(text or "").strip()
    if not text:
        raise ValueError("빈 글은 맞춰 볼 수 없음")
    tok = handle.tokenizer(lang)
    ids = [t for t in tok.encode(" " + text) if t < tok.eot]      # 앞 공백이 없으면 첫 토큰이 달라진다
    if not ids:
        raise ValueError("빈 글은 맞춰 볼 수 없음")
    if len(tok.sot_sequence) + 1 + len(ids) + 1 > MAX_TOKENS:
        raise ValueError(f"글이 {len(ids)}토큰: 디코더 한도 {MAX_TOKENS}토큰을 넘음(나눠서 넣을 것)")
    m = handle.model
    r = m.model.align(enc, tok.sot_sequence, [ids], int(n_frames), median_filter_width=7)[0]
    return _force_result(tok, ids, r.text_token_probs, r.alignments, int(n_frames),
                         float(getattr(m, "tokens_per_second", 50)))


# ------------------------------------------------------------------ 백엔드(시험용 주입)
class FasterWhisperBackend:
    """실제 백엔드: Silero VAD + faster-whisper. 가짜 백엔드는 vad_name, vad, load, transcribe 를 가진다(identify 는 선택)."""
    name = "faster-whisper"
    vad_name = "silero-v6"

    def vad(self, audio) -> list[tuple[int, int]]:
        return speech_regions(audio)

    def identify(self, name: str) -> dict:
        """모델을 열지 않고 알 수 있는 것(저장소, 스냅숏, 라이브러리 버전). 단계 지문에 들어간다."""
        _, _, repo, snap, versions = resolve_model(name)
        return {"repo": repo, "snapshot": snap, "versions": versions}

    def load(self, name: str, device: str, threads: int, probe: bool = True) -> ModelHandle:
        return load_model(name, device=device, threads=threads, lazy=True, probe=probe)

    def transcribe(self, handle: ModelHandle, clip, lang: str, hotwords: str | None = None,
                   initial_prompt: str | None = None) -> tuple[list[dict], list[dict]]:
        return transcribe_clip(handle, clip, lang, hotwords=hotwords, initial_prompt=initial_prompt)


class _State:
    backend: Any = None


def set_backend(backend) -> None:
    """가짜 백엔드 끼우기(시험). None 이면 faster-whisper.

    백엔드: vad_name(문자열), vad(audio) -> [(시작 표본, 끝 표본)], load(name, device, threads) -> ModelHandle,
    transcribe(handle, clip, lang, hotwords=None, initial_prompt=None) -> (words, segs) (transcribe_clip 과 같은 꼴).
    identify(name) -> {"repo","snapshot","versions"} 가 있으면 건너뛸지 정할 때 load 를 부르지 않는다.
    """
    _State.backend = backend


def get_backend():
    if _State.backend is None:
        _State.backend = FasterWhisperBackend()
    return _State.backend


# ------------------------------------------------------------------ 캐시
def vad_path(video_sha1: str) -> Path:
    return paths.ASR_CACHE / video_sha1 / "vad.json"


def windows_path(video_sha1: str) -> Path:
    """전체 덮기 창 두 벌의 범위: {"key", "n_samples", "a": [[a, b]..], "b": [[a, b]..]} (소리 표본 번호)."""
    return paths.ASR_CACHE / video_sha1 / "windows.json"


def lock_path(video_sha1: str) -> Path:
    return paths.ASR_CACHE / video_sha1 / ".lock"


def chunk_path(video_sha1: str, tag: str, a: int, b: int, sub: str | None = None) -> Path:
    """묶음 캐시 파일. a, b 는 덧대기 전 묶음 범위(소리 표본 번호).

    sub: 말 구간 묶음(plain)이 아닌 기록을 두는 하위 폴더. 전체 덮기 창은 WINDOW_DIR("full").
    힌트를 준 받아쓰기(ctx)를 만들 때도 하위 폴더를 따로 줘야 한다: 파일 이름에는 범위만 있고 힌트는 열쇠에만
    들어 있어, 같은 폴더에 두면 같은 범위의 plain 기록과 서로 덮어쓴다.
    """
    d = paths.ASR_CACHE / video_sha1 / tag
    return (d / sub if sub else d) / f"{a:010d}-{b:010d}.json"


def _audio_kind(tm: media.TimeMap) -> str:
    return "padded" if tm.mode == "padded" else "raw"


def chunk_key(handle: ModelHandle, lang: str, a: int, b: int, pa: int, pb: int, audio_kind: str,
              hotwords: str | None = None, initial_prompt: str | None = None) -> str:
    """묶음 캐시 열쇠: 디코딩에 영향을 주는 것 전부. pass id 는 넣지 않는다(같은 모델 결과를 다른 id 로 다시 쓴다)."""
    return _sha1_json({"decode": DECODE_VERSION, "model": handle.key(),
                       "options": dict(DECODE, hotwords=hotwords, initial_prompt=initial_prompt),
                       "lang": lang, "a": a, "b": b, "pa": pa, "pb": pb, "audio": audio_kind})


def _word_ok(w: Any) -> bool:
    return (isinstance(w, dict) and isinstance(w.get("w"), str)
            and all(isinstance(w.get(k), (int, float)) and not isinstance(w.get(k), bool) for k in ("s", "e"))
            and (w.get("p") is None or isinstance(w.get("p"), (int, float))))


def _read_cached(path: Path, key: str) -> dict | None:
    """열쇠가 맞고 꼴이 온전한 묶음 기록. 없거나 잘렸거나 망가졌으면 None(그 묶음만 다시 받아쓴다)."""
    if not path.exists():
        return None
    try:
        rec = util.read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict) or rec.get("key") != key or not isinstance(rec.get("words"), list):
        return None
    if not all(_word_ok(w) for w in rec["words"]) or not isinstance(rec.get("segs", []), list):
        return None
    return rec


def vad_regions(video_sha1: str, wav, tm: media.TimeMap, backend=None,
                compute: bool = True) -> tuple[list[tuple[int, int]], int]:
    """(말 구간 [(시작 표본, 끝 표본)], wav 표본 수). cache/asr/{sha1}/vad.json 에 있으면 읽고, 없으면 영화 전체 소리를
    올려 찾은 뒤 쓴다(배열은 곧바로 지움).

    알려진 한계: Silero 는 앞 소리의 상태를 이어 가므로 영화 전체를 한 번에 넣으면 조용한 말·소음 속 말의 확률이
    눌린다(추격자 소리 150~175초: 처음부터 넣으면 최대 0.08, 그 자리부터 넣으면 0.88). 놓친 대사는 rescue pass 가 찾는다.
    compute 가 False 면 캐시에 없을 때 찾지 않고 NotCached.
    """
    backend = backend or get_backend()
    n = media.wav_samples(wav)
    key = _sha1_json({"v": VAD_VERSION, "vad": backend.vad_name, "options": VAD_OPTIONS, "n_samples": n,
                      "audio": _audio_kind(tm)})
    p = vad_path(video_sha1)
    if p.exists():
        try:
            rec = util.read_json(p)
        except (OSError, ValueError):
            rec = None
        if isinstance(rec, dict) and rec.get("key") == key:
            return [(int(a), int(b)) for a, b in rec["regions"]], n
    if not compute:
        raise NotCached("말 구간 캐시(vad.json)가 없거나 낡음")
    t = time.time()
    with media.AudioReader(wav) as r:
        audio = r.read_all()
    try:
        regions = _clean_regions(backend.vad(audio), n)
    finally:
        del audio
        gc.collect()
    util.write_json(p, {"key": key, "vad": backend.vad_name, "options": VAD_OPTIONS, "n_samples": n,
                        "regions": [[a, b] for a, b in regions], "seconds": round(time.time() - t, 1)})
    return regions, n


# ------------------------------------------------------------------ 한 영화에 한 프로세스
def _pid_alive(pid: int) -> bool:
    """그 프로세스가 살아 있는가. 알 수 없으면(psutil 없음) 살아 있다고 본다."""
    try:
        import psutil
    except ImportError:
        return True
    try:
        return bool(psutil.pid_exists(pid))
    except Exception:  # noqa: BLE001
        return True


class _FilmLock:
    """cache/asr/{sha1}/.lock: 같은 영화를 두 프로세스가 함께 받아쓰지 못하게 한다(묶음 캐시와 asr.json 이 어긋난다).

    파일에는 프로세스 번호를 적는다. 그 프로세스가 죽어 있으면(강제 종료로 남은 파일) 가져간다.
    """

    def __init__(self, video_sha1: str):
        self.path = lock_path(video_sha1)

    def __enter__(self) -> "_FilmLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    pid = int(self.path.read_text(encoding="utf-8").strip() or 0)
                except (OSError, ValueError):
                    pid = 0
                if pid and _pid_alive(pid):
                    raise RuntimeError(f"다른 프로세스(pid {pid})가 이 영화를 받아쓰는 중: {self.path} "
                                       "(그 프로세스가 끝났는데도 남아 있으면 이 파일을 지우세요)") from None
                try:
                    self.path.unlink()                       # 죽은 프로세스가 남긴 것
                except OSError:
                    pass
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"{os.getpid()}\n")
            return self
        raise RuntimeError(f"잠금 파일을 만들 수 없음: {self.path}")

    def __exit__(self, *exc) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass


# ------------------------------------------------------------------ 전체 덮기 창(rescue)
def energy_envelope(wav) -> list[int]:
    """소리 크기 포락선: 0.1초 틀마다 표본 제곱의 합(정수). 전체 소리를 60초씩 읽는다(메모리 약 8MB).

    정수로만 셈하므로 어느 기계에서나 같은 값이다(창 경계가 묶음 캐시 열쇠에 들어간다). 끝의 반쪽 틀은 온 틀 크기로 맞춘다.
    """
    import numpy as np
    step = int(round(ENV_STEP_S * SR))
    block = step * 600
    out: list[int] = []
    with media.AudioReader(wav) as r:
        for a in range(0, r.n_samples, block):
            x = r.read_int16(a, a + block).astype(np.int64)
            k = len(x) // step
            if k:
                out.extend(int(v) for v in (x[:k * step].reshape(k, step) ** 2).sum(axis=1))
            rest = x[k * step:]
            if len(rest):
                out.append(int((rest ** 2).sum()) * step // len(rest))
    return out


def _quietest(env: list[int], lo: int, hi: int) -> int | None:
    """표본 범위 [lo, hi) 에 온전히 든 틀 가운데 가장 조용한 틀의 가운데 표본. 크기가 같으면 뒤쪽 틀. 없으면 None."""
    step = int(round(ENV_STEP_S * SR))
    k0, k1 = -(-max(lo, 0) // step), min(hi // step, len(env))
    best = None
    for k in range(k1 - 1, k0 - 1, -1):
        if best is None or env[k] < env[best]:
            best = k
    return None if best is None else best * step + step // 2


def cover_sets(env: list[int], n_total: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """소리 전체 [0, n_total) 를 빈틈·겹침 없이 덮는 창 두 벌 (A, B). 범위는 소리 표본 번호.

    벌 A: 앞 경계 + 25초를 기준 경계로 잡고, 그 앞 5초 안에서 가장 조용한 0.1초 틀의 가운데를 경계로 한다(창 20~25초).
          남은 소리가 30초 이하면 마지막 창 하나로 끝낸다(마지막 창 5~30초).
    벌 B: 경계를 벌 A 의 창마다 그 가운데 ±2.5초(짧은 창은 길이의 1/4) 안의 가장 조용한 곳에 둔다. 그래서 첫 창은
          약 12.5초이고, 두 벌의 경계는 늘 엇갈린다(벌 A 경계에서 창 길이의 1/4 이상 떨어짐). 창은 30초를 넘지 않는다.
    따로 25초씩 세면 두 벌의 경계가 같은 조용한 곳에서 만나 그 뒤로 창이 똑같아질 수 있어 벌 A 에 맞춰 둔다.
    """
    W, S, MX = int(WINDOW_S * SR), int(WINDOW_SEARCH_S * SR), int(MAX_CLIP_S * SR)
    if n_total <= 0:
        return [], []
    cuts = [0]
    while n_total - cuts[-1] > MX:
        nominal = cuts[-1] + W
        c = _quietest(env, nominal - S, nominal)
        cuts.append(nominal if c is None else c)
    cuts.append(n_total)
    set_a = list(zip(cuts, cuts[1:]))
    bc = [0]
    for x, y in set_a:
        mid, r = x + (y - x) // 2, min(S // 2, (y - x) // 4)
        lo, hi = max(mid - r, bc[-1] + 1), min(mid + r, bc[-1] + MX)
        c = _quietest(env, lo, hi) if hi > lo else None
        if c is None:
            c = min(max(mid, bc[-1] + 1), bc[-1] + MX)
        if bc[-1] < c < n_total:
            bc.append(c)
    bc.append(n_total)
    return set_a, list(zip(bc, bc[1:]))


def full_windows(video_sha1: str, wav, tm: media.TimeMap) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """전체 덮기 창 두 벌 (A, B). cache/asr/{sha1}/windows.json 에 있으면 읽고 없으면 포락선을 재서 만든다."""
    n = media.wav_samples(wav)
    key = _sha1_json({"v": WINDOW_VERSION, "n_samples": n, "audio": _audio_kind(tm), "window": WINDOW_S,
                      "search": WINDOW_SEARCH_S, "step": ENV_STEP_S, "max": MAX_CLIP_S})
    p = windows_path(video_sha1)
    if p.exists():
        try:
            rec = util.read_json(p)
        except (OSError, ValueError):
            rec = None
        if isinstance(rec, dict) and rec.get("key") == key:
            return [(int(a), int(b)) for a, b in rec["a"]], [(int(a), int(b)) for a, b in rec["b"]]
    set_a, set_b = cover_sets(energy_envelope(wav), n)
    util.write_json(p, {"key": key, "n_samples": n, "a": [[a, b] for a, b in set_a], "b": [[a, b] for a, b in set_b]})
    return set_a, set_b


# ------------------------------------------------------------------ 묶음 기록 -> 낱말
def _first_word_start(w: dict, rec: dict) -> float:
    """묶음 첫 낱말의 시작(소리 시각). 잘라 낸 조각의 시작(pa)에 붙어 있으면 묶음 시작(a) + 0.05초로 당긴다.

    Whisper 의 낱말 정렬은 조각의 0초에서 시작하므로 첫 낱말이 앞 덧댐(0.1초)과 말 구간 여유(0.15초)를 삼킨다.
    당긴 뒤에도 낱말 길이 0.1초는 남기고, 원래 시작보다 앞으로는 가지 않는다.
    """
    s, e = float(w["s"]), float(w["e"])
    a, pa = rec.get("a"), rec.get("pa")
    if a is None or pa is None or s - pa / SR >= FIRST_WORD_GLUE_S:
        return s
    return min(max(s, a / SR + FIRST_WORD_LEAD_S), max(e - FIRST_WORD_MIN_S, s))


def record_words(rec: dict) -> list[dict]:
    """묶음 기록 하나의 낱말 [{"w","s","e","p"}] (소리 시각): 첫 낱말 시작을 보정하고 U+FFFD 가 든 낱말을 뺀다."""
    out = []
    for j, w in enumerate(rec["words"]):
        if BAD_CHAR in w["w"]:
            continue
        s = _first_word_start(w, rec) if j == 0 else float(w["s"])
        out.append({"w": w["w"], "s": s, "e": max(float(w["e"]), s), "p": w["p"]})
    return out


def pass_words(recs: list[dict]) -> list[dict]:
    """묶음 기록 목록 -> pass 낱말(소리 시각). 묶음 차례·묶음 안 차례 그대로."""
    return [w for rec in recs for w in record_words(rec)]


def _seg_of(starts: list[float], t: float) -> int:
    return bisect.bisect_right(starts, t + 1e-4) - 1


def pass_word_stats(recs: list[dict]) -> list[dict]:
    """pass_words(recs) 와 같은 차례·같은 수의 낱말별 세그먼트 통계(asr.json 의 words[i] 와 짝).

    [{"chunk": 묶음 번호, "avg_logprob", "compression_ratio", "temperature", "no_speech_prob"}].
    temperature > 0 인 낱말은 무작위로 다시 뽑은 결과라 확률 p 가 높아도 믿기 어렵다.
    """
    out = []
    for k, rec in enumerate(recs):
        segs = sorted((s for s in rec.get("segs") or [] if "start" in s), key=lambda s: s["start"])
        starts = [s["start"] for s in segs]
        for w in rec["words"]:
            if BAD_CHAR in w["w"]:
                continue
            i = _seg_of(starts, w["s"])
            seg = segs[max(i, 0)] if segs else {}
            out.append({"chunk": k, **{x: seg.get(x) for x in ("avg_logprob", "compression_ratio", "temperature",
                                                                "no_speech_prob")}})
    return out


def speech_holes(regions: list[tuple[int, int]], words: list[dict]) -> list[tuple[float, float]]:
    """말 구간(표본 번호) 안에서 낱말 없이 HOLE_MIN_S 넘게 빈 자리 [(시작, 끝)] (소리 시각 초).

    말 구간 앞뒤 여유(speech_pad)는 뺀다. words 는 소리 시각 낱말(pass_words). 받아쓰기가 말을 조용히 버린 자리일 수도,
    말 구간 찾기가 비명·군중 소리를 말로 본 자리일 수도 있다.
    """
    pad = VAD_OPTIONS["speech_pad_ms"] / 1000.0
    spans = sorted((float(w["s"]), float(w["e"])) for w in words)
    starts = [s for s, _ in spans]
    out = []
    for a, b in regions:
        lo, hi = a / SR + pad, b / SR - pad
        if hi - lo < HOLE_MIN_S:
            continue
        cur = lo
        i = max(bisect.bisect_left(starts, lo) - 1, 0)
        while i < len(spans) and spans[i][0] < hi:
            s, e = spans[i]
            if e > cur:
                if s - cur >= HOLE_MIN_S:
                    out.append((cur, s))
                cur = max(cur, e)
            i += 1
        if hi - cur >= HOLE_MIN_S:
            out.append((cur, hi))
    return out


# ------------------------------------------------------------------ rescue: 안정적인 조각만
def _norm_text(text: str) -> str:
    """글자 닮음을 잴 글: NFC, 대소문자 무시, 글자·숫자만(공백·문장부호 뺌)."""
    return "".join(c for c in util.nfc(util.nfc(text or "").casefold()) if c.isalnum())


def _loop_marks(toks: list[str]) -> list[bool]:
    """되풀이 헛소리 표시: 같은 낱말이 RESCUE_MAX_REPEAT 번 넘게 잇달아 나온 곳, 한 낱말이 같은 글자만
    RESCUE_MAX_REPEAT 번 넘게인 곳. 글자가 없는 낱말(문장부호뿐)은 잇달음을 끊지 않는다."""
    marks = [bool(t) and len(t) > RESCUE_MAX_REPEAT and len(set(t)) == 1 for t in toks]
    idx = [i for i, t in enumerate(toks) if t]
    k = 0
    while k < len(idx):
        j = k
        while j + 1 < len(idx) and toks[idx[j + 1]] == toks[idx[k]]:
            j += 1
        if j - k + 1 > RESCUE_MAX_REPEAT:
            for i in idx[k:j + 1]:
                marks[i] = True
        k = j + 1
    return marks


def _best_span_ratio(text: str, toks: list[str], ok=None) -> float:
    """text 와, toks 의 이어진 범위를 붙인 글 사이 글자 닮음(rapidfuzz ratio, 0~1)의 최댓값.

    벌 B 쪽은 앞뒤 여유 때문에 이웃 낱말이 섞여 들어오므로 통째로 견주지 않고 가장 잘 맞는 범위를 찾는다.
    문턱(RESCUE_MIN_SIM)에 닿을 수 없는 길이의 범위는 재지 않는다. ok(i, j) 를 주면 그것이 참인 범위 toks[i..j] 만 센다.
    """
    from rapidfuzz import fuzz
    n = len(text)
    if not n or not toks:
        return 0.0
    lo, hi = n * RESCUE_MIN_SIM / (2 - RESCUE_MIN_SIM), n * (2 - RESCUE_MIN_SIM) / RESCUE_MIN_SIM
    best = 0.0
    for i in range(len(toks)):
        size = 0
        for j in range(i, len(toks)):
            size += len(toks[j])
            if size > hi:
                break
            if size >= lo:
                r = fuzz.ratio(text, "".join(toks[i:j + 1])) / 100.0
                if r > best and (ok is None or ok(i, j)):
                    best = r
    return best


def _same_time(a_s: float, a_e: float, s: float, e: float) -> bool:
    """낱말 [s, e] 가 조각 [a_s, a_e] 와 시간이 겹치는가. 맞닿기만 한 것은 겹침이 아니다. 길이 0 인 쪽은 다른 쪽 안에
    (끝 포함) 있으면 겹친 것으로 본다(Whisper 는 길이 0 낱말을 가끔 낸다)."""
    if min(a_e, e) - max(a_s, s) > 0:
        return True
    return (s == e and a_s <= s <= a_e) or (a_s == a_e and s <= a_s <= e)


def rescue_fragments(plain_pad: list[tuple[int, int]], recs_a: list[dict], recs_b: list[dict],
                     plain_words: list[dict] | None = None) -> tuple[list[dict], dict]:
    """벌 A 의 낱말을 조각내 조각마다 판정을 붙인다. (조각 목록, 낱말 통계). 모델 없이 캐시 기록만으로 정한다.

    plain_pad: plain 묶음의 덧댄 범위(표본 번호). recs_a, recs_b: 전체 덮기 창 두 벌의 기록.
    plain_words: plain 의 낱말(소리 시각, pass_words). 주면 (나)를 본다.
    벌 A 의 낱말 가운데
      (가) 낱말 가운데 시각이 어느 plain 묶음의 덧댄 범위 안에도 없고,
      (나) plain 이 이미 낱말로 낸 소리가 아니고: 어느 plain 낱말과 겹친 시간이 둘 가운데 짧은 쪽 길이의 절반
          (RESCUE_DUP_SHARE) 이상이거나, 글이 같은 plain 낱말과 맞닿아 있으면(사이 RESCUE_DUP_GAP_S 이하) 같은 소리다.
          전체 덮기 창에서는 쉼 옆의 낱말이 1초 넘게 늘어져 나오는 일이 흔해 가운데 시각이 묶음 밖으로 밀린다
          (추격자: plain '예' 363.42~363.61초, 벌 A '예' 363.41~364.81초. 맞닿은 예: plain 묶음의 첫 낱말 '야'
          4405.61~4405.71초, 벌 A '야' 4405.07~4405.57초). (가)만으로는 묶음 밖 벌 A 낱말 939개 가운데 185개가 plain
          낱말을 한 번 더 넣는 것이었다. plain 이 다르게 들은 낱말도 같은 소리면 뺀다(rescue 는 빈 자리만 채운다).
      (다) 되풀이 헛소리가 아닌 것만(_loop_marks: 한 창 안에서 같은 낱말이 4번 넘게 잇달음)
    세그먼트 경계와 0.6초 넘는 쉼으로 조각낸다. 조각마다 그 시간대(앞뒤 1초 여유)에 가운데 시각이 든 벌 B 낱말의
    이어진 범위 가운데 가장 닮은 것을 찾아, 글자 닮음이 0.6 이상이면 안정적인 조각("kept")이다. 그 범위는
      - 글자의 절반(RESCUE_MIN_COLOC) 이상이 그 조각과 시간이 겹치는 낱말의 것이어야 한다. 같은 소리에 반응한 같은
        헛소리가 두 벌에서 이웃한 시각에 나오는 것을 가린다(추격자 소리 6753초 '다음 영상에서 만나요': 벌 B 는 창 끝,
        벌 A 는 0.8초 뒤). 범위 전체가 조각에 걸치기만 하면 되는 것으로는 모자란다: 이웃한 같은 글에 조각과 겹치는
        다른 낱말 하나가 붙으면 통과한다. 진짜 대사는 두 벌의 낱말 시각이 거의 같아 글자 전부가 겹친다.
      - 벌 A 창과 같은 표본에서 시작하는 벌 B 창의 낱말은 쓰지 않는다. 두 벌의 첫 창은 모두 0 에서 시작해 같은
        헛소리를 같은 시각에 낸다(추격자 맨 앞 '한글자막 by 한효정'). 그래서 영화 맨 앞의 벌 B 첫 창 길이만큼
        (약 12초)은 rescue 로 되찾지 못한다.
    조각: {"s", "e"(소리 시각), "words"(벌 A 낱말, record_words 꼴), "window"(벌 A 창의 시작 표본), "text"(견준 글),
           "sim"(닮음 0~1), "verdict"}. verdict 는 "kept" 또는 버린 까닭:
      "b_silent"    벌 B 가 그 시간대에 아무 낱말도 내지 않음
      "b_differs"   벌 B 가 그 시간대에 다른 글을 냄(조각에 글자가 없는 경우 포함)
      "b_elsewhere" 닮은 글이 근처에 있지만 그 조각과 시간이 겹치지 않음
      "same_start"  닮은 글을 같은 시각에 낸 것이 같은 표본에서 시작한 창뿐
      "habit"       벌 B 와 맞았지만 조각의 이어진 낱말 범위 가운데 버릇 글귀와 닮은(0.6 이상) 것이 있음. 버릇 글귀는
                    7자 이상인 조각 글 가운데 벌 A 의 서로 다른 창 3개 이상에서 똑같이 나오고 그 조각의 절반 넘게가
                    벌 B 와 맞지 않은 것이다(여러 번 나오는 진짜 대사는 두 벌이 대개 맞는다). Whisper 는 말 없는 소리
                    (음악·소음·무음)에 배운 버릇대로 같은 글을 내는데(추격자 벌 A: '다음 영상에서 만나요' 창 38개,
                    '한글자막 by 한효정' 15개, '이 시각 세계였습니다' 8개, '한국국토정보공사' 3개), 드물게 두 벌이 같은
                    시각에 그 글이나 그 변형을 낸다(소리 1952초: 벌 A '다음 영상에서 만나요', 벌 B '다음 시간에 만나요'.
                    6120초: 벌 A '한글자막 by 김태운입니다', 벌 B '한글자막 by …'). 대사가 plain 이 놓친 자리에서만 글자
                    그대로 세 번 되풀이되는 일은 드물다(추격자에서 7자 이상 글 가운데 이 넷 말고는 모두 창 1개).
    통계: a_words, covered((가)에 걸림), duplicate((나)에 걸림), loop (낱말 수), habits {버릇 글귀: 나온 창 수}.
    """
    cover = sorted((pa / SR, pb / SR) for pa, pb in plain_pad)
    cstart = [a for a, _ in cover]

    def covered(t: float) -> bool:
        i = bisect.bisect_right(cstart, t) - 1
        return i >= 0 and t < cover[i][1]

    pw = sorted((float(w["s"]), float(w["e"]), _norm_text(w["w"])) for w in plain_words or [])
    pstart = [x[0] for x in pw]

    def duplicate(s: float, e: float, text: str) -> bool:
        j = bisect.bisect_right(pstart, e + RESCUE_DUP_GAP_S) - 1
        while j >= 0 and pstart[j] > s - MAX_CLIP_S:                 # 낱말은 창(30초)보다 길 수 없다
            ps, pe, pt = pw[j]
            if _same_time(s, e, ps, pe) and min(e, pe) - max(s, ps) >= RESCUE_DUP_SHARE * min(e - s, pe - ps):
                return True
            if text and pt == text and max(s, ps) - min(e, pe) <= RESCUE_DUP_GAP_S:
                return True                                          # 글이 같고 맞닿아 있다
            j -= 1
        return False

    bw = sorted(((w["s"] + w["e"]) / 2, _norm_text(w["w"]), w["s"], w["e"], -1 if rec.get("a") is None else rec["a"])
                for rec in recs_b for w in record_words(rec))
    bw = [x for x in bw if x[1]]
    bmid = [x[0] for x in bw]
    frags: list[dict] = []
    st: Counter = Counter()

    def judge(run: list[dict], start) -> None:
        text = _norm_text("".join(w["w"] for w in run))
        a_s, a_e = run[0]["s"], run[-1]["e"]
        near = bw[bisect.bisect_left(bmid, a_s - RESCUE_MARGIN_S):bisect.bisect_right(bmid, a_e + RESCUE_MARGIN_S)]

        def best(cand: list[tuple], colocated: bool) -> float:
            size = [len(c[1]) for c in cand]
            hit = [_same_time(a_s, a_e, c[2], c[3]) for c in cand]

            def ok(i: int, j: int) -> bool:
                return (sum(size[k] for k in range(i, j + 1) if hit[k])
                        >= RESCUE_MIN_COLOC * sum(size[i:j + 1]))

            return _best_span_ratio(text, [c[1] for c in cand], ok if colocated else None)

        other = [c for c in near if start is None or c[4] != start]     # 같은 자리에서 시작한 창은 증거가 못 된다
        sim = best(other, True) if text else 0.0
        if sim >= RESCUE_MIN_SIM:
            verdict = "kept"
        elif not near:
            verdict = "b_silent"
        elif not text or best(near, False) < RESCUE_MIN_SIM:
            verdict = "b_differs"
        elif best(near, True) >= RESCUE_MIN_SIM:
            verdict = "same_start"
        else:
            verdict = "b_elsewhere"
        frags.append({"s": a_s, "e": a_e, "words": run, "window": start, "text": text, "sim": sim, "verdict": verdict})

    for rec in recs_a:
        starts = sorted(s["start"] for s in rec.get("segs") or [] if "start" in s)
        raw = [w for w in rec["words"] if BAD_CHAR not in w["w"]]
        words = record_words(rec)                                   # raw 와 같은 차례·같은 수
        loops = _loop_marks([_norm_text(w["w"]) for w in words])
        st["a_words"] += len(words)
        run: list[dict] = []
        prev_seg, prev_end = None, None
        for w0, w, loop in zip(raw, words, loops):
            seg = _seg_of(starts, w0["s"])
            if run and (seg != prev_seg or w0["s"] - prev_end > RESCUE_GAP_S):
                judge(run, rec.get("a"))
                run = []
            prev_seg, prev_end = seg, w0["e"]
            if covered((w["s"] + w["e"]) / 2):
                st["covered"] += 1
            elif duplicate(w["s"], w["e"], _norm_text(w["w"])):
                st["duplicate"] += 1
            elif loop:
                st["loop"] += 1
            else:
                run.append(w)
                continue
            if run:                                                 # 빠진 낱말에서 조각을 끊는다
                judge(run, rec.get("a"))
                run = []
        if run:
            judge(run, rec.get("a"))
    habits = _habit_texts(frags)
    for f in frags if habits else ():
        if f["verdict"] == "kept":
            toks = [t for t in (_norm_text(w["w"]) for w in f["words"]) if t]
            if any(_best_span_ratio(h, toks) >= RESCUE_MIN_SIM for h in habits):
                f["verdict"] = "habit"
    return frags, dict({k: st[k] for k in ("a_words", "covered", "duplicate", "loop")}, habits=habits)


def _habit_texts(frags: list[dict]) -> dict[str, int]:
    """버릇 글귀 {글: 나온 창 수}: RESCUE_HABIT_CHARS 자 이상인 조각 글 가운데 서로 다른 창 RESCUE_HABIT_WINDOWS 개
    이상에서 똑같이 나오고, 그 조각의 절반 넘게가 벌 B 와 맞지 않은 것. 여러 번 나오는 진짜 대사(구호·노랫말)는 두 벌이
    대개 맞으므로 걸리지 않는다. 창을 모르는 조각(window 가 None)은 세지 않는다."""
    seen: dict[str, list] = {}
    for f in frags:
        if len(f["text"]) >= RESCUE_HABIT_CHARS and f["window"] is not None:
            wins, n = seen.setdefault(f["text"], [set(), [0, 0]])
            wins.add(f["window"])
            n[f["verdict"] == "kept"] += 1
    return {t: len(wins) for t, (wins, (lost, kept)) in sorted(seen.items())
            if len(wins) >= RESCUE_HABIT_WINDOWS and lost > kept}


def rescue_words(plain_pad: list[tuple[int, int]], recs_a: list[dict], recs_b: list[dict],
                 plain_words: list[dict] | None = None) -> tuple[list[dict], dict]:
    """rescue pass 의 낱말(소리 시각, 시각 순)과 통계: rescue_fragments 가 "kept" 로 본 조각의 낱말.

    통계: a_words, covered, duplicate, loop, unstable, kept (낱말 수), fragments, kept_fragments (조각 수),
          dropped {"b_silent", "b_differs", "b_elsewhere", "same_start", "habit": 버린 조각 수},
          habits {버릇 글귀: 나온 창 수} (까닭과 버릇 글귀는 rescue_fragments).
    """
    frags, st = rescue_fragments(plain_pad, recs_a, recs_b, plain_words)
    kept = [f for f in frags if f["verdict"] == "kept"]
    out = sorted((w for f in kept for w in f["words"]), key=lambda w: (w["s"], w["e"]))
    n_kept = sum(len(f["words"]) for f in kept)
    drops = Counter(f["verdict"] for f in frags if f["verdict"] != "kept")
    return out, dict(st, unstable=sum(len(f["words"]) for f in frags) - n_kept, kept=n_kept, fragments=len(frags),
                     kept_fragments=len(kept), dropped={k: drops[k] for k in RESCUE_DROPS})


# ------------------------------------------------------------------ 받아쓰기·조립
def _model_string(handle: ModelHandle, backend, tm: media.TimeMap) -> str:
    v = VAD_OPTIONS
    dec = f"beam{DECODE['beam_size']}" + ("" if DECODE["condition_on_previous_text"] else " nocond") + " tfallback"
    vad = (f"{backend.vad_name} thr{v['threshold']:g} min{v['min_speech_duration_ms']} "
           f"sil{v['min_silence_duration_ms']} pad{v['speech_pad_ms']} max{v['max_speech_duration_s']:g}")
    return (f"{handle.describe()} | {dec} | {vad} | join{JOIN_GAP_S:.1f} pad{CLIP_PAD_S:.1f} | timemap {tm.mode} | "
            f"firstword +{FIRST_WORD_LEAD_S:g} | sinkprep.asr {VERSION} decode{DECODE_VERSION}")


def _rescue_string(handle: ModelHandle, backend, tm: media.TimeMap) -> str:
    return (f"{_model_string(handle, backend, tm)} | rescue: full windows x2 win{WINDOW_S:g} quiet{WINDOW_SEARCH_S:g} "
            f"staggered novad | outside plain chunks, not plain words dup{RESCUE_DUP_SHARE:g} "
            f"same+gap{RESCUE_DUP_GAP_S:g} | "
            f"frag seg+gap{RESCUE_GAP_S:g} | "
            f"sim{RESCUE_MIN_SIM:g} margin{RESCUE_MARGIN_S:g} coloc{RESCUE_MIN_COLOC:g} otherstart rapidfuzz-ratio "
            f"nfc-alnum | loop>{RESCUE_MAX_REPEAT} habit{RESCUE_HABIT_CHARS}x{RESCUE_HABIT_WINDOWS}")


def transcribe_pass(film_id: str, pid: str, handle: ModelHandle, backend, reader: media.AudioReader,
                    groups: list[tuple[int, int]], pad: list[tuple[int, int]], video_sha1: str, lang: str,
                    audio_kind: str, sub: str | None = None, cache_only: bool = False) -> list[dict]:
    """묶음마다 받아쓴다. 캐시에 있는 묶음은 건너뛴다. 묶음 차례대로 캐시 기록 목록을 돌려준다.

    groups: 묶음 범위, pad: 실제로 잘라 받아쓸 범위(전체 덮기 창은 둘이 같다). sub: 캐시 하위 폴더(chunk_path).
    cache_only 면 캐시에 없는 묶음이 하나라도 있을 때 아무것도 받아쓰지 않고 NotCached.
    기록: key, a, b, pa, pb, lang, words[{"w","s","e","p"}](영화 전체 기준 소리 시각), segs[], seconds.
    """
    n = len(groups)
    keys = [chunk_key(handle, lang, a, b, pa, pb, audio_kind) for (a, b), (pa, pb) in zip(groups, pad)]
    files = [chunk_path(video_sha1, handle.tag, a, b, sub) for a, b in groups]
    recs: list[dict | None] = [_read_cached(f, k) for f, k in zip(files, keys)]
    todo = sum(r is None for r in recs)
    where = f"{handle.tag}/{sub}" if sub else handle.tag
    if cache_only and todo:
        raise NotCached(f"{pid}: 묶음 {n}개 가운데 {todo}개가 캐시에 없음({where})")
    print(f"[asr] {film_id} {pid}: 묶음 {n}개 가운데 캐시 {n - todo}개, 받아쓸 것 {todo}개 ({where})", flush=True)
    t0 = time.time()
    done = 0
    spent = 0.0
    for k, ((a, b), (pa, pb)) in enumerate(zip(groups, pad)):
        if recs[k] is not None:
            continue
        clip = reader.read_samples(pa, pb)
        t = time.time()
        words, segs = backend.transcribe(handle, clip, lang)
        sec = time.time() - t
        off = pa / SR
        rec = {"key": keys[k], "a": a, "b": b, "pa": pa, "pb": pb, "lang": lang,
               "words": [{"w": w["w"], "s": round(off + w["s"], 6), "e": round(off + w["e"], 6), "p": w["p"]}
                         for w in words],
               "segs": [dict(s, start=round(off + s["start"], 6), end=round(off + s["end"], 6))
                        if "start" in s and "end" in s else dict(s) for s in segs],
               "seconds": round(sec, 2)}
        util.write_json(files[k], rec)
        recs[k] = rec
        done += 1
        spent += sec
        left = spent / done * (todo - done)
        print(f"[asr] {film_id} {pid} {k + 1}/{n} {100 * (k + 1) // n}% 경과 {_dur(time.time() - t0)} "
              f"남은 약 {_dur(left)} | 묶음 {(pb - pa) / SR:.1f}초 → {sec:.1f}초", flush=True)
    return [r for r in recs if r is not None]


def assemble(video_sha1: str, lang: str, tm: media.TimeMap, regions: list[tuple[int, int]],
             groups: list[tuple[int, int]], passes: list) -> dict:
    """asr.json 내용. 모든 시각은 영상 시각, 소수 3자리.

    passes 의 항목은 (id, model 문자열, 묶음 기록 목록) 또는 {"id", "model", "context", "words"}.
    앞의 것은 pass_words 로 낱말을 뽑고(첫 낱말 보정, U+FFFD 낱말 버림) context 는 null,
    뒤의 것은 words(소리 시각 낱말 [{"w","s","e","p"}])를 그 차례대로 쓴다.
    """
    def v(sample_or_sec: float, samples: bool = True) -> float:
        return round(tm.to_video(sample_or_sec / SR if samples else sample_or_sec), 3)

    out_passes = []
    for ps in passes:
        if not isinstance(ps, dict):
            pid, model, recs = ps
            ps = {"id": pid, "model": model, "context": None, "words": pass_words(recs)}
        words: list[dict] = []
        for w in ps["words"]:
            s = v(w["s"], samples=False)
            e = max(v(w["e"], samples=False), s)
            words.append({"i": len(words), "w": util.nfc(w["w"]), "s": s, "e": e, "p": w["p"]})
        out_passes.append({"id": ps["id"], "model": ps["model"], "context": ps.get("context"), "words": words})
    return {"schema": SCHEMA, "video_sha1": video_sha1, "lang": lang, "passes": out_passes,
            "vad": [[v(a), max(v(b), v(a))] for a, b in regions],
            "chunks": [{"start": v(a), "end": max(v(b), v(a)), "lang": lang} for a, b in groups]}


def _mean_p(words: list[dict]) -> float | None:
    ps = [w["p"] for w in words if isinstance(w.get("p"), (int, float))]
    return sum(ps) / len(ps) if ps else None


def out_path(film_id: str) -> Path:
    return paths.FILMS / film_id / "work" / "asr.json"


def _identify(backend, name: str, handle_of) -> str:
    """지문에 넣을 모델 신원(저장소@스냅숏, 라이브러리 버전). 장치는 넣지 않는다(--device auto 가 그때그때 달리 풀려도
    끝난 결과를 다시 만들지 않게: 장치는 인자 문자열로 따로 들어간다)."""
    ident = getattr(backend, "identify", None)
    if ident is not None:
        d = ident(name)
    else:
        h = handle_of(name)
        d = {"repo": h.repo, "snapshot": h.snapshot, "versions": h.versions}
    vers = " ".join(f"{k} {v}" for k, v in sorted((d.get("versions") or {}).items()))
    return f"{d.get('repo')}@{d.get('snapshot')} | {vers}"


def process(doc: dict, *, passes: str | None = None, device: str | None = None, threads: int | None = None,
            force: bool = False, rescue: bool | None = None, cache_only: bool = False) -> tuple[str, str]:
    """한 영화(대표 문서 행)를 받아써 films/{film_id}/work/asr.json 을 쓴다. ("ok"|"skip"|"wait", 설명).

    passes, device, rescue 가 None 이면 그 영화의 단계 지문에 적힌 값(앞서 받아쓴 방식)을 이어 쓰고, 지문이 없으면
    DEFAULT_PASSES, DEFAULT_DEVICE, rescue 없음. 끝난 영화를 옵션 없이 다시 돌려도 다른 모델로 갈아 끼우지 않는다.
    rescue: 전체 덮기 창 두 벌을 plain(없으면 첫 pass)의 모델로 받아쓰고 pass "rescue" 를 더한다.
    force 는 지문이 같아도 다시 조립한다. 묶음 캐시는 그대로 쓴다(받아쓰기부터 다시 하려면 cache/asr/{sha1} 을 지운다).
    같은 영화를 다른 프로세스가 받아쓰는 중이면 RuntimeError.
    cache_only(sinkprep all): 모델을 열지 않고 소리 추출·말 구간 찾기도 하지 않는다. 캐시만으로 조립할 수 있으면 조립하고,
    받아쓴 적이 없거나 캐시에 없는 것이 있으면 asr.json 을 건드리지 않고 ("wait", 까닭)을 돌려준다.
    """
    doc_id = doc["doc_id"]
    film_id = doc.get("film_id") or ""
    sha1 = doc.get("video_sha1") or ""
    if not film_id or not doc.get("video_path") or not sha1:
        raise ValueError("영상이 등록되지 않음(python -I -m sinkprep video <영상> --docs <doc_id>)")
    old = util.stage_fp(STAGE, film_id) or {}
    how = f"python -I -m sinkprep asr --docs {doc_id}"
    if cache_only and not old:
        return "wait", f"받아쓴 적이 없음(따로 돌릴 것: {how})"
    passes = passes or old.get("passes") or DEFAULT_PASSES
    device = device or old.get("device") or DEFAULT_DEVICE
    threads = threads or DEFAULT_THREADS
    rescue = bool(old.get("rescue")) if rescue is None else bool(rescue)
    spec = parse_passes(passes)
    if rescue and any(p == RESCUE_ID for p, _ in spec):
        raise ValueError(f"--rescue 를 쓸 때는 --passes 에 id {RESCUE_ID!r} 를 쓸 수 없음")
    video_lang = (doc.get("video_lang") or "").strip()
    lang = whisper_lang(video_lang)
    video = media.resolve_video(doc["video_path"])
    if not video.is_file():
        raise FileNotFoundError(f"영상 파일이 없음: {doc['video_path']}")
    # 이 영상을 처음 받아쓸 때는 sha1 캐시(크기·고친 시각)를 믿지 않고 한 번 다시 읽는다
    if media.file_sha1(video, fresh=old.get("video_sha1") != sha1) != sha1:
        raise ValueError(f"영상 파일이 catalog 의 video_sha1 과 다름: {doc['video_path']} (sinkprep video 로 다시 등록)")

    with _FilmLock(sha1):
        try:
            return _process(doc_id, film_id, sha1, video, video_lang, lang, spec, device, threads, force, rescue, old,
                            cache_only)
        except NotCached as e:
            return "wait", f"캐시만으로는 조립할 수 없음: {e} (따로 돌릴 것: {how})"


def _process(doc_id: str, film_id: str, sha1: str, video: Path, video_lang: str, lang: str,
             spec: list[tuple[str, str]], device: str, threads: int, force: bool, rescue: bool,
             old: dict, cache_only: bool = False) -> tuple[str, str]:
    backend = get_backend()
    if cache_only and not media.audio_path(sha1).exists():
        raise NotCached("소리 캐시(wav)가 없음")
    wav = media.extract_audio(video, sha1)
    tm = media.build_timemap(video, sha1, wav)
    regions, n_total = vad_regions(sha1, wav, tm, backend, compute=not cache_only)
    groups = chunks(regions)
    pad = pads(groups, n_total)
    rescue_model = (dict(spec).get(BASE_ID) or spec[0][1]) if rescue else None
    wins = full_windows(sha1, wav, tm) if rescue else None

    handles: dict[str, ModelHandle] = {}
    state = {"open": None}

    def handle_of(name: str) -> ModelHandle:
        if name not in handles:
            if cache_only:                                   # 꼬리표·열쇠만 필요하다: cuda 도 열어 보지 않는다
                try:
                    handles[name] = backend.load(name, device, threads, probe=False)
                except TypeError:                            # probe 를 받지 않는 백엔드(시험용)
                    handles[name] = backend.load(name, device, threads)
            else:
                handles[name] = backend.load(name, device, threads)
        return handles[name]

    def use(name: str) -> ModelHandle:
        if state["open"] not in (None, name):
            handles[state["open"]].close()                   # 모델은 한 번에 하나만
        state["open"] = name
        return handle_of(name)

    out = out_path(film_id)
    try:
        fp = {"version": VERSION, "decode": DECODE_VERSION, "video_sha1": sha1, "lang": video_lang,
              "passes": ",".join(f"{p}={m}" for p, m in spec), "device": device, "rescue": rescue,
              "models": {m: _identify(backend, m, handle_of) for m in dict.fromkeys(m for _, m in spec)},
              "timemap": _sha1_json(tm.to_json()),
              "chunks": _sha1_json({"regions": regions, "groups": groups, "pad": pad, "vad": backend.vad_name}),
              "windows": _sha1_json(wins) if rescue else None}
        if not force and {k: old.get(k) for k in fp} == fp and out.exists() \
                and util.sha1_file(out) == old.get("out_sha1"):
            return "skip", ""

        speech = sum(b - a for a, b in regions) / SR
        print(f"[asr] {film_id} 소리 {n_total / SR:.0f}초(시각 대응 {tm.mode}, 틈 {tm.gaps}곳 {tm.gap_total:.3f}초), "
              f"말 구간 {len(regions)}개 {speech:.0f}초, 묶음 {len(groups)}개"
              + (f", 전체 덮기 창 {len(wins[0])}+{len(wins[1])}개" if rescue else ""), flush=True)
        done: list[dict] = []
        tags: dict[str, str] = {}
        base: tuple[list[dict], list[dict]] | None = None      # 구멍을 셀 pass 의 (묶음 기록, 낱말)
        rstats: dict | None = None
        with media.AudioReader(wav) as reader:
            for pid, name in spec:
                h = use(name)
                recs = transcribe_pass(film_id, pid, h, backend, reader, groups, pad, sha1, lang, _audio_kind(tm),
                                       cache_only=cache_only)
                words = pass_words(recs)
                done.append({"id": pid, "model": _model_string(h, backend, tm), "context": None, "words": words})
                tags[pid] = h.tag
                if base is None or pid == BASE_ID:
                    base = (recs, words)
            if rescue:
                h = use(rescue_model)
                kind = _audio_kind(tm)
                recs_a = transcribe_pass(film_id, "rescue-a", h, backend, reader, wins[0], wins[0], sha1, lang, kind,
                                         WINDOW_DIR, cache_only)
                recs_b = transcribe_pass(film_id, "rescue-b", h, backend, reader, wins[1], wins[1], sha1, lang, kind,
                                         WINDOW_DIR, cache_only)
                words, rstats = rescue_words(pad, recs_a, recs_b, base[1] if base else None)
                done.append({"id": RESCUE_ID, "model": _rescue_string(h, backend, tm), "context": RESCUE_CONTEXT,
                             "words": words})
                tags[RESCUE_ID] = h.tag
    finally:
        for h in handles.values():
            h.close()

    data = assemble(sha1, video_lang, tm, regions, groups, done)
    util.write_json(out, data)
    probs = []
    notes = []
    for ps in data["passes"]:
        mp = _mean_p(ps["words"])
        notes.append(f"{ps['id']} 낱말 {len(ps['words'])}개 평균 p {'-' if mp is None else f'{mp:.3f}'}")
        if ps["id"] != RESCUE_ID and mp is not None and mp < LOW_CONF:
            probs.append({"level": "film", "doc_id": doc_id, "film_id": film_id, "item_id": None,
                          "problem": "asr_low_conf", "score": round(mp, 3),
                          "detail": f"받아쓰기 {ps['id']} 낱말 {len(ps['words'])}개의 평균 확률이 {LOW_CONF} 미만",
                          "path": f"films/{film_id}/work/asr.json"})
    # 말 구간인데 낱말이 나오지 않은 자리(받아쓰기가 조용히 버린 말일 수 있다)
    empty = sum(not r["words"] for r in base[0]) if base else 0
    holes = speech_holes(regions, base[1]) if base else []
    hole_s = sum(b - a for a, b in holes)
    notes.append(f"말 구간 안 낱말 없는 자리({HOLE_MIN_S:g}초 이상) {len(holes)}곳 {hole_s:.1f}초, 낱말 0개 묶음 {empty}개")
    if empty or (speech > 0 and hole_s / speech > HOLE_WARN_SHARE):
        probs.append({"level": "film", "doc_id": doc_id, "film_id": film_id, "item_id": None, "problem": "other",
                      "kind": "asr_hole", "score": round(hole_s / speech, 3) if speech > 0 else None,
                      "detail": f"받아쓰기: 말 구간 {speech:.0f}초 가운데 낱말 없이 {HOLE_MIN_S:g}초 넘게 빈 자리 "
                                f"{len(holes)}곳 {hole_s:.1f}초, 낱말이 하나도 안 나온 묶음 {empty}개",
                      "path": f"films/{film_id}/work/asr.json"})
    if rstats is not None:
        d = rstats["dropped"]
        notes.append(f"rescue: 벌 A 낱말 {rstats['a_words']}개 가운데 plain 묶음 안 {rstats['covered']}, "
                     f"plain 낱말과 같은 소리 {rstats['duplicate']}, "
                     f"되풀이 {rstats['loop']}, 벌 B 와 안 맞음 {rstats['unstable']}, 남김 {rstats['kept']}"
                     f"(조각 {rstats['kept_fragments']}/{rstats['fragments']}, 버린 조각: 벌 B 가 조용함 {d['b_silent']}, "
                     f"다른 글 {d['b_differs']}, 같은 글이 다른 시각에 {d['b_elsewhere']}, "
                     f"시작이 같은 창뿐 {d['same_start']}, 버릇 글귀 {d['habit']}), "
                     f"버릇 글귀 {len(rstats['habits'])}가지"
                     + ("" if not rstats["habits"] else
                        "(" + ", ".join(f"{t} {n}창" for t, n in sorted(rstats["habits"].items(), key=lambda x: -x[1])[:5])
                        + ")"))
    util.set_problems(STAGE, film_id, probs)
    # tags: pass 마다 묶음 캐시 폴더(비교하지 않는 칸). rescue 의 두 벌은 그 폴더 아래 WINDOW_DIR 에 있다
    util.set_stage_fp(STAGE, film_id, dict(fp, tags=tags, out_sha1=util.sha1_file(out)))
    return "ok", f"말 구간 {len(regions)}개 {speech:.0f}초, 묶음 {len(groups)}개, " + ", ".join(notes)


# ------------------------------------------------------------------ 끝난 받아쓰기의 캐시 읽기(뒤 단계용)
def _guess_tag(video_sha1: str, fp: dict, pid: str) -> str:
    """지문에 꼬리표가 없을 때(0.1.0 이 쓴 지문) 폴더 이름으로 찾는다: {모델 이름}@*, 장치 인자가 cpu·cuda 면 그것."""
    name = dict(parse_passes(fp.get("passes") or "")).get(pid)
    root = paths.ASR_CACHE / video_sha1
    if not name or not root.is_dir():
        raise FileNotFoundError(f"묶음 캐시 폴더를 찾을 수 없음: pass {pid!r} ({root})")
    safe = re.sub(r"[^A-Za-z0-9._@-]", "_", name)
    cands = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name.startswith(safe + "@"))
    if fp.get("device") in ("cpu", "cuda"):
        cands = [c for c in cands if f"-{fp['device']}-" in c] or cands
    if len(cands) != 1:
        raise FileNotFoundError(f"묶음 캐시 폴더를 하나로 정할 수 없음: pass {pid!r}, 후보 {cands} ({root})")
    return cands[0]


def film_records(film_id: str, which: str = BASE_ID) -> list[dict]:
    """끝난 받아쓰기의 묶음 기록을 캐시에서 읽는다(세그먼트 통계 avg_logprob·compression_ratio·temperature 가 여기 있다).

    which: pass id("plain" 등) 또는 "rescue_a"·"rescue_b"(전체 덮기 창 두 벌). 모델을 열지 않는다.
    돌려주는 차례는 asr.json 을 조립한 차례와 같다: pass_words(film_records(f, "plain")) 가 그 pass 의 words 와 짝이고
    pass_word_stats 로 낱말별 통계를 얻는다. 지문이나 기록이 없으면 FileNotFoundError.
    """
    fp = util.stage_fp(STAGE, film_id)
    if not fp or not fp.get("video_sha1"):
        raise FileNotFoundError(f"받아쓰기 지문이 없음: {film_id} (python -I -m sinkprep asr 를 먼저 돌리세요)")
    sha1 = fp["video_sha1"]
    tags = fp.get("tags") or {}
    if which in ("rescue_a", "rescue_b"):
        tag = tags.get(RESCUE_ID)
        if not tag or not windows_path(sha1).exists():
            raise FileNotFoundError(f"rescue 를 돌린 적이 없음: {film_id} (python -I -m sinkprep asr --rescue)")
        ranges = [(int(a), int(b)) for a, b in util.read_json(windows_path(sha1))[which[-1]]]
        sub = WINDOW_DIR
    else:
        tag = tags.get(which) or _guess_tag(sha1, fp, which)
        if not vad_path(sha1).exists():
            raise FileNotFoundError(f"말 구간 캐시가 없음: {vad_path(sha1)}")
        ranges = chunks([(int(a), int(b)) for a, b in util.read_json(vad_path(sha1))["regions"]])
        sub = None
    recs = []
    for a, b in ranges:
        p = chunk_path(sha1, tag, a, b, sub)
        if not p.exists():
            raise FileNotFoundError(f"묶음 기록이 없음: {p}")
        recs.append(util.read_json(p))
    return recs


def film_rescue_fragments(film_id: str, base: str = "a") -> list[dict]:
    """끝난 rescue 의 조각 판정을 캐시에서 다시 낸다(rescue_fragments 의 조각 목록, 소리 시각). 모델을 열지 않는다.

    base "a": 벌 A 의 조각을 벌 B 와 견준 것. verdict 가 "kept" 인 조각의 낱말이 asr.json 의 pass rescue 다.
    base "b": 거꾸로 벌 B 의 조각을 벌 A 와 견준 것. asr.json 에는 들어가지 않는다. 벌 A 창이 망가져 벌 B 만 맞게
              받아쓴 대사는 여기서 verdict 가 "b_differs"·"b_silent" 인 조각으로 나온다(추격자 소리 6427~6437초
              '오빠, 오빠 저예요 …'). 같은 꼴로 헛소리도 나오므로 대본 같은 다른 근거가 있을 때만 쓴다.
    지문이나 기록이 없으면 FileNotFoundError.
    """
    return _film_rescue(film_id, base)[0]


def film_habits(film_id: str) -> dict[str, int]:
    """끝난 rescue 가 찾은 버릇 글귀 {글(문장부호·공백 뺀 것): 나온 창 수}: 이 영화에서 모델이 말 없는 소리에 되풀이해 낸 글.

    rescue 조각만 거른다. plain 같은 다른 pass 의 낱말에도 같은 글이 있을 수 있다(추격자 plain 묶음 하나가 '한글자막 by
    한효정'): 뒤 단계가 그런 낱말을 가릴 때 쓴다. 지문이나 기록이 없으면 FileNotFoundError.
    """
    return dict(_film_rescue(film_id, "a")[1]["habits"])


def _film_rescue(film_id: str, base: str) -> tuple[list[dict], dict]:
    if base not in ("a", "b"):
        raise ValueError(f"base 는 'a' 또는 'b': {base!r}")
    recs = {x: film_records(film_id, f"rescue_{x}") for x in ("a", "b")}
    fp = util.stage_fp(STAGE, film_id)
    ids = [p for p, _ in parse_passes(fp.get("passes") or "")]
    plain = pass_words(film_records(film_id, BASE_ID if BASE_ID in ids else ids[0]))   # process 가 쓴 것과 같은 pass
    vad = util.read_json(vad_path(fp["video_sha1"]))
    groups = chunks([(int(a), int(b)) for a, b in vad["regions"]])
    pad = pads(groups, int(vad["n_samples"]))
    other = "b" if base == "a" else "a"
    return rescue_fragments(pad, recs[base], recs[other], plain)


# ------------------------------------------------------------------ 명령
def select_docs(args) -> list[dict]:
    """영상이 등록된 영화의 대표 문서 행(영화마다 하나).

    --docs 로 준 문서가 대표가 아니어도 그 영화에 영상이 등록돼 있으면 그 영화의 대표 문서를 고른다
    (video 명령은 그 영화의 아무 문서 id 나 받으므로 같은 id 로 asr 를 돌릴 수 있어야 한다).
    """
    from . import catalog
    rows = catalog.load()
    docs = [d for d in rows
            if d.get("status") != "excluded" and d.get("video_path") and d.get("is_primary") == "true"]
    if getattr(args, "docs", None):
        want = set(args.docs.split(","))
        films = {r.get("film_id") for r in rows
                 if r["doc_id"] in want and r.get("film_id") and r.get("status") != "excluded"}
        docs = [d for d in docs if d["doc_id"] in want or d.get("film_id") in films]
    if getattr(args, "limit", None):
        docs = docs[: args.limit]
    return docs


def _explain_docs(args, docs: list[dict]) -> None:
    """--docs 로 준 문서가 그대로 대상이 되지 않은 까닭을 알린다."""
    if not getattr(args, "docs", None):
        return
    from . import catalog
    rows = {r["doc_id"]: r for r in catalog.load()}
    by_film = {d.get("film_id"): d["doc_id"] for d in docs}
    picked = {d["doc_id"] for d in docs}
    for doc_id in args.docs.split(","):
        if doc_id in picked:
            continue
        r = rows.get(doc_id)
        if r is None:
            why = "catalog 에 없는 doc_id"
        elif r.get("status") == "excluded":
            why = f"제외된 문서({r.get('note', '')})"
        elif r.get("film_id") in by_film:
            why = f"대표 문서가 아님: 같은 영화({r['film_id']})의 대표 문서 {by_film[r['film_id']]} 로 돌림"
        elif not r.get("video_path"):
            why = "영상이 등록되지 않음(python -I -m sinkprep video <영상> --docs " + doc_id + ")"
        else:
            why = f"그 영화({r.get('film_id')})에 대표 문서(is_primary)가 없음"
        print(f"  {doc_id}: {why}", flush=True)


def run(args) -> int:
    docs = select_docs(args)
    passes = getattr(args, "passes", None) or None
    device = getattr(args, "device", None) or None
    threads = getattr(args, "threads", None) or None
    rescue = getattr(args, "rescue", None)
    force = bool(getattr(args, "force", False))
    cache_only = getattr(args, "cmd", None) == "all"         # all 안에서는 모델을 부르지 않는다(ALL_CACHE_ONLY)
    t0 = time.time()
    counts: Counter = Counter()
    _explain_docs(args, docs)
    if not docs:
        print("asr: 대상 없음(영상이 등록된 대표 문서가 없음: python -I -m sinkprep video <영상> --docs <doc_id>)",
              flush=True)
        return 0
    for k, d in enumerate(docs, 1):
        key = d.get("film_id") or d["doc_id"]
        try:
            st, msg = process(d, passes=passes, device=device, threads=threads, force=force, rescue=rescue,
                              cache_only=cache_only)
        except Exception as e:  # noqa: BLE001
            st, msg = "error", ""
            print(f"  ! {key}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        counts[st] += 1
        if st != "skip":
            print(f"[{k}/{len(docs)}] {key} {st}" + (f": {msg}" if msg else ""), flush=True)
    print(f"{STAGE}: {dict(counts)} ({time.time() - t0:.0f}초)", flush=True)
    return 1 if (counts["fail"] or counts["error"]) else 0
