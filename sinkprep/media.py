"""영상·소리 도구: ffmpeg/ffprobe 호출, 소리 추출, 소리 시각 ↔ 영상 시각 대응표, wav 조각 읽기.

시각이 두 가지다.
- 소리 시각: cache/audio/{video_sha1}.wav 의 표본 번호 ÷ 16000.
- 영상 시각: 컨테이너 PTS − 컨테이너 시작 시각(format start_time). 플레이어·자막·ffmpeg -ss 가 쓰는 시각이고
  catalog 의 video_duration(컨테이너 길이)과 같은 기준이다. asr.json·final.json 의 시각은 모두 이것이다.
  mp4 는 시작 시각이 0 이라 PTS 그대로지만 mpegts 는 보통 1.4초쯤에서 시작한다.

조각을 이어 붙인 mp4 는 조각 경계마다 PTS 가 조금씩 건너뛴다(추격자: 750곳, 합 42초). 소리는 틈을 메우지 않고
그대로 꺼낸다(무음을 끼우면 낱말 한가운데 구멍이 생긴다). 그러면 소리 시각이 영상 시각보다 점점 일러지므로,
패킷 표로 만든 대응표(TimeMap)로 둘을 잇는다. 대응표를 못 믿을 때만 틈을 무음으로 메워 다시 뽑는다(mode "padded").
메운 소리는 다른 이름으로 뽑아 길이를 확인한 뒤에만 wav 를 바꾼다(실패하면 wav 는 그대로다).

경로는 호출 시점에 paths.X 를 읽는다(시험이 바꿔 끼운다).
"""
from __future__ import annotations

import bisect
import json
import os
import shutil
import subprocess
import wave
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

from . import paths, util

SR = 16000
VERSION = "0.2.0"         # 대응표 규칙이 바뀌면 올린다(MADE 가 달라져 대응표를 다시 만든다)
MADE = f"sinkprep.media {VERSION}"
JUMP_S = 0.002            # (영상 시각 − 소리 시각)이 이만큼 넘게 바뀌면 새 구간
CHECK_S = 0.1             # 대응표 검증 허용 오차(초)
LOOSE_S = 1.0             # 소리 끝 시각을 컨테이너 길이로만 알 때의 허용 오차(소리가 영상보다 조금 일찍 끝날 수 있다)
PAD_FILTER = "aresample=async=1:min_hard_comp=0.01:first_pts=0"   # 틈을 무음으로 메워 PTS 에 맞춤
FRAME_PROBE = 40          # 패킷 길이를 재려고 디코딩해 보는 프레임 수
T_PROBE = 300.0           # ffprobe(정보·프레임 표본) 시간 제한(초)
T_PACKETS = 1800.0        # ffprobe(패킷 표) 시간 제한
T_EXTRACT = 7200.0        # ffmpeg(소리 추출) 시간 제한


class MediaError(RuntimeError):
    """ffmpeg·ffprobe 가 없거나 실패함, 영상에서 소리·시각을 얻을 수 없음."""


# ------------------------------------------------------------------ 도구 찾기·실행
def _tool(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise MediaError(f"{name} 를 찾을 수 없습니다(PATH 에 ffmpeg 를 넣으세요)")
    return exe


def ffmpeg() -> str:
    return _tool("ffmpeg")


def ffprobe() -> str:
    return _tool("ffprobe")


def _run(cmd: list[str], what: str, timeout: float | None = None) -> str:
    """외부 프로그램을 돌려 표준 출력을 돌려준다. 실패하거나 timeout 초 안에 끝나지 않으면 MediaError."""
    try:
        r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise MediaError(f"{what} 이(가) {timeout:.0f}초 안에 끝나지 않음") from e
    except OSError as e:
        raise MediaError(f"{what} 실행 실패: {e}") from e
    if r.returncode != 0:
        tail = r.stderr.decode("utf-8", errors="replace").strip()[-400:]
        raise MediaError(f"{what} 실패(rc={r.returncode}): {tail}")
    return r.stdout.decode("utf-8", errors="replace")


# ------------------------------------------------------------------ 경로
def audio_path(video_sha1: str) -> Path:
    return paths.AUDIO / f"{video_sha1}.wav"


def timemap_path(video_sha1: str) -> Path:
    return paths.AUDIO / f"{video_sha1}.timemap.json"


def resolve_video(video_path: str) -> Path:
    """catalog 의 video_path -> 실제 경로. 상대 경로는 paths.ROOT 기준."""
    p = Path(video_path)
    return p if p.is_absolute() else paths.ROOT / p


# ------------------------------------------------------------------ 파일 sha1(크기·고친 시각으로 캐시)
def _sha1_key(p: Path) -> str:
    return util.sha1_bytes(str(p).encode("utf-8"))[:16]


def file_sha1(path, *, fresh: bool = False) -> str:
    """파일 sha1. (경로, 크기, 고친 시각)이 앞서 본 것과 같으면 그때 값을 쓴다(2GB 넘는 영상을 매번 읽지 않게).

    한계: 크기와 고친 시각을 그대로 둔 채 내용만 바꾼 파일은 알아채지 못한다. fresh 면 캐시를 보지 않고 다시 읽는다
    (asr 가 새 영상을 처음 받아쓸 때 한 번 그렇게 한다).
    """
    p = Path(path).resolve()
    st = p.stat()
    old = {} if fresh else (util.stage_fp("filesha1", _sha1_key(p)) or {})
    if old.get("path") == str(p) and old.get("size") == st.st_size and old.get("mtime_ns") == st.st_mtime_ns \
            and old.get("sha1"):
        return old["sha1"]
    h = util.sha1_file(p)
    remember_sha1(p, h)
    return h


def remember_sha1(path, sha1: str) -> None:
    """이미 아는 sha1 을 캐시에 적는다(방금 해시한 파일의 하드링크 등)."""
    p = Path(path).resolve()
    st = p.stat()
    util.set_stage_fp("filesha1", _sha1_key(p),
                      {"path": str(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha1": sha1})


# ------------------------------------------------------------------ ffprobe
def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def _i(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _fps(v: Any) -> float | None:
    try:
        fr = Fraction(str(v))
    except (ValueError, ZeroDivisionError):
        return None
    return float(fr) if fr > 0 else None


def probe(video) -> dict:
    """영상 정보: {"duration", "start_time", "audio": 첫 소리 스트림 | None, "video": 첫 영상 스트림 | None}.

    duration 은 컨테이너(format) 길이(초), start_time 은 컨테이너 시작 시각(영상 시각의 0 점).
    audio: index, codec, sample_rate, channels, time_base, start_time, duration, nb_frames.
    video: index, codec, width, height, fps, time_base, start_time, duration, nb_frames.
    ffprobe 가 값을 주지 않은 칸은 None. 스트림의 start_time 은 컨테이너 PTS 그대로다(시작 시각을 빼지 않은 값).
    """
    out = _run([ffprobe(), "-v", "error", "-show_entries",
                "format=duration,start_time:stream=index,codec_type,codec_name,sample_rate,channels,time_base,"
                "start_time,duration,nb_frames,avg_frame_rate,width,height",
                "-of", "json", str(video)], "ffprobe", T_PROBE)
    try:
        data = json.loads(out)
    except ValueError as e:
        raise MediaError(f"ffprobe 출력을 읽을 수 없음: {e}") from e
    fmt = data.get("format") or {}
    info: dict = {"duration": _f(fmt.get("duration")), "start_time": _f(fmt.get("start_time")),
                  "audio": None, "video": None}
    for st in data.get("streams") or []:
        kind = st.get("codec_type")
        base = {"index": _i(st.get("index")), "codec": st.get("codec_name"), "time_base": st.get("time_base"),
                "start_time": _f(st.get("start_time")), "duration": _f(st.get("duration")),
                "nb_frames": _i(st.get("nb_frames"))}
        if kind == "audio" and info["audio"] is None:
            info["audio"] = dict(base, sample_rate=_i(st.get("sample_rate")), channels=_i(st.get("channels")))
        elif kind == "video" and info["video"] is None:
            info["video"] = dict(base, width=_i(st.get("width")), height=_i(st.get("height")),
                                 fps=_fps(st.get("avg_frame_rate")))
    return info


def frame_seconds(video) -> float | None:
    """첫 소리 스트림의 프레임 길이(초): 앞쪽 FRAME_PROBE 프레임을 디코딩해 본 표본 수의 최빈값 ÷ 표본율. 못 재면 None.

    mkv 처럼 timestamp 가 1ms 단위인 컨테이너는 패킷 길이가 21ms 로 반올림돼 있어(실제 21.333ms) 패킷 표의 길이로는
    소리 시각을 셀 수 없다. 디코더가 내놓는 표본 수는 컨테이너와 상관없이 정확하다.
    """
    try:
        out = _run([ffprobe(), "-v", "error", "-select_streams", "a:0", "-show_entries",
                    "stream=sample_rate:frame=nb_samples", "-read_intervals", f"%+#{FRAME_PROBE}",
                    "-of", "json", str(video)], "ffprobe(프레임 표본)", T_PROBE)
        data = json.loads(out)
    except (MediaError, ValueError):
        return None
    rate = next((_i(st.get("sample_rate")) for st in data.get("streams") or [] if _i(st.get("sample_rate"))), None)
    sizes = Counter(n for n in (_i(fr.get("nb_samples")) for fr in data.get("frames") or []) if n and n > 0)
    if not rate or not sizes:
        return None
    return sizes.most_common(1)[0][0] / rate


def packet_table(video, time_base: str | None = None) -> tuple[list[float], float]:
    """첫 소리 스트림의 패킷 표: (패킷마다 PTS 초, 패킷 길이 초). PTS 는 컨테이너 값 그대로다(시작 시각을 빼지 않음).

    ffprobe 의 pts_time·duration_time 은 소수 6자리라 35만 패킷에서 0.1초 넘게 쌓인다. 정수 pts 에 스트림
    time_base 를 곱해 쓴다. 패킷 길이는 디코딩해 본 프레임 길이(frame_seconds)이고, 못 재면 패킷 표 duration 의 최빈값.
    PTS 가 없는 패킷이 있거나 길이를 알 길이 없으면 MediaError.
    """
    if time_base is None:
        a = probe(video)["audio"]
        if a is None:
            raise MediaError("소리 스트림이 없음")
        time_base = a.get("time_base")
    try:
        tb = Fraction(str(time_base))
    except (ValueError, ZeroDivisionError) as e:
        raise MediaError(f"소리 스트림 time_base 를 읽을 수 없음: {time_base!r}") from e
    if tb <= 0:
        raise MediaError(f"소리 스트림 time_base 가 이상함: {time_base!r}")
    out = _run([ffprobe(), "-v", "error", "-select_streams", "a:0", "-show_entries", "packet=pts,duration",
                "-of", "csv=p=0", str(video)], "ffprobe(패킷 표)", T_PACKETS)
    num, den = tb.numerator, tb.denominator
    pts: list[float] = []
    durs: Counter = Counter()
    for ln in out.splitlines():
        parts = ln.strip().split(",")
        if not parts[0]:
            continue
        try:
            p = int(parts[0])
        except ValueError as e:
            raise MediaError(f"패킷 표에 PTS 가 없는 줄: {ln.strip()!r}") from e
        pts.append(p * num / den)
        d = _i(parts[1]) if len(parts) > 1 else None        # mkv 는 길이가 N/A 인 패킷이 있다
        if d and d > 0:
            durs[d] += 1
    if not pts:
        raise MediaError("소리 패킷이 없음")
    frame = frame_seconds(video)
    if frame is None:
        if not durs:
            raise MediaError("소리 패킷 길이를 알 수 없음")
        frame = durs.most_common(1)[0][0] * num / den
    if frame <= 0:
        raise MediaError("소리 패킷 길이가 0")
    return pts, frame


# ------------------------------------------------------------------ 소리 추출
def extract_audio(video, video_sha1: str, *, padded: bool = False, force: bool = False) -> Path:
    """영상의 첫 소리 스트림 -> cache/audio/{video_sha1}.wav (16 kHz 모노 16비트). 있으면 건너뜀.

    기본은 틈을 메우지 않고 표본을 그대로 꺼낸다. padded 면 PTS 틈을 무음으로 메워(wav 시각 = 영상 시각)
    늘 다시 뽑는다. .part 로 쓰고 끝나면 이름을 바꾼다(중간에 죽어도 반쪽 wav 가 남지 않음).
    build_timemap 은 메운 소리를 이 함수로 뽑지 않는다(길이를 확인하기 전에 wav 를 바꾸지 않으려고).
    """
    out = audio_path(video_sha1)
    if out.exists() and not (padded or force):
        return out
    part = out.with_name(f"{out.name}.{os.getpid()}.part")     # 프로세스마다 다른 이름: 둘이 함께 뽑아도 서로 덮어쓰지 않는다
    try:
        _extract_to(video, part, padded=padded)
        try:
            os.replace(part, out)
        except OSError as e:
            raise MediaError(f"wav 를 바꿀 수 없음(다른 프로그램이 열어 두었는지 확인): {out}: {e}") from e
    finally:
        _unlink(part)
    return out


def _extract_to(video, dst: Path, *, padded: bool = False) -> None:
    """영상의 첫 소리 스트림을 dst 에 16 kHz 모노 16비트 wav 로 쓴다(이름을 바꾸지 않고 그 경로에 바로)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(video),
           "-map", "0:a:0", "-vn"]
    if padded:
        cmd += ["-af", PAD_FILTER]
    cmd += ["-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", "-f", "wav", str(dst)]
    _run(cmd, "ffmpeg(소리 추출)", T_EXTRACT)


def _unlink(p: Path) -> None:
    try:
        Path(p).unlink()
    except OSError:
        pass


def _open_wav(path) -> wave.Wave_read:
    try:
        w = wave.open(str(path), "rb")
    except (OSError, wave.Error, EOFError) as e:
        raise MediaError(f"wav 를 열 수 없음: {path}: {e}") from e
    if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != SR:
        got = (w.getnchannels(), w.getsampwidth() * 8, w.getframerate())
        w.close()
        raise MediaError(f"16 kHz 모노 16비트 wav 가 아님: {path} (채널 {got[0]}, {got[1]}비트, {got[2]} Hz)")
    return w


def wav_samples(path) -> int:
    """wav 의 표본 수(머리만 읽음)."""
    w = _open_wav(path)
    try:
        return w.getnframes()
    finally:
        w.close()


class AudioReader:
    """cache/audio 의 wav(16 kHz 모노 16비트)를 조각으로 읽는다. 시각은 소리 시각(표본 번호 ÷ 16000).

    with AudioReader(wav) as r: clip = r.read(12.0, 14.5)
    """

    def __init__(self, path):
        self.path = Path(path)
        self._w = _open_wav(self.path)
        self.sr = SR
        self.n_samples = self._w.getnframes()

    @property
    def duration(self) -> float:
        return self.n_samples / SR

    def read_int16(self, a: int, b: int):
        """표본 [a, b) -> int16 배열(파일 값 그대로). 범위는 파일 안으로 자른다."""
        import numpy as np
        a = min(max(int(a), 0), self.n_samples)
        b = min(max(int(b), a), self.n_samples)
        if b <= a:
            return np.zeros(0, dtype="<i2")
        self._w.setpos(a)
        return np.frombuffer(self._w.readframes(b - a), dtype="<i2")

    def read_samples(self, a: int, b: int):
        """표본 [a, b) -> float32 배열(-1~1). 범위는 파일 안으로 자른다."""
        import numpy as np
        return self.read_int16(a, b).astype(np.float32) / 32768.0

    def read(self, a_sec: float, b_sec: float):
        """소리 시각 [a_sec, b_sec) 초 -> float32 배열."""
        return self.read_samples(round(a_sec * SR), round(b_sec * SR))

    def read_all(self):
        return self.read_samples(0, self.n_samples)

    def close(self) -> None:
        self._w.close()

    def __enter__(self) -> "AudioReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ------------------------------------------------------------------ 소리 시각 ↔ 영상 시각
class TimeMap:
    """소리 시각(wav) ↔ 영상 시각(PTS) 대응표.

    segments 는 [[a_start, v_start], ...] (소리 시각 오름차순). 구간 안에서는 영상 시각 = 소리 시각 + (v_start − a_start).
    구간 길이는 다음 구간의 a_start 까지(마지막은 audio_duration 까지). 구간 사이의 영상 시각 틈에는 소리가 없다.
    mode: "identity"(틈 없음, 두 시각이 같음), "map"(틈·밀림 있음), "padded"(틈을 무음으로 메워 뽑아 두 시각이 같음).
    값은 소수 6자리로 맞춰 둔다(파일로 썼다 읽어도 같은 결과).
    """

    def __init__(self, video_sha1: str, mode: str, frame: float, segments: Sequence[Sequence[float]],
                 audio_duration: float, made: str = MADE):
        if not segments:
            raise ValueError("segments 가 비어 있음")
        self.video_sha1 = video_sha1
        self.mode = mode
        self.frame = round(float(frame), 9)
        self.segments = [[round(float(a), 6), round(float(v), 6)] for a, v in segments]
        self.audio_duration = round(float(audio_duration), 6)
        self.made = made
        self._a = [a for a, _ in self.segments]
        self._v = [v for _, v in self.segments]
        self._off = [v - a for a, v in self.segments]
        if any(y <= x for x, y in zip(self._a, self._a[1:])):
            raise ValueError("segments 의 소리 시각이 오름차순이 아님")
        if any(y < x - 1e-9 for x, y in zip(self._off, self._off[1:])):
            raise ValueError("segments 의 영상 시각이 뒤로 감(겹침)")

    # -- 변환
    def to_video(self, t: float) -> float:
        """소리 시각 -> 영상 시각(0 이상). 범위 밖은 [0, audio_duration] 으로 자른 뒤 옮긴다.

        소리가 영상 시각 0 보다 앞에서 시작하는 영상(첫 구간의 영상 시각이 음수)은 그 앞부분을 0 으로 자른다.
        """
        t = min(max(float(t), 0.0), self.audio_duration)
        i = max(bisect.bisect_right(self._a, t) - 1, 0)
        return max(t + self._off[i], 0.0)

    def to_audio(self, T: float) -> float:
        """영상 시각 -> 소리 시각. 틈 안의 시각은 앞 구간 끝으로, 범위 밖은 [0, audio_duration] 으로 자른다."""
        T = float(T)
        i = bisect.bisect_right(self._v, T) - 1
        if i < 0:
            return 0.0
        end = self._a[i + 1] if i + 1 < len(self._a) else self.audio_duration
        return min(max(T - self._off[i], 0.0), end)

    # -- 요약
    @property
    def video_duration(self) -> float:
        """소리 끝의 영상 시각."""
        return round(self.audio_duration + self._off[-1], 6)

    @property
    def gaps(self) -> int:
        """영상 시각이 건너뛰는 곳 수."""
        return sum(1 for x, y in zip(self._off, self._off[1:]) if y - x > 1e-9)

    @property
    def gap_total(self) -> float:
        """건너뛴 시간의 합(초)."""
        return round(sum(max(y - x, 0.0) for x, y in zip(self._off, self._off[1:])), 6)

    # -- 파일
    def to_json(self) -> dict:
        return {"video_sha1": self.video_sha1, "mode": self.mode, "frame": self.frame, "segments": self.segments,
                "audio_duration": self.audio_duration, "video_duration": self.video_duration,
                "gaps": self.gaps, "gap_total": self.gap_total, "made": self.made}

    @classmethod
    def from_json(cls, d: dict) -> "TimeMap":
        return cls(d["video_sha1"], d["mode"], d["frame"], d["segments"], d["audio_duration"], d.get("made", ""))

    @classmethod
    def identity(cls, video_sha1: str, audio_duration: float, mode: str = "identity", frame: float = 0.0) -> "TimeMap":
        return cls(video_sha1, mode, frame, [[0.0, 0.0]], audio_duration)

    @classmethod
    def from_packets(cls, video_sha1: str, pts: Sequence[float], frame: float, audio_duration: float,
                     *, jump: float = JUMP_S, start: float = 0.0) -> "TimeMap":
        """패킷 표 -> 대응표. 패킷 k 의 소리 시각 = k × frame − trim, 영상 시각 = pts[k] − start.

        start: 컨테이너 시작 시각(format start_time). 영상 시각의 0 점이다.
        trim: 첫 PTS 가 음수면(코덱 준비 표본) 디코더가 그만큼 버렸는지를 wav 길이로 가린다.
        (영상 시각 − 소리 시각)이 지금 구간 값에서 jump 넘게 바뀌는 패킷마다 새 구간.
        PTS 가 뒤로 가면(겹침) ValueError.
        """
        trim = packet_trim(pts, frame, audio_duration)
        segs: list[list[float]] = []
        off0 = 0.0
        for k, p in enumerate(pts):
            a = k * frame - trim
            v = p - start
            off = v - a
            if not segs or abs(off - off0) > jump:
                segs.append([a, v])
                off0 = off
        a0, v0 = segs[0]
        segs[0] = [0.0, v0 - a0]                       # 첫 구간은 소리 시각 0 에서 시작(같은 밀림)
        mode = "identity" if len(segs) == 1 and abs(segs[0][1]) < 0.0005 else "map"
        return cls(video_sha1, mode, frame, segs, audio_duration)


def packet_trim(pts: Sequence[float], frame: float, audio_duration: float) -> float:
    """디코더가 앞에서 버린 길이(초): 첫 PTS 가 음수이고 그만큼 뺀 쪽이 wav 길이와 더 맞으면 −pts[0], 아니면 0."""
    if not pts or pts[0] >= 0:
        return 0.0
    total = len(pts) * frame
    cand = -pts[0]
    return cand if abs(total - cand - audio_duration) < abs(total - audio_duration) else 0.0


def _near_container_end(t: float, container_duration: float | None) -> bool:
    """t 가 컨테이너 길이를 넘지 않고(0.1초 여유) 1초 넘게 모자라지도 않는가(소리는 영상보다 조금 일찍 끝날 수 있다)."""
    return bool(container_duration) and -LOOSE_S <= t - container_duration <= CHECK_S


def check_packets(pts: Sequence[float], frame: float, audio_duration: float,
                  stream_duration: float | None, start: float = 0.0,
                  container_duration: float | None = None) -> tuple[TimeMap | None, str]:
    """패킷 표로 대응표를 만들고 검증한다. (대응표, "") 또는 (None, 못 쓰는 까닭).

    stream_duration: 소리 끝의 영상 시각(ffprobe 가 말한 소리 스트림 시작 시각 + 길이 − start). 모르면 None.
    start: 컨테이너 시작 시각(format start_time). pts 에서 이만큼 뺀 것이 영상 시각이다.
    container_duration: 컨테이너 길이(format duration). mpegts 처럼 스트림 길이가 어림값인 컨테이너에서 쓴다.
    검증 1: |패킷 수 × 패킷 길이 − wav 길이| ≤ 0.1초.
    검증 2: 소리 끝의 영상 시각이 스트림 끝 시각과 0.1초 안(알 때). 다르더라도 컨테이너 길이와 맞으면
            (넘지 않고 1초 넘게 모자라지 않음) 받아들인다.
    패킷마다 표본 수가 다른 코덱이나 PTS 가 겹치는 영상은 여기서 걸린다.
    """
    trim = packet_trim(pts, frame, audio_duration)
    total = len(pts) * frame - trim
    if abs(total - audio_duration) > CHECK_S:
        return None, f"패킷 {len(pts)}개 × {frame:.6f}초 = {total:.3f}초 인데 wav 는 {audio_duration:.3f}초"
    try:
        tm = TimeMap.from_packets("", pts, frame, audio_duration, start=start)
    except ValueError as e:
        return None, str(e)
    if stream_duration is not None and abs(tm.video_duration - stream_duration) > CHECK_S \
            and not _near_container_end(tm.video_duration, container_duration):
        return None, f"소리 끝의 영상 시각 {tm.video_duration:.3f}초 가 소리 스트림 길이 {stream_duration:.3f}초 와 다름"
    return tm, ""


def _read_timemap(p: Path) -> TimeMap | None:
    if not p.exists():
        return None
    try:
        return TimeMap.from_json(util.read_json(p))
    except (ValueError, KeyError, TypeError, OSError):
        return None


def _fits(tm: TimeMap | None, dur: float) -> bool:
    """대응표가 그 길이의 wav 와 짝인가."""
    return tm is not None and abs(tm.audio_duration - dur) < 0.0005


def build_timemap(video, video_sha1: str, wav=None, *, force: bool = False) -> TimeMap:
    """대응표를 만들어 cache/audio/{video_sha1}.timemap.json 에 쓴다. 이미 있고 wav 와 맞으면 읽기만 한다.

    wav 는 늘 audio_path(video_sha1) 이다(인자는 예전 호출 꼴을 받으려는 것이고 다른 경로면 ValueError).
    패킷 표 검증에 실패하면 틈을 무음으로 메워 wav 를 다시 뽑고(같은 경로를 바꿔 씀) 항등 대응표(mode "padded")를 쓴다.
    그러니 이 함수를 부른 뒤에 wav 를 연다. 메운 소리는 다른 이름으로 뽑아 길이를 확인한 뒤에만 wav 와 바꾸고,
    확인에 실패하면 MediaError 를 내고 wav 는 건드리지 않는다.
    디스크의 wav 가 메워 뽑은 것이면(예전 대응표가 padded 이거나 대응표와 길이가 안 맞음) 먼저 깨끗하게 다시 뽑아
    패킷 표를 한 번 더 맞춰 본다(한 번 padded 로 떨어진 영상이 영영 padded 로 남지 않게).
    """
    wavp = audio_path(video_sha1)
    if wav is not None and Path(wav).resolve() != wavp.resolve():
        raise ValueError(f"build_timemap 의 wav 는 audio_path(video_sha1) 이어야 함: {wav}")
    out = timemap_path(video_sha1)
    old = _read_timemap(out)
    dur = wav_samples(wavp) / SR
    if not force and _fits(old, dur) and old.video_sha1 == video_sha1 and old.made == MADE:
        return old
    info = probe(video)
    a = info["audio"]
    if a is None:
        raise MediaError(f"소리 스트림이 없음: {video}")
    start = info.get("start_time") or 0.0            # 영상 시각의 0 점
    end = None                                       # 소리 끝의 영상 시각(스트림 시작 시각 + 길이 − 컨테이너 시작 시각)
    if a.get("duration") is not None:
        end = (a.get("start_time") or 0.0) + a["duration"] - start
    if old is not None and (old.mode == "padded" or not _fits(old, dur)):
        extract_audio(video, video_sha1, force=True)             # 디스크의 wav 가 메운 것일 수 있다
        dur = wav_samples(wavp) / SR
    why = ""
    tm = None
    frame = 0.0
    last = None                                      # 마지막 패킷 끝의 영상 시각
    try:
        pts, frame = packet_table(video, a.get("time_base"))
        last = pts[-1] + frame - start
        tm, why = check_packets(pts, frame, dur, end, start, info.get("duration"))
    except MediaError as e:
        why = str(e)
    if tm is not None:
        tm = TimeMap(video_sha1, tm.mode, tm.frame, tm.segments, tm.audio_duration)
        util.write_json(out, tm.to_json())
        return tm

    print(f"  ! 패킷 표로 시각 대응표를 만들 수 없음({why}): 틈을 무음으로 메워 소리를 다시 뽑음", flush=True)
    part = wavp.with_name(f"{wavp.name}.padded.{os.getpid()}.part")
    try:
        _extract_to(video, part, padded=True)
        pdur = wav_samples(part) / SR
        bad = _padded_mismatch(pdur, end, last, info.get("duration"))
        if bad:
            if not _fits(_read_timemap(out), wav_samples(wavp) / SR):
                _unlink(out)                         # 남은 대응표가 지금 wav 와 짝이 아니면 지운다
            raise MediaError(f"메워 뽑은 wav 길이 {pdur:.3f}초 가 {bad}: {video}")
        tm = TimeMap.identity(video_sha1, pdur, mode="padded", frame=frame)
        util.write_json(out, tm.to_json())           # 대응표 먼저: 여기서 죽으면 다음 실행이 길이 차이로 알아챈다
        try:
            os.replace(part, wavp)
        except OSError as e:
            _unlink(out)
            raise MediaError(f"메워 뽑은 wav 로 바꿀 수 없음(다른 프로그램이 wav 를 열어 두었는지 확인): {e}") from e
    finally:
        _unlink(part)
    return tm


def _padded_mismatch(pdur: float, end: float | None, last: float | None, fmt_dur: float | None) -> str:
    """메워 뽑은 wav 길이(= 소리 끝의 영상 시각)가 믿을 만한가. 맞으면 "", 아니면 무엇과 다른지.

    아는 것 가운데 하나와 맞으면 된다: 마지막 소리 패킷 끝 시각(0.1초 안), 소리 스트림 끝 시각(0.1초 안),
    컨테이너 길이(넘지 않고 1초 넘게 모자라지 않음: 영상 줄이 더 길 수 있고, mpegts 는 스트림 길이가 어림값이다).
    아는 것이 하나도 없으면 맞다고 본다.
    """
    refs = [(name, t) for name, t in (("마지막 소리 패킷 끝 시각", last), ("소리 스트림 끝 시각", end)) if t is not None]
    if any(abs(pdur - t) <= CHECK_S for _, t in refs) or _near_container_end(pdur, fmt_dur):
        return ""
    if fmt_dur:
        refs.append(("컨테이너 길이", fmt_dur))
    return ", ".join(f"{name} {t:.3f}초" for name, t in refs) + " 와 다름" if refs else ""


def load_timemap(video_sha1: str) -> TimeMap:
    """cache/audio/{video_sha1}.timemap.json 읽기. 없으면 MediaError(먼저 sinkprep video 나 asr 를 돌린다).

    wav 가 있으면 길이가 대응표와 같은지 본다(다르면 MediaError: 짝이 아닌 대응표로 시각을 옮기면 틀린다).
    """
    p = timemap_path(video_sha1)
    if not p.exists():
        raise MediaError(f"시각 대응표가 없음: {p} (python -I -m sinkprep video 또는 asr 를 먼저 돌리세요)")
    tm = TimeMap.from_json(util.read_json(p))
    wavp = audio_path(video_sha1)
    if wavp.exists():
        dur = wav_samples(wavp) / SR
        if not _fits(tm, dur):
            raise MediaError(f"시각 대응표(소리 {tm.audio_duration:.3f}초)가 wav({dur:.3f}초)와 짝이 아님: {p} "
                             "(python -I -m sinkprep video 또는 asr 를 다시 돌리세요)")
    return tm
