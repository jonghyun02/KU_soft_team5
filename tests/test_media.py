"""영상·소리 도구 시험: 소리 시각 ↔ 영상 시각 대응표(TimeMap), 검증 실패 때 메워 뽑기, wav 조각 읽기, sha1 캐시.

ffmpeg 를 부르는 시험은 ffmpeg 가 없으면 건너뛴다. 나머지는 패킷 표를 손으로 만들어 돌린다.
"""
import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from sinkprep import media, paths, util

FRAME = 1024 / 48000
SHA = "0123456789abcdef0123456789abcdef01234567"
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def packets(runs, start=0.0):
    """runs: [(패킷 수, 그 앞의 틈 초)] -> 패킷 PTS 목록. 소리는 이어지고 영상 시각만 틈만큼 건너뛴다."""
    pts, t = [], start
    for n, gap in runs:
        t += gap
        for _ in range(n):
            pts.append(t)
            t += FRAME
    return pts


def write_wav(path, seconds=None, samples=None, sr=media.SR, channels=1):
    """값이 표본 번호(mod 30000)인 wav. 읽은 자리를 값으로 알 수 있다."""
    n = samples if samples is not None else int(round(seconds * sr))
    data = (np.arange(n) % 30000).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.repeat(data, channels).tobytes())
    return n


@pytest.fixture
def proj(tmp_path, monkeypatch):
    for k, sub in (("ROOT", "."), ("CACHE", "cache"), ("AUDIO", "cache/audio"), ("STATE", "cache/state"),
                   ("PROBLEMS", "cache/problems")):
        monkeypatch.setattr(paths, k, tmp_path / sub)
    return tmp_path


# ---------------------------------------------------------------- TimeMap

def test_timemap_from_packets_with_gaps():
    """틈 두 곳(0.05초, 0.06초): 구간 3개, 틈 뒤 소리는 그만큼 늦은 영상 시각으로 간다."""
    pts = packets([(500, 0.0), (400, 0.05), (300, 0.06)])
    dur = len(pts) * FRAME
    tm = media.TimeMap.from_packets(SHA, pts, FRAME, dur)
    assert tm.mode == "map" and len(tm.segments) == 3
    assert tm.gaps == 2 and tm.gap_total == pytest.approx(0.11, abs=1e-5)
    assert tm.segments[0] == [0.0, 0.0]
    assert tm.segments[1] == [pytest.approx(500 * FRAME, abs=1e-5), pytest.approx(500 * FRAME + 0.05, abs=1e-5)]
    assert tm.audio_duration == pytest.approx(dur, abs=1e-5)
    assert tm.video_duration == pytest.approx(dur + 0.11, abs=1e-5)
    a1, a2 = 500 * FRAME, 900 * FRAME
    assert tm.to_video(1.0) == pytest.approx(1.0)
    assert tm.to_video(a1 + 1.0) == pytest.approx(a1 + 1.05, abs=1e-5)
    assert tm.to_video(a2 + 1.0) == pytest.approx(a2 + 1.11, abs=1e-5)
    assert tm.to_video(dur) == pytest.approx(tm.video_duration, abs=1e-5)


def test_timemap_roundtrip_and_gap_clamp():
    """소리 -> 영상 -> 소리 왕복이 같고, 틈 안의 영상 시각은 앞 구간 끝으로 잘린다. to_video 는 줄지 않는다."""
    pts = packets([(500, 0.0), (400, 0.05), (300, 0.06)])
    dur = len(pts) * FRAME
    tm = media.TimeMap.from_packets(SHA, pts, FRAME, dur)
    grid = [k * dur / 997 for k in range(998)]
    for t in grid:
        assert tm.to_audio(tm.to_video(t)) == pytest.approx(t, abs=1e-6)
    vs = [tm.to_video(t) for t in grid]
    assert all(b >= a for a, b in zip(vs, vs[1:]))
    a1 = tm.segments[1][0]
    assert tm.to_audio(a1 + 0.02) == pytest.approx(a1, abs=1e-6)          # 틈(영상 a1 ~ a1+0.05) 안
    assert tm.to_audio(a1 + 0.05 + 0.3) == pytest.approx(a1 + 0.3, abs=1e-5)
    assert tm.to_audio(-5.0) == 0.0 and tm.to_audio(1e9) == pytest.approx(dur, abs=1e-5)
    assert tm.to_video(-5.0) == 0.0 and tm.to_video(1e9) == pytest.approx(tm.video_duration, abs=1e-5)
    avs = [tm.to_audio(v * tm.video_duration / 500) for v in range(501)]
    assert all(b >= a for a, b in zip(avs, avs[1:]))


def test_timemap_identity_without_gaps():
    pts = packets([(1000, 0.0)])
    dur = len(pts) * FRAME
    tm = media.TimeMap.from_packets(SHA, pts, FRAME, dur)
    assert tm.mode == "identity" and tm.segments == [[0.0, 0.0]] and tm.gaps == 0 and tm.gap_total == 0.0
    for t in (0.0, 0.123, 7.5, dur):
        assert tm.to_video(t) == pytest.approx(t, abs=1e-6) and tm.to_audio(t) == pytest.approx(t, abs=1e-6)
    assert tm.video_duration == tm.audio_duration


def test_timemap_small_jitter_is_one_segment():
    """PTS 가 1ms 안쪽으로 흔들려도 구간을 나누지 않는다(문턱 2ms)."""
    pts = [k * FRAME + (0.001 if k % 7 == 0 else 0.0) for k in range(800)]
    tm = media.TimeMap.from_packets(SHA, pts, FRAME, len(pts) * FRAME)
    assert len(tm.segments) == 1


def test_timemap_trims_negative_first_pts():
    """첫 PTS 가 음수(코덱 준비 표본)이고 wav 가 그만큼 짧으면 디코더가 버린 것으로 보고 밀림 0 으로 맞춘다."""
    pts = packets([(1000, 0.0)], start=-FRAME)
    tm = media.TimeMap.from_packets(SHA, pts, FRAME, 999 * FRAME)
    assert media.packet_trim(pts, FRAME, 999 * FRAME) == pytest.approx(FRAME)
    assert tm.mode == "identity" and tm.to_video(1.0) == pytest.approx(1.0, abs=1e-6)
    # wav 가 패킷 수만큼 길면(버리지 않음) 소리가 영상 시각 -FRAME 에서 시작한 것으로 본다
    tm2 = media.TimeMap.from_packets(SHA, pts, FRAME, 1000 * FRAME)
    assert tm2.to_video(1.0) == pytest.approx(1.0 - FRAME, abs=1e-6)


def test_timemap_delayed_audio_start():
    """소리가 영상 시각 0.5초에서 시작: 틈은 없지만 항등이 아니다."""
    pts = packets([(1000, 0.0)], start=0.5)
    tm = media.TimeMap.from_packets(SHA, pts, FRAME, 1000 * FRAME)
    assert tm.mode == "map" and tm.gaps == 0 and tm.segments == [[0.0, 0.5]]
    assert tm.to_video(0.0) == pytest.approx(0.5) and tm.to_audio(0.2) == 0.0 and tm.to_audio(1.5) == pytest.approx(1.0)


def test_timemap_json_roundtrip():
    pts = packets([(500, 0.0), (400, 0.05)])
    tm = media.TimeMap.from_packets(SHA, pts, FRAME, len(pts) * FRAME)
    d = tm.to_json()
    assert list(d) == ["video_sha1", "mode", "frame", "segments", "audio_duration", "video_duration", "gaps",
                       "gap_total", "made"]
    assert d["video_sha1"] == SHA and d["made"] == media.MADE and d["gaps"] == 1
    back = media.TimeMap.from_json(d)
    assert back.to_json() == d
    for t in (0.0, 3.3, 12.0, tm.audio_duration):
        assert back.to_video(t) == tm.to_video(t)


def test_check_packets_rejects_length_mismatch_and_overlap():
    """패킷 수 × 길이가 wav 와 0.1초 넘게 다르거나, 끝 시각이 스트림 길이와 다르거나, PTS 가 뒤로 가면 못 쓴다."""
    pts = packets([(1000, 0.0)])
    dur = 1000 * FRAME
    tm, why = media.check_packets(pts, FRAME, dur, dur)
    assert tm is not None and why == ""
    assert media.check_packets(pts, FRAME, dur, None)[0] is not None           # 스트림 길이를 모르면 그 검사는 생략
    tm, why = media.check_packets(pts, FRAME, dur + 0.5, dur)
    assert tm is None and "wav" in why
    tm, why = media.check_packets(pts, FRAME, dur, dur + 0.5)
    assert tm is None and "스트림" in why
    back = packets([(500, 0.0), (500, -0.2)])
    tm, why = media.check_packets(back, FRAME, dur, None)
    assert tm is None and why


# ---------------------------------------------------------------- build_timemap

def fake_probe(stream_duration):
    def probe(video):
        return {"duration": stream_duration, "start_time": 0.0, "video": None,
                "audio": {"index": 0, "codec": "aac", "time_base": "1/48000", "start_time": 0.0,
                          "duration": stream_duration, "nb_frames": None, "sample_rate": 48000, "channels": 2}}
    return probe


def test_build_timemap_writes_and_reuses_file(proj, monkeypatch):
    """대응표를 파일로 쓰고, 다시 부르면 패킷 표를 읽지 않고 그 파일을 쓴다."""
    pts = packets([(500, 0.0), (400, 0.05)])
    dur = len(pts) * FRAME
    wav = media.audio_path(SHA)
    write_wav(wav, samples=int(round(dur * media.SR)))
    calls = []
    monkeypatch.setattr(media, "probe", fake_probe(dur + 0.05))
    monkeypatch.setattr(media, "packet_table", lambda v, tb=None: calls.append(v) or (pts, FRAME))
    tm = media.build_timemap("x.mp4", SHA, wav)
    assert tm.mode == "map" and tm.gaps == 1 and tm.video_sha1 == SHA and calls == ["x.mp4"]
    assert util.read_json(media.timemap_path(SHA)) == tm.to_json()
    again = media.build_timemap("x.mp4", SHA, wav)
    assert again.to_json() == tm.to_json() and calls == ["x.mp4"]
    assert media.load_timemap(SHA).to_json() == tm.to_json()
    media.build_timemap("x.mp4", SHA, wav, force=True)
    assert len(calls) == 2


def fake_extractor(made, padded_seconds, clean_seconds=2.0):
    """media._extract_to 대신 끼울 것: 메워 뽑으면 padded_seconds, 그냥 뽑으면 clean_seconds 짜리 wav 를 쓴다."""
    def extract_to(video, dst, *, padded=False):
        made.append(padded)
        write_wav(dst, seconds=padded_seconds if padded else clean_seconds)
    return extract_to


def boom(v, tb=None):
    raise media.MediaError("패킷 표에 PTS 가 없는 줄")


def test_build_timemap_falls_back_to_padded(proj, monkeypatch):
    """패킷 표가 wav 길이와 안 맞으면(패킷마다 표본 수가 다른 코덱 등) 틈을 메워 다시 뽑고 항등 대응표를 쓴다."""
    wav = media.audio_path(SHA)
    write_wav(wav, seconds=2.0)
    made = []
    monkeypatch.setattr(media, "probe", fake_probe(3.0))
    monkeypatch.setattr(media, "packet_table", lambda v, tb=None: (packets([(50, 0.0)]), FRAME))   # 1.07초뿐
    monkeypatch.setattr(media, "_extract_to", fake_extractor(made, 3.0))
    tm = media.build_timemap("x.mkv", SHA, wav)
    assert made == [True]
    assert tm.mode == "padded" and tm.segments == [[0.0, 0.0]] and tm.audio_duration == pytest.approx(3.0)
    assert tm.to_video(1.234) == pytest.approx(1.234) and tm.gaps == 0
    assert media.wav_samples(wav) == 3 * media.SR and not list(wav.parent.glob("*.part"))    # wav 가 메운 것으로 바뀜
    assert media.load_timemap(SHA).mode == "padded"
    assert media.build_timemap("x.mkv", SHA, wav).mode == "padded" and made == [True]             # 다시 뽑지 않음


def test_build_timemap_padded_when_packet_table_fails(proj, monkeypatch):
    """패킷 표를 못 읽어도(PTS 없는 패킷) 메워 뽑기로 넘어간다. 스트림 길이를 모르면 컨테이너 길이로 확인한다."""
    wav = media.audio_path(SHA)
    write_wav(wav, seconds=2.0)
    probe = fake_probe(2.0)

    def no_stream_duration(video):                              # mkv: 소리 스트림 duration 이 없다
        info = probe(video)
        info["audio"]["duration"] = None
        return info

    monkeypatch.setattr(media, "probe", no_stream_duration)
    monkeypatch.setattr(media, "packet_table", boom)
    monkeypatch.setattr(media, "_extract_to", fake_extractor([], 1.95))     # 소리가 영상보다 0.05초 일찍 끝남
    assert media.build_timemap("x.mkv", SHA, wav).mode == "padded"
    assert media.wav_samples(wav) == int(1.95 * media.SR)


def test_failed_padded_extract_leaves_wav_and_timemap_as_a_pair(proj, monkeypatch):
    """메워 뽑은 길이가 틀리면 MediaError 이고, 깨끗한 wav 와 예전 대응표는 짝인 채 그대로 남는다(회귀: 예전에는
    wav 만 메운 것으로 바뀌어 예전 대응표와 어긋났다)."""
    pts = packets([(100, 0.0), (88, 0.05)])
    dur = len(pts) * FRAME                                       # 4.0107초
    wav = media.audio_path(SHA)
    n = write_wav(wav, samples=int(round(dur * media.SR)))
    monkeypatch.setattr(media, "probe", fake_probe(dur + 0.05))
    monkeypatch.setattr(media, "packet_table", lambda v, tb=None: (pts, FRAME))
    tm = media.build_timemap("x.mp4", SHA, wav)
    assert tm.mode == "map" and tm.gaps == 1
    before = wav.read_bytes()
    # 다시 만들 때 패킷 표를 못 읽고, 메워 뽑은 소리는 9초(스트림·컨테이너 길이 4.06초와 다름)
    monkeypatch.setattr(media, "packet_table", boom)
    monkeypatch.setattr(media, "_extract_to", fake_extractor([], 9.0))
    with pytest.raises(media.MediaError) as e:
        media.build_timemap("x.mp4", SHA, wav, force=True)
    assert "메워 뽑은 wav 길이" in str(e.value)
    assert wav.read_bytes() == before and media.wav_samples(wav) == n          # 깨끗한 wav 그대로
    assert not list(wav.parent.glob("*.part"))
    again = media.load_timemap(SHA)                                            # 예전 대응표도 그대로이고 wav 와 짝
    assert again.mode == "map" and again.to_json() == tm.to_json()
    assert again.to_video(3.0) == pytest.approx(3.05, abs=1e-5)


def test_stale_padded_wav_is_reextracted_clean(proj, monkeypatch):
    """wav 가 메운 것으로 바뀐 채 map 대응표가 남은 상태(예전 판이 남길 수 있던 짝): load_timemap 이 거절하고,
    build_timemap 은 깨끗하게 다시 뽑아 map 으로 돌아온다(영영 padded 로 남지 않음)."""
    pts = packets([(100, 0.0), (88, 0.05)])
    dur = len(pts) * FRAME
    clean = int(round(dur * media.SR))
    wav = media.audio_path(SHA)
    write_wav(wav, samples=clean)
    monkeypatch.setattr(media, "probe", fake_probe(dur + 0.05))
    monkeypatch.setattr(media, "packet_table", lambda v, tb=None: (pts, FRAME))
    assert media.build_timemap("x.mp4", SHA, wav).mode == "map"
    write_wav(wav, samples=clean + int(0.05 * media.SR))         # 메운 wav 로 바뀜(틈 0.05초만큼 김)
    with pytest.raises(media.MediaError) as e:
        media.load_timemap(SHA)
    assert "짝이 아님" in str(e.value)
    made = []
    monkeypatch.setattr(media, "_extract_to", fake_extractor(made, 9.0, clean_seconds=clean / media.SR))
    tm = media.build_timemap("x.mp4", SHA, wav)
    assert made == [False] and tm.mode == "map" and tm.gaps == 1 and media.wav_samples(wav) == clean
    assert media.load_timemap(SHA).to_json() == tm.to_json()
    # 예전 대응표가 padded 였으면 다시 만들 때(force) 먼저 깨끗하게 뽑아 패킷 표를 맞춰 본다
    util.write_json(media.timemap_path(SHA), media.TimeMap.identity(SHA, clean / media.SR, mode="padded").to_json())
    made.clear()
    assert media.build_timemap("x.mp4", SHA, wav, force=True).mode == "map" and made == [False]


def test_build_timemap_rejects_other_wav_path(proj, tmp_path):
    """wav 인자는 audio_path(video_sha1) 만 받는다(다른 경로를 주면 엉뚱한 파일 길이를 읽던 것을 막음)."""
    write_wav(media.audio_path(SHA), seconds=1.0)
    other = tmp_path / "other.wav"
    write_wav(other, seconds=1.0)
    with pytest.raises(ValueError):
        media.build_timemap("x.mp4", SHA, other)


def test_load_timemap_missing_raises(proj):
    with pytest.raises(media.MediaError):
        media.load_timemap(SHA)


def test_format_start_time_is_the_zero_of_video_time():
    """컨테이너 시작 시각이 0 이 아닌 영상(mpegts): 영상 시각 = PTS − 시작 시각. 길이 검증도 같은 기준이다."""
    start = 1.4
    pts = packets([(500, 0.0), (400, 0.05)], start=start)
    dur = len(pts) * FRAME
    tm = media.TimeMap.from_packets(SHA, pts, FRAME, dur, start=start)
    assert tm.segments[0] == [0.0, 0.0] and tm.gaps == 1
    assert tm.to_video(1.0) == pytest.approx(1.0) and tm.video_duration == pytest.approx(dur + 0.05, abs=1e-5)
    ok, why = media.check_packets(pts, FRAME, dur, dur + 0.05, start)
    assert ok is not None and why == ""
    # 시작 시각을 빼지 않으면 1.4초 늦은 시각이 나오고 스트림 끝 시각과도 어긋난다
    assert media.TimeMap.from_packets(SHA, pts, FRAME, dur).to_video(1.0) == pytest.approx(2.4)
    bad, why = media.check_packets(pts, FRAME, dur, dur + 0.05)
    assert bad is None and "스트림" in why
    # 스트림 길이가 어림값이라 0.15초 어긋나도(mpegts) 컨테이너 길이와 맞으면 받아들인다
    ok, why = media.check_packets(pts, FRAME, dur, dur - 0.1, start, container_duration=dur + 0.04)
    assert ok is not None and why == ""
    bad, why = media.check_packets(pts, FRAME, dur, dur - 0.1, start, container_duration=dur + 3.0)
    assert bad is None


def test_negative_first_video_time_is_clamped_to_zero():
    """첫 소리 PTS 가 음수인데 디코더가 버리지 않은 영상: 소리 앞부분의 영상 시각을 0 으로 자른다(음수 시각이
    asr.json 에 들어가면 스키마에서 떨어진다)."""
    pts = packets([(1000, 0.0)], start=-FRAME)
    tm = media.TimeMap.from_packets(SHA, pts, FRAME, 1000 * FRAME)
    assert tm.segments[0][1] == pytest.approx(-FRAME, abs=1e-6)
    assert tm.to_video(0.0) == 0.0 and tm.to_video(FRAME / 2) == 0.0
    assert tm.to_video(1.0) == pytest.approx(1.0 - FRAME, abs=1e-6)
    assert all(tm.to_video(k / 100) >= 0.0 for k in range(200))


def test_run_timeout_becomes_media_error(monkeypatch):
    """멈춘 ffmpeg·ffprobe 는 시간 제한 뒤 MediaError 가 된다."""
    seen = {}

    def hang(cmd, **kw):
        seen.update(kw)
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))

    monkeypatch.setattr(media.subprocess, "run", hang)
    monkeypatch.setattr(media.shutil, "which", lambda name: name)
    with pytest.raises(media.MediaError) as e:
        media.probe("x.mp4")
    assert "끝나지 않음" in str(e.value) and seen["timeout"] == media.T_PROBE
    with pytest.raises(media.MediaError):
        media._extract_to("x.mp4", Path("never.wav"))
    assert seen["timeout"] == media.T_EXTRACT
    with pytest.raises(media.MediaError):
        media.packet_table("x.mp4", "1/48000")
    assert seen["timeout"] == media.T_PACKETS


# ---------------------------------------------------------------- wav·경로·sha1

def test_audio_reader_reads_ranges(tmp_path):
    wav = tmp_path / "a.wav"
    n = write_wav(wav, seconds=2.0)
    with media.AudioReader(wav) as r:
        assert r.n_samples == n == 32000 and r.duration == 2.0 and r.sr == 16000
        x = r.read_samples(100, 105)
        assert x.dtype == np.float32 and np.allclose(x * 32768, [100, 101, 102, 103, 104])
        y = r.read(0.5, 0.5 + 3 / 16000)                       # 소리 시각(초)
        assert np.allclose(y * 32768, [8000, 8001, 8002])
        assert len(r.read_samples(31990, 40000)) == 10 and len(r.read_samples(-5, 3)) == 3     # 파일 안으로 자름
        assert len(r.read_samples(50, 50)) == 0 and len(r.read_all()) == n
        assert np.allclose(r.read_samples(10, 12) * 32768, [10, 11])                               # 뒤로 돌아가 읽기
    assert media.wav_samples(wav) == n


def test_audio_reader_rejects_other_formats(tmp_path):
    for name, kw in (("st.wav", {"channels": 2}), ("r8.wav", {"sr": 8000})):
        p = tmp_path / name
        write_wav(p, seconds=0.5, **kw)
        with pytest.raises(media.MediaError):
            media.AudioReader(p)
    with pytest.raises(media.MediaError):
        media.AudioReader(tmp_path / "none.wav")


def test_resolve_video_relative_to_root(proj):
    assert media.resolve_video("videos/local_0001.mp4") == proj / "videos" / "local_0001.mp4"
    absolute = str(proj / "elsewhere" / "v.mp4")
    assert media.resolve_video(absolute) == proj / "elsewhere" / "v.mp4"


def test_file_sha1_caches_by_size_and_mtime(proj, monkeypatch):
    """크기·고친 시각이 같으면 파일을 다시 읽지 않고, 내용이 바뀌면 다시 계산한다."""
    f = proj / "v.bin"
    f.write_bytes(b"abc" * 1000)
    h = media.file_sha1(f)
    assert h == util.sha1_file(f)
    real = util.sha1_file
    calls = []
    monkeypatch.setattr(util, "sha1_file", lambda p, chunk=1 << 20: calls.append(p) or real(p))
    assert media.file_sha1(f) == h and calls == []
    f.write_bytes(b"abcd" * 1000)
    h2 = media.file_sha1(f)
    assert h2 != h and len(calls) == 1 and h2 == real(f)


def test_file_sha1_fresh_ignores_cache(proj):
    """크기와 고친 시각을 그대로 둔 채 내용만 바뀐 파일은 캐시로는 못 알아챈다(알려진 한계). fresh 면 다시 읽는다."""
    f = proj / "v.bin"
    f.write_bytes(b"a" * 5000)
    h = media.file_sha1(f)
    st = f.stat()
    f.write_bytes(b"b" * 5000)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert media.file_sha1(f) == h                                # 예전 값(한계)
    h2 = media.file_sha1(f, fresh=True)
    assert h2 == util.sha1_file(f) and h2 != h
    assert media.file_sha1(f) == h2                               # fresh 로 읽은 값이 캐시에 남는다


def test_missing_tool_raises_media_error(monkeypatch):
    monkeypatch.setattr(media.shutil, "which", lambda name: None)
    with pytest.raises(media.MediaError):
        media.ffmpeg()
    with pytest.raises(media.MediaError):
        media.probe("x.mp4")


# ---------------------------------------------------------------- 실제 ffmpeg

def make_video(path, seconds=3, audio_delay=None):
    """검은 화면 + 440Hz 소리의 짧은 mp4. audio_delay 초만큼 소리를 늦게 시작시킬 수 있다."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-f", "lavfi", "-i", f"color=c=black:s=64x64:r=10:d={seconds}"]
    if audio_delay:
        cmd += ["-itsoffset", str(audio_delay)]
    cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds - (audio_delay or 0)}",
            "-c:v", "mpeg4", "-c:a", "aac", "-b:a", "96k", str(path)]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, capture_output=True)
    return path


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg 가 없음")
def test_probe_extract_and_timemap_on_real_video(proj):
    """진짜 영상: probe, 소리 추출(16 kHz 모노), 패킷 표, 틈 없는 대응표."""
    video = make_video(proj / "v.mp4")
    info = media.probe(video)
    assert info["duration"] == pytest.approx(3.0, abs=0.15)
    assert info["audio"]["codec"] == "aac" and info["audio"]["sample_rate"] == 48000
    assert info["video"]["width"] == 64 and info["video"]["fps"] == pytest.approx(10.0)
    sha = media.file_sha1(video)
    wav = media.extract_audio(video, sha)
    assert wav == media.audio_path(sha) and not list(wav.parent.glob("*.part"))
    with media.AudioReader(wav) as r:
        assert r.duration == pytest.approx(3.0, abs=0.1)
        assert float(np.abs(r.read(1.0, 2.0)).max()) > 0.05
    mtime = wav.stat().st_mtime_ns
    assert media.extract_audio(video, sha) == wav and wav.stat().st_mtime_ns == mtime     # 있으면 건너뜀
    pts, frame = media.packet_table(video)
    assert frame == pytest.approx(1024 / 48000) and len(pts) > 100
    tm = media.build_timemap(video, sha, wav)
    assert tm.mode in ("identity", "map") and tm.gaps == 0
    assert tm.to_video(1.5) == pytest.approx(1.5, abs=0.03)
    assert media.timemap_path(sha).exists()


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg 가 없음")
def test_delayed_audio_maps_to_video_time(proj):
    """소리가 영상보다 0.5초 늦게 시작하는 영상: 소리가 처음 들리는 자리를 영상 시각으로 옮기면 0.5초 근처다."""
    video = make_video(proj / "d.mp4", audio_delay=0.5)
    sha = media.file_sha1(video)
    wav = media.extract_audio(video, sha)
    tm = media.build_timemap(video, sha, wav)
    with media.AudioReader(media.audio_path(sha)) as r:            # 메워 뽑기로 바뀌었을 수 있어 다시 연다
        x = np.abs(r.read_all())
    first = int(np.argmax(x > 0.02)) / media.SR
    assert tm.to_video(first) == pytest.approx(0.5, abs=0.06)
    # 메워 뽑기(대응표를 못 믿을 때의 길): 빈 앞부분이 무음으로 채워져 wav 시각이 곧 영상 시각이 된다
    media.extract_audio(video, sha, padded=True)
    with media.AudioReader(media.audio_path(sha)) as r:
        y = np.abs(r.read_all())
        assert r.duration == pytest.approx(3.0, abs=0.1)
    assert int(np.argmax(y > 0.02)) / media.SR == pytest.approx(0.5, abs=0.06)
    assert float(y[: int(0.4 * media.SR)].max()) < 0.01


# ---------------------------------------------------------------- 실제 ffmpeg: 틈·시작 시각·mkv

FF = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]


def ff(*args):
    subprocess.run(FF + [str(a) for a in args], check=True, stdin=subprocess.DEVNULL, capture_output=True)


def make_gap_video(box):
    """440Hz 2초 + 영상 시각 틈 약 0.22초 + 880Hz 2초인 mp4. 조각 둘을 이어 붙이되 첫 조각 길이를 2.25초라고 적는다."""
    box.mkdir(parents=True, exist_ok=True)
    for name, freq in (("a.ts", 440), ("b.ts", 880)):
        ff("-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d=2", "-f", "lavfi", "-i",
           f"sine=frequency={freq}:sample_rate=48000:duration=2", "-c:v", "mpeg4", "-c:a", "aac", "-b:a", "96k",
           "-f", "mpegts", box / name)
    (box / "list.txt").write_text("ffconcat version 1.0\nfile a.ts\nduration 2.25\nfile b.ts\n", encoding="utf-8")
    out = box / "gap.mp4"
    ff("-f", "concat", "-safe", "0", "-i", box / "list.txt", "-c", "copy", "-bsf:a", "aac_adtstoasc", out)
    return out


def onset_880(wav):
    """wav 에서 880Hz 가 시작하는 자리(초): 20ms 창의 영교차 수가 440Hz(약 18회)에서 880Hz(약 35회)로 뛰는 곳."""
    with media.AudioReader(wav) as r:
        x = r.read_all()
    win = 320
    for i in range(0, len(x) - win, 16):
        z = int((np.diff(np.signbit(x[i:i + win]).astype(np.int8)) != 0).sum())
        if z > 25 and float(np.abs(x[i:i + win]).max()) > 0.02:
            return (i + win / 2) / media.SR
    raise AssertionError("880Hz 가 없음")


def padded_onset(video, box):
    """ffmpeg 가 틈을 메워 뽑은 소리(wav 시각 = 영상 시각)에서 880Hz 가 시작하는 자리: 영상 시각의 참값."""
    ref = box / (video.name + ".ref.wav")
    ff("-i", video, "-map", "0:a:0", "-vn", "-af", media.PAD_FILTER, "-ac", "1", "-ar", media.SR,
       "-c:a", "pcm_s16le", "-f", "wav", ref)
    return onset_880(ref)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg 가 없음")
def test_real_gap_video_map_agrees_with_padded_extraction(proj):
    """틈이 있는 진짜 영상: 소리는 틈을 메우지 않고 뽑히고, ffprobe 패킷 표로 만든 대응표가 옮긴 영상 시각이
    ffmpeg 가 틈을 메워 뽑은 소리의 시각과 같다. mp4, 같은 내용의 mkv(1ms 단위 시각), mpegts(시작 시각 1.4초) 모두."""
    box = proj / "vids"
    mp4 = make_gap_video(box)
    mkv, ts = box / "gap.mkv", box / "gap.ts"
    ff("-i", mp4, "-c", "copy", mkv)
    ff("-i", mp4, "-c", "copy", "-f", "mpegts", ts)
    truth = padded_onset(mp4, box)
    assert truth == pytest.approx(2.26, abs=0.03)
    for video in (mp4, mkv, ts):
        info = media.probe(video)
        sha = media.file_sha1(video)
        wav = media.extract_audio(video, sha)
        tm = media.build_timemap(video, sha, wav)
        assert tm.mode == "map" and tm.gaps == 1, video.name                    # mkv 도 메워 뽑기로 가지 않는다
        assert tm.gap_total == pytest.approx(0.223, abs=0.02), video.name
        assert tm.frame == pytest.approx(1024 / 48000, abs=1e-7), video.name    # mkv 의 21ms 가 아니라 디코딩한 프레임 길이
        with media.AudioReader(wav) as r:
            assert r.duration == pytest.approx(tm.video_duration - tm.gap_total, abs=0.001)   # 틈을 메우지 않은 소리
        t = onset_880(wav)
        assert t == pytest.approx(truth - tm.gap_total, abs=0.03), video.name
        assert tm.to_video(t) == pytest.approx(truth, abs=0.01), video.name
        assert tm.to_video(t) == pytest.approx(padded_onset(video, box), abs=0.01), video.name
        assert tm.video_duration <= info["duration"] + 0.1, video.name           # 낱말 시각이 영상 길이를 넘지 않는다
        assert media.load_timemap(sha).to_json() == tm.to_json()
    assert media.probe(ts)["start_time"] == pytest.approx(1.4, abs=0.05)         # 이 시작 시각을 빼야 한다


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg 가 없음")
def test_real_padded_branch_on_nonzero_start_time(proj, monkeypatch):
    """패킷 표를 못 쓸 때의 길(메워 뽑기)을 진짜 파일로: 시작 시각이 0 이 아닌 mpegts 도 길이 검증을 통과하고,
    wav 시각이 곧 영상 시각이 된다(회귀: 예전에는 길이 검증이 시작 시각만큼 어긋나 MediaError 였다)."""
    box = proj / "vids"
    mp4 = make_gap_video(box)
    ts = box / "gap.ts"
    ff("-i", mp4, "-c", "copy", "-f", "mpegts", ts)
    truth = padded_onset(ts, box)
    sha = media.file_sha1(ts)
    wav = media.extract_audio(ts, sha)
    clean = media.wav_samples(wav)
    monkeypatch.setattr(media, "packet_table", boom)
    tm = media.build_timemap(ts, sha, wav)
    assert tm.mode == "padded" and tm.gaps == 0
    assert media.wav_samples(wav) > clean + int(0.15 * media.SR)                 # 틈만큼 길어진 wav 로 바뀜
    assert onset_880(wav) == pytest.approx(truth, abs=0.005)
    assert tm.to_video(onset_880(wav)) == pytest.approx(truth, abs=0.005)
    assert not list(wav.parent.glob("*.part"))
    # 패킷 표를 다시 읽을 수 있게 되면 깨끗하게 다시 뽑아 map 으로 돌아온다
    monkeypatch.undo()
    for k, sub in (("ROOT", "."), ("CACHE", "cache"), ("AUDIO", "cache/audio"), ("STATE", "cache/state"),
                   ("PROBLEMS", "cache/problems")):
        monkeypatch.setattr(paths, k, proj / sub)
    tm2 = media.build_timemap(ts, sha, wav, force=True)
    assert tm2.mode == "map" and tm2.gaps == 1 and media.wav_samples(wav) == clean
    assert tm2.to_video(onset_880(wav)) == pytest.approx(truth, abs=0.01)
