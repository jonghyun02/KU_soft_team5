"""받아쓰기 단계 시험(가짜 백엔드만 씀: 모델·ffmpeg 없이 돈다).

묶음 규칙·덧댐, asr.json 형식과 영상 시각 옮기기, 다시 돌려도 바이트 동일, 묶음 캐시 이어 하기, pass id 와 캐시,
문제 기록, teacher forcing 결과 묶기. 진짜 모델 시험은 SINK_ASR_REAL=1 일 때만.
"""
import argparse
import os
import string
import sys
import types
import wave

import numpy as np
import pytest

from sinkprep import asr, catalog, media, paths, util, validate

SR = asr.SR
FILM = "local_0001"
DOC = "fm1"
WAV_S = 60.0
# 소리 20초, 40초 자리에서 영상 시각이 0.05초, 0.06초 건너뛴다
SEGMENTS = [[0.0, 0.0], [20.0, 20.05], [40.0, 40.11]]


def s(x: float) -> int:
    return int(round(x * SR))


# 말 구간(초): 앞 둘은 틈 0.5초라 한 묶음, 셋째·넷째는 틈 0.1초라 한 묶음, 다섯째(23초)와 여섯째는 틈 0.05초지만
# 합치면 25초를 넘어 따로, 마지막은 파일 끝 0.05초 앞에서 끝난다
REGIONS_S = [(1.0, 3.0), (3.5, 6.0), (10.0, 12.0), (12.1, 14.0), (21.0, 44.0), (44.05, 50.0), (55.0, 59.95)]
REGIONS = [(s(a), s(b)) for a, b in REGIONS_S]
GROUPS = [(s(1.0), s(6.0)), (s(10.0), s(14.0)), (s(21.0), s(44.0)), (s(44.05), s(50.0)), (s(55.0), s(59.95))]


class Fake:
    """가짜 백엔드: 정해 둔 말 구간과, 조각 길이로만 정해지는 낱말 셋(조각 전체를 셋으로 나눔)을 돌려준다.

    진짜 Whisper 처럼 첫 낱말은 조각의 0초에서 시작한다. empty 에 든 길이(표본 수)의 조각에서는 아무것도 안 나온다.
    """
    name = "fake"
    vad_name = "fake-vad"

    def __init__(self, regions=REGIONS, p=0.9):
        self.regions = list(regions)
        self.p = p
        self.snapshot = "ab12cd34" + "0" * 32
        self.versions = {"faster_whisper": "9.9", "ctranslate2": "8.8"}
        self.empty = set()
        self.vad_calls = 0
        self.loads = []
        self.calls = []

    def vad(self, audio):
        self.vad_calls += 1
        assert audio.dtype == np.float32 and len(audio) == s(WAV_S)
        return list(self.regions)

    def load(self, name, device, threads):
        self.loads.append((name, device, threads))
        return asr.ModelHandle(name, f"fake/{name}", self.snapshot, "cpu", "int8", threads=threads,
                               versions=dict(self.versions), model=object())

    def transcribe(self, handle, clip, lang, hotwords=None, initial_prompt=None):
        assert lang == "ko" and hotwords is None and initial_prompt is None
        self.calls.append((handle.name, len(clip)))
        if len(clip) in self.empty:
            return [], []
        dur = len(clip) / SR
        words = [{"w": f"낱말{k}", "s": round(dur * k / 3, 3), "e": round(dur * (k + 1) / 3, 3), "p": self.p}
                 for k in range(3)]
        segs = [{"start": 0.0, "end": round(dur, 3), "text": "낱말0 낱말1 낱말2", "avg_logprob": -0.2,
                 "no_speech_prob": 0.01, "compression_ratio": 1.1, "temperature": 0.0}]
        return words, segs


@pytest.fixture
def proj(tmp_path, monkeypatch):
    """임시 프로젝트: 가짜 영상 파일, 60초 wav, 틈 두 곳짜리 대응표(파일로 미리 써 둬 ffprobe 를 부르지 않는다)."""
    for k, sub in (("ROOT", "."), ("CACHE", "cache"), ("AUDIO", "cache/audio"), ("ASR_CACHE", "cache/asr"),
                   ("STATE", "cache/state"), ("PROBLEMS", "cache/problems"), ("FILMS", "films"),
                   ("VIDEOS", "videos"), ("CATALOG", "catalog.csv")):
        monkeypatch.setattr(paths, k, tmp_path / sub)
    video = paths.VIDEOS / f"{FILM}.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"not really a video" * 100)
    sha = util.sha1_file(video)
    wav = media.audio_path(sha)
    wav.parent.mkdir(parents=True)
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.zeros(s(WAV_S), dtype="<i2").tobytes())
    util.write_json(media.timemap_path(sha), media.TimeMap(sha, "map", 1024 / 48000, SEGMENTS, WAV_S).to_json())
    fake = Fake()
    asr.set_backend(fake)
    doc = {"doc_id": DOC, "film_id": FILM, "status": "parsed", "is_primary": "true", "category": "ko_film",
           "video_path": f"videos/{FILM}.mp4", "video_sha1": sha, "video_duration": "60.110", "video_lang": "ko"}
    yield types.SimpleNamespace(root=tmp_path, doc=doc, sha=sha, fake=fake, out=asr.out_path(FILM))
    asr.set_backend(None)


def run_asr(proj, **kw):
    kw.setdefault("passes", "plain=tiny")
    return asr.process(proj.doc, device="cpu", threads=2, **kw)


# ---------------------------------------------------------------- 묶음 규칙

def test_chunks_join_close_regions_and_cap_at_25s():
    assert asr.chunks(REGIONS) == GROUPS
    assert asr.chunks([]) == []
    # 25초 넘는 구간은 25초마다 자른다
    assert asr.chunks([(0, s(60.0))]) == [(0, s(25.0)), (s(25.0), s(50.0)), (s(50.0), s(60.0))]
    # 틈이 꼭 1초면 붙이지 않는다(1초 미만만)
    assert asr.chunks([(0, s(2.0)), (s(3.0), s(4.0))]) == [(0, s(2.0)), (s(3.0), s(4.0))]
    assert asr.chunks([(0, s(2.0)), (s(2.99), s(4.0))]) == [(0, s(4.0))]
    for a, b in asr.chunks(REGIONS + [(s(60.5), s(200.0))]):
        assert 0 < b - a <= s(25.0)


def test_pads_do_not_overlap_neighbours():
    """덧댐은 0.1초까지, 이웃이 가까우면 틈의 절반까지: 잘라 낸 범위끼리 겹치지 않고 파일을 벗어나지 않는다."""
    n = s(WAV_S)
    pad = asr.pads(GROUPS, n)
    assert len(pad) == len(GROUPS)
    for (a, b), (pa, pb) in zip(GROUPS, pad):
        assert 0 <= pa <= a < b <= pb <= n
        assert a - pa <= s(0.1) and pb - b <= s(0.1)
    assert all(nxt[0] >= prv[1] for prv, nxt in zip(pad, pad[1:]))
    assert pad[0] == (s(0.9), s(6.1))                                   # 이웃이 멀면 0.1초
    assert pad[2][1] == s(44.0) + s(0.025) and pad[3][0] == s(44.05) - s(0.025)     # 틈 0.05초 -> 절반씩
    assert pad[4][1] == n                                                # 파일 끝까지 0.05초뿐
    assert asr.pads([(0, s(1.0))], s(1.0)) == [(0, s(1.0))]
    # 맞닿은 묶음(25초에서 자른 것)은 덧대지 않는다
    assert asr.pads([(0, s(25.0)), (s(25.0), s(30.0))], s(40.0)) == [(0, s(25.0)), (s(25.0), s(30.1))]


def test_parse_passes_and_lang():
    assert asr.parse_passes("plain=large-v3") == [("plain", "large-v3")]
    assert asr.parse_passes("plain=turbo, ctx=large-v3") == [("plain", "large-v3-turbo"), ("ctx", "large-v3")]
    for bad in ("", "plain", "Plain=x", "1a=x", "a=x,a=y", "a="):
        with pytest.raises(ValueError):
            asr.parse_passes(bad)
    assert asr.whisper_lang("ko") == "ko" and asr.whisper_lang("ko-KR") == "ko"
    for bad in ("", None, "unknown", "mixed"):
        with pytest.raises(ValueError):
            asr.whisper_lang(bad)


# ---------------------------------------------------------------- 전체 흐름

def test_process_writes_valid_asr_json_in_video_time(proj):
    st, msg = run_asr(proj)
    assert st == "ok" and "묶음 5개" in msg
    assert validate.validate_asr(proj.out) == []
    kind, errs = validate.check_file(proj.out)
    assert kind == "asr" and errs == []
    assert [p.name for p in proj.out.parent.iterdir()] == ["asr.json"]         # work/ 에는 다른 파일을 두지 않는다
    data = util.read_json(proj.out)
    assert list(data) == ["schema", "video_sha1", "lang", "passes", "vad", "chunks"]
    assert data["schema"] == "sink.asr/1" and data["video_sha1"] == proj.sha and data["lang"] == "ko"
    (ps,) = data["passes"]
    assert list(ps) == ["id", "model", "context", "words"] and ps["id"] == "plain" and ps["context"] is None
    # 모델 문자열: 모델·라이브러리·디코딩(온도 되돌이 켜짐)·말 구간·묶음 규칙·시각 대응·첫 낱말 보정·코드 판
    assert ps["model"] == ("faster-whisper 9.9 | fake/tiny@ab12cd34 | ct2 8.8 cpu int8 | beam5 nocond tfallback | "
                           "fake-vad thr0.35 min100 sil300 pad150 max25 | join1.0 pad0.1 | timemap map | "
                           f"firstword +0.05 | sinkprep.asr {asr.VERSION} decode{asr.DECODE_VERSION}")
    words = ps["words"]
    assert len(words) == 15 and [w["i"] for w in words] == list(range(15))
    for w in words:
        assert list(w) == ["i", "w", "s", "e", "p"] and w["s"] <= w["e"] and w["p"] == 0.9
        assert round(w["s"], 3) == w["s"] and round(w["e"], 3) == w["e"]
    # 묶음 차례·묶음 안 차례 그대로이고, 시각은 영상 시각: 첫 묶음(소리 0.9~6.1초)은 그대로,
    # 둘째 묶음부터는 틈만큼 늦다. 묶음 첫 낱말은 조각 시작(묶음 − 0.1초)이 아니라 묶음 시작 + 0.05초에서 시작한다
    assert [w["w"] for w in words[:3]] == ["낱말0", "낱말1", "낱말2"]
    assert words[0]["s"] == 1.05 and words[2]["e"] == pytest.approx(0.9 + 5.2, abs=0.0015)
    assert words[1]["s"] == pytest.approx(0.9 + 5.2 / 3, abs=0.0015)         # 첫 낱말이 아니면 그대로
    assert words[6]["s"] == pytest.approx(21.05 + 0.05, abs=0.0015)          # 셋째 묶음: 소리 21.05초 -> 영상 21.10초
    assert words[9]["s"] == pytest.approx(44.10 + 0.11, abs=0.0015)          # 넷째 묶음(덧댐 0.025초): 틈 둘 뒤
    for k in range(5):                                                       # 첫 낱말이 묶음 시작보다 앞서지 않는다
        assert words[3 * k]["s"] == pytest.approx(data["chunks"][k]["start"] + 0.05, abs=0.0015)
    starts = [w["s"] for w in words]
    assert starts == sorted(starts)
    assert data["vad"][0] == [1.0, 3.0] and data["vad"][4] == [21.05, 44.11] and len(data["vad"]) == 7
    assert data["chunks"] == [{"start": 1.0, "end": 6.0, "lang": "ko"}, {"start": 10.0, "end": 14.0, "lang": "ko"},
                              {"start": 21.05, "end": 44.11, "lang": "ko"},
                              {"start": 44.16, "end": 50.11, "lang": "ko"},
                              {"start": 55.11, "end": 60.06, "lang": "ko"}]
    assert proj.fake.vad_calls == 1 and len(proj.fake.calls) == 5 and proj.fake.loads == [("tiny", "cpu", 2)]
    assert util.all_problems() == []


def test_assemble_maps_words_across_a_gap():
    """틈(소리 40초)을 걸친 낱말은 영상 시각으로 그만큼 길어지고, 뒤집힌 시각은 s 로 맞춘다. 묶음 차례를 지킨다."""
    tm = media.TimeMap("a" * 40, "map", 1024 / 48000, SEGMENTS, WAV_S)
    recs = [{"words": [{"w": "가", "s": 39.9, "e": 40.1, "p": 0.5}, {"w": "나", "s": 41.0004, "e": 41.0, "p": 1.0}]},
            {"words": [{"w": "다", "s": 50.0, "e": 50.5, "p": 0.25}]}]
    data = asr.assemble("a" * 40, "ko", tm, [(s(39.0), s(51.0))], [(s(39.0), s(51.0))], [("plain", "m", recs)])
    assert data["passes"][0]["words"] == [{"i": 0, "w": "가", "s": 39.95, "e": 40.21, "p": 0.5},
                                          {"i": 1, "w": "나", "s": 41.11, "e": 41.11, "p": 1.0},
                                          {"i": 2, "w": "다", "s": 50.11, "e": 50.61, "p": 0.25}]
    assert data["vad"] == [[39.05, 51.11]] and data["chunks"] == [{"start": 39.05, "end": 51.11, "lang": "ko"}]
    # 낱말을 직접 주는 꼴(rescue): context 를 적고 낱말을 그 차례대로 영상 시각으로 옮긴다
    data = asr.assemble("a" * 40, "ko", tm, [], [], [("plain", "m", recs), {
        "id": "rescue", "model": "r", "context": "full_window_stable",
        "words": [{"w": "라", "s": 5.0, "e": 5.5, "p": 0.7}, {"w": "마", "s": 45.0, "e": 45.2, "p": 0.6}]}])
    assert [p["id"] for p in data["passes"]] == ["plain", "rescue"] and data["passes"][1]["context"] == "full_window_stable"
    assert data["passes"][1]["words"] == [{"i": 0, "w": "라", "s": 5.0, "e": 5.5, "p": 0.7},
                                          {"i": 1, "w": "마", "s": 45.11, "e": 45.31, "p": 0.6}]


def rec_of(a, b, words, pad=0.1, segs=None):
    """묶음 기록(소리 시각). words: [(글, 시작, 끝)] 또는 [(글, 시작, 끝, p)]. segs: [(시작, 끝)] (없으면 낱말 전체 하나)."""
    ws = [{"w": w[0], "s": w[1], "e": w[2], "p": w[3] if len(w) > 3 else 0.9} for w in words]
    if segs is None:
        segs = [(ws[0]["s"], ws[-1]["e"])] if ws else []
    return {"key": "k", "a": s(a), "b": s(b), "pa": s(a - pad), "pb": s(b + pad), "lang": "ko", "words": ws,
            "segs": [{"start": x, "end": y, "text": "", "avg_logprob": -0.3, "no_speech_prob": 0.0,
                      "compression_ratio": 1.2, "temperature": 0.0} for x, y in segs], "seconds": 1.0}


def test_first_word_start_is_pulled_to_chunk_start():
    """묶음 첫 낱말이 잘라 낸 조각의 시작에 붙어 있으면 묶음 시작 + 0.05초로 당긴다(추격자: 514묶음 가운데 508개가
    붙어 있었고 첫 낱말이 평균 0.14초 길었다). 캐시 기록은 고치지 않는다."""
    r = rec_of(10.0, 12.0, [("안녕", 9.9, 10.5), ("하세요", 10.5, 11.2)])
    before = [dict(w) for w in r["words"]]
    assert asr.record_words(r) == [{"w": "안녕", "s": 10.05, "e": 10.5, "p": 0.9},
                                   {"w": "하세요", "s": 10.5, "e": 11.2, "p": 0.9}]
    assert r["words"] == before
    # 이웃 묶음이 가까워 덧댐이 0.025초뿐인 묶음도 묶음 시작 + 0.05초
    assert asr.record_words(rec_of(10.0, 12.0, [("안녕", 9.975, 10.5)], pad=0.025))[0]["s"] == pytest.approx(10.05)
    # 짧은 첫 낱말은 길이 0.1초를 남긴다. 그보다 짧으면 그대로 둔다(s ≤ e)
    assert asr.record_words(rec_of(10.0, 12.0, [("네", 9.9, 10.1)]))[0]["s"] == pytest.approx(10.0)
    assert asr.record_words(rec_of(10.0, 12.0, [("네", 9.9, 9.95)]))[0] == {"w": "네", "s": 9.9, "e": 9.95, "p": 0.9}
    # 조각 시작에 붙지 않은 첫 낱말(faster-whisper 가 이미 줄인 긴 첫 낱말)은 그대로
    assert asr.record_words(rec_of(10.0, 12.0, [("안녕", 11.0, 11.5)]))[0]["s"] == 11.0
    # 25초에서 잘려 덧대지 않은 묶음: 조각 시작 = 묶음 시작이라 0.05초만 당긴다
    assert asr.record_words(rec_of(25.0, 30.0, [("그래서", 25.0, 25.6)], pad=0.0))[0]["s"] == pytest.approx(25.05)
    # 범위 칸이 없는 기록은 건드리지 않는다
    assert asr.record_words({"words": [{"w": "가", "s": 1.0, "e": 1.5, "p": 0.5}]})[0]["s"] == 1.0
    for w in asr.record_words(r):
        assert w["s"] <= w["e"]


def test_undecodable_words_are_dropped_at_assembly():
    """글자를 못 푼 낱말(U+FFFD)은 asr.json 에 넣지 않는다. 낱말별 통계도 같은 낱말을 빼고 차례가 맞는다."""
    tm = media.TimeMap.identity("a" * 40, WAV_S)
    r1 = rec_of(1.0, 5.0, [("이�年前,", 0.9, 1.4, 0.007), ("그래", 1.5, 2.0), ("tą�", 2.0, 2.2, 0.0), ("맞아", 3.5, 4.0)],
                segs=[(0.9, 2.2), (3.5, 4.0)])
    r1["segs"][1].update(temperature=1.0, avg_logprob=-1.3)
    r2 = rec_of(8.0, 9.0, [("끝", 7.9, 8.5)])
    data = asr.assemble("a" * 40, "ko", tm, [], [], [("plain", "m", [r1, r2])])
    words = data["passes"][0]["words"]
    assert [(w["i"], w["w"]) for w in words] == [(0, "그래"), (1, "맞아"), (2, "끝")]
    assert words[0]["s"] == 1.5                                   # 첫 낱말이 버려졌으면 둘째 낱말을 당기지 않는다
    stats = asr.pass_word_stats([r1, r2])
    assert len(stats) == len(asr.pass_words([r1, r2])) == 3
    assert [x["chunk"] for x in stats] == [0, 0, 1]
    assert [x["temperature"] for x in stats] == [0.0, 1.0, 0.0] and stats[1]["avg_logprob"] == -1.3


def test_process_rerun_skips_and_force_is_byte_identical(proj):
    """같은 입력이면 skip. --force 로 다시 조립해도(묶음 캐시에서) 바이트가 같고 받아쓰기를 다시 부르지 않는다."""
    assert run_asr(proj)[0] == "ok"
    first = proj.out.read_bytes()
    assert run_asr(proj) == ("skip", "")
    assert run_asr(proj, force=True)[0] == "ok"
    assert proj.out.read_bytes() == first
    assert proj.fake.vad_calls == 1 and len(proj.fake.calls) == 5
    # 결과 파일이 손상되면 지문이 같아도 다시 만든다
    proj.out.write_bytes(first + b" ")
    assert run_asr(proj)[0] == "ok" and proj.out.read_bytes() == first and len(proj.fake.calls) == 5


def test_process_resumes_from_chunk_cache(proj):
    """중간에 죽은 것처럼 묶음 캐시 하나와 결과를 지우면, 다시 돌릴 때 그 묶음만 받아쓴다."""
    run_asr(proj)
    first = proj.out.read_bytes()
    tag = "tiny@ab12cd34-cpu-int8"
    files = sorted((paths.ASR_CACHE / proj.sha / tag).glob("*.json"))
    assert [f.name for f in files] == [f"{a:010d}-{b:010d}.json" for a, b in GROUPS]
    rec = util.read_json(files[2])
    assert {"key", "a", "b", "pa", "pb", "lang", "words", "segs", "seconds"} <= set(rec)
    assert rec["words"][0]["s"] == pytest.approx(20.9) and rec["segs"][0]["avg_logprob"] == -0.2   # 소리 시각, 세그먼트 통계
    files[2].unlink()
    proj.out.unlink()
    assert run_asr(proj)[0] == "ok"
    assert len(proj.fake.calls) == 6 and proj.fake.calls[-1] == ("tiny", GROUPS[2][1] - GROUPS[2][0] + s(0.1) + s(0.025))
    assert proj.out.read_bytes() == first and proj.fake.vad_calls == 1


def test_pass_id_is_not_part_of_chunk_cache_key(proj):
    """pass id 를 바꿔도 같은 모델의 묶음 캐시를 다시 쓴다. 같은 모델을 두 pass 에 쓰면 한 번만 받아쓴다."""
    run_asr(proj)
    assert run_asr(proj, passes="other=tiny")[0] == "ok"
    assert len(proj.fake.calls) == 5
    data = util.read_json(proj.out)
    assert [p["id"] for p in data["passes"]] == ["other"] and len(data["passes"][0]["words"]) == 15
    assert run_asr(proj, passes="plain=tiny,again=tiny")[0] == "ok"
    assert len(proj.fake.calls) == 5 and len(proj.fake.loads) == 3          # 모델은 실행마다 한 번, pass 마다가 아님
    data = util.read_json(proj.out)
    assert [p["id"] for p in data["passes"]] == ["plain", "again"]
    assert data["passes"][0]["words"] == data["passes"][1]["words"]
    assert validate.validate_asr(proj.out) == []
    # 다른 모델은 다른 폴더에 따로 받아쓴다
    assert run_asr(proj, passes="plain=tiny,big=base")[0] == "ok"
    assert len(proj.fake.calls) == 10 and proj.fake.loads[-2:] == [("tiny", "cpu", 2), ("base", "cpu", 2)]
    assert sorted(p.name for p in (paths.ASR_CACHE / proj.sha).iterdir()) == \
        ["base@ab12cd34-cpu-int8", "tiny@ab12cd34-cpu-int8", "vad.json"]


def test_decode_version_invalidates_chunk_cache(proj, monkeypatch):
    run_asr(proj)
    monkeypatch.setattr(asr, "DECODE_VERSION", "test2")
    assert run_asr(proj)[0] == "ok" and len(proj.fake.calls) == 10


def test_low_confidence_is_recorded_and_cleared(proj, monkeypatch):
    """낱말 평균 확률이 0.5 미만이면 영화 수준 asr_low_conf 를 남기고, 나아지면 지운다."""
    proj.fake.p = 0.2
    run_asr(proj)
    (p,) = util.all_problems()
    assert p["level"] == "film" and p["film_id"] == FILM and p["doc_id"] == DOC and p["problem"] == "asr_low_conf"
    assert p["score"] == 0.2 and p["path"] == f"films/{FILM}/work/asr.json"
    assert (paths.PROBLEMS / "asr" / f"{FILM}.json").exists()
    proj.fake.p = 0.8
    monkeypatch.setattr(asr, "DECODE_VERSION", "again")
    run_asr(proj)
    assert util.all_problems() == []


def test_process_rejects_changed_video_and_unknown_lang(proj):
    with pytest.raises(ValueError):
        asr.process(dict(proj.doc, video_lang="unknown"), passes="plain=tiny")
    with pytest.raises(ValueError):
        asr.process(dict(proj.doc, video_sha1="0" * 40), passes="plain=tiny")
    with pytest.raises(FileNotFoundError):
        asr.process(dict(proj.doc, video_path="videos/none.mp4"), passes="plain=tiny")
    with pytest.raises(ValueError):
        asr.process(dict(proj.doc, video_path=""), passes="plain=tiny")
    assert not proj.out.exists() and proj.fake.calls == []


def test_empty_vad_gives_valid_empty_pass(proj):
    proj.fake.regions = []
    assert run_asr(proj)[0] == "ok"
    data = util.read_json(proj.out)
    assert data["passes"][0]["words"] == [] and data["vad"] == [] and data["chunks"] == []
    assert validate.validate_asr(proj.out) == []


def catalog_row(doc_id, film, **kw):
    r = {k: "" for k in catalog.FIELDS}
    r.update(doc_id=doc_id, post_id=doc_id[2:], category="ko_film", files=f"filmmakers_scripts/x/{doc_id}.hwp",
             sha1=f"{int(doc_id[2:]):040x}", real_format="hwp5", file_formats="hwp5", title_raw="가", title_norm="가",
             script_lang="ko", is_translation="false", version_label="unknown", film_id=film, film_title="가",
             is_primary="false", status="parsed")
    r.update(kw)
    return r


def test_run_selects_primary_docs_with_video(proj, capsys):
    """명령: 영상이 등록된 영화의 대표 문서만 받아쓴다. 실패한 영화가 있으면 종료 코드 1."""
    video = {k: proj.doc[k] for k in ("video_path", "video_sha1", "video_duration", "video_lang")}
    rows = [catalog_row("fm1", FILM, is_primary="true", **video),
            catalog_row("fm2", FILM, **video),                               # 대표가 아님
            catalog_row("fm3", "local_0002", is_primary="true"),            # 영상 없음
            catalog_row("fm4", FILM, status="excluded", note="dup", **video)]
    catalog.write_catalog(rows)
    args = argparse.Namespace(cmd="asr", docs=None, limit=None, force=False, workers=None, paths=[],
                              passes="plain=tiny", threads=2, device="cpu")
    assert [d["doc_id"] for d in asr.select_docs(args)] == ["fm1"]
    assert asr.run(args) == 0
    out = capsys.readouterr().out
    assert f"[1/1] {FILM} ok" in out and "asr: {'ok': 1}" in out and f"[asr] {FILM} plain 5/5 100%" in out
    assert validate.validate_asr(proj.out) == []
    assert asr.run(args) == 0 and "asr: {'skip': 1}" in capsys.readouterr().out
    args.docs = "fm3"
    assert asr.run(args) == 0 and "대상 없음" in capsys.readouterr().out
    (paths.VIDEOS / f"{FILM}.mp4").unlink()
    args.docs = None
    assert asr.run(args) == 1
    assert f"! {FILM}: FileNotFoundError" in capsys.readouterr().err


# ---------------------------------------------------------------- 모델 고르기

def test_pick_device_and_handle_strings(monkeypatch):
    assert asr.pick_device("cpu") == ("cpu", "int8") and asr.pick_device("cuda") == ("cuda", "float16")
    monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: 1))
    monkeypatch.setattr(asr, "free_vram_mib", lambda: 900)
    assert asr.pick_device("auto") == ("cpu", "int8")                        # GPU 가 다른 일로 차 있음
    monkeypatch.setattr(asr, "free_vram_mib", lambda: 8000)
    assert asr.pick_device("auto") == ("cuda", "float16")
    monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: 0))
    assert asr.pick_device("auto") == ("cpu", "int8")
    h = asr.ModelHandle("large-v3-turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
                        "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf", "cpu", "int8",
                        versions={"faster_whisper": "1.2.1", "ctranslate2": "4.8.3"}, model=object())
    assert h.tag == "large-v3-turbo@0a363e91-cpu-int8"
    assert h.describe() == "faster-whisper 1.2.1 | mobiuslabsgmbh/faster-whisper-large-v3-turbo@0a363e91 | ct2 4.8.3 cpu int8"
    assert asr._repo_snapshot(r"C:\hf\hub\models--Systran--faster-whisper-large-v3\snapshots\edaa852e", "large-v3") == \
        ("Systran/faster-whisper-large-v3", "edaa852e")
    k1 = asr.chunk_key(h, "ko", 10, 20, 5, 25, "raw")
    assert k1 == asr.chunk_key(h, "ko", 10, 20, 5, 25, "raw")
    for other in (asr.chunk_key(h, "en", 10, 20, 5, 25, "raw"), asr.chunk_key(h, "ko", 10, 20, 6, 25, "raw"),
                  asr.chunk_key(h, "ko", 10, 20, 5, 25, "padded"), asr.chunk_key(h, "ko", 10, 20, 5, 25, "raw", "힌트")):
        assert other != k1


# ---------------------------------------------------------------- teacher forcing 결과 묶기

EOT = 10 ** 6


class CharTok:
    """글자 하나가 토큰 하나인 가짜 토크나이저(faster-whisper Tokenizer 의 공백 기준 낱말 나누기를 흉내)."""
    eot = EOT
    sot_sequence = [EOT + 1, EOT + 2, EOT + 3]

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(t) for t in ids if t < EOT)

    def split_to_word_tokens(self, tokens):
        words, toks = [], []
        for t in tokens:
            sub = "" if t >= EOT else chr(t)
            if t >= EOT or sub.startswith(" ") or sub.strip() in string.punctuation or not words:
                words.append(sub)
                toks.append([t])
            else:
                words[-1] += sub
                toks[-1].append(t)
        return words, toks


def pairs_from_starts(starts, end):
    """토큰마다 시작 프레임(1/50초) -> (토큰 번호, 시간 번호) 짝. 끝 토큰(EOT) 줄 포함."""
    edges = list(starts) + [end, end + 1]
    return [(k, f) for k in range(len(starts) + 1) for f in range(edges[k], edges[k + 1])]


def test_force_result_groups_tokens_into_words():
    """토큰 확률·정렬 짝 -> 낱말. 문장부호는 앞 낱말에 붙고(시각·확률은 낱말 것), 점수는 문장부호를 빼고도 낸다."""
    tok = CharTok()
    text = " 안녕 하세요. 네!"
    ids = tok.encode(text)                                   # 11토큰
    probs = [0.5, 0.8, 0.6, 0.9, 0.7, 0.7, 0.7, 0.1, 0.9, 0.4, 0.2]
    starts = [k * 5 for k in range(len(ids))]                # 토큰마다 0.1초
    r = asr._force_result(tok, ids, probs, pairs_from_starts(starts, 55), n_frames=200)
    assert r["n_tokens"] == 11 and r["tokens"] == ids
    assert r["token_logp"] == pytest.approx(list(np.log(probs)))
    assert r["logp_sum"] == pytest.approx(float(np.log(probs).sum()))
    assert r["logp_mean"] == pytest.approx(float(np.log(probs).mean()))
    assert r["punct"] == [False] * 7 + [True, False, False, True]
    keep = [p for p, k in zip(probs, r["punct"]) if not k]
    assert r["n_tokens_nopunct"] == 9 and r["logp_sum_nopunct"] == pytest.approx(float(np.log(keep).sum()))
    assert r["logp_mean_nopunct"] == pytest.approx(float(np.log(keep).mean()))
    assert [w["w"] for w in r["words"]] == ["안녕", "하세요.", "네!"]
    assert [(w["s"], w["e"]) for w in r["words"]] == [(0.0, 0.3), (0.3, 0.7), (0.8, 1.0)]
    assert [(w["t0"], w["t1"]) for w in r["words"]] == [(0, 3), (3, 8), (8, 11)]
    assert [w["p"] for w in r["words"]] == [round(float(np.mean(probs[0:3])), 3), round(float(np.mean(probs[3:7])), 3),
                                            round(float(np.mean(probs[8:10])), 3)]
    # 낱말 토큰 범위로 줄 하나의 점수를 다시 낼 수 있다
    w = r["words"][1]
    assert sum(r["token_logp"][w["t0"]:w["t1"]]) == pytest.approx(float(np.log(probs[3:8]).sum()))


def test_force_result_shortens_first_word_after_pause():
    """조각 앞 5초가 말 없는 소리일 때: 첫 낱말이 쉼을 삼키지 않게 줄인다(faster-whisper 와 같은 규칙)."""
    tok = CharTok()
    ids = tok.encode(" 네 그래요 알았어요")
    starts = [0, 1, 260, 261, 265, 270, 275, 280, 285, 290, 295]
    r = asr._force_result(tok, ids, [0.9] * len(ids), pairs_from_starts(starts, 300), n_frames=700)
    # 정렬 그대로면 '네' 가 0.0~5.2초. 낱말 길이 중앙값 0.5초의 두 배(1.0초)로 줄인다
    assert [(w["w"], w["s"], w["e"]) for w in r["words"]] == [("네", 4.2, 5.2), ("그래요", 5.2, 5.5),
                                                              ("알았어요", 5.5, 6.0)]
    # 쉼이 없으면 그대로 둔다
    r2 = asr._force_result(tok, ids, [0.9] * len(ids), pairs_from_starts(range(0, 55, 5), 55), n_frames=700)
    assert [(w["s"], w["e"]) for w in r2["words"]] == [(0.0, 0.2), (0.2, 0.6), (0.6, 1.1)]


class FakeAligner:
    def __init__(self):
        self.seen = []
        self.tokens_per_second = 50
        self.model = self

    def align(self, enc, sot, tokens, n_frames, median_filter_width=7):
        self.seen.append((enc, list(sot), [list(t) for t in tokens], n_frames, median_filter_width))
        n = len(tokens[0])
        return [types.SimpleNamespace(text_token_probs=[0.5] * n, alignments=pairs_from_starts(range(0, n * 2, 2), n * 2))]


class FakeHandle(asr.ModelHandle):
    def tokenizer(self, lang):
        return CharTok()


def test_force_text_calls_align_and_checks_limits():
    """가짜 handle: 글 앞에 공백을 붙여 토큰으로 바꾸고 인코더 출력과 함께 align 에 넘긴다. 빈 글·448토큰 초과는 ValueError."""
    al = FakeAligner()
    h = FakeHandle("m", "r/m", "s" * 40, "cpu", "int8", model=al)
    r = asr.force_text(h, "ENC", 150, "어디 가", "ko")
    (enc, sot, toks, n_frames, width), = al.seen
    assert enc == "ENC" and sot == CharTok.sot_sequence and n_frames == 150 and width == 7
    assert "".join(chr(t) for t in toks[0]) == " 어디 가"
    assert r["n_tokens"] == 5 and [w["w"] for w in r["words"]] == ["어디", "가"]
    assert r["logp_mean"] == pytest.approx(float(np.log(0.5)))
    with pytest.raises(ValueError):
        asr.force_text(h, "ENC", 150, "   ", "ko")
    with pytest.raises(ValueError):
        asr.force_text(h, "ENC", 150, "가" * 450, "ko")
    asr.force_text(h, "ENC", 150, "가" * 442, "ko")              # 3(시작 열) + 1 + 443 + 1 = 448: 한도 안


def test_encode_clip_rejects_long_or_empty_clip():
    h = FakeHandle("m", "r/m", "s" * 40, "cpu", "int8", model=object())
    with pytest.raises(ValueError):
        asr.encode_clip(h, np.zeros(SR * 30 + 1, dtype=np.float32))
    with pytest.raises(ValueError):
        asr.encode_clip(h, np.zeros(0, dtype=np.float32))


def test_backend_default_and_missing_package_message(monkeypatch):
    asr.set_backend(None)
    assert isinstance(asr.get_backend(), asr.FasterWhisperBackend) and asr.get_backend().vad_name == "silero-v6"
    monkeypatch.setitem(sys.modules, "faster_whisper", None)                 # 설치되지 않은 것처럼
    with pytest.raises(asr.AsrUnavailable) as e:
        asr.load_model("large-v3-turbo", device="cpu")
    assert "pip install -e .[asr]" in str(e.value)


# ---------------------------------------------------------------- 진짜 모델(켤 때만)

@pytest.mark.skipif(os.environ.get("SINK_ASR_REAL") != "1", reason="SINK_ASR_REAL=1 일 때만(모델을 메모리에 올림)")
def test_real_model_transcribes_and_forces():
    pytest.importorskip("faster_whisper")
    h = asr.load_model("large-v3-turbo", device="cpu", threads=4)
    try:
        t = np.arange(SR * 3) / SR
        clip = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        words, segs = asr.transcribe_clip(h, clip, "ko")
        assert isinstance(words, list) and all({"w", "s", "e", "p"} == set(w) for w in words)
        enc, n = asr.encode_clip(h, clip)
        r = asr.force_text(h, enc, n, "안녕하세요.", "ko")
        assert n == 300 and r["n_tokens"] > 0 and r["words"][0]["w"] == "안녕하세요."
        assert r["logp_sum"] < 0 and r["logp_mean_nopunct"] is not None
    finally:
        h.close()


# ---------------------------------------------------------------- 검토에서 나온 경우(회귀 시험)

class FakeWhisper:
    """faster_whisper.WhisperModel 흉내: transcribe 가 (세그먼트 반복자, 정보)를 돌려준다."""

    def __init__(self, segs):
        self.segs = segs
        self.kw = None

    def transcribe(self, clip, **kw):
        self.kw = kw
        return iter(self.segs), None


def fw_seg(start, end, text, words, **kw):
    stats = dict(avg_logprob=-0.25, no_speech_prob=3.2e-7, compression_ratio=1.3, temperature=0.0)
    stats.update(kw)
    return types.SimpleNamespace(start=start, end=end, text=text, words=[
        types.SimpleNamespace(word=w, start=a, end=b, probability=p) for w, a, b, p in words], **stats)


def test_transcribe_clip_cleans_words_and_passes_decode_options():
    """진짜 받아쓰기 호출부를 가짜 모델로: 디코딩 옵션 전달, 공백 떼기, 빈 낱말 버리기, NFC, 시각을 조각 안으로 자르기."""
    nfd = "\u1112\u1161\u11ab"                                    # '한' 을 자모로 푼 것
    model = FakeWhisper([
        fw_seg(-0.2, 1.0, " 안녕 하세요 ", [(" 안녕", -0.2, 0.4, 0.91234), ("   ", 0.4, 0.5, 0.5), (" 하세요", 0.5, 1.0, 1.2)]),
        fw_seg(1.5, 9.0, " " + nfd, [(" " + nfd, 1.5, 9.0, float("nan")), (" 끝", 2.5, 2.0, 0.3)],
               temperature=1.0, avg_logprob=-1.23456789, compression_ratio=float("inf")),
    ])
    h = asr.ModelHandle("m", "r/m", "s" * 40, "cpu", "int8", model=model)
    clip = np.zeros(2 * SR, dtype=np.float32)                     # 2초 조각
    words, segs = asr.transcribe_clip(h, clip, "ko")
    assert model.kw == dict(language="ko", hotwords=None, initial_prompt=None, **asr.DECODE)
    assert "temperature" not in model.kw and model.kw["vad_filter"] is False     # 온도는 faster-whisper 기본값
    assert words == [{"w": "안녕", "s": 0.0, "e": 0.4, "p": 0.912}, {"w": "하세요", "s": 0.5, "e": 1.0, "p": 1.0},
                     {"w": "한", "s": 1.5, "e": 2.0, "p": 0.0}, {"w": "끝", "s": 2.0, "e": 2.0, "p": 0.3}]
    assert [x["text"] for x in segs] == ["안녕 하세요", "한"]
    assert segs[0] == {"start": -0.2, "end": 1.0, "text": "안녕 하세요", "avg_logprob": -0.25,
                       "no_speech_prob": 3.2e-7, "compression_ratio": 1.3, "temperature": 0.0}
    assert segs[1]["temperature"] == 1.0 and segs[1]["avg_logprob"] == -1.2346 and segs[1]["compression_ratio"] is None
    # 힌트는 그대로 넘어간다(빈 글은 None)
    asr.transcribe_clip(h, clip, "en", hotwords="엄중호", initial_prompt="")
    assert model.kw["hotwords"] == "엄중호" and model.kw["initial_prompt"] is None and model.kw["language"] == "en"


def test_clean_regions_sorts_clips_and_removes_overlap():
    got = asr._clean_regions([(50, 80), (-5, 10), (5, 20), (70, 90), (200, 300), (95, 95)], 250)
    assert got == [(0, 10), (10, 20), (50, 80), (80, 90), (200, 250)]
    assert asr._clean_regions([], 100) == []


def test_speech_regions_runs_installed_silero():
    """설치된 판의 VadOptions 가 VAD_OPTIONS 를 받고, 2초 무음에서는 말 구간이 없다(Silero 는 작아 바로 돈다)."""
    pytest.importorskip("faster_whisper")
    assert asr.speech_regions(np.zeros(2 * SR, dtype=np.float32)) == []


def test_load_model_lazy_opens_on_first_use(monkeypatch):
    opened = []
    monkeypatch.setattr(asr, "resolve_model", lambda name: (name, "X:/model", "org/" + name, "c0ffee00" + "0" * 32,
                                                             {"faster_whisper": "1", "ctranslate2": "2"}))
    monkeypatch.setattr(asr, "_open", lambda h: opened.append(h.name) or object())
    h = asr.load_model("large-v3-turbo", device="cpu", lazy=True)
    assert not h.opened and opened == [] and h.tag == "large-v3-turbo@c0ffee00-cpu-int8"
    assert h.key()["snapshot"].startswith("c0ffee00")              # 열쇠·꼬리표는 모델을 열지 않고 구한다
    assert h.model is not None and h.opened and opened == ["large-v3-turbo"]
    h.close()
    assert not h.opened
    assert asr.load_model("large-v3-turbo", device="cpu").opened and opened == ["large-v3-turbo"] * 2


def test_default_arguments_continue_the_previous_run(proj):
    """옵션 없이 다시 돌리면 그 영화를 앞서 받아쓴 모델·장치를 이어 쓴다(회귀: 예전에는 기본값 plain=large-v3 로
    처음부터 다시 받아써 asr.json 을 갈아 끼웠다)."""
    from sinkprep.__main__ import build_parser
    ns = build_parser().parse_args(["asr"])
    assert ns.passes is None and ns.device is None and ns.threads is None and ns.rescue is None
    assert build_parser().parse_args(["asr", "--rescue"]).rescue is True
    assert build_parser().parse_args(["asr", "--no-rescue"]).rescue is False
    assert asr.DEFAULT_PASSES == "plain=large-v3-turbo"            # 실제로 재 본 모델
    assert run_asr(proj)[0] == "ok"
    first = proj.out.read_bytes()
    assert asr.process(proj.doc) == ("skip", "")
    assert proj.fake.loads == [("tiny", "cpu", 2), ("tiny", "cpu", 6)] and len(proj.fake.calls) == 5
    assert asr.process(proj.doc, force=True)[0] == "ok"
    assert proj.out.read_bytes() == first and len(proj.fake.calls) == 5 and proj.fake.loads[-1][0] == "tiny"
    # 지문이 없는 영화는 기본값으로 간다
    (paths.STATE / "asr" / f"{FILM}.json").unlink()
    assert asr.process(proj.doc)[0] == "ok" and proj.fake.loads[-1] == ("large-v3-turbo", "auto", 6)


def test_model_snapshot_or_library_change_is_not_skipped(proj):
    """모델 스냅숏이나 라이브러리 버전이 바뀌면 묶음 열쇠가 달라지므로 건너뛰지 않고 다시 받아쓴다."""
    run_asr(proj)
    assert run_asr(proj) == ("skip", "")
    proj.fake.snapshot = "ffff0000" + "1" * 32
    assert run_asr(proj)[0] == "ok" and len(proj.fake.calls) == 10
    assert "fake/tiny@ffff0000" in util.read_json(proj.out)["passes"][0]["model"]
    assert sorted(p.name for p in (paths.ASR_CACHE / proj.sha).iterdir()) == \
        ["tiny@ab12cd34-cpu-int8", "tiny@ffff0000-cpu-int8", "vad.json"]
    assert run_asr(proj) == ("skip", "")
    proj.fake.versions = {"faster_whisper": "10.0", "ctranslate2": "8.8"}
    assert run_asr(proj)[0] == "ok" and len(proj.fake.calls) == 15
    assert "faster-whisper 10.0" in util.read_json(proj.out)["passes"][0]["model"]
    fp = util.stage_fp("asr", FILM)
    assert fp["models"] == {"tiny": "fake/tiny@" + proj.fake.snapshot + " | ctranslate2 8.8 faster_whisper 10.0"}
    assert fp["tags"] == {"plain": "tiny@ffff0000-cpu-int8"} and fp["rescue"] is False


def test_identify_lets_skip_without_loading(proj):
    """백엔드가 identify 를 주면 건너뛸지 정할 때 모델 손잡이를 만들지 않는다."""
    proj.fake.identify = lambda name: {"repo": f"fake/{name}", "snapshot": proj.fake.snapshot,
                                       "versions": proj.fake.versions}
    assert run_asr(proj)[0] == "ok" and len(proj.fake.loads) == 1
    assert run_asr(proj) == ("skip", "") and len(proj.fake.loads) == 1


def test_truncated_or_broken_chunk_cache_is_transcribed_again(proj):
    """잘린 캐시 파일, 열쇠는 맞는데 낱말이 망가진 기록: 죽지 않고 그 묶음만 다시 받아쓴다."""
    run_asr(proj)
    first = proj.out.read_bytes()
    files = sorted((paths.ASR_CACHE / proj.sha / "tiny@ab12cd34-cpu-int8").glob("*.json"))
    files[1].write_bytes(files[1].read_bytes()[:40])                           # 쓰다 만 JSON
    rec = util.read_json(files[3])
    rec["words"] = [{"w": "낱말0"}, {"w": "낱말1", "s": "x", "e": 1.0, "p": 0.9}]   # 열쇠는 맞고 내용이 망가짐
    util.write_json(files[3], rec)
    proj.out.unlink()
    assert run_asr(proj)[0] == "ok"
    assert len(proj.fake.calls) == 7 and proj.out.read_bytes() == first
    assert util.read_json(files[1])["words"][0]["w"] == "낱말0" and len(util.read_json(files[3])["words"]) == 3


def test_chunk_with_no_words_and_speech_holes_are_reported(proj):
    """말 구간인데 낱말이 하나도 안 나온 묶음과 낱말 없이 빈 자리를 요약에 찍고 문제로 남긴다."""
    proj.fake.empty = {GROUPS[2][1] - GROUPS[2][0] + s(0.1) + s(0.025)}       # 셋째 묶음(23초)에서 아무것도 안 나옴
    st, msg = run_asr(proj)
    assert st == "ok" and "낱말 0개 묶음 1개" in msg and "낱말 없는 자리(1초 이상) 1곳 22.7초" in msg
    data = util.read_json(proj.out)
    assert len(data["passes"][0]["words"]) == 12 and validate.validate_asr(proj.out) == []
    (p,) = util.all_problems()
    assert p["level"] == "film" and p["problem"] == "other" and p["kind"] == "asr_hole" and p["film_id"] == FILM
    assert p["score"] == pytest.approx(22.7 / 42.3, abs=0.001) and "1개" in p["detail"]
    # 다시 받아써 낱말이 나오면 기록이 지워진다
    proj.fake.empty = set()
    for f in (paths.ASR_CACHE / proj.sha / "tiny@ab12cd34-cpu-int8").glob("*.json"):
        f.unlink()
    st, msg = run_asr(proj, force=True)
    assert "낱말 0개 묶음 0개" in msg and "0곳 0.0초" in msg and util.all_problems() == []


def test_speech_holes_ignore_region_padding():
    """말 구간 앞뒤 여유(0.15초)는 빼고, 낱말 사이·끝의 1초 넘는 빈 자리만 센다."""
    words = [{"w": "가", "s": 1.2, "e": 1.5, "p": 1}, {"w": "나", "s": 3.0, "e": 3.4, "p": 1},
             {"w": "다", "s": 10.2, "e": 10.9, "p": 1}]
    holes = asr.speech_holes([(s(1.0), s(5.0)), (s(10.0), s(11.5)), (s(20.0), s(20.9))], words)
    assert holes == [(1.5, 3.0), (3.4, pytest.approx(4.85))]
    assert asr.speech_holes([(s(20.0), s(22.0))], words) == [(pytest.approx(20.15), pytest.approx(21.85))]
    assert asr.speech_holes([], words) == [] and asr.speech_holes([(s(1.0), s(1.9))], []) == []


def test_hinted_records_need_their_own_folder():
    """힌트는 열쇠에만 들고 파일 이름에는 없다: 힌트를 준 받아쓰기는 하위 폴더를 따로 써야 plain 기록을 덮어쓰지 않는다."""
    h = asr.ModelHandle("tiny", "fake/tiny", "ab12cd34" + "0" * 32, "cpu", "int8", versions={}, model=object())
    plain = asr.chunk_key(h, "ko", 10, 20, 5, 25, "raw")
    assert asr.chunk_key(h, "ko", 10, 20, 5, 25, "raw", hotwords="엄중호") != plain
    assert asr.chunk_key(h, "ko", 10, 20, 5, 25, "raw", initial_prompt="추격자") != plain
    p0 = asr.chunk_path("f" * 40, h.tag, 10, 20)
    p1 = asr.chunk_path("f" * 40, h.tag, 10, 20, "ctx")
    assert p0.name == p1.name == "0000000010-0000000020.json" and p1.parent == p0.parent / "ctx"
    assert asr.chunk_path("f" * 40, h.tag, 10, 20, asr.WINDOW_DIR).parent.name == "full"
    # 스레드 수는 결과를 바꾸지 않으므로 열쇠에 없다
    h2 = asr.ModelHandle("tiny", "fake/tiny", "ab12cd34" + "0" * 32, "cpu", "int8", threads=1, versions={}, model=object())
    assert asr.chunk_key(h2, "ko", 10, 20, 5, 25, "raw") == plain


def test_second_process_on_same_film_is_refused(proj, monkeypatch):
    """같은 영화를 받아쓰는 프로세스가 살아 있으면 멈춘다. 죽은 프로세스가 남긴 잠금 파일은 가져간다."""
    lock = asr.lock_path(proj.sha)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("4242\n", encoding="utf-8")
    monkeypatch.setattr(asr, "_pid_alive", lambda pid: pid == 4242)
    with pytest.raises(RuntimeError) as e:
        run_asr(proj)
    assert "pid 4242" in str(e.value) and lock.exists() and not proj.out.exists() and proj.fake.calls == []
    monkeypatch.setattr(asr, "_pid_alive", lambda pid: False)                  # 그 프로세스는 죽었다
    assert run_asr(proj)[0] == "ok" and not lock.exists()
    # 받아쓰다 예외가 나도 잠금 파일은 남지 않는다
    proj.fake.transcribe = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    for f in (paths.ASR_CACHE / proj.sha / "tiny@ab12cd34-cpu-int8").glob("*.json"):
        f.unlink()
    with pytest.raises(RuntimeError):
        run_asr(proj, force=True)
    assert not lock.exists()


def test_docs_option_accepts_any_document_of_the_film(proj, capsys):
    """--docs 에 대표가 아닌 문서를 줘도 그 영화의 대표 문서로 돈다. 대상이 안 되는 까닭을 알린다."""
    video = {k: proj.doc[k] for k in ("video_path", "video_sha1", "video_duration", "video_lang")}
    catalog.write_catalog([catalog_row("fm1", FILM, is_primary="true", **video), catalog_row("fm2", FILM, **video),
                           catalog_row("fm3", "local_0002", is_primary="true"),
                           catalog_row("fm4", FILM, status="excluded", note="dup", **video)])
    args = argparse.Namespace(cmd="asr", docs="fm2", limit=None, force=False, workers=None, paths=[],
                              passes="plain=tiny", threads=2, device="cpu", rescue=None)
    assert [d["doc_id"] for d in asr.select_docs(args)] == ["fm1"]
    assert asr.run(args) == 0
    out = capsys.readouterr().out
    assert "fm2: 대표 문서가 아님" in out and "대표 문서 fm1 로 돌림" in out and f"[1/1] {FILM} ok" in out
    args.docs = "fm3,fm999,fm4"
    assert asr.select_docs(args) == [] and asr.run(args) == 0
    out = capsys.readouterr().out
    assert "fm3: 영상이 등록되지 않음" in out and "fm999: catalog 에 없는 doc_id" in out and "fm4: 제외된 문서" in out
    assert "대상 없음" in out


def test_film_records_reads_finished_cache(proj):
    """뒤 단계가 세그먼트 통계를 찾는 길: 지문에 적힌 꼬리표로 묶음 기록을 조립 차례대로 읽는다."""
    with pytest.raises(FileNotFoundError):
        asr.film_records(FILM)
    run_asr(proj)
    recs = asr.film_records(FILM)
    assert [(r["a"], r["b"]) for r in recs] == GROUPS and all(r["segs"][0]["temperature"] == 0.0 for r in recs)
    words = util.read_json(proj.out)["passes"][0]["words"]
    assert [w["w"] for w in asr.pass_words(recs)] == [w["w"] for w in words]
    assert len(asr.pass_word_stats(recs)) == len(words)
    # 0.1.0 이 쓴 지문(꼬리표 없음)도 폴더 이름으로 찾는다
    fp = util.stage_fp("asr", FILM)
    fp.pop("tags")
    util.set_stage_fp("asr", FILM, fp)
    assert len(asr.film_records(FILM, "plain")) == 5
    with pytest.raises(FileNotFoundError):
        asr.film_records(FILM, "rescue_a")                          # rescue 를 돌린 적이 없다


# ---------------------------------------------------------------- 전체 덮기 창

def test_energy_envelope_is_integer_sum_of_squares(tmp_path):
    wav = tmp_path / "e.wav"
    data = np.concatenate([np.full(1600, 3), np.full(1600, -2), np.full(800, 4)]).astype("<i2")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())
    env = asr.energy_envelope(wav)
    assert env == [1600 * 9, 1600 * 4, 800 * 16 * 2] and all(type(x) is int for x in env)   # 끝의 반쪽 틀은 온 틀 크기로


def quiet_env(seconds, quiet_frames, loud=1000):
    env = [loud] * int(round(seconds * 10))
    for k in quiet_frames:
        env[k] = 0
    return env


def test_cover_sets_tile_the_audio_and_cut_at_quiet_places():
    """두 벌 모두 [0, 끝)를 빈틈·겹침 없이 덮고, 경계는 기준 경계 앞 5초 안의 가장 조용한 곳이며, 두 벌의 경계는 엇갈린다."""
    n = s(100.0)
    # 벌 A 의 탐색 범위(20~25, 42.35~47.35, 66.05~71.05, 88.95~93.95초) 안에 조용한 틀을 하나씩,
    # 벌 B 의 탐색 범위(벌 A 창 가운데 ±2.5초) 안에도 몇 개 심는다. 범위 밖(틀 300)의 조용한 곳은 쓰이지 않는다
    env = quiet_env(100.0, [223, 460, 689, 900] + [100, 580, 790] + [300])
    A, B = asr.cover_sets(env, n)
    for ws in (A, B):
        assert ws[0][0] == 0 and ws[-1][1] == n
        assert all(x[1] == y[0] for x, y in zip(ws, ws[1:]))                   # 빈틈·겹침 없음
        assert all(0 < b - a <= s(30.0) for a, b in ws)                        # Whisper 창(30초)을 넘지 않는다
    step = s(0.1)
    cuts_a = [a for a, _ in A[1:]]
    assert [c // step for c in cuts_a] == [223, 460, 689, 900]                 # 조용한 틀 한가운데
    assert all(c % step == step // 2 for c in cuts_a)
    assert all(s(20.0) <= b - a <= s(25.0) for a, b in A[:-1]) and A[-1] == (s(90.05), n)
    cuts_b = [a for a, _ in B[1:]]
    assert len(B) == len(A) + 1
    assert [c // step for c in cuts_b[:1] + cuts_b[2:4]] == [100, 580, 790]    # 심어 둔 조용한 곳
    assert B[0] == (0, s(10.05))                                               # 첫 창은 벌 A 첫 창의 절반쯤
    assert cuts_b[1] // step == 366                                            # 조용한 곳이 없으면 범위의 맨 뒤 틀
    for c in cuts_b:                                                           # 벌 A 경계에서 멀리(창 길이의 1/4 이상)
        assert min(abs(c - a) for a in [0] + cuts_a + [n]) >= s(2.4)
    assert all(min(abs(c - a) for a in [0] + cuts_a + [n]) >= s(7.5) for c in cuts_b[:4])
    assert asr.cover_sets(env, n) == (A, B)                                    # 같은 입력이면 같은 창(캐시 열쇠가 된다)


def test_cover_sets_edge_lengths():
    assert asr.cover_sets([], 0) == ([], [])
    # 30초 이하는 벌 A 가 창 하나, 벌 B 는 그 가운데에서 나눈 둘
    A, B = asr.cover_sets(quiet_env(12.0, []), s(12.0))
    assert A == [(0, s(12.0))] and len(B) == 2 and B[0][1] == B[1][0] and s(3.0) <= B[0][1] <= s(9.0)
    # 아주 짧은 소리도 두 벌이 같은 창이 되지 않는다
    A, B = asr.cover_sets([5], 900)
    assert A == [(0, 900)] and B == [(0, 450), (450, 900)]
    # 무음뿐이면(크기가 다 같음) 기준 경계 바로 앞 틀: 창이 25초에 가깝다
    A, B = asr.cover_sets(quiet_env(60.0, [], loud=0), s(60.0))
    assert [a for a, _ in A[1:]] == [s(24.95), s(49.85)] and [a for a, _ in B[1:]] == [s(14.85), s(39.85), s(57.35)]
    # 긴 소리: 남은 길이가 30초 이하가 될 때까지 20~25초 창, 마지막 창은 5초 넘고 30초 이하
    A, B = asr.cover_sets(quiet_env(7506.0, [], loud=7), s(7506.0))
    assert s(5.0) < A[-1][1] - A[-1][0] <= s(30.0) and len(A) == 302 and len(B) == 303
    assert all(s(20.0) <= b - a <= s(25.0) for a, b in A[:-1]) and all(0 < b - a <= s(30.0) for a, b in B)


# ---------------------------------------------------------------- rescue: 안정적인 조각만

def win(a, b, utts):
    """전체 덮기 창 기록. utts: 말 한 마디마다 [(글, 시작, 끝)] (한 마디 = 세그먼트 하나)."""
    words = [w for u in utts for w in u]
    return rec_of(a, b, words, pad=0.0, segs=[(u[0][1], u[-1][2]) for u in utts])


CAR = [("차를", 16.0, 16.5), ("이렇게", 16.5, 17.2), ("대면", 17.2, 17.7), ("어떡해요?", 17.7, 18.2)]
PLAIN_PAD = [(s(0.9), s(6.1)), (s(20.9), s(44.0))]


def texts(words):
    return [w["w"] for w in words]


def test_rescue_keeps_only_fragments_both_window_sets_agree_on():
    """벌 A 의 조각 가운데 같은 시각에 벌 B 도 같은 글을 낸 것만 남긴다. 창이 달라지면 달라지는 헛소리는 버린다."""
    a = [win(0.0, 24.95, [[("한글자막", 7.0, 7.5), ("by", 7.5, 7.7), ("누구", 7.7, 8.2)], CAR])]
    b = [win(0.0, 14.85, [[("다음", 7.0, 7.4), ("영상에서", 7.4, 7.9), ("만나요.", 7.9, 8.4)]]),
         win(14.85, 39.85, [[("차를", 16.02, 16.48), ("이렇게", 16.48, 17.25), ("대면", 17.25, 17.7), ("어떡해요?", 17.7, 18.3)]])]
    words, st = asr.rescue_words(PLAIN_PAD, a, b)
    assert texts(words) == ["차를", "이렇게", "대면", "어떡해요?"]
    assert [(w["s"], w["e"]) for w in words] == [(16.0, 16.5), (16.5, 17.2), (17.2, 17.7), (17.7, 18.2)]   # 벌 A 의 시각
    assert st == {"a_words": 7, "covered": 0, "duplicate": 0, "loop": 0, "unstable": 3, "kept": 4, "fragments": 2,
                  "kept_fragments": 1, "habits": {},
                  "dropped": {"b_silent": 0, "b_differs": 1, "b_elsewhere": 0, "same_start": 0, "habit": 0}}
    # 벌 B 가 그 시각에 아무것도 내지 않았으면 버린다
    assert asr.rescue_words(PLAIN_PAD, a, [])[0] == []
    # 같은 글이라도 3초 떨어진 시각에 나온 것은 같은 소리가 아니다(앞뒤 여유 1초)
    late = [win(14.85, 39.85, [[(w, x + 3.0, y + 3.0) for w, x, y in CAR]])]
    assert asr.rescue_words(PLAIN_PAD, a, late)[0] == []
    # 0.8초 어긋난 것은 받아들인다
    near = [win(14.85, 39.85, [[(w, x + 0.8, y + 0.8) for w, x, y in CAR]])]
    assert texts(asr.rescue_words(PLAIN_PAD, a, near)[0]) == texts(a[0]["words"][3:])


def test_rescue_needs_time_overlap_not_just_nearby_same_text():
    """같은 헛소리가 두 벌에서 이웃한 시각에 나온 경우(추격자 소리 6753초: 벌 B 창 끝과 벌 A 창 안, 0.8초 떨어짐):
    글이 같고 1초 여유 안이어도 시간이 겹치지 않으면 같은 소리가 아니다."""
    outro = ["다음", "영상에서", "만나요."]
    a = [win(41.95, 64.05, [[(w, 53.85 + k, 54.6 + k) for k, w in enumerate(outro)]])]            # 53.85~56.6초
    b = [win(31.55, 53.15, [[("다음", 51.0, 51.8), ("영상에서", 51.8, 52.6), ("만나요.", 52.6, 53.03)]])]
    words, st = asr.rescue_words([], a, b)
    assert words == [] and st["unstable"] == 3
    # 벌 B 의 낱말이 조각과 조금이라도 겹치면(진짜 말은 낱말 끝 시각이 두 벌에서 거의 같다) 남긴다
    b2 = [win(31.55, 60.0, [[("다음", 53.0, 54.0), ("영상에서", 54.0, 55.2), ("만나요.", 55.2, 56.0)]])]
    assert texts(asr.rescue_words([], a, b2)[0]) == outro
    # 같은 글이 두 번 나오는 대사: 겹치는 쪽과 맞춘다(먼저 나온 같은 글에 끌려가 버리지 않는다)
    twice_a = [win(0.0, 20.0, [[("내", 2.0, 2.3), ("딸이", 2.3, 2.8), ("있어요", 2.8, 3.4)],
                              [("내", 6.7, 7.0), ("딸이", 7.0, 7.5), ("있어요", 7.5, 8.1)]])]
    twice_b = [win(0.5, 12.0, [[("우리", 1.9, 2.3), ("딸이", 2.3, 2.8), ("있어요!", 2.8, 3.4)],
                              [("이거", 6.0, 6.4), ("딸이", 6.9, 7.5), ("있어요!", 7.5, 8.1)]])]
    words, st = asr.rescue_words([], twice_a, twice_b)
    assert len(words) == 6 and st["kept_fragments"] == 2


def test_rescue_needs_the_similar_words_themselves_to_overlap():
    """닮은 글이 이웃한 시각에 있고 그 옆의 다른 낱말 하나만 조각과 겹칠 때: 범위 전체가 조각에 걸친다고 같은 소리로
    보면 안 된다. 닮은 범위의 글자 절반 이상이 조각과 겹치는 낱말의 것이어야 한다(RESCUE_MIN_COLOC)."""
    outro = ["다음", "영상에서", "만나요."]
    a = [win(100.0, 124.0, [[(w, 112.0 + k, 112.8 + k) for k, w in enumerate(outro)]])]            # 112.0~114.8초
    # 벌 B: 조각과 겹치는 '오!' 하나, 그리고 조각이 끝난 뒤(114.9초~)의 같은 글. 모두 앞뒤 1초 여유 안에 든다
    after = [win(90.0, 113.0, [[("오!", 112.2, 112.6)]]),
             win(113.0, 135.0, [[("다음", 114.9, 115.3), ("영상에서", 115.3, 115.6), ("만나요.", 115.6, 115.75)]])]
    words, st = asr.rescue_words([], a, after)
    assert words == [] and st["dropped"] == {"b_silent": 0, "b_differs": 0, "b_elsewhere": 1, "same_start": 0, "habit": 0}
    # 조각이 시작하기 전(~111.9초)의 같은 글 + 겹치는 낱말 하나도 마찬가지
    before = [win(90.0, 112.1, [[("다음", 111.0, 111.3), ("영상에서", 111.3, 111.7), ("만나요.", 111.7, 111.9)]]),
              win(112.1, 135.0, [[("오!", 112.5, 112.9)]])]
    assert asr.rescue_words([], a, before)[0] == []
    # 글자의 절반 이상이 겹치는 낱말의 것이면 같은 소리다: 낱말 하나('다음')만 조각 앞으로 삐져나온 경우
    half = [win(90.0, 135.0, [[("다음", 111.2, 111.9), ("영상에서", 112.1, 113.4), ("만나요.", 113.4, 114.6)]])]
    assert texts(asr.rescue_words([], a, half)[0]) == outro
    assert asr.RESCUE_MIN_COLOC == 0.5


def test_rescue_zero_length_words_overlap_when_inside():
    """길이 0 인 낱말(Whisper 가 가끔 낸다)은 상대 시간대 안에 있으면 겹친 것이다. 맞닿기만 한 것은 겹침이 아니다."""
    assert asr._same_time(10.0, 12.0, 11.0, 11.0) and asr._same_time(10.0, 12.0, 12.0, 12.0)
    assert asr._same_time(11.0, 11.0, 10.0, 12.0) and asr._same_time(11.0, 11.0, 11.0, 11.0)
    assert not asr._same_time(10.0, 12.0, 12.0, 13.0) and not asr._same_time(10.0, 12.0, 9.0, 10.0)
    assert not asr._same_time(10.0, 12.0, 12.5, 12.5) and not asr._same_time(11.0, 11.0, 11.5, 12.0)
    a = [win(0.0, 20.0, [[("네", 5.0, 5.0)], [("가자", 9.0, 9.6)]])]
    b = [win(1.0, 20.0, [[("네", 4.8, 5.3)], [("가자", 9.3, 9.3)]])]
    assert texts(asr.rescue_words([], a, b)[0]) == ["네", "가자"]


def test_rescue_fragments_tell_why_each_fragment_was_dropped():
    """rescue_fragments: 조각마다 판정(kept 또는 버린 까닭)과 벌 A 낱말·시각·닮음을 돌려준다."""
    credit = [("한글자막", 6.9, 8.3), ("by", 8.3, 8.4), ("한효정", 8.4, 9.0)]
    a = [win(0.0, 21.45, [credit,                                             # 같은 표본(0)에서 시작한 창만 같은 글
                          [("진짜", 12.5, 12.9), ("대사", 12.9, 13.4)],          # 시작이 다른 창이 같은 시각에 같은 글
                          [("아무도", 15.0, 15.5), ("없는", 15.5, 15.9)],         # 벌 B 가 아무것도 안 냄
                          [("저기요", 18.0, 18.6)],                            # 벌 B 가 다른 글
                          [("!?", 20.0, 20.2)]]),                            # 글자가 없는 조각
         win(21.45, 46.0, [[("다음", 30.0, 30.6), ("영상에서", 30.6, 31.4), ("만나요.", 31.4, 32.0)]])]   # 닮은 글이 옆 시각에
    b = [win(0.0, 12.05, [[("한글자막", 6.8, 8.2), ("by", 8.2, 8.4), ("한효정", 8.4, 9.1)]]),
         win(12.05, 29.95, [[("진짜", 12.5, 12.9), ("대사", 12.9, 13.4)], [("어디", 17.9, 18.3), ("가세요", 18.3, 18.8)],
                            [("네", 20.0, 20.3)], [("다음", 29.0, 29.3), ("영상에서", 29.3, 29.6), ("만나요.", 29.6, 29.9)]])]
    frags, st = asr.rescue_fragments([], a, b)
    assert [(f["verdict"], "".join(texts(f["words"]))) for f in frags] == [
        ("same_start", "한글자막by한효정"), ("kept", "진짜대사"), ("b_silent", "아무도없는"), ("b_differs", "저기요"),
        ("b_differs", "!?"), ("b_elsewhere", "다음영상에서만나요.")]
    assert st == {"a_words": 12, "covered": 0, "duplicate": 0, "loop": 0, "habits": {}}
    f = frags[1]
    assert (f["s"], f["e"], f["window"], f["text"], f["sim"]) == (12.5, 13.4, 0, "진짜대사", 1.0)
    assert frags[0]["sim"] == 0.0 and frags[5]["window"] == s(21.45)
    words, st = asr.rescue_words([], a, b)
    assert texts(words) == ["진짜", "대사"]
    assert st == {"a_words": 12, "covered": 0, "duplicate": 0, "loop": 0, "unstable": 10, "kept": 2, "fragments": 6,
                  "kept_fragments": 1, "habits": {},
                  "dropped": {"b_silent": 1, "b_differs": 2, "b_elsewhere": 1, "same_start": 1, "habit": 0}}
    assert tuple(st["dropped"]) == asr.RESCUE_DROPS


def test_rescue_drops_habit_phrases_even_when_both_sets_agree():
    """말 없는 소리에 모델이 버릇처럼 내는 글(같은 글이 벌 A 의 서로 다른 창 3개 이상, 7자 이상)은 두 벌이 같은 시각에
    내도 버린다(추격자 소리 1952초: 벌 A '다음 영상에서 만나요', 벌 B '다음 시간에 만나요')."""
    outro = ["다음", "영상에서", "만나요."]

    def say(words, t):
        return [(w, t + 0.6 * k, t + 0.6 * k + 0.5) for k, w in enumerate(words)]

    line = [("살려", 81.0, 81.5), ("주세요", 81.5, 82.1), ("제발요", 82.1, 82.7)]            # 7자 이상인 진짜 대사
    tail = ["다음", "영상에서", "만나요", "구독과", "좋아요도", "부탁드립니다"]            # 버릇 글귀에 다른 말이 길게 붙은 변형
    short = [("만나요", 96.0, 96.5)]                                                # 버릇 글귀의 한 토막일 뿐인 짧은 말
    a = [win(0.0, 25.0, [say(outro, 5.0)]), win(25.0, 50.0, [say(outro, 30.0)]),
         win(50.0, 75.0, [say(outro, 60.0)]), win(75.0, 100.0, [line, say(["네", "다음", "영상에서", "만나요"], 90.0), short]),
         win(100.0, 125.0, [say(tail, 105.0)])]
    b = [win(12.0, 37.0, [say(["다음", "시간에", "만나요"], 30.1)]),                       # 둘째 창의 것과 같은 시각·닮은 글
         win(62.0, 87.0, [line]), win(87.0, 112.0, [say(["네", "다음", "영상에서", "만나요"], 90.0), short, say(tail, 105.0)])]
    frags, st = asr.rescue_fragments([], a, b)
    assert st["habits"] == {"다음영상에서만나요": 3}                          # 넷째·다섯째 창의 것은 글이 달라 세지 않는다
    assert [f["verdict"] for f in frags] == ["b_silent", "habit", "b_silent", "kept", "habit", "kept", "habit"]
    assert frags[1]["sim"] >= 0.6 and frags[4]["sim"] == 1.0                 # 벌 B 와는 맞았다: 버릇 글귀와 닮아서 버린다
    from rapidfuzz import fuzz
    assert fuzz.ratio(frags[6]["text"], "다음영상에서만나요") / 100 < 0.6       # 통째로는 안 닮았지만 앞 세 낱말이 그 글귀다
    words, st = asr.rescue_words([], a, b)
    assert texts(words) == ["살려", "주세요", "제발요", "만나요"] and st["dropped"]["habit"] == 3
    assert st["habits"] == {"다음영상에서만나요": 3}
    # 창 2개까지는 버릇으로 보지 않는다: 두 벌이 맞으면 남긴다
    words, st = asr.rescue_words([], a[1:4], b)
    assert st["habits"] == {} and texts(words) == outro + ["살려", "주세요", "제발요", "네", "다음", "영상에서", "만나요", "만나요"]
    # 짧은 글(7자 미만)은 여러 창에서 되풀이돼도 버릇 글귀가 아니다: '네', '감사합니다' 같은 진짜 말이 걸리지 않게
    thanks = [win(25.0 * k, 25.0 * k + 25.0, [[("감사합니다", 25.0 * k + 3.0, 25.0 * k + 3.8)]]) for k in range(4)]
    words, st = asr.rescue_words([], thanks, [dict(r, a=r["a"] + s(1.0), pa=r["a"] + s(1.0)) for r in thanks])
    assert st["habits"] == {} and texts(words) == ["감사합니다"] * 4
    # 한 창 안에서 여러 번 나온 것은 창 하나로 센다
    twice = [win(0.0, 25.0, [say(outro, 2.0), say(outro, 8.0), say(outro, 14.0)])]
    assert asr.rescue_fragments([], twice, [])[1]["habits"] == {}
    # 여러 창에서 되풀이되는 진짜 대사(구호 같은 것)는 두 벌이 대개 맞는다: 절반 넘게 맞으면 버릇 글귀가 아니다
    chant = ["독재", "타도", "민주", "쟁취"]
    ca = [win(25.0 * k, 25.0 * k + 25.0, [say(chant, 25.0 * k + 5.0)]) for k in range(4)]
    cb = [win(25.0 * k + 2.0, 25.0 * k + 27.0, [say(chant, 25.0 * k + 5.0)]) for k in range(3)]      # 넷째는 벌 B 가 놓침
    frags, st = asr.rescue_fragments([], ca, cb)
    assert st["habits"] == {} and [f["verdict"] for f in frags] == ["kept", "kept", "kept", "b_silent"]
    frags, st = asr.rescue_fragments([], ca, cb[:1])                          # 한 번만 맞으면(넷 가운데 셋이 안 맞음) 버릇
    assert st["habits"] == {"독재타도민주쟁취": 4} and [f["verdict"] for f in frags] == ["habit"] + ["b_silent"] * 3
    frags, st = asr.rescue_fragments([], ca, cb[:2])                          # 반반이면 버릇이 아니다
    assert st["habits"] == {} and [f["verdict"] for f in frags] == ["kept", "kept", "b_silent", "b_silent"]
    assert (asr.RESCUE_HABIT_CHARS, asr.RESCUE_HABIT_WINDOWS) == (7, 3)


def test_rescue_ignores_evidence_from_a_window_with_the_same_start():
    """두 벌의 첫 창은 모두 0 에서 시작해 같은 헛소리를 같은 시각에 낸다(추격자 맨 앞: 벌 A [0~21.45초] 와 벌 B [0~12.05초]
    가 둘 다 '한글자막 by 한효정'). 같은 표본에서 시작한 창끼리는 서로 증거가 되지 못한다."""
    credit = [("한글자막", 6.9, 8.3), ("by", 8.3, 8.4), ("한효정", 8.4, 9.0)]
    a = [win(0.0, 21.45, [credit, [("진짜", 15.0, 15.4), ("대사", 15.4, 15.9)]])]
    b = [win(0.0, 12.05, [[("한글자막", 5.5, 6.9), ("by", 6.9, 7.0), ("한효정", 7.0, 7.6)]]),
         win(12.05, 29.95, [[("진짜", 15.0, 15.4), ("대사", 15.4, 15.9)]])]
    words, st = asr.rescue_words([], a, b)
    assert texts(words) == ["진짜", "대사"] and st["unstable"] == 3          # 맨 앞 헛소리는 버리고 뒤의 말은 남긴다
    # 시작이 다른 창이 같은 시각에 같은 글을 냈다면 남긴다(첫 창이 아닌 곳은 늘 이 경우다)
    b[0] = dict(b[0], a=s(1.0), pa=s(1.0))
    assert texts(asr.rescue_words([], a, b)[0]) == ["한글자막", "by", "한효정", "진짜", "대사"]


def test_rescue_similarity_tolerates_small_differences_and_neighbours():
    """글이 조금 달라도(닮음 0.6 이상) 남기고, 벌 B 쪽 여유 시간대에 이웃 낱말이 섞여 있어도 맞는 범위를 찾는다."""
    a = [win(0.0, 24.95, [CAR, [("괜찮아요.", 18.9, 19.5)]])]
    b = [win(14.85, 39.85, [[("차를", 16.0, 16.5), ("저렇게", 16.5, 17.2), ("대면", 17.2, 17.7), ("어떡해요", 17.7, 18.2)],
                           [("괜찮아요", 18.9, 19.5), ("금방", 19.6, 20.0), ("나올", 20.0, 20.3), ("건데요", 20.3, 20.8)]])]
    words, st = asr.rescue_words([], a, b)
    assert texts(words) == ["차를", "이렇게", "대면", "어떡해요?", "괜찮아요."] and st["kept_fragments"] == 2
    # 한 낱말짜리 조각 '괜찮아요.' 는 벌 B 의 이웃 낱말(금방 나올…)과 통째로 견주면 0.6 에 못 미친다
    from rapidfuzz import fuzz
    assert fuzz.ratio("괜찮아요", "어떡해요괜찮아요금방나올건데요") / 100 < asr.RESCUE_MIN_SIM
    assert asr._best_span_ratio("괜찮아요", ["어떡해요", "괜찮아요", "금방", "나올", "건데요"]) == 1.0
    # 전혀 다른 글은 버린다
    other = [win(14.85, 39.85, [[("이쪽으로", 16.0, 16.6), ("오세요", 16.6, 17.4), ("빨리", 17.5, 18.1)]])]
    assert asr.rescue_words([], a[:1], other)[0] == []
    assert asr._norm_text(" 어떡해요?! ") == "어떡해요" and asr._norm_text("OK, 갑시다…") == "ok갑시다"


def test_rescue_excludes_time_covered_by_plain_chunks():
    """plain 묶음의 덧댄 범위 안(낱말 가운데 시각 기준)의 낱말은 넣지 않는다. 걸친 조각은 밖의 낱말만 따로 잰다."""
    inside = [("안에", 2.0, 2.4), ("있는", 2.4, 2.8), ("말", 2.8, 3.0)]
    edge = [("여기", 5.5, 5.9), ("까지가", 5.9, 6.2), ("묶음", 6.2, 6.6), ("밖이다", 6.6, 7.1)]   # 묶음 끝 6.1초에 걸침
    a = [win(0.0, 24.95, [inside, edge])]
    b = [win(1.0, 14.85, [inside, edge])]
    words, st = asr.rescue_words(PLAIN_PAD, a, b)
    assert texts(words) == ["묶음", "밖이다"]                       # '까지가' 의 가운데(6.05초)는 묶음 안
    assert st["covered"] == 5 and st["kept"] == 2 and st["fragments"] == 1
    assert texts(asr.rescue_words([], a, b)[0]) == texts(a[0]["words"])


def test_rescue_drops_words_plain_already_has():
    """전체 덮기 창에서는 쉼 옆의 낱말이 길게 늘어져 가운데 시각이 plain 묶음 밖으로 밀린다(추격자: plain '예'
    363.42~363.61초, 벌 A '예' 363.41~364.81초). plain 낱말과 겹친 시간이 짧은 쪽 길이의 절반 이상이면 같은 소리라
    rescue 에 한 번 더 넣지 않는다."""
    pad = [(s(9.9), s(10.4)), (s(20.0), s(21.0))]
    plain = [{"w": "예", "s": 10.0, "e": 10.2, "p": 0.9}, {"w": "일단", "s": 20.7, "e": 20.93, "p": 0.9},
             {"w": "네", "s": 30.0, "e": 30.0, "p": 0.9}]
    tail = [("예", 9.99, 11.39)]                                    # 끝이 늘어짐: 가운데 10.69초는 묶음(~10.4초) 밖
    head = [("일단", 18.9, 20.93)]                                  # 시작이 늘어짐: 가운데 19.9초는 묶음(20.0초~) 밖
    near = [("거기", 18.0, 18.6)]                                   # plain 낱말과 겹치지 않는 진짜 빠진 말
    graze = [("저기", 9.2, 10.05)]                                  # plain '예' 와 0.05초(짧은 쪽의 1/4)만 겹침
    zero = [("네", 29.5, 30.5)]                                     # 길이 0 인 plain 낱말을 품음
    a = [win(0.0, 24.95, [graze, tail, near, head]), win(24.95, 49.0, [zero])]
    b = [win(1.0, 14.85, [graze, tail]), win(14.85, 39.85, [near, head, zero])]
    words, st = asr.rescue_words(pad, a, b, plain)
    assert texts(words) == ["저기", "거기"]
    assert (st["covered"], st["duplicate"], st["kept"], st["fragments"]) == (0, 3, 2, 2)
    # plain 낱말을 주지 않으면(옛 동작) 가운데 시각만 본다: 같은 소리가 한 번 더 들어간다
    assert texts(asr.rescue_words(pad, a, b)[0]) == ["저기", "예", "거기", "일단", "네"]
    # 같은 소리 낱말에서 조각이 끊긴다: 앞뒤의 다른 낱말은 따로 잰다
    run = [win(0.0, 24.95, [[("자", 9.0, 9.4), ("예", 9.99, 11.39), ("갑시다", 11.5, 12.1)]])]
    words, st = asr.rescue_words(pad, run, [dict(run[0], a=s(1.0), pa=s(1.0))], plain)
    assert texts(words) == ["자", "갑시다"] and st["fragments"] == 2 and st["duplicate"] == 1
    # 글이 같은 plain 낱말과 맞닿은 낱말(사이 0.1초 이하, 또는 절반에 못 미치게 겹침)도 같은 소리다: plain 묶음의 첫 낱말이
    # 전체 덮기 창에서는 조금 앞에 놓여 나온다(추격자 plain '야' 4405.61~4405.71초, 벌 A '야,' 4405.07~4405.57초)
    def alone(word, a, b):                                        # (rescue 낱말, 같은 소리로 빠진 수). 가운데 시각은 묶음 밖
        one = [win(0.0, 24.95, [[(word, a, b)]])]
        words, st = asr.rescue_words(pad, one, [dict(one[0], a=s(1.0), pa=s(1.0))], plain)
        assert st["covered"] == 0
        return texts(words), st["duplicate"]
    assert alone("일단?", 19.2, 20.62) == ([], 1)                  # plain '일단' 20.7초~ 와 0.08초 떨어짐, 문장부호만 다름
    assert alone("일단", 19.2, 20.75) == ([], 1)                   # 0.05초 겹침(짧은 쪽의 절반 미만), 글이 같음
    assert alone("일단", 19.2, 20.5) == (["일단"], 0)               # 0.2초 떨어지면 따로 말한 것으로 본다
    assert alone("이따", 19.2, 20.62) == (["이따"], 0)              # 맞닿아 있어도 글이 다르면 다른 낱말
    assert alone("…", 19.2, 20.62) == ([], 0)                     # 글자가 없는 낱말은 글로 견주지 않는다(같은 소리로 세지 않음)
    assert (asr.RESCUE_DUP_SHARE, asr.RESCUE_DUP_GAP_S) == (0.5, 0.1)


def test_rescue_fragments_split_at_pauses_and_segments():
    """조각은 세그먼트 경계와 0.6초 넘는 쉼에서 끊긴다: 한 창 안에서도 맞는 조각만 남는다."""
    one_seg = [("진짜", 30.0, 30.4), ("말", 30.4, 30.7), ("헛", 31.4, 31.8), ("소리", 31.8, 32.2)]      # 쉼 0.7초
    a = [win(25.0, 50.0, [one_seg, [("둘째", 32.3, 32.7), ("마디", 32.7, 33.1)]])]                   # 쉼 0.1초, 다른 세그먼트
    b = [win(15.0, 40.0, [[("진짜", 30.0, 30.4), ("말", 30.4, 30.7)], [("전혀", 31.3, 31.7), ("다른", 31.7, 32.1), ("것", 32.1, 32.4)]])]
    words, st = asr.rescue_words([], a, b)
    assert texts(words) == ["진짜", "말"] and st["fragments"] == 3 and st["kept_fragments"] == 1


def test_rescue_drops_repetition_loops():
    """같은 낱말이 4번 넘게 잇달은 되풀이는 두 벌이 똑같이 내도 버린다(쉼으로 조각이 갈려 있어도)."""
    loop = [[("아", 51.0 + k, 51.4 + k)] for k in range(6)]                    # 1초 간격: 조각 여섯
    a = [win(49.85, 60.0, loop + [[("살려주세요", 57.3, 57.9), ("제발", 57.9, 58.3)]])]
    b = [win(39.85, 59.0, loop + [[("살려주세요.", 57.3, 57.9), ("제발.", 57.9, 58.3)]])]
    words, st = asr.rescue_words([], a, b)
    assert texts(words) == ["살려주세요", "제발"] and st["loop"] == 6 and st["kept"] == 2
    # 4번까지는 되풀이로 보지 않는다. 한 낱말이 같은 글자만 5번 넘게인 것도 되풀이다
    four = [win(0.0, 20.0, [[("네", 1.0 + k, 1.3 + k) for k in range(4)], [("아아아아아", 9.0, 10.0)]])]
    words, st = asr.rescue_words([], four, [dict(four[0], a=s(0.5), pa=s(0.5))])
    assert texts(words) == ["네"] * 4 and st["loop"] == 1
    assert asr._loop_marks(["가", "", "가", "가", "", "가", "가", "나"]) == [True, False, True, True, False, True, True, False]
    assert asr._loop_marks(["가", "가", "나", "가", "가"]) == [False] * 5


def test_rescue_skips_undecodable_words():
    a = [win(0.0, 20.0, [[("이\ufffd年前,", 3.0, 3.5), ("그래", 3.5, 4.0)]])]
    words, st = asr.rescue_words([], a, [dict(a[0], a=s(0.5), pa=s(0.5))])
    assert texts(words) == ["그래"] and st["a_words"] == 1


# ---------------------------------------------------------------- rescue: 전체 흐름

class ScriptFake(Fake):
    """정해 둔 대사(소리 시각)를, 받아쓰는 범위에 든 만큼 돌려주는 가짜 백엔드. 범위는 AudioReader 를 엿봐서 안다.

    halluc: {창 끝 표본: [(글, 시작, 끝)]} 그 창에서만 나오는 헛소리.
    """

    def __init__(self, seen, utts, halluc=None, **kw):
        super().__init__(**kw)
        self.seen = seen
        self.utts = utts
        self.halluc = halluc or {}

    def transcribe(self, handle, clip, lang, hotwords=None, initial_prompt=None):
        pa, pb = self.seen["range"]
        assert pb - pa == len(clip)
        self.calls.append((handle.name, pa, pb))
        off = pa / SR
        utts = [u for u in list(self.utts) + [self.halluc.get(pb)] if u and u[0][1] >= off and u[-1][2] <= pb / SR]
        utts.sort(key=lambda u: u[0][1])
        words = [{"w": w, "s": round(x - off, 3), "e": round(y - off, 3), "p": 0.8} for u in utts for w, x, y in u]
        segs = [{"start": round(u[0][1] - off, 3), "end": round(u[-1][2] - off, 3), "text": " ".join(w for w, _, _ in u),
                 "avg_logprob": -0.4, "no_speech_prob": 0.0, "compression_ratio": 1.2, "temperature": 0.0} for u in utts]
        return words, segs


INSIDE = [("안에", 2.0, 2.4), ("있는", 2.4, 2.8), ("말", 2.8, 3.0)]            # plain 첫 묶음(0.9~6.1초) 안
LOOP = [("아", 51.0 + 0.4 * k, 51.3 + 0.4 * k) for k in range(6)]             # plain 이 덮지 않은 50.1~54.9초
HELP = [("살려주세요.", 53.6, 54.2), ("제발.", 54.2, 54.6)]
WIN_A = [(0, s(24.95)), (s(24.95), s(49.85)), (s(49.85), s(60.0))]            # 60초 무음 wav 의 두 벌
WIN_B = [(0, s(14.85)), (s(14.85), s(39.85)), (s(39.85), s(57.35)), (s(57.35), s(60.0))]


@pytest.fixture
def rproj(proj, monkeypatch):
    seen = {}
    real = media.AudioReader.read_samples

    def read_samples(self, a, b):
        seen["range"] = (a, b)
        return real(self, a, b)

    monkeypatch.setattr(media.AudioReader, "read_samples", read_samples)
    fake = ScriptFake(seen, [INSIDE, CAR, LOOP, HELP], halluc={
        s(24.95): [("한글자막", 7.0, 7.5), ("by", 7.5, 7.7), ("누구", 7.7, 8.2)],          # 벌 A 첫 창에서만
        s(14.85): [("다음", 7.0, 7.4), ("영상에서", 7.4, 7.9), ("만나요.", 7.9, 8.4)]})    # 벌 B 첫 창에서만
    asr.set_backend(fake)
    proj.fake = fake
    return proj


def test_process_with_rescue_adds_stable_words_outside_plain_chunks(rproj):
    """--rescue: plain 뒤에 전체 덮기 창 두 벌을 같은 모델로 받아써 캐시에 두고, 안정적인 조각만 pass rescue 로 더한다."""
    st, msg = run_asr(rproj, rescue=True)
    assert st == "ok"
    assert ("rescue: 벌 A 낱말 18개 가운데 plain 묶음 안 3, plain 낱말과 같은 소리 0, 되풀이 6, 벌 B 와 안 맞음 3, "
            "남김 6(조각 2/3, 버린 조각: 벌 B 가 조용함 0, 다른 글 1, 같은 글이 다른 시각에 0, 시작이 같은 창뿐 0, "
            "버릇 글귀 0), 버릇 글귀 0가지") in msg
    assert validate.validate_asr(rproj.out) == []
    data = util.read_json(rproj.out)
    plain, rescue = data["passes"]
    assert plain["id"] == "plain" and texts(plain["words"]) == ["안에", "있는", "말"] and plain["context"] is None
    assert rescue["id"] == "rescue" and rescue["context"] == "full_window_stable"
    assert rescue["model"].startswith(plain["model"] + " | rescue: full windows x2 win25 quiet5 staggered novad")
    assert rescue["model"].endswith("outside plain chunks, not plain words dup0.5 same+gap0.1 | frag seg+gap0.6 | "
                                    "sim0.6 margin1 coloc0.5 otherstart rapidfuzz-ratio nfc-alnum | loop>4 habit7x3")
    # 헛소리(창마다 다름)·되풀이·plain 이 덮은 말은 없고, 시각은 영상 시각(40초 뒤는 틈 0.11초만큼 늦다), i 는 시각 순
    assert [(w["i"], w["w"], w["s"], w["e"], w["p"]) for w in rescue["words"]] == [
        (0, "차를", 16.0, 16.5, 0.8), (1, "이렇게", 16.5, 17.2, 0.8), (2, "대면", 17.2, 17.7, 0.8),
        (3, "어떡해요?", 17.7, 18.2, 0.8), (4, "살려주세요.", 53.71, 54.31, 0.8), (5, "제발.", 54.31, 54.71, 0.8)]
    assert len(data["chunks"]) == 5 and len(data["vad"]) == 7              # 말 구간·묶음 칸은 plain 것 그대로
    # 캐시: 두 벌은 꼬리표 폴더 아래 full/ 에, 범위는 windows.json 에. work/ 에는 asr.json 뿐
    root = paths.ASR_CACHE / rproj.sha
    tag = "tiny@ab12cd34-cpu-int8"
    assert sorted(p.name for p in root.iterdir()) == [tag, "vad.json", "windows.json"]
    wins = util.read_json(root / "windows.json")
    assert [tuple(x) for x in wins["a"]] == WIN_A and [tuple(x) for x in wins["b"]] == WIN_B
    assert sorted(p.name for p in (root / tag / "full").iterdir()) == \
        sorted(f"{a:010d}-{b:010d}.json" for a, b in WIN_A + WIN_B)
    assert len(list((root / tag).glob("*.json"))) == 5
    rec = util.read_json(asr.chunk_path(rproj.sha, tag, *WIN_A[0], "full"))
    assert rec["pa"] == rec["a"] == 0 and rec["pb"] == rec["b"] == s(24.95)   # 창은 덧대지 않는다
    assert rec["key"] == asr.chunk_key(rproj.fake.load("tiny", "cpu", 2), "ko", 0, s(24.95), 0, s(24.95), "raw")
    assert [p.name for p in rproj.out.parent.iterdir()] == ["asr.json"]
    assert len(rproj.fake.calls) == 5 + 3 + 4
    fp = util.stage_fp("asr", FILM)
    assert fp["rescue"] is True and fp["windows"] and fp["tags"] == {"plain": tag, "rescue": tag}
    assert [(r["a"], r["b"]) for r in asr.film_records(FILM, "rescue_a")] == WIN_A
    assert [(r["a"], r["b"]) for r in asr.film_records(FILM, "rescue_b")] == WIN_B


def test_film_rescue_fragments_gives_both_sides_from_cache(rproj):
    """끝난 rescue 의 조각 판정을 캐시에서 다시 낸다. 벌 A 쪽의 kept 가 pass rescue 이고, 벌 B 쪽(벌 A 와 견줌)은
    asr.json 에 없는 후보다: 벌 B 만 낸 글은 kept 가 아닌 조각으로 나온다."""
    with pytest.raises(FileNotFoundError):
        asr.film_rescue_fragments(FILM)                                        # 받아쓴 적이 없음
    assert run_asr(rproj, rescue=True)[0] == "ok"
    n = len(rproj.fake.calls)
    fa = asr.film_rescue_fragments(FILM)
    assert [(f["verdict"], "".join(texts(f["words"]))) for f in fa] == [
        ("b_differs", "한글자막by누구"), ("kept", "차를이렇게대면어떡해요?"), ("kept", "살려주세요.제발.")]
    kept = [w for f in fa if f["verdict"] == "kept" for w in f["words"]]
    rescue = util.read_json(rproj.out)["passes"][1]["words"]
    assert texts(kept) == texts(rescue) and [w["p"] for w in kept] == [w["p"] for w in rescue]
    assert kept[4]["s"] == 53.6 and rescue[4]["s"] == 53.71                    # 조각은 소리 시각, asr.json 은 영상 시각
    fb = asr.film_rescue_fragments(FILM, "b")
    assert [(f["verdict"], "".join(texts(f["words"]))) for f in fb] == [
        ("b_differs", "다음영상에서만나요."), ("kept", "차를이렇게대면어떡해요?"), ("kept", "살려주세요.제발.")]
    assert fb[0]["window"] == 0 and fb[1]["window"] == s(14.85)
    assert asr.film_habits(FILM) == {}                                         # 여러 창에서 되풀이된 글이 없다
    assert len(rproj.fake.calls) == n                                          # 모델을 다시 부르지 않는다
    with pytest.raises(ValueError):
        asr.film_rescue_fragments(FILM, "c")
    # rescue 없이 받아쓴 영화
    assert run_asr(rproj, rescue=False)[0] == "ok"
    with pytest.raises(FileNotFoundError):
        asr.film_rescue_fragments(FILM)


def test_habit_phrases_are_dropped_and_listed_for_the_film(proj, monkeypatch):
    """받아쓰기 끝까지: 벌 A 의 세 창이 모두 같은 글귀를 내고 그 가운데 하나를 벌 B 가 같은 시각에 내도 rescue 에 넣지
    않는다. 요약 줄과 film_habits 로 그 글귀를 알린다."""
    seen = {}
    real = media.AudioReader.read_samples

    def read_samples(self, a, b):
        seen["range"] = (a, b)
        return real(self, a, b)

    def outro(t):
        return [("다음", t, t + 0.5), ("영상에서", t + 0.5, t + 1.1), ("만나요.", t + 1.1, t + 1.6)]

    monkeypatch.setattr(media.AudioReader, "read_samples", read_samples)
    fake = ScriptFake(seen, [INSIDE, HELP], regions=[(s(1.0), s(3.0))], halluc={
        WIN_A[0][1]: outro(7.0), WIN_A[1][1]: outro(30.0), WIN_A[2][1]: outro(56.0),       # 벌 A 는 세 창 모두
        WIN_B[1][1]: outro(30.1)})                                                         # 벌 B 는 둘째 것만, 같은 시각에
    asr.set_backend(fake)
    proj.fake = fake
    st, msg = run_asr(proj, rescue=True)
    assert st == "ok" and validate.validate_asr(proj.out) == []
    assert ("rescue: 벌 A 낱말 14개 가운데 plain 묶음 안 3, plain 낱말과 같은 소리 0, 되풀이 0, 벌 B 와 안 맞음 9, "
            "남김 2(조각 1/4, 버린 조각: 벌 B 가 조용함 2, 다른 글 0, 같은 글이 다른 시각에 0, 시작이 같은 창뿐 0, "
            "버릇 글귀 1), 버릇 글귀 1가지(다음영상에서만나요 3창)") in msg
    assert texts(util.read_json(proj.out)["passes"][1]["words"]) == ["살려주세요.", "제발."]
    assert asr.film_habits(FILM) == {"다음영상에서만나요": 3}
    assert [f["verdict"] for f in asr.film_rescue_fragments(FILM)] == ["b_silent", "habit", "kept", "b_silent"]


def test_rescue_rerun_is_byte_identical_and_option_is_remembered(rproj):
    """다시 조립해도 바이트가 같고 모델을 다시 부르지 않는다. --rescue 를 다시 주지 않아도 이어 쓰고, --no-rescue 는 뺀다."""
    assert run_asr(rproj, rescue=True)[0] == "ok"
    first = rproj.out.read_bytes()
    n = len(rproj.fake.calls)
    assert run_asr(rproj, rescue=True) == ("skip", "")
    assert run_asr(rproj) == ("skip", "")                                       # 옵션 없이: 지문의 rescue 를 이어 쓴다
    assert run_asr(rproj, force=True)[0] == "ok"
    assert rproj.out.read_bytes() == first and len(rproj.fake.calls) == n
    # 창 기록 하나가 없어지면 그 창만 다시 받아쓴다
    asr.chunk_path(rproj.sha, "tiny@ab12cd34-cpu-int8", *WIN_B[1], "full").unlink()
    rproj.out.unlink()
    assert run_asr(rproj)[0] == "ok" and rproj.out.read_bytes() == first
    assert len(rproj.fake.calls) == n + 1 and rproj.fake.calls[-1] == ("tiny",) + WIN_B[1]
    # --no-rescue: plain 만 남는다(캐시에서 조립)
    assert run_asr(rproj, rescue=False)[0] == "ok" and len(rproj.fake.calls) == n + 1
    assert [p["id"] for p in util.read_json(rproj.out)["passes"]] == ["plain"]
    assert util.stage_fp("asr", FILM)["rescue"] is False and run_asr(rproj) == ("skip", "")
    # 다시 켜면 캐시에서 같은 바이트
    assert run_asr(rproj, rescue=True)[0] == "ok" and rproj.out.read_bytes() == first
    assert len(rproj.fake.calls) == n + 1


def test_rescue_uses_the_plain_model_and_reserves_its_id(rproj):
    """두 벌은 plain 의 모델로 받아쓴다(plain 이 없으면 첫 pass). pass id rescue 는 --passes 에 쓸 수 없다."""
    assert run_asr(rproj, passes="other=base,plain=tiny", rescue=True)[0] == "ok"
    data = util.read_json(rproj.out)
    assert [p["id"] for p in data["passes"]] == ["other", "plain", "rescue"]
    assert "fake/tiny@" in data["passes"][2]["model"]
    assert {c[0] for c in rproj.fake.calls[-7:]} == {"tiny"}                   # 창 3+4개
    assert rproj.fake.loads == [("base", "cpu", 2), ("tiny", "cpu", 2)]         # 같은 모델은 다시 열지 않는다
    with pytest.raises(ValueError):
        run_asr(rproj, passes="plain=tiny,rescue=tiny", rescue=True)


def test_rescue_from_command_line(rproj, capsys):
    video = {k: rproj.doc[k] for k in ("video_path", "video_sha1", "video_duration", "video_lang")}
    catalog.write_catalog([catalog_row("fm1", FILM, is_primary="true", **video)])
    args = argparse.Namespace(cmd="asr", docs="fm1", limit=None, force=False, workers=None, paths=[],
                              passes="plain=tiny", threads=2, device="cpu", rescue=True)
    assert asr.run(args) == 0
    out = capsys.readouterr().out
    assert "전체 덮기 창 3+4개" in out and f"[asr] {FILM} rescue-a 3/3 100%" in out and f"[asr] {FILM} rescue-b 4/4 100%" in out
    assert "(tiny@ab12cd34-cpu-int8/full)" in out and "rescue 낱말 6개" in out
    args.passes = args.device = args.threads = args.rescue = None
    assert asr.run(args) == 0 and "asr: {'skip': 1}" in capsys.readouterr().out


# ---------------------------------------------------------------- sinkprep all 안에서는 캐시만

def all_args(**kw):
    ns = dict(cmd="all", docs=None, limit=None, force=False, workers=None, paths=[])
    ns.update(kw)
    return argparse.Namespace(**ns)


def test_all_stage_assembles_from_cache_and_never_transcribes(rproj, capsys, monkeypatch):
    """sinkprep all 안의 asr: 모델을 부르지 않고(ALL_CACHE_ONLY) 앞서 받아쓴 옵션 그대로 캐시만으로 조립한다.
    캐시에 없는 것이 있으면 asr.json 을 건드리지 않고 알린다."""
    assert asr.ALL_CACHE_ONLY is True
    video = {k: rproj.doc[k] for k in ("video_path", "video_sha1", "video_duration", "video_lang")}
    catalog.write_catalog([catalog_row("fm1", FILM, is_primary="true", **video)])
    fake = rproj.fake
    # 받아쓴 적이 없는 영화: 말 구간도 찾지 않고 기다린다
    assert asr.run(all_args()) == 0
    out = capsys.readouterr().out
    assert f"[1/1] {FILM} wait: 받아쓴 적이 없음" in out and "sinkprep asr --docs fm1" in out and "asr: {'wait': 1}" in out
    assert not rproj.out.exists() and fake.vad_calls == 0 and fake.calls == [] and fake.loads == []
    # 따로 받아쓴 뒤(rescue 포함)에는 건너뛴다
    assert run_asr(rproj, rescue=True)[0] == "ok"
    first, n = rproj.out.read_bytes(), len(fake.calls)
    assert asr.run(all_args()) == 0 and "asr: {'skip': 1}" in capsys.readouterr().out
    # 조립 방식이 바뀌면(VERSION) 캐시에서 다시 조립한다: 옵션(모델·rescue)은 지문 것을 이어 쓴다
    monkeypatch.setattr(asr, "VERSION", "9.9.9")
    assert asr.run(all_args()) == 0 and f"[1/1] {FILM} ok" in capsys.readouterr().out
    data = util.read_json(rproj.out)
    assert [p["id"] for p in data["passes"]] == ["plain", "rescue"] and "sinkprep.asr 9.9.9" in data["passes"][0]["model"]
    assert len(fake.calls) == n and fake.loads[-1] == ("tiny", "cpu", 6)
    second = rproj.out.read_bytes()
    # 캐시에 없는 창이 있으면 받아쓰지 않고 그대로 둔다
    monkeypatch.setattr(asr, "VERSION", "9.9.10")
    asr.chunk_path(rproj.sha, "tiny@ab12cd34-cpu-int8", *WIN_B[1], "full").unlink()
    assert asr.run(all_args(force=True)) == 0
    out = capsys.readouterr().out
    assert f"[1/1] {FILM} wait: 캐시만으로는 조립할 수 없음: rescue-b: 묶음 4개 가운데 1개가 캐시에 없음" in out
    assert rproj.out.read_bytes() == second and len(fake.calls) == n
    assert util.stage_fp("asr", FILM)["version"] == "9.9.9" and not asr.lock_path(rproj.sha).exists()
    # 소리 캐시나 말 구간 캐시가 없어도 다시 만들지 않는다
    asr.vad_path(rproj.sha).unlink()
    assert asr.run(all_args()) == 0 and "wait: 캐시만으로는 조립할 수 없음: 말 구간 캐시" in capsys.readouterr().out
    assert fake.vad_calls == 1 and rproj.out.read_bytes() == second
    # 같은 것을 asr 명령으로 돌리면 빠진 것만 받아쓴다
    assert asr.process(rproj.doc)[0] == "ok" and len(fake.calls) == n + 1 and fake.vad_calls == 2


def test_cache_only_load_does_not_probe_cuda(monkeypatch):
    """캐시만 읽을 때는 cuda 장치도 열어 보지 않는다(꼬리표와 열쇠만 있으면 된다)."""
    opened = []
    monkeypatch.setattr(asr, "resolve_model", lambda name: (name, "X:/model", "org/" + name, "c0ffee00" + "0" * 32,
                                                             {"faster_whisper": "1", "ctranslate2": "2"}))
    monkeypatch.setattr(asr, "_open", lambda h: opened.append(h.device) or object())
    h = asr.FasterWhisperBackend().load("large-v3-turbo", "cuda", 6, probe=False)
    assert h.tag == "large-v3-turbo@c0ffee00-cuda-float16" and not h.opened and opened == []
