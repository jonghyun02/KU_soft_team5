"""영상 등록 시험: catalog 영상 칸 갱신, videos/ 하드링크, 다시 돌리면 skip, 잘못된 입력.

ffmpeg 로 만든 3초짜리 시험 영상을 쓴다(ffmpeg 가 없으면 건너뜀).
"""
import argparse
import os
import re
import shutil
import subprocess

import pytest

from sinkprep import catalog, media, paths, util, validate, video

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg 가 없음")
FILM = "local_0001"


def row(doc_id, film, **kw):
    r = {k: "" for k in catalog.FIELDS}
    r.update(doc_id=doc_id, post_id=doc_id[2:], category="ko_film", files=f"filmmakers_scripts/x/{doc_id}.hwp",
             sha1=f"{int(doc_id[2:]):040x}", real_format="hwp5", file_formats="hwp5", title_raw="가", title_norm="가",
             script_lang="ko", is_translation="false", version_label="unknown", film_id=film, film_title="가",
             is_primary="false", status="parsed")
    r.update(kw)
    return r


def make_video(path, seconds=3, freq=440):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-f", "lavfi", "-i", f"color=c=black:s=64x64:r=10:d={seconds}",
                    "-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate=48000:duration={seconds}",
                    "-c:v", "mpeg4", "-c:a", "aac", "-b:a", "96k", str(path)],
                   check=True, stdin=subprocess.DEVNULL, capture_output=True)
    return path


@pytest.fixture
def proj(tmp_path, monkeypatch):
    """임시 프로젝트: 한 영화에 문서 셋(대표, 다른 판, 제외)과 다른 영화 문서 둘(하나는 외국영화)."""
    for k, sub in (("ROOT", "."), ("CACHE", "cache"), ("AUDIO", "cache/audio"), ("ASR_CACHE", "cache/asr"),
                   ("STATE", "cache/state"), ("PROBLEMS", "cache/problems"), ("FILMS", "films"),
                   ("VIDEOS", "videos"), ("CATALOG", "catalog.csv")):
        monkeypatch.setattr(paths, k, tmp_path / sub)
    catalog.write_catalog([row("fm1", FILM, is_primary="true"), row("fm2", FILM),
                           row("fm3", FILM, status="excluded", note="duplicate_file_of:fm1"),
                           row("fm4", "local_0002", is_primary="true"),
                           row("fm5", "local_0003", is_primary="true", category="foreign_film", script_lang="en")])
    return tmp_path


def by_id():
    return {r["doc_id"]: r for r in catalog.load()}


@needs_ffmpeg
def test_register_fills_catalog_links_video_and_builds_timemap(proj):
    src = make_video(proj / "밖" / "영화 원본.MP4")
    before = src.read_bytes()
    st, msg = video.register(src, "fm1")
    assert st == "ok" and "catalog 2행 갱신" in msg and "영상 시각 틈 없음" in msg
    sha = util.sha1_file(src)
    rows = by_id()
    for d in ("fm1", "fm2"):                                              # 그 영화의 제외되지 않은 모든 문서
        r = rows[d]
        assert r["video_path"] == f"videos/{FILM}.mp4" and r["video_sha1"] == sha and r["video_lang"] == "ko"
        assert re.fullmatch(r"\d+\.\d{3}", r["video_duration"]) and abs(float(r["video_duration"]) - 3.0) < 0.2
    for d in ("fm3", "fm4", "fm5"):                                       # 제외 문서와 다른 영화는 그대로
        assert all(rows[d][c] == "" for c in video.COLUMNS)
    assert rows["fm1"]["is_primary"] == "true" and rows["fm1"]["status"] == "parsed"
    dst = paths.VIDEOS / f"{FILM}.mp4"
    assert dst.is_file() and os.path.samefile(src, dst)                   # 하드링크(같은 볼륨)
    assert media.resolve_video(rows["fm1"]["video_path"]) == dst
    assert src.read_bytes() == before                                     # 원본은 그대로
    assert media.audio_path(sha).is_file() and media.load_timemap(sha).gaps == 0
    assert validate.validate_catalog(check_files=False) == []


@needs_ffmpeg
def test_register_again_skips_without_rewriting_catalog(proj):
    src = make_video(proj / "v.mp4")
    assert video.register(src, "fm1")[0] == "ok"
    raw = paths.CATALOG.read_bytes()
    mtime = paths.CATALOG.stat().st_mtime_ns
    st, msg = video.register(src, "fm1")
    assert st == "skip" and "catalog 0행 갱신" in msg
    assert paths.CATALOG.read_bytes() == raw and paths.CATALOG.stat().st_mtime_ns == mtime
    assert video.register(paths.VIDEOS / f"{FILM}.mp4", "fm2")[0] == "skip"      # 등록된 파일 자체, 같은 영화의 다른 문서
    # 언어만 바꾸면 그 칸만 갱신된다
    st, _ = video.register(src, "fm1", "en")
    assert st == "ok" and {by_id()[d]["video_lang"] for d in ("fm1", "fm2")} == {"en"}


@needs_ffmpeg
def test_register_default_lang_and_replacing_video(proj):
    src = make_video(proj / "f.mp4")
    assert video.register(src, "fm5")[0] == "ok"
    assert by_id()["fm5"]["video_lang"] == "unknown" and by_id()["fm5"]["video_path"] == "videos/local_0003.mp4"
    st, msg = video.register(src, "fm5", "en-US")                          # 지역 꼬리는 기본 언어로 줄여 적고 까닭을 알린다
    assert st == "ok" and by_id()["fm5"]["video_lang"] == "en" and "'en-US' 는 기본 언어 'en' 로 줄여 적음" in msg
    other = make_video(proj / "g.mp4", freq=880)
    with pytest.raises(media.MediaError):                                  # 같은 영화에 다른 영상: --force 없이는 거절
        video.register(other, "fm5", "en-US")
    assert by_id()["fm5"]["video_sha1"] == util.sha1_file(src)
    assert video.register(other, "fm5", "en-US", force=True)[0] == "ok"
    assert by_id()["fm5"]["video_sha1"] == util.sha1_file(other)
    assert util.sha1_file(paths.VIDEOS / "local_0003.mp4") == util.sha1_file(other) and src.is_file()


@needs_ffmpeg
def test_link_video_falls_back_to_copy(proj, monkeypatch):
    """하드링크를 못 하면(다른 볼륨 등) 복사하고 sha1 을 확인한다."""
    src = make_video(proj / "v.mp4")

    def no_link(a, b):
        raise OSError("cross-device link")

    monkeypatch.setattr(video.os, "link", no_link)
    sha = util.sha1_file(src)
    dst, how = video.link_video(src, FILM, sha)
    assert how == "copy" and dst.is_file() and not os.path.samefile(src, dst) and util.sha1_file(dst) == sha
    assert not list(dst.parent.glob("*.part"))
    assert video.link_video(src, FILM, sha) == (dst, "kept")


def test_register_rejects_bad_input(proj):
    f = proj / "v.mp4"
    f.write_bytes(b"x")
    with pytest.raises(KeyError):
        video.register(f, "fm999")
    with pytest.raises(ValueError):
        video.register(f, "fm3")                                           # 제외된 문서
    with pytest.raises(ValueError):
        video.register(f, "fm1", "korean")                                 # 언어 코드 꼴이 아님
    with pytest.raises(FileNotFoundError):
        video.register(proj / "none.mp4", "fm1")
    assert all(r["video_path"] == "" for r in catalog.load()) and not paths.VIDEOS.exists()


def test_run_needs_one_video_and_one_doc(proj, capsys):
    ns = dict(cmd="video", limit=None, force=False, workers=None, lang=None)
    assert video.run(argparse.Namespace(paths=[], docs="fm1", **ns)) == 1
    assert video.run(argparse.Namespace(paths=["a.mp4"], docs=None, **ns)) == 1
    assert video.run(argparse.Namespace(paths=["a.mp4"], docs="fm1,fm2", **ns)) == 1
    assert capsys.readouterr().err.count("쓰는 법") == 3
    assert video.run(argparse.Namespace(paths=[str(proj / "none.mp4")], docs="fm1", **ns)) == 1
    assert "FileNotFoundError" in capsys.readouterr().err


@needs_ffmpeg
def test_run_registers_and_reports(proj, capsys):
    src = make_video(proj / "v.mp4")
    args = argparse.Namespace(cmd="video", paths=[str(src)], docs="fm1", lang="ko", limit=None, force=False,
                              workers=None)
    assert video.run(args) == 0
    out = capsys.readouterr().out
    assert "[1/1] fm1 ok:" in out and f"videos/{FILM}.mp4" in out and "video: {'ok': 1}" in out
    assert video.run(args) == 0 and "video: {'skip': 1}" in capsys.readouterr().out


# ---------------------------------------------------------------- 검토에서 나온 경우(회귀 시험)

@needs_ffmpeg
def test_register_other_extension_needs_force_and_removes_old_file(proj):
    """확장자가 다른 영상도 그 영화의 '다른 영상'이다: --force 없이는 거절하고, 바꾸면 videos/ 의 옛 파일을 지운다."""
    src = make_video(proj / "v.mp4")
    assert video.register(src, "fm1")[0] == "ok"
    other = proj / "g.mkv"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i",
                    str(make_video(proj / "g.mp4", freq=880)), "-c", "copy", str(other)],
                   check=True, stdin=subprocess.DEVNULL, capture_output=True)
    raw = paths.CATALOG.read_bytes()
    with pytest.raises(media.MediaError) as e:
        video.register(other, "fm1")
    assert "--force" in str(e.value)
    assert paths.CATALOG.read_bytes() == raw
    assert sorted(p.name for p in paths.VIDEOS.iterdir()) == [f"{FILM}.mp4"]
    st, msg = video.register(other, "fm1", force=True)
    assert st == "ok" and f"{FILM}.mp4" in msg
    assert sorted(p.name for p in paths.VIDEOS.iterdir()) == [f"{FILM}.mkv"]        # 옛 mp4 는 남지 않는다
    assert {by_id()[d]["video_path"] for d in ("fm1", "fm2")} == {f"videos/{FILM}.mkv"}
    assert by_id()["fm1"]["video_sha1"] == util.sha1_file(other)
    assert src.is_file() and other.is_file()                                         # 원본은 그대로


@needs_ffmpeg
def test_register_rejects_audio_only_file(proj):
    """영상 줄이 없는 파일(소리만)은 영상으로 등록하지 않는다."""
    m4a = proj / "a.m4a"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=48000:duration=2", "-c:a", "aac", str(m4a)],
                   check=True, stdin=subprocess.DEVNULL, capture_output=True)
    with pytest.raises(media.MediaError) as e:
        video.register(m4a, "fm1")
    assert "영상 줄이 없는" in str(e.value)
    assert all(r["video_path"] == "" for r in catalog.load()) and not paths.VIDEOS.exists()


@needs_ffmpeg
def test_register_keeps_given_lang_when_rerun_without_lang(proj):
    """--lang 없이 다시 등록해도 앞서 준 언어가 게시판 기본값으로 돌아가지 않는다."""
    src = make_video(proj / "f.mp4")
    assert video.register(src, "fm5", "en")[0] == "ok"
    st, msg = video.register(src, "fm5")
    assert st == "skip" and "언어 en" in msg and by_id()["fm5"]["video_lang"] == "en"
    ko = make_video(proj / "k.mp4", freq=660)
    assert video.register(ko, "fm1", "mixed")[0] == "ok"
    assert video.register(ko, "fm1")[0] == "skip"
    assert {by_id()[d]["video_lang"] for d in ("fm1", "fm2")} == {"mixed"}


@needs_ffmpeg
def test_register_writes_catalog_only_after_audio_is_ready(proj, monkeypatch):
    """소리 추출이 실패하면 catalog 를 고치지 않는다. 다시 돌려 소리를 뽑으면 skip 이 아니라 ok 다."""
    src = make_video(proj / "v.mp4")
    real = media.extract_audio

    def fail(*a, **kw):
        raise media.MediaError("ffmpeg(소리 추출) 실패")

    monkeypatch.setattr(media, "extract_audio", fail)
    with pytest.raises(media.MediaError):
        video.register(src, "fm1")
    assert all(r["video_path"] == "" for r in catalog.load())
    monkeypatch.setattr(media, "extract_audio", real)
    st, _ = video.register(src, "fm1")
    assert st == "ok" and by_id()["fm1"]["video_path"] == f"videos/{FILM}.mp4"
    sha = util.sha1_file(src)
    # catalog 는 이미 맞는데 소리 캐시만 없어진 경우에도 다시 만든 것을 ok 로 알린다
    media.audio_path(sha).unlink()
    assert video.register(src, "fm1")[0] == "ok" and media.audio_path(sha).is_file()
    assert video.register(src, "fm1")[0] == "skip"


def test_base_lang_cuts_region_tails():
    assert video.base_lang("ko") == ("ko", "") and video.base_lang("mixed") == ("mixed", "")
    lang, why = video.base_lang("ko-KR")
    assert lang == "ko" and "기본 언어 'ko'" in why
    with pytest.raises(ValueError):
        video.base_lang("Korean")


# ---------------------------------------------------------------- 등록 해제
def _film_outputs(film=FILM):
    """맞추기·최종본이 남기는 것(시험용 가짜): films/ 결과, 단계 문제 기록, catalog status·match."""
    (paths.FILMS / film / "work").mkdir(parents=True, exist_ok=True)
    (paths.FILMS / film / "final.json").write_text("{}", encoding="utf-8")
    for stage in video.FILM_STAGES:
        util.set_problems(stage, film, [{"level": "film", "doc_id": "fm1", "film_id": film, "item_id": "",
                                         "problem": "other", "score": None, "detail": "x", "path": ""}])
    util.set_stage_fp("final", film, {"version": "x"})
    catalog.finalize(updates={"fm1": {"status": "done", "match": "0.613"}})


@needs_ffmpeg
def test_unregister_clears_catalog_problems_and_the_hardlink_but_keeps_results_and_cache(proj):
    src = make_video(proj / "밖" / "원본.mp4")
    before = src.read_bytes()
    assert video.register(src, "fm1")[0] == "ok"
    sha = util.sha1_file(src)
    _film_outputs()
    other_film = {d: dict(by_id()[d]) for d in ("fm3", "fm4", "fm5")}
    st, msg = video.unregister("fm2")                                     # 그 영화의 어느 문서로 불러도 된다
    assert st == "ok" and "catalog 2행" in msg and "문제 기록 지움(asr, align, final)" in msg
    rows = by_id()
    for d in ("fm1", "fm2"):
        assert all(rows[d][c] == "" for c in video.COLUMNS) and rows[d]["match"] == ""
    assert rows["fm1"]["status"] == "parsed" and rows["fm2"]["status"] == "parsed" and rows["fm1"]["is_primary"] == "true"
    assert {d: rows[d] for d in other_film} == other_film                   # 제외 문서와 다른 영화는 그대로
    assert not (paths.VIDEOS / f"{FILM}.mp4").exists()                    # 하드링크만 지움
    assert src.read_bytes() == before                                     # 원본은 그대로
    assert not any((paths.PROBLEMS / s / f"{FILM}.json").exists() for s in video.FILM_STAGES)
    assert (paths.FILMS / FILM / "final.json").is_file() and media.audio_path(sha).is_file()   # 결과·캐시는 남김
    assert f"films/{FILM}/ 파일 1개" in msg and f"cache/state/final/{FILM}.json" in msg and sha in msg
    assert validate.validate_catalog(check_files=False) == []
    assert video.unregister("fm1")[0] == "skip"
    assert video.register(src, "fm1")[0] == "ok" and by_id()["fm1"]["video_sha1"] == sha       # 다시 등록할 수 있다


@needs_ffmpeg
def test_unregister_keeps_a_single_link_copy_unless_forced(proj, monkeypatch):
    """하드링크를 못 해 복사한 영상(링크 하나)은 원본인지 알 수 없어 --force 일 때만 지운다."""
    src = make_video(proj / "v.mp4")

    def no_link(a, b):
        raise OSError("cross-device link")

    monkeypatch.setattr(video.os, "link", no_link)
    assert video.register(src, "fm1")[0] == "ok"
    dst = paths.VIDEOS / f"{FILM}.mp4"
    st, msg = video.unregister("fm1")
    assert st == "ok" and dst.is_file() and "videos/ 에 남김" in msg and "--force" in msg
    st, msg = video.unregister("fm1", force=True)
    assert st == "ok" and not dst.exists() and src.is_file() and f"videos/ 지움: {FILM}.mp4" in msg
    assert video.unregister("fm1", force=True)[0] == "skip"


def test_unregister_rejects_unknown_doc(proj):
    with pytest.raises(KeyError):
        video.unregister("fm999")


def test_run_unregister_takes_docs_and_no_video(proj, capsys):
    ns = dict(cmd="video", limit=None, force=False, workers=None, unregister=True)
    _film_outputs()
    assert video.run(argparse.Namespace(paths=["a.mp4"], docs="fm1", lang=None, **ns)) == 1     # 영상 경로는 받지 않음
    assert video.run(argparse.Namespace(paths=[], docs="fm1", lang="ko", **ns)) == 1
    assert capsys.readouterr().err.count("쓰는 법") == 2
    assert video.run(argparse.Namespace(paths=[], docs="fm1", lang=None, **ns)) == 0
    out = capsys.readouterr().out
    assert "[1/1] fm1 ok:" in out and "video: {'ok': 1}" in out and by_id()["fm1"]["status"] == "parsed"
