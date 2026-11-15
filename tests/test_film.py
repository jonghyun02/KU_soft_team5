"""sinkprep.film(영화 한 편을 한 번에): 인자 가리기, 대본 찾기, 장치 규칙, 단계 차례와 멈춤, 요약.

받아쓰기 모델·claude·ffprobe 없이 돈다: media.probe 는 가짜, 단계의 run 은 monkeypatch 로 가짜를 끼운다.
끝의 한 시험만 align·final·검사·자막을 진짜로 돌린다(video·asr 는 가짜).
"""
import argparse
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import align_synth as S  # noqa: E402
from sinkprep import align, asr, catalog, film, final, llm, media, paths, util, validate, video  # noqa: E402

FILM_ID = "local_0010"


def row(doc_id, film_id, **kw):
    r = {k: "" for k in catalog.FIELDS}
    n = int(re.sub(r"\D", "", doc_id))
    r.update(doc_id=doc_id, post_id=str(n), category="ko_film", files=f"filmmakers_scripts/x/{doc_id}.hwp",
             sha1=f"{n:040x}", real_format="hwp5", file_formats="hwp5", title_raw="가", title_norm="가",
             script_lang="ko", is_translation="false", version_label="unknown", film_id=film_id, film_title="가",
             is_primary="false", status="parsed")
    r.update(kw)
    return r


def fake_probe(p):
    """가짜 ffprobe: 'fake video' 로 시작하는 파일은 영상, .txt 는 글자 그림 영상(ansi), 나머지는 못 읽음."""
    p = Path(p)
    data = p.read_bytes()
    if data.startswith(b"fake video"):
        return {"duration": 7449.0, "start_time": 0.0, "audio": {"codec": "mp3"}, "video": {"codec": "mpeg4"}}
    if p.suffix.lower() == ".txt":
        return {"duration": None, "start_time": None, "audio": None, "video": {"codec": "ansi"}}
    raise media.MediaError("ffprobe 실패(rc=1): Invalid data found when processing input")


@pytest.fixture
def proj(tmp_path, monkeypatch):
    """임시 프로젝트: 써니(local_0010: 대표 fm10, 다른 판 fm11, fm11 의 중복으로 제외된 fm12), 괴물, 영어 대본, 번역 대본,
    제목이 같은 두 영화(커피). 입력 폴더에 대본 파일·중복 파일·가짜 영상."""
    for k, sub in (("ROOT", "."), ("DOCS", "docs"), ("FILMS", "films"), ("CACHE", "cache"), ("STATE", "cache/state"),
                   ("PROBLEMS", "cache/problems"), ("AUDIO", "cache/audio"), ("ASR_CACHE", "cache/asr"),
                   ("VIDEOS", "videos"), ("CATALOG", "catalog.csv"), ("QA", "qa"),
                   ("NEEDS_REVIEW", "qa/needs_review.csv")):
        monkeypatch.setattr(paths, k, tmp_path / sub)
    inp = tmp_path / "입력 폴더"
    inp.mkdir()
    script = inp / "써니.hwp"
    script.write_bytes(b"sunny script v1")
    dup = inp / "써니 사본.hwp"
    dup.write_bytes(b"sunny script copy")
    vid = inp / "써니-001.mkv"
    vid.write_bytes(b"fake video sunny")
    catalog.write_catalog([
        row("fm10", FILM_ID, title_norm="써니", film_title="써니", sha1=util.sha1_file(script), is_primary="true"),
        row("fm11", FILM_ID, title_norm="써니", film_title="써니"),
        row("fm12", FILM_ID, title_norm="써니", sha1=util.sha1_file(dup), status="excluded",
            note="duplicate_file_of:fm11"),
        row("fm20", "local_0020", title_norm="괴물", is_primary="true"),
        row("fm30", "local_0030", title_norm="Heat", is_primary="true", category="foreign_film", script_lang="en"),
        row("fm31", "local_0031", title_norm="번역", is_primary="true", is_translation="true"),
        row("fm40", "local_0040", title_norm="커피", is_primary="true"),
        row("fm41", "local_0041", title_norm="커피", is_primary="true"),
    ])
    for d in ("fm10", "fm20", "fm30", "fm31", "fm40", "fm41"):
        util.write_json(paths.DOCS / d / "script.json", {"schema": "sink.script/1"})
    monkeypatch.setattr(media, "probe", fake_probe)
    monkeypatch.setattr(llm, "stopped", lambda: None)
    monkeypatch.setattr(llm, "calls_made", lambda: 0)
    gpu(monkeypatch, "cuda")
    return SimpleNamespace(root=tmp_path, inp=inp, script=script, dup=dup, video=vid, sha=util.sha1_file(vid))


def gpu(monkeypatch, dev, *, libs_missing=(), count=1, vram=1200):
    """GPU 상태 흉내: asr.pick_device('auto') 의 답, CUDA DLL, 장치 수, 여유 VRAM."""
    monkeypatch.setattr(asr, "pick_device", lambda device="auto": (dev, "float16" if dev == "cuda" else "int8"))
    monkeypatch.setattr(film, "cuda_libs_missing", lambda: list(libs_missing))
    monkeypatch.setattr(film, "_cuda_count", lambda: count)
    monkeypatch.setattr(film, "_has_nvidia_smi", lambda: True)
    monkeypatch.setattr(asr, "free_vram_mib", lambda: vram)


def fargs(*inputs, **kw):
    base = dict(cmd="film", docs=None, limit=None, force=False, workers=None, paths=[str(p) for p in inputs],
                lang=None, passes=None, threads=None, device=None, rescue=None, llm=None, open=False)
    base.update(kw)
    return argparse.Namespace(**base)


def by_id():
    return {r["doc_id"]: r for r in catalog.load()}


# ------------------------------------------------------------------ 가짜 단계
FINAL_ITEMS = [
    {"id": "I0001", "type": "dialogue", "speakers": ["P01"], "speaker_from": "script", "status": "same",
     "start": 1.0, "end": 2.0, "words": [{"w": "안녕", "s": 1.0, "e": 2.0, "p": 0.9}], "heard": "안녕",
     "script_ref": {"el": "E0001", "text": "안녕"}},
    {"id": "I0002", "type": "dialogue", "speakers": [], "speaker_from": "unknown", "status": "added",
     "start": 3.0, "end": 4.0, "words": [{"w": "뭐야", "s": 3.0, "e": 4.0, "p": 0.9}], "heard": "뭐야",
     "script_ref": None},
    {"id": "I0003", "type": "dialogue", "speakers": ["P01"], "speaker_from": "script", "status": "absent",
     "start": None, "end": None, "words": [], "heard": None, "script_ref": {"el": "E0003", "text": "잘 가"}},
]


class Stages:
    """video·asr·align·final 의 run 과 validate.check_file 을 가짜로. 부른 차례와 넘긴 인자를 적는다.

    fail: {단계: 종료 코드}, no_output: 0 으로 끝나지만 결과 파일을 쓰지 않는 단계, check_errors: 검사 오류."""

    def __init__(self, proj, monkeypatch):
        self.p = proj
        self.calls: list[tuple[str, dict]] = []
        self.fail: dict[str, int] = {}
        self.no_output: set[str] = set()
        self.check_errors: list[str] = []
        self.checked: list[str] = []
        for mod, name in ((video, "video"), (asr, "asr"), (align, "align"), (final, "final")):
            monkeypatch.setattr(mod, "run", self._make(name))
        monkeypatch.setattr(validate, "check_file", self._check)

    def names(self):
        return [n for n, _ in self.calls]

    def args(self, name):
        return next(a for n, a in self.calls if n == name)

    def _make(self, name):
        def run(ns):
            self.calls.append((name, dict(vars(ns))))
            if name in self.fail:
                print(f"  ! {name}: 가짜 실패", file=sys.stderr)
                return self.fail[name]
            if name not in self.no_output:
                getattr(self, "_" + name)(ns)
            return 0
        return run

    def _video(self, ns):
        src = Path(ns.paths[0])
        dst = paths.VIDEOS / f"{FILM_ID}{src.suffix.lower()}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        sha = util.sha1_file(src)
        catalog.finalize(updates={d: {"video_path": f"videos/{dst.name}", "video_sha1": sha,
                                      "video_duration": "7449.000", "video_lang": ns.lang}
                                  for d in ("fm10", "fm11")})

    def _work(self):
        return paths.FILMS / FILM_ID / "work"

    def _asr(self, ns):
        util.write_json(self._work() / "asr.json",
                        {"schema": "sink.asr/1", "video_sha1": self.p.sha,
                         "passes": [{"id": "plain", "words": [{}] * 1200}, {"id": "lv3", "words": [{}] * 1300},
                                    {"id": "rescue", "words": [{}] * 40}]})
        util.set_problems("asr", FILM_ID, [{"level": "film", "doc_id": "fm10", "film_id": FILM_ID, "item_id": None,
                                            "problem": "other", "detail": "받아쓰기 구멍"}])

    def _align(self, ns):
        util.write_json(self._work() / "align.json",
                        {"schema": "sink.align/1", "script_doc": ns.docs,
                         "asr_sha1": util.sha1_file(self._work() / "asr.json")})
        same = {"level": "item", "doc_id": "fm10", "film_id": FILM_ID, "item_id": "I0001", "problem": "asr_low_conf",
                "detail": "same 줄 conf 0.100 < 0.2"}
        util.set_problems("align", FILM_ID, [same])

    def _final(self, ns):
        util.write_json(paths.FILMS / FILM_ID / "final.json", {
            "schema": "sink.final/1",
            "film": {"id": FILM_ID, "title": "써니", "video": {"sha1": self.p.sha, "duration": 7449.0, "lang": "ko"},
                     "script": {"doc_id": ns.docs, "match": 0.8123}},
            "people": [{"id": "P01", "name": "수지", "aliases": []}], "scenes": [], "items": FINAL_ITEMS,
            "quality": {"same": 1500, "changed": 600, "absent": 320, "absent_spoken": 300, "added": 450,
                        "match": 0.8123}})
        same = {"level": "item", "doc_id": "fm10", "film_id": FILM_ID, "item_id": "I0001", "problem": "asr_low_conf",
                "detail": "same 줄 conf 0.100 < 0.2"}                 # align 과 같은 줄: 한 번만 센다
        util.set_problems("final", FILM_ID, [same, dict(same, item_id="I0002", detail="added 줄 conf 0.050 < 0.2")])

    def _check(self, p):
        self.checked.append(Path(p).relative_to(paths.FILMS / FILM_ID).as_posix())
        return ("x", list(self.check_errors))


@pytest.fixture
def stages(proj, monkeypatch):
    return Stages(proj, monkeypatch)


# ------------------------------------------------------------------ 인자 가리기
def test_split_inputs_either_order(proj):
    want = (str(proj.script), proj.video)
    assert film.split_inputs([str(proj.script), str(proj.video)]) == want
    assert film.split_inputs([str(proj.video), str(proj.script)]) == want


def test_split_inputs_trusts_ffprobe_extension_only_helps(proj):
    """확장자가 아니라 ffprobe 로 가린다: 영상 줄이 있는 .bin 은 영상, 글자 그림 영상으로 읽힌 .txt 는 대본,
    ffprobe 가 못 읽은 .mkv 는 확장자를 보조로 영상."""
    odd = proj.inp / "영상.bin"
    odd.write_bytes(b"fake video odd")
    assert film.split_inputs([str(proj.script), str(odd)])[1] == odd
    txt = proj.inp / "써니.txt"
    txt.write_bytes("대본 글".encode())
    assert film.split_inputs([str(txt), str(proj.video)]) == (str(txt), proj.video)
    broken = proj.inp / "깨진 영상.mkv"
    broken.write_bytes(b"\x00\x01")
    assert film.split_inputs([str(broken), str(proj.script)]) == (str(proj.script), broken)


def test_split_inputs_doc_id_or_title_and_docs_option(proj):
    assert film.split_inputs(["fm10", str(proj.video)]) == ("fm10", proj.video)
    assert film.split_inputs([str(proj.video), "써니"]) == ("써니", proj.video)
    assert film.split_inputs([str(proj.video)], docs="fm10") == ("fm10", proj.video)


@pytest.mark.parametrize("make", [
    lambda p: [str(p.video), str(p.video)],                                  # 둘 다 영상
    lambda p: [str(p.script), str(p.dup)],                                   # 영상이 없음
    lambda p: [str(p.script)],                                               # 하나뿐
    lambda p: [str(p.script), str(p.inp / "없는 영상.mkv")],                  # 없는 파일
    lambda p: [str(p.script), str(p.inp)],                                   # 폴더
])
def test_split_inputs_stops_when_it_cannot_tell(proj, make):
    with pytest.raises(film.Stop):
        film.split_inputs(make(proj))


# ------------------------------------------------------------------ 대본 찾기
def test_find_doc_by_file_sha1(proj):
    r, notes = film.find_doc(str(proj.script), catalog.load())
    assert r["doc_id"] == "fm10" and "fm10" in notes[0]


def test_find_doc_excluded_duplicate_goes_to_original_then_primary(proj):
    rows = catalog.load()
    r, notes = film.find_doc(str(proj.dup), rows)
    assert r["doc_id"] == "fm11" and any("fm12" in n and "fm11" in n for n in notes)
    p, more = film.primary_of(r, rows)
    assert p["doc_id"] == "fm10" and "대표 대본이 아니다" in more[0]
    assert film.find_doc("fm12", rows)[0]["doc_id"] == "fm11"          # doc_id 로 줘도 같다


def test_find_doc_non_primary_doc_id_runs_primary(proj):
    rows = catalog.load()
    r, _ = film.find_doc("fm11", rows)
    p, notes = film.primary_of(r, rows)
    assert (r["doc_id"], p["doc_id"]) == ("fm11", "fm10") and notes
    assert film.primary_of(film.find_doc("fm10", rows)[0], rows)[1] == []


def test_find_doc_same_name_other_content_stops_and_names_doc(proj):
    """catalog 에 없는 파일인데 이름이 제목과 같으면 조용히 바꿔 쓰지 않고 그 doc_id 를 알리며 멈춘다."""
    other = proj.inp / "다른 판" / "써니 시나리오.hwp"
    other.parent.mkdir()
    other.write_bytes(b"sunny script edited by hand")
    with pytest.raises(film.Stop) as e:
        film.find_doc(str(other), catalog.load())
    assert "내용이 다르다" in str(e.value) and "fm10" in str(e.value) and "sink-film.cmd fm10" in str(e.value)


def test_find_doc_unknown_file_stops(proj):
    f = proj.inp / "새 영화.hwp"
    f.write_bytes(b"brand new")
    with pytest.raises(film.Stop, match="통일해 둔 7문서에 없는 대본"):
        film.find_doc(str(f), catalog.load())


def test_find_doc_by_title_one_many_none(proj):
    rows = catalog.load()
    assert film.find_doc("괴물", rows)[0]["doc_id"] == "fm20"
    with pytest.raises(film.Stop) as e:
        film.find_doc("커피", rows)
    assert "fm40" in str(e.value) and "fm41" in str(e.value)
    with pytest.raises(film.Stop):
        film.find_doc("써니", rows)                                     # 같은 영화의 두 판(fm10, fm11): 목록
    with pytest.raises(film.Stop):
        film.find_doc("없는 제목", rows)
    with pytest.raises(film.Stop, match="catalog 에 없는 doc_id"):
        film.find_doc("fm999", rows)


@pytest.mark.parametrize("doc, word", [("fm30", "대본 언어 en"), ("fm31", "번역 대본")])
def test_non_korean_or_translated_script_stops_before_anything(proj, stages, capsys, doc, word):
    assert film.run(fargs(doc, proj.video)) == 2
    assert stages.calls == [] and word in capsys.readouterr().err


def test_missing_script_json_stops(proj, stages, capsys):
    (paths.DOCS / "fm20" / "script.json").unlink()
    assert film.run(fargs("fm20", proj.video)) == 2
    assert stages.calls == [] and "sinkprep all" in capsys.readouterr().err


# ------------------------------------------------------------------ 장치 규칙
def test_first_asr_without_gpu_stops_before_any_stage(proj, stages, monkeypatch, capsys):
    gpu(monkeypatch, "cpu", vram=1200)
    assert film.run(fargs(proj.script, proj.video)) == 2
    err = capsys.readouterr().err
    assert stages.calls == []
    assert "여유 VRAM" in err and "1200MiB" in err and "--device cpu" in err and "asr-gpu" in err


def test_first_asr_with_missing_cuda_libs_stops(proj, stages, monkeypatch, capsys):
    gpu(monkeypatch, "cuda", libs_missing=("cuBLAS",))
    assert film.run(fargs(proj.script, proj.video)) == 2
    assert stages.calls == [] and "CUDA 라이브러리(cuBLAS)" in capsys.readouterr().err


def test_gpu_problem_reasons(monkeypatch):
    gpu(monkeypatch, "cpu", count=0)
    assert "CUDA 장치를 찾지 못했다" in film.gpu_problem()
    gpu(monkeypatch, "cpu", count=-1)
    assert "설치되지 않았다" in film.gpu_problem()
    gpu(monkeypatch, "cuda")
    assert film.gpu_problem() is None


def test_device_cpu_given_runs_on_cpu(proj, stages, monkeypatch):
    gpu(monkeypatch, "cpu")
    assert film.run(fargs(proj.script, proj.video, device="cpu")) == 0
    assert stages.args("asr")["device"] == "cpu"


def test_first_asr_with_gpu_passes_cuda(proj, stages):
    assert film.run(fargs(proj.script, proj.video)) == 0
    assert stages.args("asr")["device"] == "cuda"


def test_already_transcribed_keeps_previous_device(proj, stages, monkeypatch):
    """지문이 있는(이 영상을 받아쓴) 영화는 GPU 가 바빠도 멈추지 않고 --device 를 넘기지 않는다."""
    util.set_stage_fp("asr", FILM_ID, {"video_sha1": proj.sha, "device": "cuda", "passes": film.DEFAULT_PASSES})
    gpu(monkeypatch, "cpu")
    assert film.run(fargs(proj.script, proj.video)) == 0
    assert stages.args("asr")["device"] is None


def test_fingerprint_of_other_video_counts_as_first_time(proj, stages, monkeypatch):
    util.set_stage_fp("asr", FILM_ID, {"video_sha1": "f" * 40, "device": "cpu"})
    gpu(monkeypatch, "cpu")
    assert film.run(fargs(proj.script, proj.video, force=True)) == 2 and stages.calls == []


# ------------------------------------------------------------------ 단계 차례
def test_end_to_end_calls_stages_in_order_with_command_args(proj, stages, monkeypatch, capsys):
    opened = []
    monkeypatch.setattr(film, "open_folder", opened.append)
    assert film.run(fargs(proj.video, proj.script, open=True)) == 0          # 순서를 바꿔 줘도 된다
    out = capsys.readouterr().out
    assert stages.names() == ["video", "asr", "align", "final"]
    v = stages.args("video")
    assert (v["cmd"], v["docs"], v["paths"], v["lang"], v["force"], v["unregister"]) == \
        ("video", "fm10", [str(proj.video)], "ko", False, False)
    a = stages.args("asr")
    assert (a["cmd"], a["docs"], a["passes"], a["rescue"], a["device"], a["threads"], a["force"]) == \
        ("asr", "fm10", "plain=large-v3-turbo,lv3=large-v3", True, "cuda", None, False)
    al = stages.args("align")
    assert (al["cmd"], al["docs"], al["llm"], al["force"]) == ("align", "fm10", True, False)
    f = stages.args("final")
    assert (f["cmd"], f["docs"], f["force"]) == ("final", "fm10", False)
    assert stages.checked == ["final.json", "work/align.json", "work/asr.json"]
    for name in ("준비", "video", "asr", "align", "final", "validate", "preview", "끝"):
        assert f"== film: {name} ==" in out
    srt = (paths.FILMS / FILM_ID / "preview.srt").read_text(encoding="utf-8-sig")
    assert "수지: 안녕" in srt and "[+] 뭐야" in srt and "[없음] 수지: 잘 가" in srt
    assert "써니 (fm10, local_0010)" in out and "2:04:09 (7449.000초)" in out
    assert "plain 1,200 · lv3 1,300 · rescue 40" in out
    assert "same 1,500 · changed 600 · absent 300(말 있는 대사) · added 450, 일치율 0.812" in out
    assert "확인 필요 3건 (asr_low_conf 2, other 1)" in out
    assert str(paths.FILMS / FILM_ID / "final.json") in out and str(paths.VIDEOS / f"{FILM_ID}.mkv") in out
    assert "[+] 는 대본에 없는 말" in out and "걸린 시간" in out
    assert opened == [paths.FILMS / FILM_ID]


def test_options_reach_stages(proj, stages):
    args = fargs(proj.script, proj.video, lang="mixed", passes="plain=turbo", rescue=False, threads=4,
                 device="cuda", llm=False, workers=2)
    assert film.run(args) == 0
    assert stages.args("video")["lang"] == "mixed"
    a = stages.args("asr")
    assert (a["passes"], a["rescue"], a["threads"], a["device"]) == ("plain=turbo", False, 4, "cuda")
    assert (stages.args("align")["llm"], stages.args("align")["workers"]) == (False, 2)


@pytest.mark.parametrize("stage, after", [("video", []), ("asr", ["video"]), ("align", ["video", "asr"]),
                                          ("final", ["video", "asr", "align"])])
def test_failed_stage_stops_the_rest(proj, stages, capsys, stage, after):
    stages.fail[stage] = 1
    assert film.run(fargs(proj.script, proj.video)) == 1
    assert stages.names() == after + [stage]
    err = capsys.readouterr().err
    assert f"{stage} 단계에서 멈춤" in err and "같은 명령을 다시 치면" in err
    assert f'sink-film.cmd "{proj.script}" "{proj.video}"' in err
    assert not (paths.FILMS / FILM_ID / "preview.srt").exists()


def test_stage_that_exits_zero_without_output_counts_as_failure(proj, stages, capsys):
    stages.no_output.add("align")                    # 대상을 못 찾은 align 은 0 으로 끝난다
    assert film.run(fargs(proj.script, proj.video)) == 1
    assert stages.names() == ["video", "asr", "align"] and "align.json 이 없다" in capsys.readouterr().err


def test_stage_exception_is_a_failure(proj, stages, monkeypatch, capsys):
    def boom(ns):
        raise RuntimeError("터짐")
    monkeypatch.setattr(asr, "run", boom)
    assert film.run(fargs(proj.script, proj.video)) == 1
    err = capsys.readouterr().err
    assert "RuntimeError: 터짐" in err and "asr 단계에서 멈춤" in err and "AsrUnavailable" in err


def test_check_failure_stops_before_preview(proj, stages, capsys):
    stages.check_errors = ["[D2] 시험 오류"]
    assert film.run(fargs(proj.script, proj.video)) == 1
    out = capsys.readouterr()
    assert "FAIL films/local_0010/final.json" in out.out and "validate 단계에서 멈춤" in out.err
    assert not (paths.FILMS / FILM_ID / "preview.srt").exists()


def test_claude_unavailable_prints_notice(proj, stages, monkeypatch, capsys):
    monkeypatch.setattr(llm, "stopped", lambda: "LLMUnavailable: claude 실행 파일을 찾을 수 없음(PATH)")
    assert film.run(fargs(proj.script, proj.video)) == 0
    assert "claude 를 부를 수 없어" in capsys.readouterr().out


def test_other_video_registered_stops_unless_force(proj, stages, capsys):
    catalog.finalize(updates={d: {"video_path": "videos/local_0010.mp4", "video_sha1": "e" * 40,
                                  "video_duration": "100.000", "video_lang": "ko"} for d in ("fm10", "fm11")})
    assert film.run(fargs(proj.script, proj.video)) == 2
    assert stages.calls == [] and "--force" in capsys.readouterr().err
    assert film.run(fargs(proj.script, proj.video, force=True)) == 0
    assert stages.args("video")["force"] is True and stages.args("asr")["force"] is False
    assert by_id()["fm10"]["video_sha1"] == proj.sha


def test_rerun_after_success_runs_every_stage_again_for_their_own_skip(proj, stages):
    """다시 쳐도 film 이 단계를 건너뛰지 않는다(건너뛰기는 각 단계의 지문이 정한다). 두 번째는 지문이 있어 장치를 넘기지 않음."""
    assert film.run(fargs(proj.script, proj.video)) == 0
    util.set_stage_fp("asr", FILM_ID, {"video_sha1": proj.sha, "device": "cuda"})
    stages.calls.clear()
    assert film.run(fargs(proj.script, proj.video)) == 0
    assert stages.names() == ["video", "asr", "align", "final"] and stages.args("asr")["device"] is None


def test_again_quotes_paths_and_keeps_options():
    a = fargs(r"C:\Users\x\Desktop\Temp\써니.hwp", r"C:\a b\v.mkv", passes="plain=turbo", rescue=False, llm=False,
              device="cpu", force=True)
    assert film.again(a) == (r'sink-film.cmd "C:\Users\x\Desktop\Temp\써니.hwp" "C:\a b\v.mkv" '
                             '--passes "plain=turbo" --device cpu --no-rescue --no-llm --force')


def test_cli_parser_when_registered():
    """__main__ 에 film 이 등록되면 옵션이 film.run 이 읽는 칸으로 들어오는지(등록 전에는 건너뜀)."""
    from sinkprep.__main__ import COMMANDS, build_parser
    if "film" not in COMMANDS:
        pytest.skip("film 이 아직 sinkprep.__main__.COMMANDS 에 없음(등록 전)")
    ns = build_parser().parse_args(["film", "a.hwp", "b.mkv"])
    assert (ns.cmd, ns.paths, ns.lang, ns.passes, ns.device, ns.rescue, ns.llm, ns.open, ns.force) == \
        ("film", ["a.hwp", "b.mkv"], None, None, None, None, None, False, False)
    ns = build_parser().parse_args(["film", "a.hwp", "b.mkv", "--no-rescue", "--no-llm", "--device", "cpu",
                                    "--passes", "plain=turbo", "--threads", "4", "--lang", "ko", "--open"])
    assert (ns.rescue, ns.llm, ns.device, ns.passes, ns.threads, ns.lang, ns.open) == \
        (False, False, "cpu", "plain=turbo", 4, "ko", True)


# ------------------------------------------------------------------ align·final 을 진짜로
TURNS = [("1", "scene", "S#1 공원"), ("1", "dialogue", "오늘 날씨가 정말 좋네요."),
         ("1", "dialogue", "그러게요. 산책이나 갈까요?"), ("1", "dialogue", "저는 집에서 책을 읽고 싶어요."),
         ("2", "scene", "S#2 역"), ("2", "dialogue", "내일 아침 일찍 출발합시다.")]
SAID = ["오늘 날씨가 정말 좋네요.", "그러게요. 산책이나 갈까요?", "누구세요?", "내일 아침 일찍 출발합시다."]


def test_real_align_final_check_and_preview(proj, monkeypatch, capsys):
    """video·asr 만 가짜로 두고 align·final·검사·미리보기 자막은 진짜로: 명령 인자 그대로 받아 끝까지 간다."""
    class NoClaude:
        def __call__(self, *a, **kw):
            raise llm.LLMUnavailable("시험에서는 claude 를 부르지 않음")
    llm.set_backend(NoClaude())                          # --no-llm 이라 부르지 않지만, 혹시 불러도 claude 는 뜨지 않게
    script = S.tiny_script(TURNS)
    script["doc_id"] = "fm10"
    util.write_json(paths.DOCS / "fm10" / "script.json", script)
    words, t = [], 10.0
    for u in SAID:
        t += 1.0
        for w in u.split():
            words.append((w, round(t, 3), round(t + 0.3, 3), 0.9))
            t += 0.3
    data = S.asr_of(S.say(words), vad=[[10.0, 30.0]])
    data["video_sha1"] = proj.sha
    calls = []

    def fake_video(ns):
        calls.append("video")
        dst = paths.VIDEOS / f"{FILM_ID}.mkv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(ns.paths[0]).read_bytes())
        catalog.finalize(updates={d: {"video_path": f"videos/{dst.name}", "video_sha1": proj.sha,
                                      "video_duration": "600.000", "video_lang": "ko"} for d in ("fm10", "fm11")})
        return 0

    def fake_asr(ns):
        calls.append(("asr", ns.device))
        util.write_json(paths.FILMS / FILM_ID / "work" / "asr.json", data)
        return 0
    monkeypatch.setattr(video, "run", fake_video)
    monkeypatch.setattr(asr, "run", fake_asr)
    try:
        rc = film.run(fargs(proj.script, proj.video, device="cpu", llm=False))
    finally:
        llm.set_backend(None)
        llm.reset_run()
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert calls == ["video", ("asr", "cpu")]
    fdir = paths.FILMS / FILM_ID
    assert validate.check_file(fdir / "final.json") == ("final", [])
    assert util.read_json(fdir / "final.json")["film"]["script"]["doc_id"] == "fm10"
    assert "오늘 날씨가 정말 좋네요." in (fdir / "preview.srt").read_text(encoding="utf-8-sig")
    assert by_id()["fm10"]["status"] == "done"
    assert "ok   films/local_0010/final.json (final)" in out.out and "일치율" in out.out
