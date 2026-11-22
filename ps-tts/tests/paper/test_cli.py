"""논문 판 명령줄: 옵션 -> PaperConfig, 문장 하나와 파일 모드, 끝 코드(0, 1, 2, 130), run_paper.cmd 와 setup_paper.cmd,
requirements-paper.txt 고정 판.

끝까지 도는 시험은 cli.run_paper 를 가짜 모델을 넣은 진짜 run_paper 로 바꿔 돌린다(진짜 approx 프런트엔드).
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

from pstts.config import ROOT
from pstts.paper import cli
from pstts.paper.config import PaperConfig
from pstts.paper.io import OUTPUT_NAMES
from pstts.paper.pipeline import run_paper

EN = "We have to leave before the storm hits."
T0 = "폭풍이 오기 전에 떠나야 해."
PS_TEXTS = ["폭풍 전에 가자.", "폭풍이 치기 전에 떠나자.", "비가 오기 전에 가야 해."]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    return env


# ---------------------------------------------------------------- 옵션

def test_options_map_to_paper_config() -> None:
    args = cli.build_parser().parse_args([
        "Hi.", "--device", "cpu", "--cpu-threads", "3", "--llm", "file", "--llm-file", "c.jsonl", "--llm-chunk", "4",
        "--no-llm-cache", "--iso-batch", "5", "--iso-max", "30", "--ps-n", "20", "--ps-chunk", "1", "--ps-extra-rounds",
        "0", "--alpha", "1.2", "--beta", "0.8", "--sakoe-chiba-radius", "3", "--labse-min", "0.7", "--iso-tol-frames",
        "30", "--tts", "yourtts", "--tts-dir", "m", "--tts-checkpoint", "c.pth", "--tts-speaker", "s", "--tts-lang-ko",
        "kr", "--vowel-table", "v.json", "--nmt-ckpt", "x.pt", "--beam", "2", "--out", "o", "--seconds", "1.5"])
    cfg = cli.config_from_args(args)
    assert (cfg.device, cfg.cpu_threads, cfg.llm, cfg.llm_file, cfg.llm_chunk, cfg.llm_cache) == (
        "cpu", 3, "file", "c.jsonl", 4, False)
    assert (cfg.iso_batch, cfg.iso_max, cfg.ps_n, cfg.ps_chunk, cfg.ps_extra_rounds) == (5, 30, 20, 1, 0)
    assert (cfg.alpha, cfg.beta, cfg.sakoe_chiba_radius, cfg.labse_min, cfg.iso_tol_frames) == (1.2, 0.8, 3, 0.7, 30)
    assert (cfg.tts, cfg.tts_dir, cfg.tts_checkpoint, cfg.tts_speaker, cfg.tts_lang_ko, cfg.vowel_table) == (
        "yourtts", "m", "c.pth", "s", "kr", "v.json")
    assert (cfg.nmt_ckpt, cfg.nmt_beam, cfg.out_dir) == ("x.pt", 2, "o")
    default = cli.config_from_args(cli.build_parser().parse_args(["Hi."]))
    assert default == PaperConfig() and default.sakoe_chiba_radius is None and default.device == "cuda"


@pytest.mark.parametrize("argv, words", [
    ([], "하나만"), (["Hi.", "--input", "x.csv"], "하나만"), (["--input", "x.csv", "--seconds", "2"], "--seconds"),
    (["Hi.", "--tts", "wav"], "invalid choice")])
def test_argument_errors_exit_2(argv: list[str], words: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2 and words in capsys.readouterr().err


def test_input_and_value_errors_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--input", str(tmp_path / "none.csv"), "--out", str(tmp_path / "o")]) == 2
    assert "입력 파일이 없습니다" in capsys.readouterr().err
    assert cli.main(["Hi.", "--seconds", "-1", "--out", str(tmp_path / "o")]) == 2
    assert "seconds" in capsys.readouterr().err
    assert cli.main(["Hi.", "--tts", "yourtts", "--device", "cpu", "--out", str(tmp_path / "o")]) == 2
    assert "--tts-dir" in capsys.readouterr().err


# ---------------------------------------------------------------- 끝까지(가짜 모델)

def _patched(monkeypatch: pytest.MonkeyPatch, fakes, ps_texts: list[str], stop_stage: str | None = None) -> list:
    """cli.run_paper 를 가짜 모델 2, LaBSE, COMET, 모델 1 을 넣은 진짜 run_paper 로 바꾼다."""
    seen: list = []

    def script(stage, rnd, req):
        return ps_texts if stage == "ps" else []

    def fake_run(items, cfg):
        seen.append(cfg)
        return run_paper(items, cfg, backend=fakes.Gen(script, stop_stage=stop_stage),
                         make_drafter=lambda: fakes.Drafter(default=T0), make_labse=fakes.Labse,
                         make_comet=fakes.Comet, log=lambda m: None)
    monkeypatch.setattr(cli, "run_paper", fake_run)
    return seen


def test_single_sentence_prints_choice_and_writes(monkeypatch, fakes, tmp_path: Path,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    seen = _patched(monkeypatch, fakes, PS_TEXTS)
    out = tmp_path / "single"
    assert cli.main([EN, "--seconds", "1.5", "--device", "cpu", "--ps-n", "3", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "PS-Comet(식 20, 최종):" in text and "K': 129.2프레임(seconds 칸 1.5초, 원문 TTS 길이 1.454초, R 1.032)" in text
    assert "저장:" in text
    assert seen[0].ps_n == 3 and seen[0].device == "cpu"
    assert sorted(p.name for p in out.iterdir()) == sorted(OUTPUT_NAMES)
    header = (out / "results.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    assert header.endswith(",message,seconds")


def test_file_mode_and_exit_codes(monkeypatch, fakes, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "lines.csv"
    src.write_text(f"id,source_en,register\nx1,{EN},반말\n", encoding="utf-8-sig")
    _patched(monkeypatch, fakes, PS_TEXTS)
    assert cli.main(["--input", str(src), "--device", "cpu", "--ps-n", "3", "--out", str(tmp_path / "a")]) == 0
    got = capsys.readouterr()
    assert re.search(r"x1 \[고름\] .+ \(ISO 초벌 통과, PS-Comet 점수 [0-9.]+, DTW [0-9.]+\)", got.out)
    assert "PS-Comet 고름 1줄, 실패 0줄" in got.out
    # PS 후보가 없으면 실패한 줄이 있어 1
    _patched(monkeypatch, fakes, [])
    assert cli.main(["--input", str(src), "--device", "cpu", "--out", str(tmp_path / "b")]) == 1
    assert "x1 [실패] (없음)" in capsys.readouterr().out
    # 모델 2 를 쓸 수 없으면 1 과 주의
    _patched(monkeypatch, fakes, PS_TEXTS, stop_stage="ps")
    assert cli.main(["--input", str(src), "--device", "cpu", "--out", str(tmp_path / "c")]) == 1
    assert "모델 2 를 쓸 수 없어" in capsys.readouterr().err


@pytest.mark.parametrize("exc, code, words", [
    (KeyboardInterrupt(), 130, "멈춤(Ctrl+C)"), (RuntimeError("GPU 없음"), 1, "오류: GPU 없음"),
    (ValueError("값 틀림"), 2, "오류: 값 틀림"), (ZeroDivisionError("0"), 1, "예상하지 못한 문제")])
def test_exceptions_map_to_exit_codes(monkeypatch, tmp_path: Path, capsys, exc: BaseException, code: int,
                                      words: str) -> None:
    def boom(items, cfg):
        raise exc
    monkeypatch.setattr(cli, "run_paper", boom)
    assert cli.main(["Hi.", "--device", "cpu", "--out", str(tmp_path / "o")]) == code
    assert words in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_unused_columns_are_not_validated(monkeypatch, fakes, tmp_path: Path) -> None:
    """논문 판이 쓰지 않는 register, lip, target_syllables 값이 기존 판 꼴에 맞지 않아도 멈추지 않고 그대로 적는다."""
    src = tmp_path / "bad.csv"
    src.write_text(f"id,source_en,register,lip,target_syllables\nx1,{EN},존댓말,yes,abc\nx2,{EN},반말,off,9\n",
                   encoding="utf-8-sig")
    _patched(monkeypatch, fakes, PS_TEXTS)
    out = tmp_path / "o"
    assert cli.main(["--input", str(src), "--device", "cpu", "--ps-n", "3", "--out", str(out)]) == 0
    with (out / "results.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["register"], r["lip"], r["target_syllables"]) for r in rows] == [("존댓말", "yes", "abc"),
                                                                                ("반말", "off", "9")]
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["unused_columns"] == {"register": 2, "lip": 2, "target_syllables": 2}
    res = [json.loads(x) for x in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert res[0]["input_unused"] == {"register": "존댓말", "lip": "yes", "target_syllables": "abc",
                                      "relationship": "", "prev_en": "", "meaning_notes": ""}
    assert res[0]["extra"] == {} and res[1]["input_unused"]["target_syllables"] == 9


def test_llm_file_alone_means_file_backend(monkeypatch, fakes, tmp_path: Path,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    """재생 명령에서 --llm file 을 빠뜨려도 구독 claude 를 부르지 않는다."""
    from pstts.llm import FileBackend, make_backend

    seen = _patched(monkeypatch, fakes, PS_TEXTS)
    replay = tmp_path / "candidates.jsonl"
    replay.write_text("", encoding="utf-8")
    assert cli.main([EN, "--device", "cpu", "--ps-n", "3", "--llm-file", str(replay), "--out", str(tmp_path / "o")]) == 0
    assert seen[0].llm == "file" and seen[0].llm_file == str(replay)
    assert isinstance(make_backend(seen[0].base_config()), FileBackend)
    assert "--llm file 로 돈다" in capsys.readouterr().err


def test_llm_file_with_another_backend_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        cli.main(["Hi.", "--llm", "claude", "--llm-file", "c.jsonl"])
    assert e.value.code == 2 and "--llm file 과 함께" in capsys.readouterr().err


def test_g2p_options(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = cli.config_from_args(cli.build_parser().parse_args(["Hi.", "--g2p-en-voice", "en-us", "--g2p-en-stress"]))
    assert cfg.g2p_en_voice == "en-us" and cfg.g2p_en_stress is True
    default = cli.config_from_args(cli.build_parser().parse_args(["Hi."]))
    assert default.g2p_en_voice == "en-gb" and default.g2p_en_stress is False
    # 없는 목소리는 모델을 올리기 전에 입력·설정 오류(2)로 멈춘다
    assert cli.main(["Hi.", "--g2p-en-voice", "fr-fr", "--device", "cpu", "--out", str(tmp_path / "o")]) == 2
    assert "영어 목소리" in capsys.readouterr().err and not (tmp_path / "o").exists()


def test_module_help_runs() -> None:
    r = subprocess.run([sys.executable, "-m", "pstts.paper", "--help"], capture_output=True, text=True,
                       encoding="utf-8", cwd=ROOT, env=_env(), timeout=120)
    assert r.returncode == 0 and "run_paper.cmd" in r.stdout and "--sakoe-chiba-radius" in r.stdout


# ---------------------------------------------------------------- cmd 파일

def _above_exit(path: Path) -> bytes:
    raw = path.read_bytes()
    return raw.split(b"exit /b %PSTTS_EXIT%")[0]


@pytest.mark.parametrize("name", ["run_paper.cmd", "setup_paper.cmd"])
def test_cmd_files_are_crlf_and_ascii_above_exit(name: str) -> None:
    raw = (ROOT / name).read_bytes()
    assert raw.count(b"\n") == raw.count(b"\r\n") and b"exit /b %PSTTS_EXIT%" in raw
    assert all(b < 128 for b in _above_exit(ROOT / name))
    text = raw.decode("utf-8")
    assert 'set "PSTTS_CP=%%a"' in text and "chcp 65001 >nul" in text and "if defined PSTTS_CP chcp %PSTTS_CP% >nul" in text


def test_run_paper_cmd_matches_run_cmd_except_module() -> None:
    ours = _above_exit(ROOT / "run_paper.cmd").decode("ascii")
    theirs = _above_exit(ROOT / "run.cmd").decode("ascii")
    assert ours == theirs.replace(" -m pstts %*", " -m pstts.paper %*")


def test_run_paper_cmd_passes_exit_codes() -> None:
    cmd = os.environ.get("COMSPEC", "cmd.exe")
    ok = subprocess.run([cmd, "/c", str(ROOT / "run_paper.cmd"), "--help"], capture_output=True, cwd=ROOT, timeout=120)
    assert ok.returncode == 0 and b"run_paper.cmd" in ok.stdout
    bad = subprocess.run([cmd, "/c", str(ROOT / "run_paper.cmd")], capture_output=True, cwd=ROOT, timeout=120)
    assert bad.returncode == 2


def test_setup_paper_cmd_uses_frozen_constraints() -> None:
    text = (ROOT / "setup_paper.cmd").read_text(encoding="utf-8")
    assert '-m pip freeze > "%PSTTS_CONSTRAINTS%"' in text
    assert ('-m pip install -c "%PSTTS_CONSTRAINTS%" torchaudio==2.11.0 --index-url '
            "https://download.pytorch.org/whl/cu128") in text
    assert '-m pip install -c "%PSTTS_CONSTRAINTS%" -r "%~dp0requirements-paper.txt"' in text
    assert "nltk.download('cmudict', download_dir=d" in text and "os.path.join(sys.prefix, 'nltk_data')" in text
    assert "from pstts.paper import g2p" in text


def test_requirements_paper_pins_installed_versions() -> None:
    pins = {}
    for line in (ROOT / "requirements-paper.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, _, ver = line.partition("==")
            assert ver, f"판을 고정하지 않은 줄: {line}"
            pins[name.strip()] = ver.strip()
    assert {"phonemizer", "g2pk", "coqui-tts", "jamo", "nltk"} <= {k.lower() for k in pins}
    for name, ver in pins.items():
        assert version(name) == ver, name
