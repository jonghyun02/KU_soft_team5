"""Your-TTS 학습 경로 점검(스모크). 진짜 모델을 만들지 않는다. CPU 로 10분 안에 끝난다.

1. enko-nmt valid.tsv(읽기만)에서 영어 8줄, 한국어 8줄을 골라 합성 소리(음절마다 톤 묶음 + 작은 잡음, 22050 Hz)를 붙인다.
   화자는 언어마다 둘(기본 높이가 다른 톤). 두 언어를 따로 CSV 로 써서 LibriTTS, KMSSS 처럼 자료마다 8:1:1 로 나뉜다.
2. make_manifest.py(--csv) 로 매니페스트, phonemize_manifest.py 로 phonemes 칸과 characters.txt 를 만든다.
   글자 목록에는 valid.tsv 전체(영어 5,996줄, 한국어 5,996줄)의 토큰도 더한다(--extra-en, --extra-ko). 학습 줄은 16개
   그대로이고, 점검용 모델이 다른 문장(파이프라인 후보 등)에서 모르는 토큰 오류로 바로 멈추지 않게 하려는 것이다.
3. compute_dvectors.py 로 공개 YourTTS 화자 인코더 d-vector 를 CPU 로 만든다.
4. train_yourtts.py 를 따로 띄워(같은 명령 경로) 아주 작은 구조(--arch tiny)를 몇 걸음 CPU 로 학습한다.
5. 마지막 체크포인트를 models/yourtts_smoke 에 model_file.pth 로 내보내고 README_SMOKE.txt 를 쓴다.
6. scripts/paper_build_vowel_table.py 로 문장 몇십 개 모음표를 만든다.
7. YourTTSFrontend(make_frontend) 로 길이를 재 보고 결과를 smoke_report.json 에 남긴다.
값은 뜻이 없다. 경로가 끝까지 도는지만 본다. 모든 단계가 CPU 이고 GPU 는 가린다(CUDA_VISIBLE_DEVICES='').

실행: .venv\\Scripts\\python.exe scripts\\yourtts\\smoke_train.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _p in (str(ROOT), str(HERE), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VALID_TSV = Path("D:/transformer-base/enko-nmt/data/clean_v2/valid.tsv")
WORK = ROOT / "out" / "yourtts_smoke_work"
OUT = ROOT / "models" / "yourtts_smoke"
WORK_MARK = "smoke_work.txt"
SR = 22050
RUN_NAME = "smoke_check_not_real"
SPEAKERS = {"en": (("spk_en_a", 110.0), ("spk_en_b", 190.0)), "ko": (("spk_ko_a", 130.0), ("spk_ko_b", 220.0))}
EXAMPLES = {"en": "We have to leave before the storm hits.", "ko": "폭풍이 오기 전에 떠나야 해."}
EXPORTED = ("model_file.pth", "config.json", "d_vectors.pth", "d_vector_means.pth", "speakers.pth", "characters.txt",
            "g2p.json", "vowel_table.json", "vowel_table.csv", "train_result.json", "README_SMOKE.txt", "smoke_report.json")
_HANGUL_LINE = re.compile(r"^[가-힣 .,?!]+$")
_EN_LINE = re.compile(r"^[A-Za-z ,.?!']+$")

README = """점검용(스모크) 모델입니다. 진짜 Your-TTS 모델이 아닙니다.

- scripts\\yourtts\\smoke_train.py 가 학습 경로를 끝까지 돌려 보려고 만든 것입니다.
- 합성 소리(톤 묶음과 작은 잡음) 16개와 enko-nmt valid 의 영어 8줄, 한국어 8줄로 아주 작은 구조(--arch tiny)를
  CPU 로 {steps}걸음만 학습했습니다. 사람 목소리를 배운 적이 없습니다.
- 그래서 이 폴더의 토큰별 길이와 모음표(vowel_table.json) 값에는 뜻이 없습니다. 논문 판 결과를 내는 데 쓰지 마세요.
- 글자 목록(characters.txt)은 학습 16줄에 enko-nmt valid 전체(영어, 한국어 각 5,996줄)의 토큰을 더한 것입니다.
  경로 점검에서 모르는 토큰 오류가 덜 나게 하려는 것이고, 학습 16줄에 없던 토큰은 학습되지 않았습니다.
  그래도 목록에 없는 글자(드문 기호 등)가 나오면 run_paper 는 그 글자를 버리고(Coqui 와 같게) run.json 에 셉니다.
  YourTTSFrontend 를 직접 쓸 때만 모르는 토큰 오류가 납니다.
- 진짜 모델은 LibriTTS-360 과 KMSSS 로 scripts\\yourtts\\train_yourtts.py 를 GPU 에서 돌려 만듭니다.

만든 때: {when}
학습 결과: train_result.json, 점검 결과: smoke_report.json
"""


# ---------------------------------------------------------------------------- 1. 자료
def pick_lines(path: Path, n_each: int = 8) -> dict[str, list[str]]:
    """valid.tsv(영어<TAB>한국어<TAB>갈래)에서 고른다. 영어는 숫자 없는 60~120자, 한국어는 한글과 문장부호만 25~50자.
    파일 전체에 고르게 퍼지게 같은 간격으로 본다."""
    rows = path.read_text(encoding="utf-8").splitlines()
    picked: dict[str, list[str]] = {"en": [], "ko": []}
    step = max(1, len(rows) // 400)
    for i in range(0, len(rows), step):
        parts = rows[i].split("\t")
        if len(parts) < 2:
            continue
        en, ko = parts[0].strip(), parts[1].strip()
        if len(picked["en"]) < n_each and 60 <= len(en) <= 120 and _EN_LINE.match(en):
            picked["en"].append(en)
        if len(picked["ko"]) < n_each and 25 <= len(ko) <= 50 and _HANGUL_LINE.match(ko):
            picked["ko"].append(ko)
        if all(len(v) >= n_each for v in picked.values()):
            break
    if any(len(v) < n_each for v in picked.values()):
        raise RuntimeError(f"valid.tsv 에서 줄을 다 고르지 못했습니다: {[len(v) for v in picked.values()]}")
    return picked


def syllable_units(text: str, lang: str) -> int:
    """합성 소리 길이용 대략 음절 수: 한국어 한글 글자 수, 영어 모음 글자 묶음 수."""
    if lang == "ko":
        return sum(1 for c in text if "가" <= c <= "힣")
    return len(re.findall(r"[aeiouy]+", text.lower()))


def synth(units: int, f0: float, rng: np.random.Generator, max_sec: float = 3.0) -> np.ndarray:
    """음절마다 배음 4개 톤 묶음(한닝 창) + 쉼, 작은 잡음. 최대 max_sec 초."""
    syl, gap = 0.15, 0.04
    n = max(3, min(units, int(max_sec / (syl + gap))))
    out = [np.zeros(int(0.1 * SR))]
    for _ in range(n):
        f = f0 * (1.0 + 0.08 * rng.uniform(-1.0, 1.0))
        t = np.arange(int(syl * SR)) / SR
        tone = sum((0.6 / h) * np.sin(2 * np.pi * h * f * t + rng.uniform(0, np.pi)) for h in range(1, 5))
        out += [tone * np.hanning(t.size), np.zeros(int(gap * SR))]
    out.append(np.zeros(int(0.1 * SR)))
    wav = np.concatenate(out)
    wav = wav + 0.003 * rng.standard_normal(wav.size)
    return (0.5 * wav / np.max(np.abs(wav))).astype(np.float32)


def make_data(work: Path, lines: dict[str, list[str]]) -> list[Path]:
    """언어마다 CSV(audio,text,speaker,lang)와 wav 를 쓴다."""
    import soundfile as sf

    rng = np.random.default_rng(0)
    paths = []
    for lang, texts in lines.items():
        folder = work / "data"
        (folder / "wav").mkdir(parents=True, exist_ok=True)
        rows = []
        for i, text in enumerate(texts):
            spk, f0 = SPEAKERS[lang][i % 2]
            name = f"wav/{lang}_{i:02d}.wav"
            sf.write(folder / name, synth(syllable_units(text, lang), f0, rng), SR, subtype="PCM_16")
            rows.append({"audio": name, "text": text, "speaker": spk, "lang": lang})
        p = folder / f"smoke_{lang}.csv"
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["audio", "text", "speaker", "lang"], lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        paths.append(p)
    return paths


def extra_texts(path: Path, work: Path) -> dict[str, Path]:
    """글자 목록에만 더할 글: valid.tsv 의 영어 칸 전체, 한국어 칸 전체(한 줄 한 문장 파일)."""
    rows = [r.split("\t") for r in path.read_text(encoding="utf-8").splitlines()]
    out = {}
    for lang, col in (("en", 0), ("ko", 1)):
        p = work / f"extra_{lang}.txt"
        p.write_text("\n".join(r[col].strip() for r in rows if len(r) > col and r[col].strip()) + "\n",
                     encoding="utf-8")
        out[lang] = p
    return out


def vowel_pools(path: Path, work: Path, n: int = 1500) -> dict[str, Path]:
    """모음표 문장 후보: valid.tsv 의 짧은 줄(영어 60자, 한국어 30자 이하)을 파일 전체에서 고르게."""
    rows = path.read_text(encoding="utf-8").splitlines()
    pools: dict[str, list[str]] = {"en": [], "ko": []}
    for row in rows:
        parts = row.split("\t")
        if len(parts) < 2:
            continue
        if len(parts[0]) <= 60:
            pools["en"].append(parts[0].strip())
        if len(parts[1]) <= 30:
            pools["ko"].append(parts[1].strip())
    out = {}
    for lang, items in pools.items():
        step = max(1, len(items) // n)
        p = work / f"vowel_sentences_{lang}.txt"
        p.write_text("\n".join(items[::step][:n]) + "\n", encoding="utf-8")
        out[lang] = p
    return out


# ---------------------------------------------------------------------------- 5. 내보내기
def export(run: dict, man: Path, out: Path, steps: int) -> dict:
    """마지막 체크포인트와 필요한 파일을 out 에 모은다. config 의 경로 칸은 같은 폴더 파일 이름으로 바꾼다."""
    out.mkdir(parents=True, exist_ok=True)
    for name in EXPORTED:
        (out / name).unlink(missing_ok=True)
    run_dir = Path(run["run_dir"])
    import torch
    from trainer.io import load_fsspec

    # 추론에는 가중치만 쓰므로 옵티마이저·스케줄러 상태는 빼고 저장한다(Vits.load_checkpoint 는 state['model'] 만 읽음)
    state = load_fsspec(run["final_checkpoint"], map_location="cpu")
    torch.save({k: state[k] for k in ("model", "config", "step", "epoch", "date") if k in state}, out / "model_file.pth")
    shutil.copyfile(man / "d_vectors.pth", out / "d_vectors.pth")
    shutil.copyfile(man / "characters.txt", out / "characters.txt")
    shutil.copyfile(run_dir / "g2p.json", out / "g2p.json")                     # 학습 때 G2P(추론이 확인함)
    shutil.copyfile(run_dir / "d_vector_means.pth", out / "d_vector_means.pth")   # 추론용 화자 평균
    shutil.copyfile(run_dir / "train_result.json", out / "train_result.json")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    config["model_args"]["d_vector_file"] = "d_vectors.pth"
    config["d_vector_file"] = "d_vectors.pth"
    if (run_dir / "speakers.pth").is_file():
        shutil.copyfile(run_dir / "speakers.pth", out / "speakers.pth")
        config["model_args"]["speakers_file"] = "speakers.pth"
        config["speakers_file"] = "speakers.pth"
    config["run_description"] = "점검용(스모크) 모델. 진짜 모델 아님. 합성 소리로 몇 걸음만 학습함."
    config["output_path"] = str(out)
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=4), encoding="utf-8")
    (out / "README_SMOKE.txt").write_text(README.format(steps=steps, when=time.strftime("%Y-%m-%d %H:%M")),
                                          encoding="utf-8")
    return {"dir": str(out), "files": sorted(p.name for p in out.iterdir())}


# ---------------------------------------------------------------------------- 7. 길이 재 보기
def try_frontend(out: Path, vowel_doc: dict) -> dict:
    """make_frontend(tts='yourtts') 로 올려 길이, 라벨, 비용을 재 본다."""
    from pstts.paper.config import PaperConfig
    from pstts.paper.frontend import Frontend, make_frontend

    cfg = PaperConfig(tts="yourtts", tts_dir=str(out))
    front = make_frontend(cfg, "cpu")
    report: dict = {"is_frontend": isinstance(front, Frontend), "frame_sec": front.frame_sec, "examples": {}}
    try:
        for lang in ("en", "ko"):
            text = EXAMPLES[lang]
            toks = front.tokens(text, lang)
            bad = front.unknown_tokens(toks)
            ex: dict = {"wanted": text, "unknown_tokens": bad}
            if bad:      # 점검용 글자 목록이 작아서 모르는 토큰이 있으면 모음표 문장 하나로 바꾼다
                text = vowel_doc["picked"][lang][0]
                toks = front.tokens(text, lang)
            d1 = front.durations([toks], lang)[0]
            d2 = front.durations([toks], lang)[0]
            frames = front.duration_frames([toks], lang)[0]
            labels = front.labels(toks, lang)
            ex.update({"text": text, "n_tokens": len(toks), "deterministic": bool(np.array_equal(d1, d2)),
                       "sum_sec": round(float(d1.sum()), 4), "frames_sum": round(float(frames.sum()), 3),
                       "min_frames": round(float(frames.min()), 4), "max_frames": round(float(frames.max()), 4),
                       "first": [[t, round(float(s), 4)] for t, s in zip(toks[:12], d1[:12])],
                       "vowel_labels": sum(1 for x in labels if x is not None),
                       "m_p_shape": list(front.encoder_means(toks, lang).shape)})
            report["examples"][lang] = ex
        en_l = front.labels(front.tokens(report["examples"]["en"]["text"], "en"), "en")
        ko_l = front.labels(front.tokens(report["examples"]["ko"]["text"], "ko"), "ko")
        c = front.cost(en_l, ko_l)
        null_null = [i for i, x in enumerate(en_l) if x is None][0], [j for j, y in enumerate(ko_l) if y is None][0]
        report["cost"] = {"shape": list(c.shape), "null_null": float(c[null_null]),
                          "max_distance": front.table.max_distance, "min": float(c.min()), "max": float(c.max())}
        info = front.info()
        report["info"] = {k: info[k] for k in ("name", "device", "checkpoint", "checkpoint_sha1", "speaker",
                                               "noise_scale_dp", "length_scale", "vocab_size", "vowel_table")}
    finally:
        front.close()
    return report


# ---------------------------------------------------------------------------- 실행
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Your-TTS 학습 경로 점검(CPU, 합성 자료, 진짜 모델 아님).")
    ap.add_argument("--work", type=Path, default=WORK, help=f"작업 폴더(기본 {WORK}, 다시 만들어짐)")
    ap.add_argument("--out", type=Path, default=OUT, help=f"내보낼 폴더(기본 {OUT})")
    ap.add_argument("--steps", type=int, default=8, help="학습 걸음 수(기본 8: 한 epoch 6걸음 + 2걸음)")
    ap.add_argument("--num-workers", type=int, default=1, help="학습 데이터 읽기 프로세스 수(기본 1, Windows spawn 점검)")
    ap.add_argument("--n-vowel", type=int, default=30, help="모음표 문장 수(언어마다, 기본 30)")
    ap.add_argument("--cpu-threads", type=int, default=6, help="CPU 스레드 수")
    args = ap.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""            # 모든 단계를 CPU 로. GPU 는 건드리지 않는다
    t0 = time.perf_counter()
    times: dict[str, float] = {}
    work, out = args.work.resolve(), args.out.resolve()
    if work.exists():
        if not (work / WORK_MARK).is_file():
            raise SystemExit(f"{work} 는 스모크 작업 폴더 표시가 없어 지우지 않습니다. 다른 --work 를 주세요.")
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / WORK_MARK).write_text("smoke_train.py 작업 폴더(다시 실행하면 지워짐)\n", encoding="utf-8")

    print("1/7 자료 만들기", flush=True)
    lines = pick_lines(VALID_TSV)
    csvs = make_data(work, lines)
    times["data"] = time.perf_counter() - t0

    print("2/7 매니페스트와 음소화", flush=True)
    import make_manifest
    import phonemize_manifest

    man = work / "manifests"
    make_manifest.main([*sum((["--csv", str(p)] for p in csvs), []), "--out", str(man), "--seed", "0"])
    extra = extra_texts(VALID_TSV, work)
    phonemize_manifest.main(["--manifest-dir", str(man), "--extra-en", str(extra["en"]), "--extra-ko",
                             str(extra["ko"])])
    times["manifest"] = time.perf_counter() - t0

    print("3/7 d-vector(공개 YourTTS 화자 인코더, CPU)", flush=True)
    import compute_dvectors

    compute_dvectors.main(["--manifest-dir", str(man), "--device", "cpu", "--cpu-threads", str(args.cpu_threads)])
    times["dvectors"] = time.perf_counter() - t0

    print(f"4/7 학습(train_yourtts.py, --arch tiny, {args.steps}걸음, CPU)", flush=True)
    runs = work / "runs"
    # 배치 2: 학습 12줄(언어마다 6줄)이 한 epoch 6걸음이 된다. 검증 2줄은 표본 추출기 없이 차례로 읽는다.
    cmd = [sys.executable, str(HERE / "train_yourtts.py"), "--manifest-dir", str(man), "--out", str(runs),
           "--run-name", RUN_NAME, "--device", "cpu", "--cpu-threads", str(args.cpu_threads), "--arch", "tiny",
           "--batch-size", "2", "--eval-batch-size", "2", "--max-steps", str(args.steps),
           "--num-workers", str(args.num_workers), "--print-step", "1"]
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": ""}
    log = work / "train_log.txt"
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=480)
    if proc.returncode != 0:
        print(log.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
        raise SystemExit(f"학습이 실패했습니다(끝 코드 {proc.returncode}). 기록: {log}")
    run = json.loads((runs / "last_run.json").read_text(encoding="utf-8"))
    times["train"] = time.perf_counter() - t0

    print("5/7 내보내기", flush=True)
    exported = export(run, man, out, run["steps"])
    times["export"] = time.perf_counter() - t0

    print("6/7 모음표(문장 몇십 개)", flush=True)
    import paper_build_vowel_table

    pools = vowel_pools(VALID_TSV, work)
    paper_build_vowel_table.main(["--tts-dir", str(out), "--ko-sentences", str(pools["ko"]), "--en-sentences",
                                  str(pools["en"]), "--n-ko", str(args.n_vowel), "--n-en", str(args.n_vowel),
                                  "--seed", "0", "--device", "cpu", "--cpu-threads", str(args.cpu_threads)])
    vdoc = json.loads((out / "vowel_table.json").read_text(encoding="utf-8"))
    vowel_summary = {"ko": vdoc["ko"], "en": vdoc["en"], "selection": vdoc["selection"], "check_u": vdoc["check_u"],
                     "max_distance": vdoc["max_distance"], "n_clusters": vdoc["n_clusters"]}
    picked = {lang: paper_build_vowel_table.pick_sentences(
        paper_build_vowel_table.read_sentences(pools[lang], lang), 1, 0, _Vocab(out), lang)[0] for lang in ("en", "ko")}
    times["vowel_table"] = time.perf_counter() - t0

    print("7/7 YourTTSFrontend 로 길이 재기", flush=True)
    front_report = try_frontend(out, {"picked": picked})
    times["frontend"] = time.perf_counter() - t0

    report = {
        "note": "점검용(스모크). 값에 뜻이 없음. 진짜 모델 아님.",
        "lines": lines,
        "characters": json.loads((man / "phonemize_info.json").read_text(encoding="utf-8"))["characters"],
        "train": {k: run[k] for k in ("run_dir", "final_checkpoint", "steps", "epochs_done", "device", "arch",
                                      "batch_size", "seconds", "train_samples", "eval_samples", "cuda_initialized")},
        "export": exported,
        "vowel_table": vowel_summary,
        "frontend": front_report,
        "seconds_cumulative": {k: round(v, 1) for k, v in times.items()},
        "seconds_total": round(time.perf_counter() - t0, 1),
    }
    (out / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"train_steps": run["steps"], "train_seconds": run["seconds"],
                      "examples": {k: {kk: v[kk] for kk in ("text", "n_tokens", "sum_sec", "deterministic")}
                                   for k, v in front_report["examples"].items()},
                      "check_u": vdoc["check_u"], "seconds_total": report["seconds_total"]},
                     ensure_ascii=False, indent=1))
    print(f"저장: {out}")
    return 0


class _Vocab:
    """characters.txt 로 토큰이 모두 있는지 보는 가벼운 도우미(모델을 다시 올리지 않으려고)."""

    def __init__(self, out: Path) -> None:
        from phonemize_manifest import read_characters

        self.chars = set(read_characters(out / "characters.txt"))

    def tokens(self, text: str, lang: str) -> list[str]:
        from pstts.paper import g2p

        return g2p.tokens(text, lang)

    def unknown_tokens(self, toks: list[str]) -> list[str]:
        from pstts.paper import g2p

        return [t for t in dict.fromkeys(toks) if t != g2p.BLANK and t not in self.chars]


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
