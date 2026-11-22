"""Your-TTS 학습 매니페스트 만들기(명세 3.12, 논문 4.1절 'Datasets').

- LibriTTS(--libritts-dir): 표준 구조 <root>/<subset>/<화자>/<장>/*.wav 와 같은 이름의 .normalized.txt.
  기본 subset 은 train-clean-360(논문의 'Libri-TTS-360', 904명). 언어 en, 화자 이름 libritts_<화자 번호>.
- 그 밖의 자료(--csv, 여러 번 줄 수 있음): 열 audio,text,speaker[,lang]. AI Hub KMSSS(다화자 음성합성)는 받은 뒤 이 꼴로
  만든다. KMSSS 자료 구조를 지금 모르므로 KMSSS 전용 읽기는 짜지 않았다. audio 가 상대 경로면 CSV 폴더 기준이다.
  lang 칸이 없으면 --csv-lang(기본 ko). 화자 이름은 <자료 이름>_<speaker>(자료 이름 = CSV 파일 이름).
- 자료마다 따로 시드를 고정해 섞고 8:1:1 로 나눈다(논문 'split into train, validation, and test sets in an 8:1:1
  ratio'). 무엇 단위로 나눴는지는 논문에 없어 발화 단위로 나눈다(해석). 자르는 곳은 floor(0.8n + 0.5), floor(0.9n + 0.5).
- 출력(--out): manifest_train.csv, manifest_valid.csv, manifest_test.csv(열 audio,text,speaker,lang, utf-8)와
  manifest_info.json(자료별 수). audio 는 절대 경로다.
- 소리는 wav, flac, mp3 이고 soundfile 이 읽을 수 있어야 한다(coqui-tts 0.28 get_audio_size 와 같은 조건). 학습 때 Coqui 는
  그 밖의 소리를 경고 한 줄만 남기고 빼 버리므로, 여기서 하나라도 있으면 파일을 적고 멈춘다.

실행 예:
  .venv\\Scripts\\python.exe scripts\\yourtts\\make_manifest.py --libritts-dir D:\\data\\LibriTTS ^
      --csv D:\\data\\kmsss\\kmsss.csv --out D:\\data\\pstts_tts
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from io import StringIO
from pathlib import Path

MANIFEST_COLUMNS = ("audio", "text", "speaker", "lang")
SPLITS = ("train", "valid", "test")
LANGS = ("en", "ko")
LIBRITTS_SUBSET = "train-clean-360"
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3")      # coqui-tts 0.28 TTS/tts/datasets/dataset.py get_audio_size 가 받는 꼴


# ---------------------------------------------------------------------------- 함께 쓰는 도구
def audio_path(row: dict[str, str], folder: str | os.PathLike) -> Path:
    """행의 소리 경로. 상대 경로면 folder(매니페스트나 CSV 가 있는 폴더) 기준이다. 학습과 d-vector 가 같은 규칙을 쓴다."""
    p = Path(str(row["audio"]).strip())
    return p if p.is_absolute() else Path(folder) / p


def audio_problem(path: Path) -> str | None:
    """학습(coqui-tts 0.28 TTSDataset)이 쓸 수 있는 소리인지. 문제가 있으면 까닭(한국어), 없으면 None."""
    import soundfile as sf

    if path.suffix.lower() not in AUDIO_SUFFIXES:
        return f"확장자 {path.suffix or '(없음)'} 는 학습에서 빠짐(wav, flac, mp3 만 됨)"
    try:
        frames = sf.info(str(path)).frames
    except RuntimeError as e:
        return f"soundfile 이 읽지 못함({e})"
    return None if frames > 0 else "소리 길이가 0"


def _stop_if_bad(bad: list[str], where: Path) -> None:
    if bad:
        shown = "; ".join(bad[:5]) + (f" 외 {len(bad) - 5}개" if len(bad) > 5 else "")
        raise ValueError(f"{where}: 학습에 쓸 수 없는 소리 {len(bad)}개. wav, flac, mp3 로 바꾸세요. {shown}")
def manifest_path(folder: str | os.PathLike, split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"나누기 이름은 {SPLITS} 가운데 하나여야 합니다: {split}")
    return Path(folder) / f"manifest_{split}.csv"


def read_manifest(path: str | os.PathLike) -> list[dict[str, str]]:
    """매니페스트 CSV 를 읽는다(utf-8, BOM 있어도 됨). 필수 열이 없으면 멈춘다."""
    p = Path(path)
    with p.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in MANIFEST_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{p} 에 열 {missing} 이 없습니다(필요: {list(MANIFEST_COLUMNS)}).")
        return [dict(row) for row in reader]


def write_manifest(path: str | os.PathLike, rows: list[dict[str, str]], columns: tuple[str, ...] = MANIFEST_COLUMNS
                   ) -> None:
    """매니페스트 CSV 를 한꺼번에 바꿔 쓴다(utf-8, BOM 없음)."""
    p = Path(path)
    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n", extrasaction="raise")
    w.writeheader()
    for row in rows:
        w.writerow({c: row[c] for c in columns})
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8", newline="")
    os.replace(tmp, p)


def sample_key(row: dict[str, str]) -> str:
    """발화 하나의 키 = '언어#소리 경로'. d-vector 파일의 키이자 학습의 audio_unique_name 이다."""
    return f"{row['lang']}#{Path(row['audio']).as_posix()}"


def split_811(rows: list[dict[str, str]], seed: int) -> tuple[list[dict[str, str]], ...]:
    """시드로 섞고 8:1:1 로 나눈다. 자르는 곳 floor(0.8n + 0.5), floor(0.9n + 0.5)(정수 셈)."""
    order = list(rows)
    random.Random(seed).shuffle(order)
    n = len(order)
    c1 = (8 * n + 5) // 10
    c2 = (9 * n + 5) // 10
    return order[:c1], order[c1:c2], order[c2:]


# ---------------------------------------------------------------------------- 자료 읽기
def libritts_rows(root: Path, subset: str = LIBRITTS_SUBSET) -> tuple[list[dict[str, str]], Counter]:
    """LibriTTS 표준 구조에서 (소리, 정규화 글, 화자) 를 읽는다. 글이 없거나 빈 발화는 건너뛰고 센다."""
    base = root / subset
    if not base.is_dir():
        raise FileNotFoundError(f"LibriTTS 폴더가 없습니다: {base} (<root>/{subset}/<화자>/<장>/*.wav)")
    rows: list[dict[str, str]] = []
    skipped: Counter = Counter()
    bad: list[str] = []
    for wav in sorted(base.glob("*/*/*.wav")):
        txt = wav.with_suffix(".normalized.txt")
        if not txt.is_file():
            skipped["no_text"] += 1
            continue
        text = " ".join(txt.read_text(encoding="utf-8").split())
        if not text:
            skipped["empty_text"] += 1
            continue
        problem = audio_problem(wav)
        if problem:
            bad.append(f"{wav}: {problem}")
            continue
        speaker = wav.parent.parent.name
        rows.append({"audio": str(wav.resolve()), "text": text, "speaker": f"libritts_{speaker}", "lang": "en"})
    _stop_if_bad(bad, base)
    return rows, skipped


def csv_rows(path: Path, default_lang: str, name: str) -> tuple[list[dict[str, str]], Counter]:
    """일반 CSV(audio,text,speaker[,lang]). 소리 파일이 없거나 글이 빈 줄은 건너뛰고 센다."""
    rows: list[dict[str, str]] = []
    skipped: Counter = Counter()
    bad: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        need = [c for c in ("audio", "text", "speaker") if c not in (reader.fieldnames or [])]
        if need:
            raise ValueError(f"{path} 에 열 {need} 이 없습니다(필요: audio,text,speaker[,lang]).")
        for row in reader:
            lang = (row.get("lang") or default_lang).strip().lower()
            if lang not in LANGS:
                raise ValueError(f"{path}: 언어는 en 또는 ko 여야 합니다: {lang}")
            audio = audio_path(row, path.parent)
            if not audio.is_file():
                skipped["no_audio"] += 1
                continue
            text = " ".join(str(row["text"]).split())
            if not text:
                skipped["empty_text"] += 1
                continue
            problem = audio_problem(audio)
            if problem:
                bad.append(f"{audio}: {problem}")
                continue
            rows.append({"audio": str(audio.resolve()), "text": text, "speaker": f"{name}_{row['speaker'].strip()}",
                         "lang": lang})
    _stop_if_bad(bad, path)
    return rows, skipped


# ---------------------------------------------------------------------------- 실행
def build(datasets: list[tuple[str, list[dict[str, str]], Counter]], out: Path, seed: int) -> dict:
    """자료마다 8:1:1 로 나눠 세 매니페스트와 manifest_info.json 을 쓴다."""
    parts: dict[str, list[dict[str, str]]] = {s: [] for s in SPLITS}
    info: dict = {"seed": seed, "ratio": "8:1:1", "datasets": {}, "files": {}}
    keys: set[str] = set()
    for name, rows, skipped in datasets:
        if not rows:
            raise ValueError(f"자료 {name} 에 쓸 발화가 없습니다.")
        for r in rows:
            k = sample_key(r)
            if k in keys:
                raise ValueError(f"같은 소리 파일이 두 번 나옵니다: {r['audio']}")
            keys.add(k)
        train, valid, test = split_811(rows, seed)
        for s, chunk in zip(SPLITS, (train, valid, test)):
            parts[s].extend(chunk)
        info["datasets"][name] = {
            "rows": len(rows), "train": len(train), "valid": len(valid), "test": len(test),
            "speakers": len({r["speaker"] for r in rows}), "langs": dict(Counter(r["lang"] for r in rows)),
            "skipped": dict(skipped),
        }
    for s in SPLITS:
        p = manifest_path(out, s)
        write_manifest(p, parts[s])
        info["files"][s] = {"path": str(p.resolve()), "rows": len(parts[s])}
    (out / "manifest_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Your-TTS 학습 매니페스트(8:1:1)를 만든다.")
    ap.add_argument("--libritts-dir", type=Path, help="LibriTTS 뿌리 폴더(<root>/train-clean-360/...)")
    ap.add_argument("--libritts-subset", default=LIBRITTS_SUBSET, help=f"LibriTTS 하위 폴더(기본 {LIBRITTS_SUBSET})")
    ap.add_argument("--csv", type=Path, action="append", default=[], help="audio,text,speaker[,lang] CSV(여러 번 가능)")
    ap.add_argument("--csv-lang", default="ko", choices=LANGS, help="CSV 에 lang 칸이 없을 때 언어(기본 ko)")
    ap.add_argument("--out", type=Path, required=True, help="매니페스트를 쓸 폴더")
    ap.add_argument("--seed", type=int, default=0, help="섞기 시드(기본 0)")
    args = ap.parse_args(argv)

    datasets: list[tuple[str, list[dict[str, str]], Counter]] = []
    if args.libritts_dir is not None:
        rows, skipped = libritts_rows(args.libritts_dir, args.libritts_subset)
        datasets.append(("libritts", rows, skipped))
    for path in args.csv:
        name = path.stem
        if any(name == d[0] for d in datasets):
            raise SystemExit(f"자료 이름이 겹칩니다: {name} (CSV 파일 이름을 바꾸세요)")
        rows, skipped = csv_rows(path, args.csv_lang, name)
        datasets.append((name, rows, skipped))
    if not datasets:
        ap.error("--libritts-dir 나 --csv 가운데 하나는 있어야 합니다.")
    args.out.mkdir(parents=True, exist_ok=True)
    info = build(datasets, args.out, args.seed)
    for name, d in info["datasets"].items():
        print(f"{name}: 발화 {d['rows']}개(화자 {d['speakers']}명) -> train {d['train']}, valid {d['valid']}, "
              f"test {d['test']}, 건너뜀 {d['skipped']}")
    for s, f in info["files"].items():
        print(f"저장: {f['path']} ({f['rows']}줄)")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
