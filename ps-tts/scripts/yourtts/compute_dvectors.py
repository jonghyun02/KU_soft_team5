"""매니페스트 발화마다 d-vector 를 만들어 Coqui d_vector_file 꼴로 저장한다(명세 3.12).

화자 인코더는 공개 YourTTS 의 것(model_se.pth, config_se.json: ResNet, 16 kHz 입력, 512차원)이다. 계산은 Coqui
TTS/bin/compute_embeddings.py 와 같다. SpeakerManager(encoder_model_path, encoder_config_path) 의
compute_embedding_from_clip(파일) 이 AudioProcessor.load_wav 로 16 kHz 로 읽고(librosa, rms 정규화는 config_se 대로)
화자 인코더 compute_embedding(10조각 평균, L2 정규화)을 낸다.
저장 꼴(Coqui d_vector_file): {키: {"name": 화자 이름, "embedding": [512 실수]}}. .pth 는 torch.save, .json 은 JSON
(TTS.tts.utils.managers.save_file). 키 = make_manifest.sample_key(행) = '언어#소리 경로' 이고 학습(train_yourtts.py)의
audio_unique_name 과 같다. 소리 경로가 상대 경로면 학습과 같게 매니페스트 폴더 기준으로 연다(make_manifest.audio_path).
키는 매니페스트에 적힌 글 그대로라 학습과 맞는다.

장치: 기본 cuda. cuda 를 못 쓰면 멈춘다(pstts.scorers.resolve_device). CPU 는 --device cpu 로 명시한다.
실행 예: .venv\\Scripts\\python.exe scripts\\yourtts\\compute_dvectors.py --manifest-dir D:\\data\\pstts_tts
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_manifest import SPLITS, audio_path, manifest_path, read_manifest, sample_key  # noqa: E402

PUBLIC_YOURTTS = ROOT / "models" / "your_tts" / "tts_models--multilingual--multi-dataset--your_tts"
D_VECTOR_NAME = "d_vectors.pth"


def load_encoder(encoder_dir: Path, device: str):
    """공개 YourTTS 화자 인코더를 올린 Coqui SpeakerManager. device 는 resolve_device 를 거친 값."""
    import torch
    from TTS.tts.utils.speakers import SpeakerManager

    model_path, config_path = encoder_dir / "model_se.pth", encoder_dir / "config_se.json"
    for p in (model_path, config_path):
        if not p.is_file():
            raise FileNotFoundError(f"화자 인코더 파일이 없습니다: {p}")
    use_cuda = device.startswith("cuda")
    if use_cuda and ":" in device:
        torch.cuda.set_device(int(device.split(":", 1)[1]))
    return SpeakerManager(encoder_model_path=str(model_path), encoder_config_path=str(config_path), use_cuda=use_cuda)


def compute(rows: list[dict[str, str]], manager, out: Path, *, base: Path, resume: bool = True,
            save_every: int = 2000) -> dict:
    """발화마다 d-vector 를 만들어 out 에 저장한다. resume 이면 out 에 이미 있는 키는 다시 만들지 않는다.

    base 는 매니페스트 폴더다. 상대 소리 경로는 학습(train_yourtts.load_samples)처럼 그 폴더 기준으로 연다.
    """
    from TTS.tts.utils.managers import load_file, save_file

    mapping: dict[str, dict] = {}
    if resume and out.is_file():
        mapping = dict(load_file(str(out)))
    todo = [r for r in rows if sample_key(r) not in mapping]
    start = time.perf_counter()

    def flush() -> None:
        tmp = out.with_name(f"{out.stem}.{os.getpid()}.tmp{out.suffix}")
        save_file(mapping, str(tmp))
        os.replace(tmp, out)

    for i, row in enumerate(todo, 1):
        emb = manager.compute_embedding_from_clip(str(audio_path(row, base)))
        mapping[sample_key(row)] = {"name": row["speaker"], "embedding": [float(v) for v in emb]}
        if save_every and i % save_every == 0:
            flush()
            print(f"  d-vector {i}/{len(todo)} ({time.perf_counter() - start:.0f}초)", file=sys.stderr, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    flush()
    dims = {len(v["embedding"]) for v in mapping.values()}
    return {"path": str(out.resolve()), "vectors": len(mapping), "computed": len(todo),
            "reused": len(rows) - len(todo), "dims": sorted(dims), "speakers": len({v["name"] for v in mapping.values()}),
            "seconds": round(time.perf_counter() - start, 2)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="공개 YourTTS 화자 인코더로 발화별 d-vector 를 만든다.")
    ap.add_argument("--manifest-dir", type=Path, required=True, help="manifest_{train,valid,test}.csv 폴더")
    ap.add_argument("--encoder-dir", type=Path, default=PUBLIC_YOURTTS, help="model_se.pth, config_se.json 폴더")
    ap.add_argument("--out", type=Path, help=f"저장 파일(.pth 또는 .json, 기본 <manifest-dir>/{D_VECTOR_NAME})")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:N, cpu (기본 cuda)")
    ap.add_argument("--cpu-threads", type=int, default=None, help="CPU 스레드 수(기본 기존 Config 값)")
    ap.add_argument("--no-resume", action="store_true", help="이미 있는 벡터도 다시 만든다")
    ap.add_argument("--save-every", type=int, default=2000, help="이만큼마다 중간 저장(기본 2000)")
    args = ap.parse_args(argv)

    from pstts.scorers import resolve_device

    device = resolve_device(args.device, args.cpu_threads)
    rows: list[dict[str, str]] = []
    for split in SPLITS:
        rows.extend(read_manifest(manifest_path(args.manifest_dir, split)))
    manager = load_encoder(args.encoder_dir, device)
    out = args.out or (args.manifest_dir / D_VECTOR_NAME)
    s = compute(rows, manager, out, base=args.manifest_dir.resolve(), resume=not args.no_resume,
                save_every=args.save_every)
    print(f"d-vector {s['vectors']}개(새로 {s['computed']}, 다시 씀 {s['reused']}), 화자 {s['speakers']}명, "
          f"차원 {s['dims']}, {s['seconds']}초 -> {s['path']}")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
