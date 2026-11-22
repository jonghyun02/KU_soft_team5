"""식 (8), (9) 모음표를 학습한 Your-TTS 체크포인트로 만든다(명세 1.4, 3.5).

문장 파일은 한 줄에 한 문장(utf-8)이다. 매니페스트 CSV(make_manifest.py 꼴)를 주면 그 언어의 text 칸을 쓴다.
실제 판은 학습 매니페스트의 대사(KMSSS 한국어, LibriTTS 영어)에서 뽑는다(논문: TTS 학습 자료 한국어 6,000, 영어 5,000).
문장 고르기: 시드로 섞은 순서에서 모델 글자 목록에 토큰이 다 있는 문장을 n 개까지 고른다(빈 줄과 모르는 토큰이 든 문장은
건너뛰고 센다). 출력: --out(기본 <tts-dir>/vowel_table.json)과 같은 이름 .csv(한국어 행 x 영어 열 거리).
음소화는 체크포인트 폴더의 g2p.json(학습 때 G2P: 영어 목소리, 강세표)을 따른다. 화자 평균 파일(d_vector_means.pth)이 있으면 그것을 쓴다.

실행 예:
  .venv\\Scripts\\python.exe scripts\\paper_build_vowel_table.py --tts-dir models\\yourtts_run ^
      --ko-sentences D:\\data\\pstts_tts\\manifest_train.csv --en-sentences D:\\data\\pstts_tts\\manifest_train.csv
장치는 기본 cuda 이고 못 쓰면 멈춘다. CPU 는 --device cpu.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pstts.paper.config import PaperConfig  # noqa: E402
from pstts.paper.vowel_embed import KO_U, build_vowel_table, save_vowel_table  # noqa: E402


def read_sentences(path: Path, lang: str) -> list[str]:
    """한 줄 한 문장 파일, 또는 매니페스트 CSV(lang 이 맞는 줄의 text)."""
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not {"text", "lang"} <= set(reader.fieldnames or []):
                raise ValueError(f"{path} 에 text, lang 칸이 없습니다.")
            return [row["text"] for row in reader if row["lang"].strip().lower() == lang]
    return path.read_text(encoding="utf-8").splitlines()


def pick_sentences(lines: list[str], n: int, seed: int, front, lang: str) -> tuple[list[str], dict]:
    """시드로 섞은 순서에서 토큰이 모두 모델 글자 목록에 있는 문장을 n 개까지 고른다."""
    order = list(range(len(lines)))
    random.Random(seed).shuffle(order)
    picked: list[str] = []
    empty = oov = 0
    for i in order:
        if len(picked) >= n:
            break
        text = " ".join(str(lines[i]).split())
        if not text:
            empty += 1
            continue
        if front.unknown_tokens(front.tokens(text, lang)):
            oov += 1
            continue
        picked.append(text)
    return picked, {"pool": len(lines), "wanted": n, "picked": len(picked), "skipped_empty": empty,
                    "skipped_unknown_tokens": oov}


def render(doc: dict) -> str:
    """한국어(행) x 영어(열) 거리표 글."""
    rows = ["ko\\en " + " ".join(f"{e:>6}" for e in doc["en"])]
    for k, row in zip(doc["ko"], doc["matrix"]):
        rows.append(f"{k:>5} " + " ".join(f"{float(v):6.2f}" for v in row))
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="학습한 Your-TTS 로 식 (8), (9) 모음표를 만든다.")
    ap.add_argument("--tts-dir", type=Path, required=True, help="체크포인트 폴더(config.json + .pth)")
    ap.add_argument("--tts-checkpoint", default=None, help="체크포인트 파일 이름(기본 best_model*.pth 또는 model_file.pth)")
    ap.add_argument("--tts-speaker", default=None, help="d-vector 키나 화자 이름(기본 이름순 첫 화자)")
    ap.add_argument("--ko-sentences", type=Path, required=True, help="한국어 문장 파일(.txt 한 줄 한 문장, 또는 매니페스트 .csv)")
    ap.add_argument("--en-sentences", type=Path, required=True, help="영어 문장 파일(.txt 또는 매니페스트 .csv)")
    ap.add_argument("--n-ko", type=int, default=6000, help="한국어 문장 수(논문 6,000)")
    ap.add_argument("--n-en", type=int, default=5000, help="영어 문장 수(논문 5,000)")
    ap.add_argument("--k", type=int, default=5, help="모음마다 KMeans 중심 수(식 8 의 N = 5)")
    ap.add_argument("--seed", type=int, default=0, help="문장 고르기와 KMeans random_state(기본 0)")
    ap.add_argument("--out", type=Path, default=None, help="출력 JSON(기본 <tts-dir>/vowel_table.json)")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:N, cpu (기본 cuda. 못 쓰면 멈춤)")
    ap.add_argument("--cpu-threads", type=int, default=None, help="CPU 스레드 수")
    args = ap.parse_args(argv)

    from pstts.paper import g2p
    from pstts.paper.yourtts import YourTTSFrontend

    base = PaperConfig()
    # 모델을 학습할 때 쓴 G2P(체크포인트 폴더의 g2p.json)로 음소화한다. 없으면 YourTTSFrontend 가 알리고 멈춘다.
    settings_path = args.tts_dir / g2p.SETTINGS_NAME
    trained = g2p.read_settings(settings_path) if settings_path.is_file() else {}
    en = trained.get("en") if isinstance(trained.get("en"), dict) else {}
    cfg = PaperConfig(tts="yourtts", tts_dir=str(args.tts_dir), tts_checkpoint=args.tts_checkpoint,
                      tts_speaker=args.tts_speaker,
                      g2p_en_voice=str(en.get("voice", base.g2p_en_voice)),
                      g2p_en_stress=bool(en.get("with_stress", base.g2p_en_stress)),
                      cpu_threads=base.cpu_threads if args.cpu_threads is None else args.cpu_threads)
    front = YourTTSFrontend(cfg, args.device, load_table=False)       # 표는 지금 만든다
    try:
        ko, ko_pick = pick_sentences(read_sentences(args.ko_sentences, "ko"), args.n_ko, args.seed, front, "ko")
        en, en_pick = pick_sentences(read_sentences(args.en_sentences, "en"), args.n_en, args.seed, front, "en")
        print(f"문장: 한국어 {len(ko)}/{args.n_ko}, 영어 {len(en)}/{args.n_en} "
              f"(모르는 토큰으로 건너뜀 한국어 {ko_pick['skipped_unknown_tokens']}, 영어 {en_pick['skipped_unknown_tokens']})",
              flush=True)
        doc = build_vowel_table(front, ko, en, k=args.k, seed=args.seed)
        info = front.info()
        doc["selection"] = {"ko": ko_pick | {"file": str(args.ko_sentences)},
                            "en": en_pick | {"file": str(args.en_sentences)}}
        doc["frontend"] = {k: info[k] for k in ("speaker", "languages", "noise_scale_dp", "length_scale", "frame_sec", "g2p",
                                                 "versions")}
    finally:
        front.close()
    out = args.out or (args.tts_dir / "vowel_table.json")
    jp, cp = save_vowel_table(doc, out)
    print(f"모음: 한국어 {len(doc['ko'])}개 {''.join(doc['ko'])}, 영어 {len(doc['en'])}개 {' '.join(doc['en'])}")
    print(render(doc))
    print("한국어 모음마다 가장 가까운 영어 모음(거리):")
    for k, (e, dist) in doc["nearest"].items():
        print(f"  {k} -> {e} {dist:.3f}")
    cu = doc["check_u"]
    if cu["is_u"] is None:
        print(f"ㅜ 확인: {cu['note']}")
    else:
        same = "같음" if cu["is_u"] else "다름"
        print(f"ㅜ 확인: {KO_U} 와 가장 가까운 영어 모음 {cu['nearest_en']} (거리 {cu['distance']:.3f}, u 까지 "
              f"{cu['distance_to_u']:.3f}) -> 논문 Fig. 5c(ㅜ-u)와 {same}")
    print(f"저장: {jp}")
    print(f"저장: {cp}")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):     # IPA 기호·한글이 cp949 로 깨지지 않게
        stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
