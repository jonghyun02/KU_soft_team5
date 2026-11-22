from __future__ import annotations

import argparse
from pathlib import Path

from env_setup import (
    DEFAULT_CUDA_INDEX_URL,
    PROJECT_DIR,
    ensure_venv_and_reexec,
    get_default_checkpoint_path,
    get_default_venv_dir,
    setup_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDarch 가상환경 구축 및 CUDA 12.8 스트리밍 추론 스크립트")
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=get_default_venv_dir(),
        help="가상환경 경로 (기본값: OneDrive 밖 %%LOCALAPPDATA%%\\EDarch_venv)",
    )
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help="가상환경 자동 생성을 건너뛰고 현재 활성 파이썬 환경에서 바로 실행",
    )
    parser.add_argument(
        "--torch-index-url",
        type=str,
        default=DEFAULT_CUDA_INDEX_URL,
        help=f"PyTorch CUDA 인덱스 URL (기본값: {DEFAULT_CUDA_INDEX_URL})",
    )
    parser.add_argument("--cpu", action="store_true", help="CUDA 대신 CPU 전용으로 설치 및 실행")
    parser.add_argument("--reinstall", action="store_true", help="필수 패키지 강제 재설치")

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=get_default_checkpoint_path(),
        help="로드할 체크포인트 경로 (파일이 없으면 --preset으로 새 모델 초기화)",
    )
    parser.add_argument(
        "--preset",
        choices=("1b", "2b"),
        default="1b",
        help="체크포인트가 없을 때 사용할 모델 프리셋 (기본값: 1b)",
    )
    parser.add_argument(
        "--mimi-model",
        type=str,
        default=None,
        help="외부 사전학습 Mimi 코덱 ID 또는 로컬 경로 (예: kyutai/mimi)",
    )
    parser.add_argument(
        "--sp-model",
        type=Path,
        default=None,
        help="Inner Monologue 한국어 토큰 디코딩용 SentencePiece .model 파일 경로",
    )
    parser.add_argument(
        "--input-wav",
        type=Path,
        default=None,
        help="입력 소스 음성 WAV 경로 (미지정 시 1.6초 합성 파형으로 테스트)",
    )
    parser.add_argument(
        "--output-wav",
        type=Path,
        default=PROJECT_DIR / "outputs" / "EDarch_out.wav",
        help="복원된 타깃 음성 WAV 저장 경로",
    )
    parser.add_argument("--demo-frames", type=int, default=20, help="합성 입력 시 프레임 수 (20=1.6초)")
    parser.add_argument(
        "--max-tail-sec",
        type=float,
        default=10.0,
        help="원문 종료 후 마지막 번역이 끝날 때까지 무음을 흘려보낼 최대 꼬리 길이(초)",
    )
    parser.add_argument("--cfg-coef", type=float, default=3.0, help="Classifier-Free Guidance 계수")
    parser.add_argument("--temp-audio", type=float, default=0.8, help="오디오 샘플링 온도")
    parser.add_argument("--temp-text", type=float, default=0.4, help="텍스트 샘플링 온도")
    parser.add_argument("--top-k-audio", type=int, default=250, help="오디오 Top-k")
    parser.add_argument("--top-k-text", type=int, default=50, help="텍스트 Top-k")
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="CUDA 혼합 정밀도(BF16/FP16) 추론 비활성화 (기본값: CUDA에서 자동 활성)",
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    return parser.parse_args()


def run_inference(args: argparse.Namespace) -> None:
    import torch
    import torchaudio

    from EDarch_from_scratch import (
        EDarchConfig,
        EDarchModel,
        EDarchStreamingGenerator,
        ExternalMimiCodec,
        MiniMimiCodec,
    )

    device = setup_device(cpu_only=args.cpu, seed=args.seed, tag="Infer")

    codec_type = "mini_demo"
    if args.checkpoint is not None and args.checkpoint.exists():
        print(f"[Infer] 체크포인트 로드 중: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        cfg = EDarchConfig.from_dict(ckpt["config"])
        cfg.train_on_source_audio = False
        codec_type = ckpt.get("codec_type", "mini_demo")
        model = EDarchModel(cfg).to(device)
        model.load_state_dict_for_inference(ckpt["model_state_dict"])
        if args.mimi_model is not None:
            codec = ExternalMimiCodec(cfg, model_id=args.mimi_model).to(device)
        elif "codec_state_dict" in ckpt:
            codec = MiniMimiCodec(cfg).to(device)
            codec.load_state_dict(ckpt["codec_state_dict"])
        elif codec_type == "external_mimi":
            try:
                codec = ExternalMimiCodec(cfg, model_id="kyutai/mimi").to(device)
                print("[Infer] 사전학습 Mimi 코덱 로드 완료: kyutai/mimi")
            except RuntimeError as exc:
                print(
                    f"[Infer] WARNING: 외부 사전학습 Mimi 코덱 자동 로드 실패 ({exc}). "
                    "데모용 미학습 MiniMimiCodec으로 폴백하므로 정상 음성을 얻으려면 "
                    "transformers 또는 moshi를 설치하거나 --mimi-model을 지정하세요."
                )
                codec = MiniMimiCodec(cfg).to(device)
        else:
            codec = MiniMimiCodec(cfg).to(device)
    else:
        print(f"[Infer] 체크포인트 없음 -> '{args.preset}' 프리셋으로 새 모델 초기화")
        cfg = EDarchConfig.from_preset(args.preset)
        cfg.train_on_source_audio = False
        if args.mimi_model is not None:
            codec = ExternalMimiCodec(cfg, model_id=args.mimi_model).to(device)
        else:
            codec = MiniMimiCodec(cfg).to(device)
        model = EDarchModel(cfg).to(device)

    model.eval()
    codec.eval()

    sp_proc = None
    if args.sp_model is not None:
        if not args.sp_model.exists():
            raise FileNotFoundError(f"SentencePiece 모델 파일을 찾을 수 없습니다: {args.sp_model}")
        import sentencepiece as spm

        sp_proc = spm.SentencePieceProcessor()
        sp_proc.Load(str(args.sp_model))
        if sp_proc.eos_id() != cfg.text_eos_id or sp_proc.pad_id() != cfg.text_pad_id:
            print(
                f"[Infer] WARNING: SentencePiece 특수 토큰(eos={sp_proc.eos_id()}, pad={sp_proc.pad_id()})이 "
                f"모델 설정(eos={cfg.text_eos_id}, pad={cfg.text_pad_id})과 다릅니다."
            )
        if sp_proc.GetPieceSize() != cfg.text_card:
            print(
                f"[Infer] WARNING: SentencePiece 어휘 크기({sp_proc.GetPieceSize()})가 "
                f"모델 text_card({cfg.text_card})와 다릅니다."
            )

    if args.input_wav is not None:
        if not args.input_wav.exists():
            raise FileNotFoundError(f"입력 WAV 파일을 찾을 수 없습니다: {args.input_wav}")
        wav, sr = torchaudio.load(str(args.input_wav))
        wav = wav.mean(dim=0, keepdim=True).float()
        if sr != cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, cfg.sample_rate)
        src_wav = wav.unsqueeze(0).to(device)
        print(f"[Infer] 입력 WAV 로드 완료: {args.input_wav} ({src_wav.shape[-1] / cfg.sample_rate:.2f}초)")
    else:
        num_samples = args.demo_frames * codec.hop_length
        src_wav = torch.randn(1, 1, num_samples, device=device) * 0.1
        print(f"[Infer] 합성 데모 입력 생성: {args.demo_frames}프레임 ({num_samples / cfg.sample_rate:.2f}초)")

    if src_wav.shape[-1] == 0:
        raise ValueError("입력 오디오의 길이가 0 샘플입니다.")
    with torch.no_grad():
        src_codes = codec.encode(src_wav)
        silence_frames = torch.zeros(
            1, 1, 8 * codec.hop_length, device=device, dtype=src_wav.dtype
        )
        silence_code = codec.encode(silence_frames)[:, :, -1:]
    num_frames = src_codes.shape[-1]
    if num_frames == 0:
        raise ValueError("인코딩된 오디오 프레임 수가 0입니다.")

    streamer = EDarchStreamingGenerator(
        model,
        batch_size=1,
        cfg_coef=args.cfg_coef,
        temp_audio=args.temp_audio,
        temp_text=args.temp_text,
        top_k_audio=args.top_k_audio,
        top_k_text=args.top_k_text,
    )

    emitted_texts: list[int] = []
    emitted_audio_frames: list[torch.Tensor] = []

    use_amp = (not args.no_amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    max_tail_frames = max(1, int(args.max_tail_sec * cfg.frame_rate)) + streamer.max_delay
    eos_reached = False

    for t in range(num_frames):
        chunk = src_codes[:, :, t : t + 1]
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            step_out = streamer.step(chunk, suppress_eos=True)
        if step_out is not None:
            t_tok, a_tok = step_out
            emitted_texts.append(int(t_tok[0, 0].item()))
            emitted_audio_frames.append(a_tok)

    for _ in range(max_tail_frames):
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            step_out = streamer.step(silence_code, suppress_eos=False)
        if step_out is not None:
            t_tok, a_tok = step_out
            tok_id = int(t_tok[0, 0].item())
            emitted_texts.append(tok_id)
            emitted_audio_frames.append(a_tok)
            if tok_id == cfg.text_eos_id:
                eos_reached = True
                break

    pred_audio_codes = torch.cat(emitted_audio_frames, dim=-1)
    with torch.no_grad():
        out_wav = codec.decode(pred_audio_codes).squeeze(0).float().cpu()

    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output_wav), out_wav, sample_rate=cfg.sample_rate)

    non_pad_tokens = [tid for tid in emitted_texts if tid != cfg.text_pad_id]
    stop_reason = "EOS 감지" if eos_reached else f"최대 꼬리 길이({args.max_tail_sec}s) 도달"
    print(f"[Done] 스트리밍 추론 완료 ({len(emitted_texts)}프레임, CFG={args.cfg_coef}, 종료={stop_reason})")
    print(f"  - 전체 Inner Monologue 토큰 (앞 30개): {emitted_texts[:30]}")
    print(f"  - 비-PAD 토큰 수: {len(non_pad_tokens)} / {len(emitted_texts)}")
    if sp_proc is not None:
        piece_size = sp_proc.GetPieceSize()
        decode_ids = [
            tid
            for tid in emitted_texts
            if tid not in (cfg.text_pad_id, cfg.text_eos_id) and 0 <= tid < piece_size
        ]
        decoded_text = sp_proc.DecodeIds(decode_ids)
        print(f"  - 디코딩된 한국어 번역: {decoded_text}")
    print(f"  - 출력 WAV 저장 완료: {args.output_wav} (길이: {out_wav.shape[-1] / cfg.sample_rate:.2f}초)")


def main() -> None:
    args = parse_args()
    ensure_venv_and_reexec(
        script_path=Path(__file__),
        venv_dir=args.venv_dir,
        torch_index_url=args.torch_index_url,
        cpu_only=args.cpu,
        reinstall=args.reinstall,
        no_venv=args.no_venv,
    )
    run_inference(args)


if __name__ == "__main__":
    main()
