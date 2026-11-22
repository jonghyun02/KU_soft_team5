"""실시간 스트리밍 동시통역 생성기, 추론 실행 루프 및 코어 아키텍처 검증 모듈."""

from __future__ import annotations

import argparse
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio

from libs.codec import (
    MiniMimiCodec,
    load_mono_wav_24k,
    resolve_inference_codec,
)
from libs.config import EDarchConfig
from libs.env import setup_device
from libs.model import EDarchModel
from libs.optim import resolve_amp_context
from libs.tokenizer import load_korean_tokenizer


def sample_top_k(
    logits: torch.Tensor, temperature: float = 0.8, top_k: int = 250
) -> torch.Tensor:
    """온도 스케일링과 Top-k 필터링을 적용하여 다음 토큰 ID를 다항 분포에서 샘플링합니다."""
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    scaled = logits.float() / temperature
    if 0 < top_k < scaled.shape[-1]:
        top_vals, _ = torch.topk(scaled, k=top_k, dim=-1)
        min_keep = top_vals[..., -1:].expand_as(scaled)
        scaled = torch.where(
            scaled < min_keep, torch.full_like(scaled, float("-inf")), scaled
        )
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1).view(
        probs.shape[:-1]
    )


class EDarchStreamingGenerator:
    """순환 링 버퍼와 KV 캐시를 유지하며 80ms 프레임 단위로 실시간 동시통역을 수행하는 생성기입니다."""

    def __init__(
        self,
        model: EDarchModel,
        batch_size: int = 1,
        cfg_coef: float = 3.0,
        temp_audio: float = 0.8,
        temp_text: float = 0.4,
        top_k_audio: int = 250,
        top_k_text: int = 50,
        sim_bin: Optional[int] = None,
    ):
        """모델, 배치 크기, CFG 계수, 화자 유사도 구간, 샘플링 온도 및 순환 지연 버퍼를 초기화합니다."""
        self.model = model
        self.cfg = model.cfg
        self.batch_size = batch_size
        self.cfg_coef = cfg_coef
        self.temp_audio = temp_audio
        self.temp_text = temp_text
        self.top_k_audio = top_k_audio
        self.top_k_text = top_k_text

        device = next(model.parameters()).device
        self.device = device
        self.delays_list: list[int] = list(self.cfg.delays)
        self.delays_tensor = torch.tensor(
            self.delays_list, dtype=torch.long, device=device
        )
        self.max_delay = max(self.delays_list)
        self.ring_len = self.max_delay * 2 + 3

        self.ring_cache = torch.full(
            (batch_size, self.cfg.total_streams, self.ring_len),
            self.cfg.ungenerated_token_id,
            dtype=torch.long,
            device=device,
        )
        self.initial_tokens = model.get_initial_tokens(batch_size, device)
        self.step_offset = 0

        use_cfg = self.cfg_coef != 1.0
        self.temporal_kv = model.temporal_transformer.init_kv_caches()

        resolved_sim_bin = (
            self.cfg.num_speaker_sim_bins - 1
            if sim_bin is None
            else max(0, min(self.cfg.num_speaker_sim_bins - 1, int(sim_bin)))
        )
        self.sim_bin = resolved_sim_bin
        cond_bins = torch.full(
            (batch_size,), resolved_sim_bin, dtype=torch.long, device=device
        )
        if use_cfg:
            null_bins = torch.full(
                (batch_size,),
                model.speaker_cond.null_idx,
                dtype=torch.long,
                device=device,
            )
            all_bins = torch.cat([cond_bins, null_bins], dim=0)
        else:
            all_bins = cond_bins
        with torch.no_grad():
            self.cond_emb = model.speaker_cond(all_bins).detach()

    def reset(self) -> None:
        """순환 버퍼, 스텝 카운터 및 Temporal KV 캐시를 초기 상태로 리셋합니다."""
        self.ring_cache.fill_(self.cfg.ungenerated_token_id)
        self.step_offset = 0
        self.temporal_kv = self.model.temporal_transformer.init_kv_caches()

    @torch.no_grad()
    def step(
        self,
        source_audio_tokens: torch.Tensor,
        suppress_eos: Optional[bool] = None,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """소스 오디오 1프레임(`[B, Q, 1]`)을 입력받아 지연 정렬된 타깃 텍스트·오디오 프레임을 생성합니다."""
        bsz, q_in, s_len = source_audio_tokens.shape
        if bsz != self.batch_size or q_in != self.cfg.dep_q or s_len != 1:
            raise ValueError(
                f"예상 입력 크기 ({self.batch_size}, {self.cfg.dep_q}, 1), 실제 {tuple(source_audio_tokens.shape)}"
            )

        was_training = self.model.training
        if was_training:
            self.model.eval()

        try:
            src_start = 1 + self.cfg.dep_q
            src_delays = self.delays_list[src_start:]
            for q_idx, d in enumerate(src_delays):
                w_pos = (self.step_offset + 1 + d) % self.ring_len
                self.ring_cache[:, src_start + q_idx, w_pos] = (
                    source_audio_tokens[:, q_idx, 0]
                )

            read_pos = self.step_offset % self.ring_len
            cur_in = self.ring_cache[:, :, read_pos : read_pos + 1].clone()
            is_init = self.step_offset <= self.delays_tensor.view(1, -1, 1)
            cur_in = torch.where(is_init, self.initial_tokens, cur_in)

            use_cfg = self.cfg_coef != 1.0
            temporal_in = cur_in.repeat(2, 1, 1) if use_cfg else cur_in
            z_t, text_logits = self.model.forward_temporal(
                temporal_in, cond_emb=self.cond_emb, kv_caches=self.temporal_kv
            )
            t_logits_step = text_logits[:, 0, 0, :].float()
            if use_cfg:
                t_cond, t_uncond = t_logits_step.chunk(2, dim=0)
                t_logits_step = t_uncond + self.cfg_coef * (t_cond - t_uncond)

            do_suppress_eos = (
                self.model._suppress_early_eos if suppress_eos is None else suppress_eos
            )
            if do_suppress_eos:
                t_logits_step = t_logits_step.clone()
                t_logits_step[:, self.cfg.text_eos_id] = float("-inf")

            sampled_text = sample_top_k(
                t_logits_step, temperature=self.temp_text, top_k=self.top_k_text
            )

            dep_kv = self.model.depformer.init_kv_caches()
            prev_tok = sampled_text
            sampled_audio_list: list[torch.Tensor] = []
            for cb in range(self.cfg.dep_q):
                dep_tok_in = prev_tok.repeat(2) if use_cfg else prev_tok
                a_logits = self.model.forward_depformer_single_step(
                    cb_index=cb,
                    prev_token=dep_tok_in,
                    z_t_step=z_t,
                    dep_kv_caches=dep_kv,
                ).float()
                if use_cfg:
                    a_cond, a_uncond = a_logits.chunk(2, dim=0)
                    a_logits = a_uncond + self.cfg_coef * (a_cond - a_uncond)

                next_audio_tok = sample_top_k(
                    a_logits, temperature=self.temp_audio, top_k=self.top_k_audio
                )
                sampled_audio_list.append(next_audio_tok)
                prev_tok = next_audio_tok

            sampled_audio = torch.stack(sampled_audio_list, dim=1)

            self.step_offset += 1
            next_pos = self.step_offset % self.ring_len
            self.ring_cache[:, 0, next_pos] = sampled_text
            self.ring_cache[:, 1 : 1 + self.cfg.dep_q, next_pos] = sampled_audio

            if self.step_offset <= self.max_delay:
                return None

            gen_delays = self.delays_list[: 1 + self.cfg.dep_q]
            emit_base = self.step_offset - self.max_delay
            text_emit_pos = (emit_base + gen_delays[0]) % self.ring_len
            aligned_text = self.ring_cache[:, 0, text_emit_pos].unsqueeze(-1)
            aligned_audio = torch.stack(
                [
                    self.ring_cache[:, 1 + cb, (emit_base + gen_delays[1 + cb]) % self.ring_len]
                    for cb in range(self.cfg.dep_q)
                ],
                dim=1,
            ).unsqueeze(-1)
            return aligned_text, aligned_audio
        finally:
            if was_training:
                self.model.train()


def run_streaming_inference(args: argparse.Namespace) -> None:
    """CLI 인자를 바탕으로 체크포인트와 코덱을 로드하고 전체 스트리밍 음성·텍스트 번역 추론을 실행합니다."""
    device = setup_device(cpu_only=args.cpu, seed=args.seed, tag="Infer")

    ckpt = None
    if args.checkpoint is not None and args.checkpoint.exists():
        print(f"[Infer] 체크포인트 로드 중(CPU): {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        cfg = EDarchConfig.from_dict(ckpt["config"])
        cfg.train_on_source_audio = False
        model = EDarchModel(cfg)
        model.load_state_dict_for_inference(ckpt["model_state_dict"])
        model = model.to(device)
    else:
        print(f"[Infer] 체크포인트 없음 -> '{args.preset}' 프리셋으로 새 모델 초기화")
        cfg = EDarchConfig.from_preset(args.preset)
        cfg.train_on_source_audio = False
        model = EDarchModel(cfg).to(device)

    raw_cfg_coef = getattr(args, "cfg_coef", None)
    if raw_cfg_coef is not None:
        resolved_cfg_coef = float(raw_cfg_coef)
    elif ckpt is not None and ckpt.get("stage") == "grpo_stage2":
        resolved_cfg_coef = float(ckpt.get("rollout_cfg_coef", 1.0))
    else:
        resolved_cfg_coef = 3.0

    raw_sim_bin = getattr(args, "sim_bin", None)
    if raw_sim_bin is not None:
        resolved_sim_bin: Optional[int] = int(raw_sim_bin)
    elif ckpt is not None and ckpt.get("default_sim_bin") is not None:
        resolved_sim_bin = int(ckpt["default_sim_bin"])
    else:
        resolved_sim_bin = None

    codec = resolve_inference_codec(
        cfg=cfg, device=device, mimi_model_id=args.mimi_model, ckpt=ckpt
    )
    del ckpt
    model.eval()

    sp_proc = None
    if args.sp_model is not None:
        sp_proc = load_korean_tokenizer(args.sp_model)
        if sp_proc.eos_id() != cfg.text_eos_id or sp_proc.pad_id() != cfg.text_pad_id:
            print(
                f"[Infer] WARNING: 토크나이저 특수 토큰(eos={sp_proc.eos_id()}, pad={sp_proc.pad_id()})이 "
                f"모델 설정(eos={cfg.text_eos_id}, pad={cfg.text_pad_id})과 다릅니다."
            )
        if sp_proc.GetPieceSize() != cfg.text_card:
            print(
                f"[Infer] WARNING: 토크나이저 어휘 크기({sp_proc.GetPieceSize()})가 "
                f"모델 text_card({cfg.text_card})와 다릅니다."
            )

    if args.input_wav is not None:
        if not args.input_wav.exists():
            raise FileNotFoundError(f"입력 WAV 파일을 찾을 수 없습니다: {args.input_wav}")
        wav = load_mono_wav_24k(
            args.input_wav, target_sr=cfg.sample_rate, keep_channel_dim=True
        )
        src_wav = wav.unsqueeze(0).to(device)
        print(
            f"[Infer] 입력 WAV 로드 완료: {args.input_wav} "
            f"({src_wav.shape[-1] / cfg.sample_rate:.2f}초)"
        )
    else:
        hop_len = int(getattr(codec, "hop_length", int(cfg.sample_rate / cfg.frame_rate)))
        num_samples = args.demo_frames * hop_len
        src_wav = torch.randn(1, 1, num_samples, device=device) * 0.1
        print(
            f"[Infer] 합성 데모 입력 생성: {args.demo_frames}프레임 "
            f"({num_samples / cfg.sample_rate:.2f}초)"
        )

    if src_wav.shape[-1] == 0:
        raise ValueError("입력 오디오의 길이가 0 샘플입니다.")

    hop_length = int(getattr(codec, "hop_length", int(cfg.sample_rate / cfg.frame_rate)))
    with torch.no_grad():
        src_codes = codec.encode(src_wav)
        silence_frames = torch.zeros(
            1, 1, 8 * hop_length, device=device, dtype=src_wav.dtype
        )
        silence_code = codec.encode(silence_frames)[:, :, -1:]

    num_frames = src_codes.shape[-1]
    if num_frames == 0:
        raise ValueError("인코딩된 오디오 프레임 수가 0입니다.")

    streamer = EDarchStreamingGenerator(
        model,
        batch_size=1,
        cfg_coef=resolved_cfg_coef,
        temp_audio=args.temp_audio,
        temp_text=args.temp_text,
        top_k_audio=args.top_k_audio,
        top_k_text=args.top_k_text,
        sim_bin=resolved_sim_bin,
    )

    emitted_texts: list[int] = []
    emitted_audio_frames: list[torch.Tensor] = []

    amp = resolve_amp_context(device=device, no_amp=args.no_amp)
    max_tail_frames = max(1, int(args.max_tail_sec * cfg.frame_rate)) + streamer.max_delay
    eos_tail_frames = max(0, int(getattr(args, "eos_tail_frames", 2)))
    eos_reached = False
    remaining_after_eos = 0

    for t in range(num_frames):
        chunk = src_codes[:, :, t : t + 1]
        with torch.amp.autocast(
            device_type=device.type, dtype=amp.dtype, enabled=amp.enabled
        ):
            step_out = streamer.step(chunk, suppress_eos=True)
        if step_out is not None:
            t_tok, a_tok = step_out
            emitted_texts.append(int(t_tok[0, 0].item()))
            emitted_audio_frames.append(a_tok)

    for _ in range(max_tail_frames):
        with torch.amp.autocast(
            device_type=device.type, dtype=amp.dtype, enabled=amp.enabled
        ):
            step_out = streamer.step(silence_code, suppress_eos=False)
        if step_out is not None:
            t_tok, a_tok = step_out
            emitted_audio_frames.append(a_tok)
            if not eos_reached:
                tok_id = int(t_tok[0, 0].item())
                emitted_texts.append(tok_id)
                if tok_id == cfg.text_eos_id:
                    eos_reached = True
                    remaining_after_eos = eos_tail_frames
                    if remaining_after_eos <= 0:
                        break
            else:
                remaining_after_eos -= 1
                if remaining_after_eos <= 0:
                    break

    pred_audio_codes = torch.cat(emitted_audio_frames, dim=-1)
    with torch.no_grad():
        out_wav = codec.decode(pred_audio_codes).squeeze(0).float().cpu()

    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output_wav), out_wav, sample_rate=cfg.sample_rate)

    non_pad_tokens = [tid for tid in emitted_texts if tid != cfg.text_pad_id]
    stop_reason = "EOS 감지" if eos_reached else f"최대 꼬리 길이({args.max_tail_sec}s) 도달"
    print(
        f"[Done] 스트리밍 추론 완료 "
        f"(텍스트={len(emitted_texts)}프레임, 오디오={len(emitted_audio_frames)}프레임, "
        f"CFG={resolved_cfg_coef}, sim_bin={streamer.sim_bin}, 종료={stop_reason})"
    )
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
    print(
        f"  - 출력 WAV 저장 완료: {args.output_wav} "
        f"(길이: {out_wav.shape[-1] / cfg.sample_rate:.2f}초)"
    )


def run_bottom_up_verification(preset: str = "1b") -> None:
    """모델 초기화, 코덱 인코딩, 1스텝 역전파, 스트리밍 생성 및 파형 복원까지의 전 과정을 검증합니다."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    model_dtype = (
        torch.bfloat16
        if (use_cuda and torch.cuda.is_bf16_supported())
        else (torch.float16 if use_cuda else torch.float32)
    )

    cfg = EDarchConfig.from_preset(preset)
    codec = MiniMimiCodec(cfg).to(device=device)
    model = EDarchModel(cfg).to(device=device, dtype=model_dtype)
    model.set_gradient_checkpointing(True)

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"[1] EDarchModel('{preset}') 초기화 완료 "
        f"(Device: {device}, dtype: {model_dtype}, 파라미터 수: {total_params:,})"
    )

    batch_size = 1
    num_frames = 10
    num_samples = num_frames * codec.hop_length
    src_wav = torch.randn(batch_size, 1, num_samples, device=device)
    tgt_wav = torch.randn(batch_size, 1, num_samples, device=device)

    with torch.no_grad():
        src_codes = codec.encode(src_wav)
        tgt_codes = codec.encode(tgt_wav)
    print(f"[2] Mimi 인코딩 완료: 파형 {tuple(src_wav.shape)} -> RVQ 토큰 {tuple(src_codes.shape)}")

    tgt_text = torch.randint(4, cfg.text_card, (batch_size, 1, num_frames), device=device)
    tgt_text[:, :, :3] = cfg.text_pad_id

    full_codes = torch.cat([tgt_text, tgt_codes, src_codes], dim=1)
    sim_bins = torch.tensor([4], device=device)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, dtype=model_dtype, enabled=use_cuda):
        loss, metrics = model.compute_loss(full_codes, speaker_sim_bins=sim_bins)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print(
        f"[3] 학습 1스텝 역전파 성공: total_loss={metrics['loss']:.4f} "
        f"(text={metrics['text_loss']:.4f}, audio={metrics['audio_loss']:.4f})"
    )

    model.eval()
    streamer = EDarchStreamingGenerator(model, batch_size=1, cfg_coef=3.0)

    emitted_texts: list[int] = []
    emitted_audio_frames: list[torch.Tensor] = []

    for t in range(num_frames + streamer.max_delay):
        t_clamped = min(t, num_frames - 1)
        chunk_codes = src_codes[:1, :, t_clamped : t_clamped + 1]
        is_src_step = t < num_frames
        with torch.amp.autocast(device_type=device.type, dtype=model_dtype, enabled=use_cuda):
            step_out = streamer.step(chunk_codes, suppress_eos=is_src_step)
        if step_out is not None:
            t_tok, a_tok = step_out
            emitted_texts.append(int(t_tok[0, 0].item()))
            emitted_audio_frames.append(a_tok)

    pred_audio_codes = torch.cat(emitted_audio_frames, dim=-1)
    with torch.no_grad():
        reconstructed_wav = codec.decode(pred_audio_codes)

    print("[4] 스트리밍 추론 완료 (CFG=3.0):")
    print(f"    - 예측된 Inner Monologue 토큰 시퀀스 ({len(emitted_texts)}프레임): {emitted_texts}")
    print(
        f"    - 복원된 타깃 음성 파형 크기: {tuple(reconstructed_wav.shape)} "
        f"(기대 길이: (1, 1, {num_samples}))"
    )
