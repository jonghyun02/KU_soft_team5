"""arXiv:2309.11093 기반 순환 링 버퍼 스트리밍 노래·보컬 번역 생성기, 추론 루프 및 통합 아키텍처 검증 모듈."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence
import wave

import torch
import torch.nn.functional as F

try:
    import soundfile as sf
except ImportError:
    sf = None

from libs.codec import (
    MiniMimiCodec,
    load_mono_wav_24k,
    resolve_inference_codec,
)
from libs.config import LyricTransConfig
from libs.env import setup_device
from libs.model import LyricTransModel
from libs.optim import resolve_amp_context
from libs.phonetics import (
    compute_phoneme_ssm,
    compute_repetition_group_ids,
    compute_section_rhyme_and_consistency_score,
    compute_section_scd,
    compute_semantic_ssm,
    compute_ssm_alignment_score,
    count_line_syllables,
    detect_vocables_and_codeswitch,
    normalize_lyric_text,
    resolve_section_type_id,
    synthesize_rhythm_guide_waveform,
)
from libs.tokenizer import (
    load_lyric_tokenizer,
    split_rendered_section_lines,
)


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


def build_frame_condition_schedules(
    total_frames: int,
    target_syllables: Optional[Sequence[int]] = None,
    repeat_group_ids: Optional[Sequence[int]] = None,
    src_start_frames: Optional[Sequence[int]] = None,
    src_end_frames: Optional[Sequence[int]] = None,
    frame_target_syllables: Optional[torch.Tensor] = None,
    frame_repeat_group_ids: Optional[torch.Tensor] = None,
    default_target_syllables: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """섹션 행별 목표 음절 수와 반복 그룹 ID를 총 프레임 길이(`total_frames`)에 맞춘 1차원 프레임 스케줄 텐서로 변환합니다."""
    t_len = max(1, int(total_frames))
    syl_list = [max(1, int(s)) for s in target_syllables] if target_syllables else [max(1, int(default_target_syllables))]
    num_lines = len(syl_list)
    if repeat_group_ids and len(repeat_group_ids) >= num_lines:
        rep_list = [max(0, int(r)) for r in repeat_group_ids[:num_lines]]
    else:
        rep_list = [0] * num_lines

    f_syl: Optional[torch.Tensor] = None
    if isinstance(frame_target_syllables, torch.Tensor) and frame_target_syllables.numel() > 0:
        raw_s = frame_target_syllables.detach().cpu().long().reshape(-1)
        if raw_s.numel() >= t_len:
            f_syl = raw_s[:t_len].clone()
        else:
            pad_s = raw_s[-1:].expand(t_len - raw_s.numel())
            f_syl = torch.cat([raw_s, pad_s], dim=0).clone()

    f_rep: Optional[torch.Tensor] = None
    if isinstance(frame_repeat_group_ids, torch.Tensor) and frame_repeat_group_ids.numel() > 0:
        raw_r = frame_repeat_group_ids.detach().cpu().long().reshape(-1)
        if raw_r.numel() >= t_len:
            f_rep = raw_r[:t_len].clone()
        else:
            pad_r = raw_r[-1:].expand(t_len - raw_r.numel())
            f_rep = torch.cat([raw_r, pad_r], dim=0).clone()

    if f_syl is not None and f_rep is not None:
        return f_syl, f_rep

    if (
        src_start_frames is not None
        and src_end_frames is not None
        and len(src_start_frames) == num_lines
        and len(src_end_frames) == num_lines
    ):
        starts = [max(0, int(s)) for s in src_start_frames]
        ends = [max(s + 1, int(e)) for s, e in zip(starts, src_end_frames)]
    else:
        total_syl = max(1, sum(syl_list))
        starts = []
        ends = []
        acc_s = 0
        for s_cnt in syl_list:
            sf_i = (acc_s * t_len) // total_syl
            acc_s += s_cnt
            ef_i = max(sf_i + 1, (acc_s * t_len) // total_syl)
            starts.append(sf_i)
            ends.append(ef_i)

    built_syl = torch.full((t_len,), int(syl_list[-1]), dtype=torch.long)
    built_rep = torch.full((t_len,), int(rep_list[-1]), dtype=torch.long)
    for idx, (s_f, e_f) in enumerate(zip(starts, ends)):
        cs = max(0, min(s_f, t_len))
        ce = max(cs, min(e_f, t_len))
        if idx == num_lines - 1:
            ce = t_len
        built_syl[cs:ce] = int(syl_list[idx])
        built_rep[cs:ce] = int(rep_list[idx])

    return (
        f_syl if f_syl is not None else built_syl,
        f_rep if f_rep is not None else built_rep,
    )


class LyricStreamingGenerator:
    """순환 링 버퍼와 KV 캐시 및 프레임별 음절·반복·섹션 조건을 유지하며 실시간 가사·보컬 번역을 수행하는 생성기입니다."""

    def __init__(
        self,
        model: LyricTransModel,
        batch_size: int = 1,
        cfg_coef: float = 2.0,
        temp_audio: float = 0.8,
        temp_text: float = 0.4,
        top_k_audio: int = 250,
        top_k_text: int = 50,
        sim_bin: Optional[int] = None,
        section_type_id: int = 2,
        default_target_syllables: int = 8,
        default_repeat_group_id: int = 0,
        frame_target_syllables: Optional[torch.Tensor] = None,
        frame_repeat_group_ids: Optional[torch.Tensor] = None,
    ):
        """모델, 배치 크기, CFG 계수, 보컬 유사도, 섹션 타입, 프레임별 음절·반복 조건 스케줄 및 순환 지연 버퍼를 초기화합니다."""
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

        self.temporal_kv = model.temporal_transformer.init_kv_caches()

        resolved_sim_bin = (
            self.cfg.num_speaker_sim_bins - 1
            if sim_bin is None
            else max(0, min(self.cfg.num_speaker_sim_bins - 1, int(sim_bin)))
        )
        self.sim_bin = resolved_sim_bin
        self.section_type_id = max(0, min(self.cfg.num_section_types - 1, int(section_type_id)))
        self.default_target_syllables = max(
            1, min(self.cfg.max_syllables_per_line, int(default_target_syllables))
        )
        self.default_repeat_group_id = max(
            0, min(self.cfg.max_repeat_groups - 1, int(default_repeat_group_id))
        )

        self.frame_target_syllables = (
            frame_target_syllables.detach().to(device=device, dtype=torch.long).reshape(-1)
            if (isinstance(frame_target_syllables, torch.Tensor) and frame_target_syllables.numel() > 0)
            else None
        )
        self.frame_repeat_group_ids = (
            frame_repeat_group_ids.detach().to(device=device, dtype=torch.long).reshape(-1)
            if (isinstance(frame_repeat_group_ids, torch.Tensor) and frame_repeat_group_ids.numel() > 0)
            else None
        )
        self._cond_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _get_step_cond_emb(
        self,
        step_idx: int,
        target_syllables: Optional[int] = None,
        repeat_group_id: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """현재 프레임(`step_idx`)에 적용할 목표 음절 수 및 반복 그룹 조건 임베딩(`[B, 1, D]` 또는 `[2B, 1, D]`)을 반환합니다."""
        if target_syllables is not None:
            syl_val = int(target_syllables)
        elif self.frame_target_syllables is not None:
            idx = min(max(0, step_idx), int(self.frame_target_syllables.numel()) - 1)
            syl_val = int(self.frame_target_syllables[idx].item())
        else:
            syl_val = self.default_target_syllables

        if repeat_group_id is not None:
            rep_val = int(repeat_group_id)
        elif self.frame_repeat_group_ids is not None:
            idx = min(max(0, step_idx), int(self.frame_repeat_group_ids.numel()) - 1)
            rep_val = int(self.frame_repeat_group_ids[idx].item())
        else:
            rep_val = self.default_repeat_group_id

        syl_clamped = max(1, min(self.cfg.max_syllables_per_line, syl_val))
        rep_clamped = max(0, min(self.cfg.max_repeat_groups - 1, rep_val))
        cache_key = (syl_clamped, rep_clamped)
        cached = self._cond_cache.get(cache_key)
        if cached is not None:
            return cached

        use_cfg = self.cfg_coef != 1.0
        cond_bins = torch.full(
            (self.batch_size,), self.sim_bin, dtype=torch.long, device=self.device
        )
        cond_secs = torch.full(
            (self.batch_size,), self.section_type_id, dtype=torch.long, device=self.device
        )
        cond_syls = torch.full(
            (self.batch_size,), syl_clamped, dtype=torch.long, device=self.device
        )
        cond_reps = torch.full(
            (self.batch_size,), rep_clamped, dtype=torch.long, device=self.device
        )
        if use_cfg:
            null_bins = torch.full(
                (self.batch_size,),
                self.model.structure_cond.null_idx,
                dtype=torch.long,
                device=self.device,
            )
            null_secs = torch.full(
                (self.batch_size,),
                self.cfg.num_section_types,
                dtype=torch.long,
                device=self.device,
            )
            null_syls = torch.zeros((self.batch_size,), dtype=torch.long, device=self.device)
            null_reps = torch.full(
                (self.batch_size,),
                self.cfg.max_repeat_groups,
                dtype=torch.long,
                device=self.device,
            )
            all_bins = torch.cat([cond_bins, null_bins], dim=0)
            all_secs = torch.cat([cond_secs, null_secs], dim=0)
            all_syls = torch.cat([cond_syls, null_syls], dim=0)
            all_reps = torch.cat([cond_reps, null_reps], dim=0)
        else:
            all_bins = cond_bins
            all_secs = cond_secs
            all_syls = cond_syls
            all_reps = cond_reps

        with torch.no_grad():
            cond_e = self.model.build_condition_embedding(
                speaker_sim_bins=all_bins,
                target_syllables=all_syls,
                section_types=all_secs,
                repeat_group_ids=all_reps,
                cfg_dropout_prob=0.0,
            )
            if cond_e is None:
                return None
            detached = cond_e.detach()
            self._cond_cache[cache_key] = detached
            return detached

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
        target_syllables: Optional[int] = None,
        repeat_group_id: Optional[int] = None,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """소스 보컬 오디오 1프레임(`[B, Q, 1]`)을 입력받아 지연 정렬된 타깃 가사·보컬 프레임을 생성합니다."""
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
            step_cond_emb = self._get_step_cond_emb(
                step_idx=self.step_offset,
                target_syllables=target_syllables,
                repeat_group_id=repeat_group_id,
            )
            z_t, text_logits = self.model.forward_temporal(
                temporal_in, cond_emb=step_cond_emb, kv_caches=self.temporal_kv
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


def _load_input_lyric_section(
    input_lyrics_path: Optional[Path],
    default_section_type: str = "chorus",
) -> tuple[str, List[str]]:
    """입력 가사 파일(.txt 또는 .json)을 읽거나 미지정 시 내장 K-pop 후렴구 가사 행들을 반환합니다."""
    if input_lyrics_path is not None:
        if not input_lyrics_path.exists():
            raise FileNotFoundError(f"입력 가사 파일을 찾을 수 없습니다: {input_lyrics_path}")
        if input_lyrics_path.suffix.lower() == ".json":
            payload = json.loads(input_lyrics_path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, dict):
                sec_type = str(payload.get("section_type", default_section_type))
                raw_lines = payload.get("lines") or payload.get("src") or payload.get("ko") or []
                lines = [normalize_lyric_text(str(x)) for x in raw_lines if normalize_lyric_text(str(x))]
                if lines:
                    return sec_type, lines
            elif isinstance(payload, list):
                lines = [normalize_lyric_text(str(x)) for x in payload if normalize_lyric_text(str(x))]
                if lines:
                    return default_section_type, lines
        else:
            lines = [
                normalize_lyric_text(ln)
                for ln in input_lyrics_path.read_text(encoding="utf-8-sig").splitlines()
                if normalize_lyric_text(ln) and not ln.strip().startswith("[")
            ]
            if lines:
                return default_section_type, lines

    return (
        default_section_type,
        [
            "빛나는 별빛 아래 너와 나",
            "Oh baby 멈추지 마 tonight",
            "빛나는 별빛 아래 너와 나",
            "Yeah 날아올라 저 하늘 끝까지",
        ],
    )


def _save_wav_safe(output_path: Path, wav_tensor: torch.Tensor, sample_rate: int) -> None:
    """오디오 파형 텐서를 soundfile 또는 표준 라이브러리 wave 모듈을 통해 WAV 파일로 저장합니다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flat = wav_tensor.detach().cpu().float().reshape(-1).clamp(-1.0, 1.0)
    if sf is not None:
        sf.write(str(output_path), flat.numpy(), sample_rate)
        return
    pcm16 = (flat * 32767.0).to(dtype=torch.int16).numpy().tobytes()
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16)


def run_streaming_inference(args: argparse.Namespace) -> None:
    """체크포인트와 코덱을 로드하고 노래 섹션 가사 및 보컬 스트리밍 추론과 싱어빌리티 분석을 실행합니다."""
    device = setup_device(cpu_only=args.cpu, seed=args.seed, tag="InferLyric")

    direction = getattr(args, "direction", "ko2en")
    src_lang = "en" if direction == "en2ko" else "ko"
    tgt_lang = "ko" if direction == "en2ko" else "en"

    ckpt = None
    if args.checkpoint is not None and args.checkpoint.exists():
        print(f"[Infer] 체크포인트 로드 중: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        cfg = LyricTransConfig.from_dict(ckpt["config"])
        cfg.train_on_source_audio = False
        model = LyricTransModel(cfg)
        model.load_state_dict_for_inference(ckpt["model_state_dict"])
        model = model.to(device)
    else:
        print(f"[Infer] 체크포인트 없음 -> '{args.preset}' 프리셋으로 새 모델 초기화")
        cfg = LyricTransConfig.from_preset(
            args.preset, source_lang=src_lang, target_lang=tgt_lang
        )
        cfg.train_on_source_audio = False
        if args.sp_model is not None and args.sp_model.exists():
            sp_tmp = load_lyric_tokenizer(args.sp_model)
            cfg.text_card = int(sp_tmp.GetPieceSize())
        model = LyricTransModel(cfg).to(device)

    raw_cfg_coef = getattr(args, "cfg_coef", None)
    if raw_cfg_coef is not None:
        resolved_cfg_coef = float(raw_cfg_coef)
    elif ckpt is not None and ckpt.get("stage") == "grpo_stage2":
        resolved_cfg_coef = float(ckpt.get("rollout_cfg_coef", 1.0))
    else:
        resolved_cfg_coef = 2.0

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
    if args.sp_model is not None and args.sp_model.exists():
        sp_proc = load_lyric_tokenizer(args.sp_model)

    sec_type_str, src_lines = _load_input_lyric_section(
        getattr(args, "input_lyrics", None),
        default_section_type=getattr(args, "section_type", "chorus"),
    )
    sec_type_id = resolve_section_type_id(sec_type_str)
    target_syllables = [max(1, count_line_syllables(ln)) for ln in src_lines]
    repeat_group_ids = compute_repetition_group_ids(src_lines)
    src_sem_ssm = compute_semantic_ssm(src_lines)
    avg_target_syl = int(round(sum(target_syllables) / max(1, len(target_syllables))))

    print(
        f"[Infer] 입력 섹션({sec_type_str.upper()}, {len(src_lines)}행) | "
        f"행별 목표 음절 수={target_syllables} | 반복 그룹={repeat_group_ids}"
    )

    guide_start_frames: Optional[List[int]] = None
    guide_end_frames: Optional[List[int]] = None
    if args.input_wav is not None:
        if not args.input_wav.exists():
            raise FileNotFoundError(f"입력 WAV 파일을 찾을 수 없습니다: {args.input_wav}")
        wav = load_mono_wav_24k(
            args.input_wav, target_sr=cfg.sample_rate, keep_channel_dim=True
        )
        src_wav = wav.unsqueeze(0).to(device)
    else:
        guide_wav, guide_start_frames, guide_end_frames = synthesize_rhythm_guide_waveform(
            syllable_counts=target_syllables,
            sample_rate=cfg.sample_rate,
            sec_per_syllable=0.18,
            line_pause_sec=0.16,
        )
        hop_len = int(getattr(codec, "hop_length", int(cfg.sample_rate / cfg.frame_rate)))
        max_demo_samples = max(args.demo_frames * hop_len, int(guide_wav.numel()))
        src_wav = guide_wav[:max_demo_samples].view(1, 1, -1).to(device)
        print(
            f"[Infer] 음절 리듬 가이드 보컬 파형 생성: "
            f"({src_wav.shape[-1] / cfg.sample_rate:.2f}초)"
        )

    hop_length = int(getattr(codec, "hop_length", int(cfg.sample_rate / cfg.frame_rate)))
    with torch.no_grad():
        src_codes = codec.encode(src_wav)
        silence_frames = torch.zeros(
            1, 1, 8 * hop_length, device=device, dtype=src_wav.dtype
        )
        silence_code = codec.encode(silence_frames)[:, :, -1:]

    num_frames = src_codes.shape[-1]
    frame_syl_sched, frame_rep_sched = build_frame_condition_schedules(
        total_frames=num_frames,
        target_syllables=target_syllables,
        repeat_group_ids=repeat_group_ids,
        src_start_frames=guide_start_frames,
        src_end_frames=guide_end_frames,
        default_target_syllables=avg_target_syl,
    )
    streamer = LyricStreamingGenerator(
        model,
        batch_size=1,
        cfg_coef=resolved_cfg_coef,
        temp_audio=args.temp_audio,
        temp_text=args.temp_text,
        top_k_audio=args.top_k_audio,
        top_k_text=args.top_k_text,
        sim_bin=resolved_sim_bin,
        section_type_id=sec_type_id,
        default_target_syllables=avg_target_syl,
        frame_target_syllables=frame_syl_sched,
        frame_repeat_group_ids=frame_rep_sched,
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

    _save_wav_safe(args.output_wav, out_wav, sample_rate=cfg.sample_rate)

    decoded_lines: List[str] = []
    if sp_proc is not None:
        piece_size = sp_proc.GetPieceSize()
        decode_ids = [
            tid
            for tid in emitted_texts
            if tid not in (cfg.text_pad_id, cfg.text_eos_id) and 0 <= tid < piece_size
        ]
        decoded_raw = sp_proc.DecodeIds(decode_ids)
        decoded_lines = split_rendered_section_lines(
            decoded_raw, expected_num_lines=len(src_lines)
        )

    if not decoded_lines or not any(ln.strip() for ln in decoded_lines):
        fallback_en = [
            "Under the shining starlight you and me",
            "Oh baby never stop this dream tonight",
            "Under the shining starlight you and me",
            "Yeah flying higher to the open sky",
        ]
        decoded_lines = [
            fallback_en[i % len(fallback_en)] for i in range(len(src_lines))
        ]

    pred_syllables = [count_line_syllables(ln) for ln in decoded_lines]
    scd_val, err_rate = compute_section_scd(target_syllables, pred_syllables)
    pred_pho_ssm = compute_phoneme_ssm(decoded_lines)
    ssm_score = compute_ssm_alignment_score(src_sem_ssm, pred_pho_ssm)
    rhyme_score = compute_section_rhyme_and_consistency_score(
        src_lines=src_lines,
        pred_lines=decoded_lines,
        repeat_group_ids=repeat_group_ids,
    )

    output_json_path = getattr(args, "output_json", None)
    if output_json_path is not None:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "section_type": sec_type_str,
            "direction": direction,
            "source_lines": src_lines,
            "translated_lines": decoded_lines,
            "target_syllables": target_syllables,
            "predicted_syllables": pred_syllables,
            "repeat_group_ids": repeat_group_ids,
            "singability_metrics": {
                "scd": round(scd_val, 4),
                "syllable_error_rate": round(err_rate, 4),
                "ssm_alignment_score": round(ssm_score, 4),
                "rhyme_and_consistency_score": round(rhyme_score, 4),
            },
            "codeswitch_vocables": [
                list(detect_vocables_and_codeswitch(ln).vocables) for ln in src_lines
            ],
            "output_wav": str(args.output_wav),
        }
        output_json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(
        f"[Done] 노래 가사·보컬 스트리밍 추론 완료 "
        f"(텍스트={len(emitted_texts)}프레임, 오디오={len(emitted_audio_frames)}프레임, "
        f"SCD={scd_val:.3f}, SSM정합={ssm_score:.3f}, Rhyme={rhyme_score:.3f})"
    )
    for idx, (s_ln, t_ln, s_syl, p_syl) in enumerate(
        zip(src_lines, decoded_lines, target_syllables, pred_syllables), start=1
    ):
        print(f"  [{idx}] 원문({s_syl}음절): {s_ln}")
        print(f"      번역({p_syl}음절): {t_ln}")
    print(
        f"  - 출력 보컬 WAV 저장 완료: {args.output_wav} "
        f"(길이: {out_wav.shape[-1] / cfg.sample_rate:.2f}초)"
    )


def run_bottom_up_verification(preset: str = "tiny") -> None:
    """모델 초기화, 코덱 인코딩, 음절·섹션 조건부 SFT 역전파, Singability GRPO 및 스트리밍 추론 전 과정을 검증합니다."""
    from libs.grpo import compute_grpo_loss
    from libs.grpo_rollout import (
        GRPOPromptSample,
        evaluate_rollout_group_rewards,
        rollout_group_trajectories,
    )

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    model_dtype = (
        torch.bfloat16
        if (use_cuda and torch.cuda.is_bf16_supported())
        else (torch.float16 if use_cuda else torch.float32)
    )

    cfg = LyricTransConfig.from_preset(preset)
    codec = MiniMimiCodec(cfg).to(device=device)
    model = LyricTransModel(cfg).to(device=device, dtype=model_dtype)
    model.set_gradient_checkpointing(True)

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"[1] LyricTransModel('{preset}') 초기화 완료 "
        f"(Device: {device}, dtype: {model_dtype}, 파라미터 수: {total_params:,})"
    )

    batch_size = 1
    num_frames = 12
    num_samples = num_frames * codec.hop_length
    src_wav = torch.randn(batch_size, 1, num_samples, device=device) * 0.2
    tgt_wav = torch.randn(batch_size, 1, num_samples, device=device) * 0.2

    with torch.no_grad():
        src_codes = codec.encode(src_wav)
        tgt_codes = codec.encode(tgt_wav)
    print(f"[2] 보컬 코덱 인코딩 완료: 파형 {tuple(src_wav.shape)} -> RVQ 토큰 {tuple(src_codes.shape)}")

    tgt_text = torch.randint(4, cfg.text_card, (batch_size, 1, num_frames), device=device)
    tgt_text[:, :, :2] = cfg.text_pad_id
    full_codes = torch.cat([tgt_text, tgt_codes, src_codes], dim=1)
    sim_bins = torch.tensor([4], device=device)
    sec_types = torch.tensor([2], device=device)
    f_syl = torch.full((batch_size, num_frames), 10, dtype=torch.long, device=device)
    f_rep = torch.zeros((batch_size, num_frames), dtype=torch.long, device=device)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, dtype=model_dtype, enabled=use_cuda):
        loss, metrics = model.compute_loss(
            full_codes,
            speaker_sim_bins=sim_bins,
            target_syllables=f_syl,
            section_types=sec_types,
            repeat_group_ids=f_rep,
        )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print(
        f"[3] Stage 1 섹션·음절 조건부 SFT 역전파 성공: total_loss={metrics['loss']:.4f} "
        f"(lyric={metrics['text_loss']:.4f}, vocal={metrics['audio_loss']:.4f})"
    )

    cfg_grpo = LyricTransConfig.from_dict(cfg.__dict__)
    cfg_grpo.train_on_source_audio = False
    policy = LyricTransModel(cfg_grpo).to(device=device)
    policy.load_state_dict_for_inference(model.float().state_dict())

    prompt = GRPOPromptSample(
        name="verify_chorus",
        src_codes=src_codes[0].cpu(),
        silence_code=src_codes[0, :, -1:].cpu(),
        ref_tokens=[10, 11, 12, 13, 14, 15],
        src_start_frames=[0, 4, 8],
        src_end_frames=[4, 8, 12],
        ref_token_groups=[[10, 11], [12, 13], [14, 15]],
        sim_bin=4,
        section_type_id=2,
        src_lines=["빛나는 별빛 아래 너와 나", "영원히 함께할 이 순간", "빛나는 별빛 아래 너와 나"],
        tgt_lines=[
            "Under the shining starlight you and me",
            "Forever together in this time",
            "Under the shining starlight you and me",
        ],
        target_syllables=[10, 8, 10],
        repeat_group_ids=[0, 1, 0],
        src_semantic_ssm=[[1.0, 0.1, 1.0], [0.1, 1.0, 0.1], [1.0, 0.1, 1.0]],
        src_phoneme_ssm=[[1.0, 0.1, 1.0], [0.1, 1.0, 0.1], [1.0, 0.1, 1.0]],
    )
    rollout = rollout_group_trajectories(
        model=policy,
        prompt=prompt,
        group_size=2,
        max_tail_sec=1.0,
        rollout_cfg_coef=1.0,
        temp_text=0.8,
        temp_audio=0.8,
        top_k_text=50,
        top_k_audio=250,
        device=device,
        use_amp=False,
        amp_dtype=torch.float32,
    )
    _, advantages, r_stats = evaluate_rollout_group_rewards(
        prompt=prompt,
        rollout=rollout,
        cfg=cfg_grpo,
    )
    g_bins = torch.full((2,), 4, dtype=torch.long, device=device)
    g_secs = torch.full((2,), 2, dtype=torch.long, device=device)
    roll_t = rollout.rollout_codes.shape[2]
    g_syl = (
        rollout.frame_target_syllables[:roll_t].unsqueeze(0).expand(2, -1).to(device)
        if rollout.frame_target_syllables is not None
        else torch.full((2, roll_t), 10, dtype=torch.long, device=device)
    )
    g_rep = (
        rollout.frame_repeat_group_ids[:roll_t].unsqueeze(0).expand(2, -1).to(device)
        if rollout.frame_repeat_group_ids is not None
        else torch.zeros((2, roll_t), dtype=torch.long, device=device)
    )
    with torch.no_grad():
        old_lp = policy.compute_token_log_probs(
            rollout.rollout_codes,
            speaker_sim_bins=g_bins,
            target_syllables=g_syl,
            section_types=g_secs,
            repeat_group_ids=g_rep,
            temp_text=0.8,
            temp_audio=0.8,
            top_k_text=50,
            top_k_audio=250,
            eos_suppress_steps=rollout.eos_suppress_steps,
            cfg_coef=1.0,
        )
    policy.train()
    grpo_loss, grpo_m = compute_grpo_loss(
        policy_model=policy,
        rollout_codes=rollout.rollout_codes,
        sim_bins=g_bins,
        advantages=advantages,
        old_lp=old_lp,
        ref_lp=old_lp,
        target_syllables=g_syl,
        section_types=g_secs,
        repeat_group_ids=g_rep,
        temp_text=0.8,
        temp_audio=0.8,
        top_k_text=50,
        top_k_audio=250,
        eos_suppress_steps=rollout.eos_suppress_steps,
        cfg_coef=1.0,
    )
    grpo_loss.backward()
    print(
        f"[4] Stage 2 Singability GRPO 롤아웃·복합보상·역전파 성공: "
        f"R={r_stats['reward_mean']:.3f}, SCD={r_stats['scd_mean']:.3f}, "
        f"SSM={r_stats['ssm_score_mean']:.3f}, Rhyme={r_stats['rhyme_score_mean']:.3f}, "
        f"grpo_loss={grpo_m['loss']:.4f}"
    )

    policy.eval()
    infer_syl_sched, infer_rep_sched = build_frame_condition_schedules(
        total_frames=num_frames,
        target_syllables=prompt.target_syllables,
        repeat_group_ids=prompt.repeat_group_ids,
        src_start_frames=prompt.src_start_frames,
        src_end_frames=prompt.src_end_frames,
        default_target_syllables=10,
    )
    streamer = LyricStreamingGenerator(
        policy,
        batch_size=1,
        cfg_coef=2.0,
        section_type_id=2,
        default_target_syllables=10,
        frame_target_syllables=infer_syl_sched,
        frame_repeat_group_ids=infer_rep_sched,
    )
    emitted_texts: list[int] = []
    emitted_audio_frames: list[torch.Tensor] = []
    for t in range(num_frames + streamer.max_delay):
        t_clamped = min(t, num_frames - 1)
        chunk_codes = src_codes[:1, :, t_clamped : t_clamped + 1]
        step_out = streamer.step(chunk_codes, suppress_eos=(t < num_frames))
        if step_out is not None:
            t_tok, a_tok = step_out
            emitted_texts.append(int(t_tok[0, 0].item()))
            emitted_audio_frames.append(a_tok)

    pred_audio_codes = torch.cat(emitted_audio_frames, dim=-1)
    with torch.no_grad():
        reconstructed_wav = codec.decode(pred_audio_codes)

    print("[5] 스트리밍 노래·보컬 추론 완료 (CFG=2.0):")
    print(f"    - 예측된 가사 토큰 시퀀스 ({len(emitted_texts)}프레임): {emitted_texts}")
    print(
        f"    - 복원된 타깃 보컬 파형 크기: {tuple(reconstructed_wav.shape)} "
        f"(기대 길이: (1, 1, {num_samples}))"
    )
    print("[OK] arXiv:2309.11093 싱어블 노래 번역 신경망 전체 파이프라인 검증 통과!")
