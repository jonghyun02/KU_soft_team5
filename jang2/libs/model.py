"""EDarch 핵심 멀티스트림 트랜스포머 모델 및 한국어 어휘 어댑테이션 모듈."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.config import EDarchConfig
from libs.layers import (
    LayerKVCache,
    LowRankScaledEmbedding,
    RMSNormFP32,
    SpeakerSimilarityConditioner,
    StreamingTransformerStack,
    delay_sequence,
    undelay_sequence,
)


@dataclass
class EDarchForwardOutput:
    """지연 복원된 텍스트·오디오 로짓과 유효 프레임 마스크를 담는 순전파 결과 데이터클래스입니다."""

    text_logits: torch.Tensor
    text_mask: torch.Tensor
    audio_logits: torch.Tensor
    audio_mask: torch.Tensor


@dataclass
class EDarchSequenceLogProbs:
    """강화학습 정책 평가를 위해 추출된 시퀀스별 텍스트·오디오 토큰 로그 확률과 마스크입니다."""

    text_log_probs: torch.Tensor
    text_mask: torch.Tensor
    audio_log_probs: torch.Tensor
    audio_mask: torch.Tensor


class EDarchModel(nn.Module):
    """시간축 Temporal Transformer와 깊이축 Depformer를 결합한 영-한 동시통역 멀티스트림 신경망입니다."""

    def __init__(self, cfg: EDarchConfig):
        """설정에 따라 멀티스트림 임베딩, 화자 조건화기, Temporal Transformer 및 Depformer를 초기화합니다."""
        super().__init__()
        self.cfg = cfg

        self.train_dep_q = cfg.n_q if cfg.train_on_source_audio else cfg.dep_q
        self._suppress_early_eos: bool = False

        self.text_emb = LowRankScaledEmbedding(
            cfg.text_card + 1, cfg.dim, zero_idx=cfg.zero_token_id
        )
        self.audio_embs = nn.ModuleList(
            [
                LowRankScaledEmbedding(
                    cfg.audio_card + 1, cfg.dim, zero_idx=cfg.zero_token_id
                )
                for _ in range(cfg.n_q)
            ]
        )
        self.speaker_cond = SpeakerSimilarityConditioner(cfg.num_speaker_sim_bins, cfg.dim)

        self.temporal_transformer = StreamingTransformerStack(
            num_layers=cfg.num_layers,
            dim=cfg.dim,
            num_heads=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            use_rope=True,
            rope_max_period=cfg.rope_max_period,
            max_context=cfg.context_frames,
            step_schedule=None,
            norm_eps=cfg.norm_eps,
        )
        self.out_norm = RMSNormFP32(cfg.dim, eps=cfg.norm_eps)
        self.text_linear = nn.Linear(cfg.dim, cfg.text_card, bias=False)

        if len(cfg.depformer_weights_schedule) != cfg.num_codebooks_per_stream:
            raise ValueError(
                f"depformer_weights_schedule 길이({len(cfg.depformer_weights_schedule)})가 "
                f"num_codebooks_per_stream({cfg.num_codebooks_per_stream})과 다릅니다."
            )

        if cfg.train_on_source_audio:
            offset = max(cfg.depformer_weights_schedule) + 1
            full_schedule = cfg.depformer_weights_schedule + [
                offset + s for s in cfg.depformer_weights_schedule
            ]
        else:
            full_schedule = list(cfg.depformer_weights_schedule)
        self.dep_schedule = full_schedule

        num_proj_weights = max(full_schedule) + 1
        self.depformer_in = nn.ModuleList(
            [
                nn.Linear(cfg.dim, cfg.depformer_dim, bias=False)
                for _ in range(num_proj_weights)
            ]
        )
        self.depformer_text_emb = LowRankScaledEmbedding(
            cfg.text_card + 1,
            cfg.depformer_dim,
            zero_idx=cfg.zero_token_id,
            low_rank=cfg.depformer_low_rank_emb,
        )
        self.depformer_audio_embs = nn.ModuleList(
            [
                LowRankScaledEmbedding(
                    cfg.audio_card + 1,
                    cfg.depformer_dim,
                    zero_idx=cfg.zero_token_id,
                    low_rank=cfg.depformer_low_rank_emb,
                )
                for _ in range(self.train_dep_q - 1)
            ]
        )
        self.depformer = StreamingTransformerStack(
            num_layers=cfg.depformer_num_layers,
            dim=cfg.depformer_dim,
            num_heads=cfg.depformer_num_heads,
            dim_feedforward=cfg.depformer_dim_feedforward,
            use_rope=False,
            max_context=None,
            step_schedule=full_schedule,
            norm_eps=cfg.norm_eps,
        )
        self.depformer_linears = nn.ModuleList(
            [
                nn.Linear(cfg.depformer_dim, cfg.audio_card, bias=False)
                for _ in range(self.train_dep_q)
            ]
        )

    def load_state_dict_for_inference(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """소스 오디오 보조 손실이 포함된 체크포인트에서 추론용 타깃 스트림 가중치만 선별해 로드합니다."""
        if self.cfg.train_on_source_audio:
            self.load_state_dict(state_dict, strict=True)
            return

        q = self.cfg.dep_q
        base_proj_count = max(self.cfg.depformer_weights_schedule) + 1
        filtered: dict[str, torch.Tensor] = {}

        for key, val in state_dict.items():
            parts = key.split(".")
            if parts[0] == "depformer_in" and int(parts[1]) >= base_proj_count:
                continue
            if parts[0] == "depformer_audio_embs" and int(parts[1]) >= q - 1:
                continue
            if parts[0] == "depformer_linears" and int(parts[1]) >= q:
                continue
            if (
                parts[0] == "depformer"
                and len(parts) >= 6
                and parts[1] == "layers"
                and parts[4] in ("linears", "ffns")
                and int(parts[5]) >= base_proj_count
            ):
                continue
            filtered[key] = val

        self.load_state_dict(filtered, strict=True)

    def get_initial_tokens(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """시퀀스 시작 시 모든 텍스트·오디오 스트림에 공급할 BOS 토큰 텐서를 생성합니다."""
        t_bos = torch.full(
            (batch_size, 1, 1), self.cfg.text_bos_id, dtype=torch.long, device=device
        )
        a_bos = torch.full(
            (batch_size, self.cfg.n_q, 1),
            self.cfg.audio_bos_id,
            dtype=torch.long,
            device=device,
        )
        return torch.cat([t_bos, a_bos], dim=1)

    def apply_early_eos_suppression_(self, enabled: bool = True) -> None:
        """스트리밍 생성 시 기본 조기 EOS 억제 플래그를 설정합니다."""
        self._suppress_early_eos = enabled

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Temporal Transformer와 Depformer 양쪽에 Activation Checkpointing 활성 여부를 설정합니다."""
        self.temporal_transformer.gradient_checkpointing = enabled
        self.depformer.gradient_checkpointing = enabled

    def forward_temporal(
        self,
        seq_codes: torch.Tensor,
        cond_emb: Optional[torch.Tensor] = None,
        kv_caches: Optional[list[LayerKVCache]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """모든 스트림 임베딩 합을 Temporal Transformer에 통과시켜 은닉 상태 `z_t`와 텍스트 로짓을 계산합니다."""
        x = self.text_emb(seq_codes[:, 0, :])
        for q_idx in range(self.cfg.n_q):
            x = x + self.audio_embs[q_idx](seq_codes[:, 1 + q_idx, :])
        if cond_emb is not None:
            x = x + cond_emb.to(dtype=x.dtype)

        z_t = self.temporal_transformer(x, kv_caches=kv_caches)
        z_t = self.out_norm(z_t)
        text_logits = self.text_linear(z_t).unsqueeze(1)
        return z_t, text_logits

    def forward_depformer_parallel(
        self, target_delayed_codes: torch.Tensor, z_t: torch.Tensor
    ) -> torch.Tensor:
        """학습 및 로그 확률 평가 시 모든 시간 프레임과 코드북 스텝을 병렬로 Depformer에 통과시킵니다."""
        bsz, _, seq_len = target_delayed_codes.shape
        ka = self.train_dep_q
        proj_cache: dict[int, torch.Tensor] = {}
        step_inputs = []
        for cb in range(ka):
            proj_idx = self.dep_schedule[cb]
            proj_z = proj_cache.get(proj_idx)
            if proj_z is None:
                proj_z = self.depformer_in[proj_idx](z_t)
                proj_cache[proj_idx] = proj_z
            if cb == 0:
                tok_emb = self.depformer_text_emb(target_delayed_codes[:, 0, :])
            else:
                tok_emb = self.depformer_audio_embs[cb - 1](
                    target_delayed_codes[:, cb, :]
                )
            step_inputs.append(proj_z + tok_emb)

        dep_in = torch.stack(step_inputs, dim=2).view(bsz * seq_len, ka, -1)
        dep_out = self.depformer(dep_in, kv_caches=None, step_idx=None)

        per_cb_logits = []
        for cb in range(ka):
            cb_logit = self.depformer_linears[cb](dep_out[:, cb, :]).view(
                bsz, seq_len, -1
            )
            per_cb_logits.append(cb_logit)
        return torch.stack(per_cb_logits, dim=1)

    def forward_depformer_single_step(
        self,
        cb_index: int,
        prev_token: torch.Tensor,
        z_t_step: torch.Tensor,
        dep_kv_caches: list[LayerKVCache],
    ) -> torch.Tensor:
        """스트리밍 추론 시 단일 시간 프레임 내에서 지정된 코드북 인덱스의 로짓을 순차 계산합니다."""
        proj_idx = self.dep_schedule[cb_index]
        h = self.depformer_in[proj_idx](z_t_step)
        tok_in = prev_token.unsqueeze(1)
        if cb_index == 0:
            h = h + self.depformer_text_emb(tok_in)
        else:
            h = h + self.depformer_audio_embs[cb_index - 1](tok_in)

        dep_out = self.depformer(h, kv_caches=dep_kv_caches, step_idx=cb_index)
        return self.depformer_linears[cb_index](dep_out[:, 0, :])

    def _forward_delayed(
        self,
        codes: torch.Tensor,
        speaker_sim_bins: Optional[torch.Tensor] = None,
        cfg_dropout_prob: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """지연 공간(Delayed Space)에서 Temporal 및 병렬 Depformer 순전파를 수행합니다."""
        bsz, num_streams, _ = codes.shape
        if num_streams != self.cfg.total_streams:
            raise ValueError(
                f"입력 스트림 수({num_streams})가 설정값({self.cfg.total_streams})과 다릅니다."
            )

        initial = self.get_initial_tokens(bsz, codes.device)
        delayed = delay_sequence(self.cfg.delays, codes, initial)
        delayed_with_bos = torch.cat([initial, delayed], dim=2)

        cond_emb: Optional[torch.Tensor] = None
        if speaker_sim_bins is not None:
            bins = speaker_sim_bins.clone()
            if self.training and cfg_dropout_prob > 0.0:
                drop_mask = torch.rand(bsz, device=codes.device) < cfg_dropout_prob
                bins = torch.where(
                    drop_mask,
                    torch.full_like(bins, self.speaker_cond.null_idx),
                    bins,
                )
            cond_emb = self.speaker_cond(bins)

        z_t, delayed_text_logits = self.forward_temporal(
            delayed_with_bos[:, :, :-1], cond_emb=cond_emb, kv_caches=None
        )

        dep_target_inputs = delayed_with_bos[:, :, 1:]
        delayed_audio_logits = self.forward_depformer_parallel(
            dep_target_inputs, z_t
        )

        dep_step_valid = (
            (dep_target_inputs[:, : self.train_dep_q, :] != self.cfg.zero_token_id)
            .long()
            .cumprod(dim=1)
            .bool()
        )
        return delayed_text_logits, delayed_audio_logits, dep_target_inputs, dep_step_valid

    def forward(
        self,
        codes: torch.Tensor,
        speaker_sim_bins: Optional[torch.Tensor] = None,
        cfg_dropout_prob: float = 0.1,
    ) -> EDarchForwardOutput:
        """멀티스트림 입력에 대해 시간축 지연이 복원된 텍스트·오디오 로짓과 마스크를 반환합니다."""
        (
            delayed_text_logits,
            delayed_audio_logits,
            _,
            dep_step_valid,
        ) = self._forward_delayed(
            codes, speaker_sim_bins=speaker_sim_bins, cfg_dropout_prob=cfg_dropout_prob
        )

        audio_delays = self.cfg.delays[1 : 1 + self.train_dep_q]
        audio_logits, audio_mask = undelay_sequence(audio_delays, delayed_audio_logits)
        undelayed_dep_valid, _ = undelay_sequence(
            audio_delays, dep_step_valid.unsqueeze(-1)
        )
        audio_mask = (
            audio_mask
            & undelayed_dep_valid.squeeze(-1)
            & (codes[:, 1 : 1 + self.train_dep_q, :] != self.cfg.zero_token_id)
        )

        text_logits, text_mask = undelay_sequence(
            self.cfg.delays[:1], delayed_text_logits
        )
        text_mask = text_mask & (codes[:, :1, :] != self.cfg.zero_token_id)

        return EDarchForwardOutput(
            text_logits=text_logits,
            text_mask=text_mask,
            audio_logits=audio_logits,
            audio_mask=audio_mask,
        )

    def compute_loss(
        self,
        codes: torch.Tensor,
        speaker_sim_bins: torch.Tensor,
        cfg_dropout_prob: float = 0.1,
        pad_text_weight: float = 0.5,
        semantic_audio_weight: float = 10.0,
        source_audio_weight: float = 0.2,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Stage 1 지도학습을 위해 지연 공간에서 가중 교차 엔트로피 텍스트·오디오 손실을 계산합니다."""
        (
            delayed_text_logits,
            delayed_audio_logits,
            dep_target_inputs,
            dep_step_valid,
        ) = self._forward_delayed(
            codes, speaker_sim_bins=speaker_sim_bins, cfg_dropout_prob=cfg_dropout_prob
        )
        seq_len = codes.shape[2]
        s_idx = torch.arange(seq_len, device=codes.device).view(1, 1, seq_len)

        t_delay = self.cfg.delays[0]
        t_targets_grid = dep_target_inputs[:, :1, :]
        t_mask = (s_idx >= t_delay) & (t_targets_grid != self.cfg.zero_token_id)
        t_targets = t_targets_grid[t_mask]
        t_logits = delayed_text_logits[t_mask]
        if t_targets.numel() > 0:
            t_ce = F.cross_entropy(t_logits.float(), t_targets, reduction="none")
            t_weights = torch.where(
                t_targets == self.cfg.text_pad_id,
                torch.full_like(t_ce, pad_text_weight, dtype=torch.float32),
                torch.ones_like(t_ce, dtype=torch.float32),
            )
            text_loss = (t_ce * t_weights).sum() / t_weights.sum().clamp_min(1e-6)
        else:
            text_loss = delayed_text_logits.float().sum() * 0.0

        q_dep = self.cfg.dep_q
        cb_weights = torch.ones(self.train_dep_q, device=codes.device, dtype=torch.float32)
        cb_weights[0] = semantic_audio_weight
        if self.train_dep_q > q_dep:
            cb_weights[q_dep:] = source_audio_weight

        audio_delays = torch.tensor(
            self.cfg.delays[1 : 1 + self.train_dep_q],
            device=codes.device,
            dtype=torch.long,
        ).view(1, self.train_dep_q, 1)
        a_targets_grid = dep_target_inputs[:, 1 : 1 + self.train_dep_q, :]
        a_mask = (
            (s_idx >= audio_delays)
            & dep_step_valid
            & (a_targets_grid != self.cfg.zero_token_id)
        )
        a_weight_grid = cb_weights.view(1, self.train_dep_q, 1).expand_as(a_mask)
        a_targets = a_targets_grid[a_mask]
        a_logits = delayed_audio_logits[a_mask]
        a_weights = a_weight_grid[a_mask]
        if a_targets.numel() > 0:
            a_ce = F.cross_entropy(a_logits.float(), a_targets, reduction="none")
            audio_loss = (a_ce * a_weights).sum() / a_weights.sum().clamp_min(1e-6)
        else:
            audio_loss = delayed_audio_logits.float().sum() * 0.0

        total_loss = text_loss + audio_loss
        metrics = {
            "loss": float(total_loss.detach().cpu()),
            "text_loss": float(text_loss.detach().cpu()),
            "audio_loss": float(audio_loss.detach().cpu()),
        }
        return total_loss, metrics

    def compute_token_log_probs(
        self,
        codes: torch.Tensor,
        speaker_sim_bins: Optional[torch.Tensor] = None,
        temp_text: float = 1.0,
        temp_audio: float = 1.0,
    ) -> EDarchSequenceLogProbs:
        """Stage 2 GRPO 강화학습을 위해 샘플링된 텍스트·오디오 토큰의 로그 확률을 메모리 효율적으로 추출합니다."""
        if temp_text <= 0.0 or temp_audio <= 0.0:
            raise ValueError(
                f"temp_text({temp_text}) 및 temp_audio({temp_audio})는 양수여야 합니다."
            )
        (
            delayed_text_logits,
            delayed_audio_logits,
            dep_target_inputs,
            dep_step_valid,
        ) = self._forward_delayed(
            codes, speaker_sim_bins=speaker_sim_bins, cfg_dropout_prob=0.0
        )
        seq_len = codes.shape[2]
        s_idx = torch.arange(seq_len, device=codes.device).view(1, 1, seq_len)

        t_delay = self.cfg.delays[0]
        t_targets_grid = dep_target_inputs[:, :1, :]
        t_mask = (
            (s_idx >= t_delay)
            & (t_targets_grid >= 0)
            & (t_targets_grid < self.cfg.text_card)
        )
        safe_t_targets = torch.where(
            t_mask, t_targets_grid, torch.zeros_like(t_targets_grid)
        )
        t_logits_scaled = (
            delayed_text_logits.float() / temp_text
            if temp_text != 1.0
            else delayed_text_logits.float()
        )
        t_logp_flat = -F.cross_entropy(
            t_logits_scaled.reshape(-1, self.cfg.text_card),
            safe_t_targets.reshape(-1),
            reduction="none",
        )
        text_log_probs = t_logp_flat.view_as(t_targets_grid).masked_fill(~t_mask, 0.0)

        audio_delays = torch.tensor(
            self.cfg.delays[1 : 1 + self.train_dep_q],
            device=codes.device,
            dtype=torch.long,
        ).view(1, self.train_dep_q, 1)
        a_targets_grid = dep_target_inputs[:, 1 : 1 + self.train_dep_q, :]
        a_mask = (
            (s_idx >= audio_delays)
            & dep_step_valid
            & (a_targets_grid >= 0)
            & (a_targets_grid < self.cfg.audio_card)
        )
        safe_a_targets = torch.where(
            a_mask, a_targets_grid, torch.zeros_like(a_targets_grid)
        )
        a_logits_scaled = (
            delayed_audio_logits.float() / temp_audio
            if temp_audio != 1.0
            else delayed_audio_logits.float()
        )
        a_logp_flat = -F.cross_entropy(
            a_logits_scaled.reshape(-1, self.cfg.audio_card),
            safe_a_targets.reshape(-1),
            reduction="none",
        )
        audio_log_probs = a_logp_flat.view_as(a_targets_grid).masked_fill(~a_mask, 0.0)

        return EDarchSequenceLogProbs(
            text_log_probs=text_log_probs,
            text_mask=t_mask,
            audio_log_probs=audio_log_probs,
            audio_mask=a_mask,
        )


def adapt_EDarch_for_korean_vocab(
    model: EDarchModel,
    new_ko_vocab_size: int,
    preserve_special_tokens: bool = True,
) -> EDarchModel:
    """사전학습된 EDarchModel의 백본과 특수 토큰 가중치를 보존하며 한국어 어휘 크기에 맞게 텍스트 레이어를 교체합니다."""
    cfg = model.cfg
    min_vocab_size = max(cfg.text_eos_id, cfg.text_pad_id) + 1
    if new_ko_vocab_size < min_vocab_size:
        raise ValueError(
            f"new_ko_vocab_size({new_ko_vocab_size})는 특수 토큰 ID(EOS={cfg.text_eos_id}, "
            f"PAD={cfg.text_pad_id})를 포함할 수 있도록 최소 {min_vocab_size} 이상이어야 합니다."
        )

    old_bos_id = cfg.text_bos_id
    old_text_emb = model.text_emb
    old_text_linear = model.text_linear
    old_dep_text_emb = model.depformer_text_emb

    cfg.text_card = new_ko_vocab_size
    new_bos_id = cfg.text_bos_id

    device = old_text_emb.emb.weight.device
    dtype = old_text_emb.emb.weight.dtype

    new_text_emb = LowRankScaledEmbedding(
        new_ko_vocab_size + 1, cfg.dim, zero_idx=cfg.zero_token_id
    ).to(device=device, dtype=dtype)

    new_text_linear = nn.Linear(cfg.dim, new_ko_vocab_size, bias=False).to(
        device=device, dtype=dtype
    )

    new_dep_text_emb = LowRankScaledEmbedding(
        new_ko_vocab_size + 1,
        cfg.depformer_dim,
        zero_idx=cfg.zero_token_id,
        low_rank=cfg.depformer_low_rank_emb,
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        for new_w, old_w in (
            (new_text_emb.emb.weight, old_text_emb.emb.weight),
            (new_text_linear.weight, old_text_linear.weight),
            (new_dep_text_emb.emb.weight, old_dep_text_emb.emb.weight),
        ):
            w_f = old_w.float()
            mean = float(w_f.mean().item())
            raw_std = (
                float(w_f.std(correction=0).item())
                if w_f.numel() <= 1
                else float(w_f.std().item())
            )
            std = 1e-4 if (math.isnan(raw_std) or raw_std < 1e-4) else raw_std
            nn.init.normal_(new_w, mean=mean, std=std)

    if preserve_special_tokens:
        with torch.no_grad():
            for tok_id in (cfg.text_eos_id, cfg.text_pad_id):
                if tok_id < old_text_emb.emb.num_embeddings:
                    new_text_emb.emb.weight[tok_id].copy_(old_text_emb.emb.weight[tok_id])
                if tok_id < old_dep_text_emb.emb.num_embeddings:
                    new_dep_text_emb.emb.weight[tok_id].copy_(old_dep_text_emb.emb.weight[tok_id])
                if tok_id < old_text_linear.out_features:
                    new_text_linear.weight[tok_id].copy_(old_text_linear.weight[tok_id])

            if old_bos_id < old_text_emb.emb.num_embeddings:
                new_text_emb.emb.weight[new_bos_id].copy_(old_text_emb.emb.weight[old_bos_id])
            if old_bos_id < old_dep_text_emb.emb.num_embeddings:
                new_dep_text_emb.emb.weight[new_bos_id].copy_(
                    old_dep_text_emb.emb.weight[old_bos_id]
                )

    if (
        old_dep_text_emb.low_rank_proj is not None
        and new_dep_text_emb.low_rank_proj is not None
    ):
        with torch.no_grad():
            new_dep_text_emb.low_rank_proj.weight.copy_(
                old_dep_text_emb.low_rank_proj.weight
            )

    model.text_emb = new_text_emb
    model.text_linear = new_text_linear
    model.depformer_text_emb = new_dep_text_emb

    return model
