"""노래 번역 계층형 트랜스포머를 구성하는 신경망 레이어, 음절·섹션 조건화기, KV 캐시 및 지연 정렬 모듈."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNormFP32(nn.Module):
    """혼합 정밀도 환경에서도 FP32 정밀도로 제곱평균제곱근 정규화를 수행하는 레이어입니다."""

    def __init__(self, dim: int, eps: float = 1e-5):
        """정규화 차원 크기와 수치 안정성 입실론을 초기화합니다."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력 텐서의 마지막 차원에 대해 FP32 RMS 정규화 및 스케일 가중치를 적용합니다."""
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = (x_fp32 * rms).to(dtype=x.dtype)
        return normed * self.weight.to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    """스트리밍 KV 캐시 오프셋을 지원하는 회전 위치 임베딩(RoPE) 모듈입니다."""

    def __init__(self, head_dim: int, max_period: float = 10000.0):
        """어텐션 헤드 차원(짝수)에 대한 주파수 버퍼를 생성합니다."""
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim은 짝수여야 합니다: {head_dim}")
        half = head_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def apply_rotary(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """쿼리(Q)와 키(K) 텐서에 위치 오프셋이 반영된 회전 변환을 적용합니다."""
        seq_len = q.shape[2]
        positions = torch.arange(
            offset, offset + seq_len, device=q.device, dtype=torch.float32
        )
        angles = torch.outer(positions, self.freqs)
        cos = torch.cos(angles)[None, None, :, :]
        sin = torch.sin(angles)[None, None, :, :]

        def _rotate(x: torch.Tensor) -> torch.Tensor:
            """짝수·홀수 채널 쌍에 2차원 평면 회전을 적용합니다."""
            x_f = x.float()
            x1 = x_f[..., 0::2]
            x2 = x_f[..., 1::2]
            out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            return out.flatten(-2).to(dtype=x.dtype)

        return _rotate(q), _rotate(k)


class SwiGLUFFN(nn.Module):
    """게이트 선형 유닛과 SiLU 활성화를 결합한 피드포워드 네트워크입니다."""

    def __init__(self, dim: int, hidden_dim: int):
        """입력·은닉·출력 선형 투영 레이어를 바이어스 없이 초기화합니다."""
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU 순전파 연산을 수행합니다."""
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class LowRankScaledEmbedding(nn.Module):
    """패딩 마스킹 토큰(`zero_idx`) 영벡터 처리와 선택적 저계수(Low-Rank) 투영을 지원하는 임베딩 레이어입니다."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        zero_idx: int = -1,
        low_rank: Optional[int] = None,
    ):
        """임베딩 테이블과 선택적 저계수 선형 투영 행렬을 초기화합니다."""
        super().__init__()
        self.zero_idx = zero_idx
        inner_dim = low_rank if low_rank is not None else embedding_dim
        self.emb = nn.Embedding(num_embeddings, inner_dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=1.0 / math.sqrt(inner_dim))
        self.low_rank_proj = (
            nn.Linear(inner_dim, embedding_dim, bias=False)
            if low_rank is not None
            else None
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """토큰 ID 텐서를 임베딩 벡터로 변환하되 `zero_idx` 위치는 0 벡터로 마스킹합니다."""
        is_zero = token_ids == self.zero_token_value
        safe_ids = torch.where(is_zero, torch.zeros_like(token_ids), token_ids)
        out = self.emb(safe_ids)
        if self.low_rank_proj is not None:
            out = self.low_rank_proj(out)
        return out.masked_fill(is_zero.unsqueeze(-1), 0.0)

    @property
    def zero_token_value(self) -> int:
        """영벡터로 마스킹되는 특수 토큰 ID 값을 반환합니다."""
        return self.zero_idx


def _build_schedule_runs(schedule: Optional[list[int]]) -> list[tuple[int, int, int]]:
    """가중치 공유 스케줄 리스트에서 동일한 가중치 인덱스가 연속되는 구간들을 압축 추출합니다."""
    if not schedule:
        return []
    runs: list[tuple[int, int, int]] = []
    run_start = 0
    cur_w = schedule[0]
    for idx in range(1, len(schedule)):
        if schedule[idx] != cur_w:
            runs.append((run_start, idx, cur_w))
            run_start = idx
            cur_w = schedule[idx]
    runs.append((run_start, len(schedule), cur_w))
    return runs


def _init_step_schedule(
    schedule: Optional[list[int]],
) -> tuple[Optional[list[int]], list[tuple[int, int, int]], int]:
    """스텝 스케줄의 유효성을 검증하고 연속 실행 구간 리스트와 필요한 가중치 개수를 반환합니다."""
    if schedule is not None and len(schedule) == 0:
        raise ValueError("schedule은 빈 리스트일 수 없습니다.")
    runs = _build_schedule_runs(schedule)
    num_weights = (max(schedule) + 1) if schedule is not None else 1
    return schedule, runs, num_weights


def _forward_step_indexed(
    modules: nn.ModuleList,
    schedule: Optional[list[int]],
    runs: list[tuple[int, int, int]],
    x: torch.Tensor,
    step_idx: Optional[int] = None,
) -> torch.Tensor:
    """단일 스텝(`step_idx`) 또는 전체 스텝 시퀀스에 대해 스케줄된 모듈 리스트를 적용합니다."""
    if schedule is None:
        return modules[0](x)

    if step_idx is not None:
        if step_idx < 0 or step_idx >= len(schedule):
            raise ValueError(
                f"step_idx({step_idx})가 schedule 범위(0..{len(schedule) - 1})를 벗어났습니다."
            )
        return modules[schedule[step_idx]](x)

    seq_len = x.shape[1]
    if seq_len > len(schedule):
        raise ValueError(
            f"입력 길이({seq_len})가 schedule 길이({len(schedule)})를 초과합니다."
        )
    outputs: list[torch.Tensor] = []
    for start, end, w_idx in runs:
        if start >= seq_len:
            break
        seg_end = min(end, seq_len)
        outputs.append(modules[w_idx](x[:, start:seg_end, :]))
    if len(outputs) == 1:
        return outputs[0]
    return torch.cat(outputs, dim=1)


class StepIndexedLinear(nn.Module):
    """Depformer 코드북 스텝 스케줄에 따라 가중치를 공유하거나 분리 적용하는 선형 레이어입니다."""

    def __init__(
        self, in_features: int, out_features: int, schedule: Optional[list[int]] = None
    ):
        """스케줄에서 요구하는 고유 가중치 개수만큼 선형 모듈 리스트를 생성합니다."""
        super().__init__()
        self.schedule, self._runs, num_weights = _init_step_schedule(schedule)
        self.linears = nn.ModuleList(
            [nn.Linear(in_features, out_features, bias=False) for _ in range(num_weights)]
        )

    def forward(
        self, x: torch.Tensor, step_idx: Optional[int] = None
    ) -> torch.Tensor:
        """단일 스텝(`step_idx`) 또는 전체 스텝 시퀀스에 대해 스케줄된 선형 변환을 적용합니다."""
        return _forward_step_indexed(
            self.linears, self.schedule, self._runs, x, step_idx=step_idx
        )


class StepIndexedSwiGLU(nn.Module):
    """Depformer 코드북 스텝 스케줄에 따라 가중치를 공유하거나 분리 적용하는 SwiGLU FFN입니다."""

    def __init__(
        self, dim: int, hidden_dim: int, schedule: Optional[list[int]] = None
    ):
        """스케줄에서 요구하는 고유 가중치 개수만큼 SwiGLUFFN 모듈 리스트를 생성합니다."""
        super().__init__()
        self.schedule, self._runs, num_weights = _init_step_schedule(schedule)
        self.ffns = nn.ModuleList(
            [SwiGLUFFN(dim, hidden_dim) for _ in range(num_weights)]
        )

    def forward(
        self, x: torch.Tensor, step_idx: Optional[int] = None
    ) -> torch.Tensor:
        """단일 스텝(`step_idx`) 또는 전체 스텝 시퀀스에 대해 스케줄된 SwiGLU FFN을 적용합니다."""
        return _forward_step_indexed(
            self.ffns, self.schedule, self._runs, x, step_idx=step_idx
        )


@dataclass
class LayerKVCache:
    """단일 트랜스포머 블록의 스트리밍 추론용 Key/Value 텐서와 누적 위치 오프셋을 보관합니다."""

    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    offset: int = 0


class StreamingTransformerBlock(nn.Module):
    """슬라이딩 윈도우 인과 어텐션, RoPE, KV 캐시 및 스텝 스케줄 가중치를 통합 지원하는 트랜스포머 블록입니다."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_feedforward: int,
        use_rope: bool = True,
        rope_max_period: float = 10000.0,
        max_context: Optional[int] = None,
        step_schedule: Optional[list[int]] = None,
        norm_eps: float = 1e-5,
    ):
        """어텐션 투영층, RoPE, 정규화층 및 SwiGLU FFN을 초기화합니다."""
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim({dim})이 num_heads({num_heads})로 나누어떨어지지 않습니다.")
        if max_context is not None and max_context < 1:
            raise ValueError(f"max_context({max_context})는 1 이상이어야 합니다.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_context = max_context

        self.norm1 = RMSNormFP32(dim, eps=norm_eps)
        self.q_proj = StepIndexedLinear(dim, dim, schedule=step_schedule)
        self.k_proj = StepIndexedLinear(dim, dim, schedule=step_schedule)
        self.v_proj = StepIndexedLinear(dim, dim, schedule=step_schedule)
        self.o_proj = StepIndexedLinear(dim, dim, schedule=step_schedule)
        self.rope = (
            RotaryEmbedding(self.head_dim, max_period=rope_max_period)
            if use_rope
            else None
        )

        self.norm2 = RMSNormFP32(dim, eps=norm_eps)
        self.ffn = StepIndexedSwiGLU(dim, dim_feedforward, schedule=step_schedule)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[LayerKVCache] = None,
        step_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """인과 마스크 및 KV 캐시를 갱신하며 멀티헤드 셀프 어텐션과 FFN 연산을 수행합니다."""
        bsz, seq_len, _ = x.shape
        h_in = self.norm1(x)

        q = (
            self.q_proj(h_in, step_idx=step_idx)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(h_in, step_idx=step_idx)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(h_in, step_idx=step_idx)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        offset = kv_cache.offset if kv_cache is not None else 0
        if self.rope is not None:
            q, k = self.rope.apply_rotary(q, k, offset=offset)

        if kv_cache is not None:
            if kv_cache.k is None:
                k_cat, v_cat = k, v
            else:
                k_cat = torch.cat([kv_cache.k, k], dim=2)
                v_cat = torch.cat([kv_cache.v, v], dim=2)

            if self.max_context is not None and k_cat.shape[2] > self.max_context:
                kv_cache.k = k_cat[:, :, -self.max_context :, :]
                kv_cache.v = v_cat[:, :, -self.max_context :, :]
            else:
                kv_cache.k, kv_cache.v = k_cat, v_cat
            kv_cache.offset += seq_len

            if seq_len == 1:
                k_full, v_full = kv_cache.k, kv_cache.v
                attn_mask = None
                use_is_causal = False
            else:
                k_full, v_full = k_cat, v_cat
                k_len = k_full.shape[2]
                q_pos = torch.arange(k_len - seq_len, k_len, device=x.device)
                k_pos = torch.arange(k_len, device=x.device)
                causal = q_pos[:, None] >= k_pos[None, :]
                if self.max_context is not None:
                    causal = causal & ((q_pos[:, None] - k_pos[None, :]) < self.max_context)
                attn_mask = causal
                use_is_causal = False
        else:
            k_full, v_full = k, v
            if self.max_context is None or seq_len <= self.max_context:
                attn_mask = None
                use_is_causal = True
            else:
                idx = torch.arange(seq_len, device=x.device)
                causal = (idx[:, None] >= idx[None, :]) & (
                    (idx[:, None] - idx[None, :]) < self.max_context
                )
                attn_mask = causal
                use_is_causal = False

        attn_out = F.scaled_dot_product_attention(
            q, k_full, v_full, attn_mask=attn_mask, is_causal=use_is_causal
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        x = x + self.o_proj(attn_out, step_idx=step_idx)
        x = x + self.ffn(self.norm2(x), step_idx=step_idx)
        return x


class StreamingTransformerStack(nn.Module):
    """다수의 StreamingTransformerBlock을 적층하고 Activation Checkpointing을 지원하는 스택 모듈입니다."""

    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_heads: int,
        dim_feedforward: int,
        use_rope: bool,
        rope_max_period: float = 10000.0,
        max_context: Optional[int] = None,
        step_schedule: Optional[list[int]] = None,
        norm_eps: float = 1e-5,
    ):
        """지정된 층수만큼 트랜스포머 블록을 초기화합니다."""
        super().__init__()
        self.layers = nn.ModuleList(
            [
                StreamingTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    use_rope=use_rope,
                    rope_max_period=rope_max_period,
                    max_context=max_context,
                    step_schedule=step_schedule,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.gradient_checkpointing: bool = False

    def init_kv_caches(self) -> list[LayerKVCache]:
        """각 트랜스포머 층에 대응하는 빈 LayerKVCache 리스트를 생성합니다."""
        return [LayerKVCache() for _ in range(len(self.layers))]

    def forward(
        self,
        x: torch.Tensor,
        kv_caches: Optional[list[LayerKVCache]] = None,
        step_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """모든 트랜스포머 층을 순차적으로 통과시키며 필요 시 그래디언트 체크포인팅을 적용합니다."""
        use_ckpt = (
            self.gradient_checkpointing
            and self.training
            and kv_caches is None
            and x.requires_grad
        )
        for idx, layer in enumerate(self.layers):
            if use_ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, None, step_idx, use_reentrant=False
                )
            else:
                cache = kv_caches[idx] if kv_caches is not None else None
                x = layer(x, kv_cache=cache, step_idx=step_idx)
        return x


class SyllableConditioner(nn.Module):
    """arXiv:2309.11093의 행별 목표 음절 수(1..max_syllables 및 비조건부 0)를 조건 임베딩으로 변환합니다."""

    def __init__(self, max_syllables: int, out_dim: int):
        """최대 음절 수와 비조건부(null=0) 인덱스를 포함한 음절 조건 임베딩 테이블을 초기화합니다."""
        super().__init__()
        self.max_syllables = max_syllables
        self.null_idx = 0
        self.lut = nn.Embedding(max_syllables + 2, out_dim)
        nn.init.normal_(self.lut.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.lut.weight[0].zero_()

    def forward(self, syllable_counts: torch.Tensor) -> torch.Tensor:
        """목표 음절 수 텐서(`[B]` 또는 `[B, T]`)를 `[B, 1, D]` 또는 `[B, T, D]` 임베딩으로 변환합니다."""
        clamped = syllable_counts.clamp(0, self.max_syllables + 1)
        if clamped.ndim == 1:
            return self.lut(clamped).unsqueeze(1)
        return self.lut(clamped)


class SectionStructureConditioner(nn.Module):
    """섹션 종류(Verse/Chorus/Bridge), 행 반복 그룹 ID, 보컬 음색 유사도 구간을 결합하는 조건화 레이어입니다."""

    LABELS = ("very_bad", "bad", "neutral", "good", "very_good")

    def __init__(
        self,
        num_section_types: int,
        max_repeat_groups: int,
        num_speaker_sim_bins: int,
        out_dim: int,
    ):
        """섹션 타입, 반복 그룹, 보컬 음색 유사도 각각의 임베딩 룩업 테이블을 초기화합니다."""
        super().__init__()
        self.num_section_types = num_section_types
        self.max_repeat_groups = max_repeat_groups
        self.num_speaker_sim_bins = num_speaker_sim_bins
        self.null_idx = num_speaker_sim_bins

        self.sim_lut = nn.Embedding(num_speaker_sim_bins + 1, out_dim)
        self.sec_lut = nn.Embedding(num_section_types + 1, out_dim)
        self.rep_lut = nn.Embedding(max_repeat_groups + 1, out_dim)

        nn.init.normal_(self.sim_lut.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.sec_lut.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.rep_lut.weight, mean=0.0, std=0.02)

    def forward(
        self,
        sim_bin_ids: torch.Tensor,
        section_type_ids: Optional[torch.Tensor] = None,
        repeat_group_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """보컬 유사도·섹션 타입·반복 행 구조를 합산한 조건 임베딩(`[B, 1, D]` 또는 `[B, T, D]`)을 반환합니다."""
        sim_clamped = sim_bin_ids.clamp(0, self.null_idx)
        out = self.sim_lut(sim_clamped).unsqueeze(1)

        if section_type_ids is not None:
            sec_clamped = section_type_ids.clamp(0, self.num_section_types)
            sec_emb = self.sec_lut(sec_clamped)
            if sec_emb.ndim == 2:
                sec_emb = sec_emb.unsqueeze(1)
            out = out + sec_emb

        if repeat_group_ids is not None:
            rep_clamped = repeat_group_ids.clamp(0, self.max_repeat_groups)
            rep_emb = self.rep_lut(rep_clamped)
            if rep_emb.ndim == 2:
                rep_emb = rep_emb.unsqueeze(1)
            out = out + rep_emb

        return out


def delay_sequence(
    delays: list[int],
    codes: torch.Tensor,
    initial_tokens: torch.Tensor,
    zero_token_id: int = -1,
) -> torch.Tensor:
    """각 스트림의 지정된 프레임 지연(`delays`)만큼 시퀀스를 우측으로 밀고 초기 토큰으로 앞을 채웁니다."""
    bsz, _, seq_len = codes.shape
    delayed = torch.empty_like(codes)
    for k, d in enumerate(delays):
        if d == 0:
            delayed[:, k, :] = codes[:, k, :]
        else:
            bos_pad = initial_tokens[:, k, :]
            if seq_len > 0:
                is_masked_stream = codes[:, k, :1] == zero_token_id
                pad_tok = torch.where(
                    is_masked_stream,
                    torch.full_like(bos_pad, zero_token_id),
                    bos_pad,
                )
            else:
                pad_tok = bos_pad
            if d < seq_len:
                pad = pad_tok.expand(bsz, d)
                delayed[:, k, :] = torch.cat([pad, codes[:, k, :-d]], dim=-1)
            else:
                delayed[:, k, :] = pad_tok.expand(bsz, seq_len)
    return delayed


def undelay_sequence(
    delays: list[int], logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """지연된 스트림별 출력 텐서를 원래 시간축으로 재정렬하고 유효 프레임 마스크를 함께 반환합니다."""
    bsz, k_sub, seq_len, _ = logits.shape
    if all(d == 0 for d in delays):
        valid_mask = torch.ones(
            (bsz, k_sub, seq_len), dtype=torch.bool, device=logits.device
        )
        return logits, valid_mask

    realigned = torch.zeros_like(logits)
    valid_mask = torch.zeros((bsz, k_sub, seq_len), dtype=torch.bool, device=logits.device)
    for k, d in enumerate(delays):
        if d == 0:
            realigned[:, k, :, :] = logits[:, k, :, :]
            valid_mask[:, k, :] = True
        elif d < seq_len:
            valid_len = seq_len - d
            realigned[:, k, :valid_len, :] = logits[:, k, d:, :]
            valid_mask[:, k, :valid_len] = True
    return realigned, valid_mask
