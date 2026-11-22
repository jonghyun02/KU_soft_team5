from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EDarchConfig:
    source_lang: str = "en"
    target_lang: str = "ko"

    sample_rate: int = 24000
    frame_rate: float = 12.5
    audio_card: int = 2048
    num_codebooks_per_stream: int = 16
    codec_dim: int = 512
    codec_vq_dim: int = 256
    codec_strides: tuple[int, ...] = (8, 6, 5, 4, 2)

    text_card: int = 32000
    text_eos_id: int = 2
    text_pad_id: int = 3
    zero_token_id: int = -1
    ungenerated_token_id: int = -2

    dim: int = 2560
    num_heads: int = 20
    num_layers: int = 24
    dim_feedforward: int = 7040
    context_frames: int = 1000
    rope_max_period: float = 10000.0
    norm_eps: float = 1e-5

    depformer_dim: int = 1024
    depformer_num_heads: int = 16
    depformer_num_layers: int = 4
    depformer_dim_feedforward: int = 2048
    depformer_low_rank_emb: Optional[int] = 128

    depformer_weights_schedule: list[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8]
    )

    num_speaker_sim_bins: int = 5
    train_on_source_audio: bool = True

    def __post_init__(self) -> None:
        if self.source_lang != "en" or self.target_lang != "ko":
            raise ValueError(
                f"EDarch는 영어(en) -> 한국어(ko) 번역만 지원합니다: "
                f"{self.source_lang} -> {self.target_lang}"
            )

    @property
    def n_q(self) -> int:
        return 2 * self.num_codebooks_per_stream

    @property
    def dep_q(self) -> int:
        return self.num_codebooks_per_stream

    @property
    def total_streams(self) -> int:
        return 1 + self.n_q

    @property
    def audio_bos_id(self) -> int:
        return self.audio_card

    @property
    def text_bos_id(self) -> int:
        return self.text_card

    @property
    def delays(self) -> list[int]:
        q = self.num_codebooks_per_stream
        audio_stream_delay = [0] + [1] * (q - 1)
        return [0] + audio_stream_delay + audio_stream_delay

    @classmethod
    def EDarch_2b(cls) -> "EDarchConfig":
        return cls()

    @classmethod
    def EDarch_m_1b(cls) -> "EDarchConfig":
        return cls(
            num_codebooks_per_stream=8,
            dim=2048,
            num_heads=16,
            num_layers=16,
            dim_feedforward=5632,
            depformer_weights_schedule=list(range(8)),
        )

    @classmethod
    def from_preset(cls, preset: str) -> "EDarchConfig":
        presets = {
            "1b": cls.EDarch_m_1b,
            "2b": cls.EDarch_2b,
        }
        if preset not in presets:
            raise ValueError(f"알 수 없는 프리셋입니다: {preset} (지원: {tuple(presets.keys())})")
        return presets[preset]()

    @classmethod
    def from_dict(cls, raw_cfg: Mapping[str, Any]) -> "EDarchConfig":
        cfg_dict = dict(raw_cfg)
        if "codec_strides" in cfg_dict and isinstance(cfg_dict["codec_strides"], list):
            cfg_dict["codec_strides"] = tuple(cfg_dict["codec_strides"])
        if "depformer_weights_schedule" in cfg_dict and isinstance(
            cfg_dict["depformer_weights_schedule"], tuple
        ):
            cfg_dict["depformer_weights_schedule"] = list(
                cfg_dict["depformer_weights_schedule"]
            )
        return cls(**cfg_dict)


class RMSNormFP32(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = (x_fp32 * rms).to(dtype=x.dtype)
        return normed * self.weight.to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_period: float = 10000.0):
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
        seq_len = q.shape[2]
        positions = torch.arange(
            offset, offset + seq_len, device=q.device, dtype=torch.float32
        )
        angles = torch.outer(positions, self.freqs)
        cos = torch.cos(angles)[None, None, :, :]
        sin = torch.sin(angles)[None, None, :, :]

        def _rotate(x: torch.Tensor) -> torch.Tensor:
            x_f = x.float()
            x1 = x_f[..., 0::2]
            x2 = x_f[..., 1::2]
            out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            return out.flatten(-2).to(dtype=x.dtype)

        return _rotate(q), _rotate(k)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class LowRankScaledEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        zero_idx: int = -1,
        low_rank: Optional[int] = None,
    ):
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
        is_zero = token_ids == self.zero_token_value
        safe_ids = torch.where(is_zero, torch.zeros_like(token_ids), token_ids)
        out = self.emb(safe_ids)
        if self.low_rank_proj is not None:
            out = self.low_rank_proj(out)
        return out.masked_fill(is_zero.unsqueeze(-1), 0.0)

    @property
    def zero_token_value(self) -> int:
        return self.zero_idx


def _build_schedule_runs(schedule: Optional[list[int]]) -> list[tuple[int, int, int]]:
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


class StepIndexedLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, schedule: Optional[list[int]] = None
    ):
        super().__init__()
        if schedule is not None and len(schedule) == 0:
            raise ValueError("schedule은 빈 리스트일 수 없습니다.")
        self.schedule = schedule
        self._runs = _build_schedule_runs(schedule)
        num_weights = (max(schedule) + 1) if schedule is not None else 1
        self.linears = nn.ModuleList(
            [nn.Linear(in_features, out_features, bias=False) for _ in range(num_weights)]
        )

    def forward(
        self, x: torch.Tensor, step_idx: Optional[int] = None
    ) -> torch.Tensor:
        if self.schedule is None:
            return self.linears[0](x)

        if step_idx is not None:
            if step_idx < 0 or step_idx >= len(self.schedule):
                raise ValueError(
                    f"step_idx({step_idx})가 schedule 범위(0..{len(self.schedule) - 1})를 벗어났습니다."
                )
            return self.linears[self.schedule[step_idx]](x)

        seq_len = x.shape[1]
        if seq_len > len(self.schedule):
            raise ValueError(
                f"입력 길이({seq_len})가 schedule 길이({len(self.schedule)})를 초과합니다."
            )
        outputs: list[torch.Tensor] = []
        for start, end, w_idx in self._runs:
            if start >= seq_len:
                break
            seg_end = min(end, seq_len)
            outputs.append(self.linears[w_idx](x[:, start:seg_end, :]))
        if len(outputs) == 1:
            return outputs[0]
        return torch.cat(outputs, dim=1)


class StepIndexedSwiGLU(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int, schedule: Optional[list[int]] = None
    ):
        super().__init__()
        if schedule is not None and len(schedule) == 0:
            raise ValueError("schedule은 빈 리스트일 수 없습니다.")
        self.schedule = schedule
        self._runs = _build_schedule_runs(schedule)
        num_weights = (max(schedule) + 1) if schedule is not None else 1
        self.ffns = nn.ModuleList(
            [SwiGLUFFN(dim, hidden_dim) for _ in range(num_weights)]
        )

    def forward(
        self, x: torch.Tensor, step_idx: Optional[int] = None
    ) -> torch.Tensor:
        if self.schedule is None:
            return self.ffns[0](x)

        if step_idx is not None:
            if step_idx < 0 or step_idx >= len(self.schedule):
                raise ValueError(
                    f"step_idx({step_idx})가 schedule 범위(0..{len(self.schedule) - 1})를 벗어났습니다."
                )
            return self.ffns[self.schedule[step_idx]](x)

        seq_len = x.shape[1]
        if seq_len > len(self.schedule):
            raise ValueError(
                f"입력 길이({seq_len})가 schedule 길이({len(self.schedule)})를 초과합니다."
            )
        outputs: list[torch.Tensor] = []
        for start, end, w_idx in self._runs:
            if start >= seq_len:
                break
            seg_end = min(end, seq_len)
            outputs.append(self.ffns[w_idx](x[:, start:seg_end, :]))
        if len(outputs) == 1:
            return outputs[0]
        return torch.cat(outputs, dim=1)


@dataclass
class LayerKVCache:
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    offset: int = 0


class StreamingTransformerBlock(nn.Module):
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
        return [LayerKVCache() for _ in range(len(self.layers))]

    def forward(
        self,
        x: torch.Tensor,
        kv_caches: Optional[list[LayerKVCache]] = None,
        step_idx: Optional[int] = None,
    ) -> torch.Tensor:
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


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.left_pad = kernel_size - stride
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_pad, 0)))


class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv_tr = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=2 * stride, stride=stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_tr(x)
        return out[..., : -self.stride]


class VectorQuantizerLayer(nn.Module):
    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def encode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_t = z.transpose(1, 2)
        cb = self.codebook.weight
        dist = (
            z_t.pow(2).sum(dim=-1, keepdim=True)
            - 2.0 * torch.matmul(z_t, cb.t())
            + cb.pow(2).sum(dim=-1)
        )
        indices = dist.argmin(dim=-1)
        quantized = self.codebook(indices).transpose(1, 2)
        quantized_st = z + (quantized - z).detach()
        return indices, quantized_st

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return self.codebook(indices).transpose(1, 2)


class SplitResidualVectorQuantizer(nn.Module):
    def __init__(self, latent_dim: int, vq_dim: int, num_codebooks: int, bins: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.sem_in = nn.Conv1d(latent_dim, vq_dim, kernel_size=1)
        self.sem_out = nn.Conv1d(vq_dim, latent_dim, kernel_size=1)
        self.sem_vq = VectorQuantizerLayer(vq_dim, bins)

        self.ac_in = nn.Conv1d(latent_dim, vq_dim, kernel_size=1)
        self.ac_out = nn.Conv1d(vq_dim, latent_dim, kernel_size=1)
        self.ac_vqs = nn.ModuleList(
            [VectorQuantizerLayer(vq_dim, bins) for _ in range(num_codebooks - 1)]
        )

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        codes = []
        sem_idx, _ = self.sem_vq.encode(self.sem_in(z))
        codes.append(sem_idx)

        residual = self.ac_in(z)
        for vq in self.ac_vqs:
            idx, q_vec = vq.encode(residual)
            codes.append(idx)
            residual = residual - q_vec
        return torch.stack(codes, dim=1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        z_sem = self.sem_out(self.sem_vq.decode(codes[:, 0, :]))
        z_ac_inner = torch.zeros(
            codes.shape[0],
            self.ac_in.out_channels,
            codes.shape[2],
            device=codes.device,
            dtype=z_sem.dtype,
        )
        for q_idx, vq in enumerate(self.ac_vqs, start=1):
            if q_idx < codes.shape[1]:
                z_ac_inner = z_ac_inner + vq.decode(codes[:, q_idx, :])
        return z_sem + self.ac_out(z_ac_inner)


class MiniMimiCodec(nn.Module):
    def __init__(self, cfg: EDarchConfig):
        super().__init__()
        self.hop_length = math.prod(cfg.codec_strides)

        enc_layers: list[nn.Module] = [CausalConv1d(1, cfg.codec_dim, kernel_size=7)]
        for s in cfg.codec_strides:
            enc_layers.extend([nn.ELU(), CausalConv1d(cfg.codec_dim, cfg.codec_dim, 2 * s, s)])
        self.encoder = nn.Sequential(*enc_layers)

        self.quantizer = SplitResidualVectorQuantizer(
            latent_dim=cfg.codec_dim,
            vq_dim=cfg.codec_vq_dim,
            num_codebooks=cfg.num_codebooks_per_stream,
            bins=cfg.audio_card,
        )

        dec_layers: list[nn.Module] = []
        for s in reversed(cfg.codec_strides):
            dec_layers.extend([nn.ELU(), CausalConvTranspose1d(cfg.codec_dim, cfg.codec_dim, s)])
        dec_layers.extend([nn.ELU(), CausalConv1d(cfg.codec_dim, 1, kernel_size=3), nn.Tanh()])
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        rem = waveform.shape[-1] % self.hop_length
        if rem != 0:
            waveform = F.pad(waveform, (0, self.hop_length - rem))
        z = self.encoder(waveform)
        return self.quantizer.encode(z)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        z_q = self.quantizer.decode(codes)
        return self.decoder(z_q)


class ExternalMimiCodec(nn.Module):
    def __init__(self, cfg: EDarchConfig, model_id: str = "kyutai/mimi"):
        super().__init__()
        if cfg.num_codebooks_per_stream < 1:
            raise ValueError(
                f"num_codebooks_per_stream({cfg.num_codebooks_per_stream})은 1 이상이어야 합니다."
            )
        self.hop_length = int(round(cfg.sample_rate / cfg.frame_rate))
        self.num_codebooks = cfg.num_codebooks_per_stream
        self.model_id = model_id
        self._backend: str = ""
        try:
            from transformers import MimiModel

            self.mimi = MimiModel.from_pretrained(model_id)
            self._backend = "transformers"
        except Exception as tf_exc:
            try:
                from huggingface_hub import hf_hub_download
                from moshi.models import loaders

                mimi_weight = hf_hub_download(model_id, loaders.MIMI_NAME)
                mimi = loaders.get_mimi(mimi_weight, device="cpu")
                mimi.set_num_codebooks(self.num_codebooks)
                self.mimi = mimi
                self._backend = "moshi"
            except Exception as moshi_exc:
                raise RuntimeError(
                    f"사전학습 Mimi 코덱({model_id}) 로드에 실패했습니다. "
                    f"transformers 오류: {tf_exc} | moshi 오류: {moshi_exc}"
                ) from tf_exc

        if self._backend == "transformers":
            mimi_cfg = getattr(self.mimi, "config", None)
            codebook_size = int(getattr(mimi_cfg, "codebook_size", 2048))
            max_q = getattr(mimi_cfg, "num_quantizers", None)
            mimi_sr = getattr(mimi_cfg, "sampling_rate", None)
        else:
            codebook_size = int(getattr(self.mimi, "cardinality", 2048))
            max_q = getattr(self.mimi, "total_codebooks", None)
            mimi_sr = getattr(self.mimi, "sample_rate", None)

        if cfg.audio_card < codebook_size:
            raise ValueError(
                f"모델 audio_card({cfg.audio_card})가 외부 Mimi 코덱({model_id})의 "
                f"codebook_size({codebook_size})보다 작습니다. 1b 또는 2b 프리셋(audio_card=2048)을 사용하세요."
            )
        if max_q is not None and self.num_codebooks > int(max_q):
            raise ValueError(
                f"요청한 코드북 수({self.num_codebooks})가 외부 Mimi 코덱({model_id})의 "
                f"최대 코드북 수({max_q})를 초과합니다."
            )
        if mimi_sr is not None and int(mimi_sr) != cfg.sample_rate:
            raise ValueError(
                f"모델 sample_rate({cfg.sample_rate})와 외부 Mimi 코덱 sample_rate({mimi_sr})가 불일치합니다."
            )

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        ref_param = next(self.mimi.parameters(), None)
        if ref_param is not None:
            waveform = waveform.to(device=ref_param.device, dtype=ref_param.dtype)
        rem = waveform.shape[-1] % self.hop_length
        if rem != 0:
            waveform = F.pad(waveform, (0, self.hop_length - rem))
        if self._backend == "transformers":
            out = self.mimi.encode(waveform, num_quantizers=self.num_codebooks)
            return out.audio_codes.long()
        return self.mimi.encode(waveform)[:, : self.num_codebooks, :].long()

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        ref_param = next(self.mimi.parameters(), None)
        if ref_param is not None:
            codes = codes.to(device=ref_param.device)
        codes = codes[:, : self.num_codebooks, :].long()
        if self._backend == "transformers":
            return self.mimi.decode(codes).audio_values
        return self.mimi.decode(codes)


def delay_sequence(
    delays: list[int], codes: torch.Tensor, initial_tokens: torch.Tensor
) -> torch.Tensor:
    bsz, _, seq_len = codes.shape
    delayed = torch.empty_like(codes)
    for k, d in enumerate(delays):
        if d == 0:
            delayed[:, k, :] = codes[:, k, :]
        elif d < seq_len:
            pad = initial_tokens[:, k, :].expand(bsz, d)
            delayed[:, k, :] = torch.cat([pad, codes[:, k, :-d]], dim=-1)
        else:
            delayed[:, k, :] = initial_tokens[:, k, :].expand(bsz, seq_len)
    return delayed


def undelay_sequence(
    delays: list[int], logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
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


class SpeakerSimilarityConditioner(nn.Module):
    LABELS = ("very_bad", "bad", "neutral", "good", "very_good")

    def __init__(self, num_bins: int, out_dim: int):
        super().__init__()
        self.null_idx = num_bins
        self.lut = nn.Embedding(num_bins + 1, out_dim)
        nn.init.normal_(self.lut.weight, mean=0.0, std=0.02)

    def forward(self, bin_ids: torch.Tensor) -> torch.Tensor:
        return self.lut(bin_ids).unsqueeze(1)


@dataclass
class EDarchForwardOutput:
    text_logits: torch.Tensor
    text_mask: torch.Tensor
    audio_logits: torch.Tensor
    audio_mask: torch.Tensor


@dataclass
class EDarchSequenceLogProbs:
    text_log_probs: torch.Tensor
    text_mask: torch.Tensor
    audio_log_probs: torch.Tensor
    audio_mask: torch.Tensor


class EDarchModel(nn.Module):
    def __init__(self, cfg: EDarchConfig):
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
        self._suppress_early_eos = enabled

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.temporal_transformer.gradient_checkpointing = enabled
        self.depformer.gradient_checkpointing = enabled

    def forward_temporal(
        self,
        seq_codes: torch.Tensor,
        cond_emb: Optional[torch.Tensor] = None,
        kv_caches: Optional[list[LayerKVCache]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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


def sample_top_k(
    logits: torch.Tensor, temperature: float = 0.8, top_k: int = 250
) -> torch.Tensor:
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
    def __init__(
        self,
        model: EDarchModel,
        batch_size: int = 1,
        cfg_coef: float = 3.0,
        temp_audio: float = 0.8,
        temp_text: float = 0.4,
        top_k_audio: int = 250,
        top_k_text: int = 50,
    ):
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

        very_good_id = self.cfg.num_speaker_sim_bins - 1
        cond_bins = torch.full(
            (batch_size,), very_good_id, dtype=torch.long, device=device
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
        self.ring_cache.fill_(self.cfg.ungenerated_token_id)
        self.step_offset = 0
        self.temporal_kv = self.model.temporal_transformer.init_kv_caches()

    @torch.no_grad()
    def step(
        self,
        source_audio_tokens: torch.Tensor,
        suppress_eos: Optional[bool] = None,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
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


def run_bottom_up_verification(preset: str = "1b") -> None:
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
        f"[1] EDarchModel('{preset}', {cfg.source_lang}->{cfg.target_lang}, ko_vocab={cfg.text_card}) 초기화 완료 "
        f"(Device: {device}, dtype: {model_dtype}, 파라미터 수: {total_params:,})"
    )

    batch_size = 1
    num_frames = 10
    num_samples = num_frames * codec.hop_length
    en_wav = torch.randn(batch_size, 1, num_samples, device=device)
    ko_wav = torch.randn(batch_size, 1, num_samples, device=device)

    with torch.no_grad():
        en_codes = codec.encode(en_wav)
        ko_codes = codec.encode(ko_wav)
    print(f"[2] 영/한 Mimi 인코딩 완료: 영어 파형 {tuple(en_wav.shape)} -> RVQ 토큰 {tuple(en_codes.shape)}")

    ko_text = torch.randint(4, cfg.text_card, (batch_size, 1, num_frames), device=device)
    ko_text[:, :, :3] = cfg.text_pad_id

    full_codes = torch.cat([ko_text, ko_codes, en_codes], dim=1)
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
        f"[3] 영->한 학습 1스텝 역전파 성공: total_loss={metrics['loss']:.4f} "
        f"(ko_text={metrics['text_loss']:.4f}, audio={metrics['audio_loss']:.4f})"
    )

    model.eval()
    model.apply_early_eos_suppression_()
    streamer = EDarchStreamingGenerator(model, batch_size=1, cfg_coef=3.0)

    emitted_ko_texts: list[int] = []
    emitted_ko_audio_frames: list[torch.Tensor] = []

    for t in range(num_frames + streamer.max_delay):
        t_clamped = min(t, num_frames - 1)
        en_chunk_codes = en_codes[:1, :, t_clamped : t_clamped + 1]
        with torch.amp.autocast(device_type=device.type, dtype=model_dtype, enabled=use_cuda):
            step_out = streamer.step(en_chunk_codes)
        if step_out is not None:
            t_tok, a_tok = step_out
            emitted_ko_texts.append(int(t_tok[0, 0].item()))
            emitted_ko_audio_frames.append(a_tok)

    pred_ko_audio_codes = torch.cat(emitted_ko_audio_frames, dim=-1)
    with torch.no_grad():
        reconstructed_ko_wav = codec.decode(pred_ko_audio_codes)

    print("[4] 영->한 스트리밍 추론 완료 (CFG=3.0):")
    print(f"    - 예측된 한국어 Inner Monologue 토큰 시퀀스 ({len(emitted_ko_texts)}프레임): {emitted_ko_texts}")
    print(
        f"    - 복원된 한국어 음성 파형 크기: {tuple(reconstructed_ko_wav.shape)} "
        f"(기대 길이: (1, 1, {num_samples}))"
    )


if __name__ == "__main__":
    run_bottom_up_verification()
