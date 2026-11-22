"""오디오 파형 로드·리샘플링 유틸리티 및 내장/외부 Mimi 신경망 오디오 코덱 모듈."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from libs.config import EDarchConfig

SAMPLE_RATE = 24_000
FRAME_RATE = 12.5
SAMPLES_PER_FRAME = int(SAMPLE_RATE / FRAME_RATE)

_RESAMPLER_CACHE: dict[tuple[int, int, str, str], torchaudio.transforms.Resample] = {}


def resample_wav(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """변환기를 캐싱하여 오디오 파형 텐서를 목표 샘플링 레이트로 고속 리샘플링합니다."""
    if orig_sr == target_sr:
        return wav
    key = (orig_sr, target_sr, str(wav.dtype), str(wav.device))
    resampler = _RESAMPLER_CACHE.get(key)
    if resampler is None:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr, new_freq=target_sr, dtype=wav.dtype
        ).to(wav.device)
        _RESAMPLER_CACHE[key] = resampler
    return resampler(wav)


def load_mono_wav_24k(
    wav_path: Path,
    target_sr: int = SAMPLE_RATE,
    keep_channel_dim: bool = False,
) -> torch.Tensor:
    """오디오 파일을 읽어 모노 채널로 평균화하고 지정된 샘플링 레이트(기본 24kHz)로 변환합니다."""
    wav, sr = torchaudio.load(str(wav_path))
    mono = wav.mean(dim=0, keepdim=keep_channel_dim).float()
    return resample_wav(mono, sr, target_sr)


class CausalConv1d(nn.Module):
    """미래 샘플을 참조하지 않도록 좌측 패딩만 적용하는 인과적 1차원 합성곱 레이어입니다."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        """입력·출력 채널, 커널 크기 및 스트라이드에 따른 좌측 패딩 길이를 설정합니다."""
        super().__init__()
        self.left_pad = kernel_size - stride
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """좌측 인과 패딩 후 1차원 합성곱을 수행합니다."""
        return self.conv(F.pad(x, (self.left_pad, 0)))


class CausalConvTranspose1d(nn.Module):
    """인과적 업샘플링을 수행하는 1차원 전치 합성곱 레이어입니다."""

    def __init__(self, in_ch: int, out_ch: int, stride: int):
        """스트라이드의 2배 커널을 갖는 전치 합성곱 모듈을 초기화합니다."""
        super().__init__()
        self.stride = stride
        self.conv_tr = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=2 * stride, stride=stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """전치 합성곱 결과의 우측 초과 구간을 잘라 인과성을 유지합니다."""
        out = self.conv_tr(x)
        return out[..., : -self.stride]


class VectorQuantizerLayer(nn.Module):
    """유클리드 거리 최근접 탐색과 Straight-Through 추정기를 사용하는 단일 코드북 벡터 양자화기입니다."""

    def __init__(self, dim: int, codebook_size: int):
        """지정된 크기와 차원의 코드북 임베딩 테이블을 균등 분포로 초기화합니다."""
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def encode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """연속 잠재 벡터에서 가장 가까운 코드북 인덱스와 양자화 벡터를 반환합니다."""
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
        """코드북 인덱스 텐서를 연속 잠재 벡터로 역변환합니다."""
        return self.codebook(indices).transpose(1, 2)


class SplitResidualVectorQuantizer(nn.Module):
    """첫 번째 의미(Semantic) 코드북과 나머지 음향(Acoustic) 잔차 코드북들을 분리 양자화하는 모듈입니다."""

    def __init__(self, latent_dim: int, vq_dim: int, num_codebooks: int, bins: int):
        """의미 분기 및 음향 잔차 분기의 투영층과 벡터 양자화기들을 초기화합니다."""
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
        """잠재 표현을 다중 코드북 이산 인덱스 텐서 `[B, Q, T]`로 인코딩합니다."""
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
        """다중 코드북 인덱스 텐서 `[B, Q, T]`를 연속 잠재 표현으로 복원합니다."""
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
    """외부 가중치 다운로드 없이 파이프라인 검증 및 데모 실행을 수행하는 경량 내장 Mimi 오디오 코덱입니다."""

    def __init__(self, cfg: EDarchConfig):
        """설정된 스트라이드와 코드북 수에 맞춰 인코더, Split-RVQ, 디코더를 구성합니다."""
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
        """오디오 파형 `[B, 1, L]`을 프레임 단위로 패딩한 뒤 코드북 토큰 `[B, Q, T]`로 인코딩합니다."""
        rem = waveform.shape[-1] % self.hop_length
        if rem != 0:
            waveform = F.pad(waveform, (0, self.hop_length - rem))
        z = self.encoder(waveform)
        return self.quantizer.encode(z)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """코드북 토큰 `[B, Q, T]`를 오디오 파형 `[B, 1, L]`로 복원합니다."""
        z_q = self.quantizer.decode(codes)
        return self.decoder(z_q)


class ExternalMimiCodec(nn.Module):
    """HuggingFace transformers 또는 moshi 패키지의 사전학습 Mimi 코덱을 동일한 인터페이스로 감싸는 래퍼입니다."""

    def __init__(self, cfg: EDarchConfig, model_id: str = "kyutai/mimi"):
        """사전학습 Mimi 가중치를 로드하고 모델 설정과의 코드북·샘플레이트 정합성을 검증합니다."""
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
        """오디오 파형을 사전학습 Mimi 코덱으로 양자화하여 `[B, Q, T]` 정수 코드 텐서를 반환합니다."""
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
        """Mimi 코드북 토큰 `[B, Q, T]`를 연속 오디오 파형 텐서로 디코딩합니다."""
        ref_param = next(self.mimi.parameters(), None)
        if ref_param is not None:
            codes = codes.to(device=ref_param.device)
        codes = codes[:, : self.num_codebooks, :].long()
        if self._backend == "transformers":
            return self.mimi.decode(codes).audio_values
        return self.mimi.decode(codes)


def resolve_inference_codec(
    cfg: EDarchConfig,
    device: torch.device,
    mimi_model_id: Optional[str] = None,
    ckpt: Optional[Mapping[str, Any]] = None,
) -> nn.Module:
    """명시된 모델 ID 또는 체크포인트 메타데이터에 따라 적절한 오디오 코덱을 로드해 반환합니다."""
    if mimi_model_id is not None:
        codec: nn.Module = ExternalMimiCodec(cfg, model_id=mimi_model_id).to(device)
    elif ckpt is not None and "codec_state_dict" in ckpt:
        mini = MiniMimiCodec(cfg).to(device)
        mini.load_state_dict(ckpt["codec_state_dict"])
        codec = mini
    elif ckpt is not None and ckpt.get("codec_type") == "external_mimi":
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
    return codec.eval()
