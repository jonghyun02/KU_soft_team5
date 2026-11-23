"""보컬 오디오 파형 부분 Seek 로드·리샘플링 유틸리티 및 내장/외부 Mimi 신경망 오디오 코덱 모듈."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Optional
import wave

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import torchaudio
except ImportError:
    torchaudio = None

from libs.config import LyricTransConfig

SAMPLE_RATE = 24_000
FRAME_RATE = 12.5
SAMPLES_PER_FRAME = int(SAMPLE_RATE / FRAME_RATE)

_RESAMPLER_CACHE: dict[tuple[int, int, str, str], Any] = {}


def resample_wav(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """변환기를 캐싱하거나 선형 보간을 사용하여 오디오 파형 텐서를 목표 샘플링 레이트로 리샘플링합니다."""
    if orig_sr == target_sr:
        return wav
    if torchaudio is not None:
        key = (orig_sr, target_sr, str(wav.dtype), str(wav.device))
        resampler = _RESAMPLER_CACHE.get(key)
        if resampler is None:
            if len(_RESAMPLER_CACHE) >= 16:
                _RESAMPLER_CACHE.clear()
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=target_sr, dtype=wav.dtype
            ).to(wav.device)
            _RESAMPLER_CACHE[key] = resampler
        return resampler(wav)

    orig_shape = wav.shape
    flat = wav.reshape(1, 1, -1).float()
    target_len = max(1, int(round(flat.shape[-1] * float(target_sr) / float(orig_sr))))
    out = F.interpolate(flat, size=target_len, mode="linear", align_corners=False)
    return out.reshape(*orig_shape[:-1], target_len).to(dtype=wav.dtype)


def _read_wav_stdlib(
    path_str: str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> tuple[torch.Tensor, int]:
    """표준 라이브러리 wave 모듈을 사용하여 16-bit PCM WAV 파일의 전체 또는 일부 구간을 로드합니다."""
    with wave.open(path_str, "rb") as wf:
        sr = int(wf.getframerate())
        n_ch = int(wf.getnchannels())
        total_frames = int(wf.getnframes())
        s_frame = max(0, min(int(start_sec * sr), total_frames)) if start_sec is not None else 0
        e_frame = (
            max(s_frame, min(int(end_sec * sr) + 1, total_frames))
            if end_sec is not None
            else total_frames
        )
        if e_frame <= s_frame:
            return torch.zeros(0, dtype=torch.float32), sr
        wf.setpos(s_frame)
        raw_bytes = wf.readframes(e_frame - s_frame)
        pcm = torch.frombuffer(bytearray(raw_bytes), dtype=torch.int16).float() / 32768.0
        if n_ch > 1:
            pcm = pcm.view(-1, n_ch).mean(dim=1)
        return pcm.contiguous(), sr


def load_mono_wav_24k(
    wav_path: Path | str,
    target_sr: int = SAMPLE_RATE,
    keep_channel_dim: bool = False,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> torch.Tensor:
    """보컬 오디오 파일 전체 또는 지정된 시간 구간만 부분 Seek하여 모노 24kHz 텐서로 반환합니다."""
    path_str = str(wav_path)
    try:
        if sf is None:
            raise RuntimeError("soundfile 미설치")
        if start_sec is not None and end_sec is not None:
            with sf.SoundFile(path_str, mode="r") as sfile:
                sr = int(sfile.samplerate)
                total_frames = len(sfile)
                s_frame = max(0, min(int(start_sec * sr), total_frames))
                e_frame = max(s_frame, min(int(end_sec * sr) + 1, total_frames))
                if e_frame <= s_frame:
                    empty = torch.zeros(0, dtype=torch.float32)
                    return empty.unsqueeze(0) if keep_channel_dim else empty
                sfile.seek(s_frame)
                data = sfile.read(e_frame - s_frame, dtype="float32", always_2d=True)
        else:
            data, sr = sf.read(path_str, dtype="float32", always_2d=True)

        if data.size == 0:
            empty = torch.zeros(0, dtype=torch.float32)
            return empty.unsqueeze(0) if keep_channel_dim else empty
        mono = torch.from_numpy(data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]).float()
        if int(sr) != target_sr:
            mono = resample_wav(mono, int(sr), target_sr)
        mono = mono.contiguous()
        return mono.unsqueeze(0) if keep_channel_dim else mono
    except Exception:
        if torchaudio is not None:
            try:
                wav, sr = torchaudio.load(path_str)
                mono = wav.mean(dim=0, keepdim=False).float()
                if sr != target_sr:
                    mono = resample_wav(mono, sr, target_sr)
                if start_sec is not None and end_sec is not None:
                    s_i = max(0, min(int(start_sec * target_sr), mono.numel()))
                    e_i = max(s_i, min(int(end_sec * target_sr), mono.numel()))
                    mono = mono[s_i:e_i].contiguous()
                return mono.unsqueeze(0) if keep_channel_dim else mono
            except Exception:
                pass
        mono, sr = _read_wav_stdlib(path_str, start_sec=start_sec, end_sec=end_sec)
        if sr != target_sr and mono.numel() > 0:
            mono = resample_wav(mono, sr, target_sr)
        return mono.unsqueeze(0) if keep_channel_dim else mono


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
    """외부 가중치 다운로드 없이 보컬 파이프라인 검증 및 데모 실행을 수행하는 경량 내장 오디오 코덱입니다."""

    def __init__(self, cfg: LyricTransConfig):
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
        """보컬 파형 `[B, 1, L]`을 프레임 단위로 패딩한 뒤 코드북 토큰 `[B, Q, T]`로 인코딩합니다."""
        rem = waveform.shape[-1] % self.hop_length
        if rem != 0:
            waveform = F.pad(waveform, (0, self.hop_length - rem))
        z = self.encoder(waveform)
        return self.quantizer.encode(z)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """코드북 토큰 `[B, Q, T]`를 보컬 오디오 파형 `[B, 1, L]`로 복원합니다."""
        z_q = self.quantizer.decode(codes)
        return self.decoder(z_q)


class ExternalMimiCodec(nn.Module):
    """HuggingFace transformers 또는 moshi 패키지의 사전학습 Mimi 코덱을 동일한 인터페이스로 감싸는 래퍼입니다."""

    def __init__(self, cfg: LyricTransConfig, model_id: str = "kyutai/mimi"):
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
    cfg: LyricTransConfig,
    device: torch.device,
    mimi_model_id: Optional[str] = None,
    ckpt: Optional[Mapping[str, Any]] = None,
) -> nn.Module:
    """명시된 모델 ID 또는 체크포인트 메타데이터에 따라 적절한 보컬 오디오 코덱을 로드해 반환합니다."""
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
                "내장 MiniMimiCodec으로 폴백합니다."
            )
            codec = MiniMimiCodec(cfg).to(device)
    else:
        codec = MiniMimiCodec(cfg).to(device)
    return codec.eval()
