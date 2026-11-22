"""논문 판 Your-TTS 학습(명세 3.12, 논문 4.1절). coqui-tts 0.28.0 의 Vits 를 Coqui Trainer(coqui-tts-trainer 0.4.0)로 학습한다.

논문 값(4.1절 'Training and Inference'):
- AdamW beta1 0.8, beta2 0.99, weight decay 0.01, 학습률 2e-4(생성기·판별기 모두), 200 epoch, batch 48(A100 4장).
- 자료 LibriTTS-360(904명) + KMSSS(550명), 8:1:1(make_manifest.py).
Your-TTS 구조(공개 YourTTS config.json 의 model_args 와 같은 값, YOURTTS_ARCH): 텍스트 인코더 트랜스포머 10층(192 + 언어 4),
확률 길이 예측기 SDP(use_sdp), 길이 예측기 화자 조건(condition_dp_on_speaker), 언어 임베딩 en/ko 4차원, d-vector 512
(공개 화자 인코더, compute_dvectors.py), 빈 토큰(add_blank), HiFi-GAN 디코더 resblock 2, 업샘플 8x8x2x2.
소리: 22050 Hz, hop 256, 창 1024, FFT 1024, mel 80. 논문은 TTS 소리 설정을 적지 않았다. 22.05 kHz 와 hop 256 은 논문이
원문 음성(SA) 쉼 검출에 쓴 프레임(3.1절)과 같게 골라 TTS 길이 프레임과 SA 프레임이 같게 한 것이다(해석). mel 80 은 VITS
값이다. 공개 YourTTS 는 16 kHz(hop 256) 모델이다. 다른 샘플레이트 소리는 읽을 때 바꾼다.
글: phonemize_manifest.py 가 미리 음소화한 phonemes 칸. 글자 목록 = characters.txt, use_phonemes False, 정리기 없음.
G2P 설정 g2p.json 과 추론용 화자 평균 d-vector 파일(d_vector_means.pth)을 학습 폴더에 함께 둔다(pstts.paper.yourtts 가 씀).

표본 추출(PsVits.get_sampler):
- 학습, 언어 균형(기본): coqui-tts 0.28 그대로. 언어 무게 복원 추출 + 길이 묶음 배치(BucketBatchSampler)라 epoch 마다 섞인다.
- 학습, --no-language-balance: epoch 마다 새로 섞는 길이 묶음 배치(RandomSampler + BucketBatchSampler). Coqui 그대로면
  표본 추출기가 없어 batch_group_size 0 에서 epoch 마다 같은 짧은 것부터 긴 것 순서가 되어서 바꿨다.
- 학습, --no-language-balance --start-by-longest: Coqui 그대로 길이 순서이고 첫 배치에 가장 긴 발화가 든다(메모리 재기용).
  언어 균형을 켜면 첫 배치가 무작위라 --start-by-longest 는 뜻이 없다.
- 검증: 늘 표본 추출기 없이 검증 자료를 한 번씩 차례로 읽는다(--eval-batch-size, 마지막 배치도 씀). coqui-tts 0.28
  Vits.get_data_loader 는 get_sampler 에 is_eval 을 넘기지 않아 언어 균형일 때 검증도 학습 배치 크기, 복원 추출,
  drop_last 로 읽었다(--eval-batch-size 가 무시되고 검증 줄이 배치보다 적으면 검증 배치가 0개). 그래서 고쳤다(해석).
--max-steps 를 주면 그 걸음에서 학습을 멈추고, 검증도 한 번에 최대 그 수만큼의 배치만 읽는다(점검을 짧게).

논문과 다른 점(README_paper 의 남는 차이):
- 논문은 Your-TTS 의 길이 예측기를 Parallel Tacotron 2 의 미분 가능한 길이 모델 [39] 로 바꿨다. 이 레시피에는 그것이 없다
  (Your-TTS 기본 SDP 를 쓴다). 추론(pstts.paper.yourtts)은 SDP 의 올리기 전 실수 길이를 쓴다.
- 논문에 없는 학습 값은 공개 YourTTS 설정이나 Coqui VITS 기본값을 쓴다(해석): 학습률 감쇠 ExponentialLR(gamma 0.999875),
  AdamW eps 1e-9, grad_clip [5, 5], 언어 균형 표본 추출(공개 YourTTS 의 use_language_weighted_sampler), 시드 54321,
  체크포인트 10000걸음마다 5개, 최고 모델은 10000걸음 뒤부터. 공개 YourTTS 의 화자 일관성 손실(SCL)은 논문에 없어 넣지 않았다.
- 12GB 카드: batch 48 은 A100 4장 값이다. Coqui Trainer 0.4.0 은 옵티마이저가 둘인 모델(VITS 의 판별기·생성기)에서
  grad_accum_steps 를 막는다(trainer/trainer.py Trainer.train_step 의 ValueError). 그래서 그래디언트 누적으로 48 을 맞출 수
  없고 --batch-size 를 줄여 학습한다. GPU 가 빌 때 --max-steps, --start-by-longest, --no-language-balance 로 가장 긴 배치의
  메모리를 먼저 잰다.
- --init-from 은 공개 YourTTS 가중치에서 모양이 같은 층만 가져온다. 공개 모델은 16 kHz 라 hop 256 이 16 ms 프레임이고,
  이 레시피(22.05 kHz)에서는 11.6 ms 프레임이다. 사후 인코더, 길이 예측기, HiFi-GAN 디코더가 다른 프레임과 스펙트럼 칸을
  배운 값이라 거친 출발점일 뿐이다. 논문은 처음부터 학습했으므로 기본은 끈다.

Windows 에서 안전하게 읽기: coqui-tts 0.28 의 VitsDataset 은 torchaudio.load 로 소리를 읽는데 torch 2.9 부터 이것은
torchcodec(FFmpeg 공유 DLL 필요)을 부른다. 여기서는 soundfile 로 읽고 샘플레이트가 다르면 librosa.resample(soxr_hq)로 바꾸는
PsVitsDataset 을 쓴다. 데이터 읽기 프로세스는 Windows 에서 spawn 이라 클래스를 모듈 맨 위에 둔다.
장치: 기본 cuda 이고 못 쓰면 멈춘다(pstts.scorers.resolve_device). --device cpu 면 GPU 를 가리고(CUDA_VISIBLE_DEVICES='')
Coqui Trainer 가 배치를 GPU 로 옮기지 않게 한다(Trainer.format_batch 의 to_cuda 는 torch.cuda.is_available() 만 본다).

실행 예(진짜 학습, GPU):
  .venv\\Scripts\\python.exe scripts\\yourtts\\train_yourtts.py --manifest-dir D:\\data\\pstts_tts --batch-size 16
이어하기: --continue-path <run 폴더>. 공개 YourTTS 가중치로 거칠게 시작: --init-from <model_file.pth>(기본 끔).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from make_manifest import audio_path, manifest_path, read_manifest, sample_key  # noqa: E402
from phonemize_manifest import CHARACTERS_NAME, PHONEMES, read_characters  # noqa: E402
from torch.utils.data import RandomSampler  # noqa: E402
from trainer import Trainer, TrainerArgs  # noqa: E402
from TTS.config.shared_configs import BaseAudioConfig  # noqa: E402
from TTS.tts.configs.shared_configs import CharactersConfig  # noqa: E402
from TTS.tts.configs.vits_config import VitsArgs, VitsConfig  # noqa: E402
from TTS.tts.models.vits import Vits, VitsDataset  # noqa: E402
from TTS.utils.samplers import BucketBatchSampler  # noqa: E402

from pstts.paper.g2p import SETTINGS_NAME as G2P_SETTINGS_NAME  # noqa: E402
from pstts.paper.yourtts import SPEAKER_MEANS_NAME, write_speaker_means  # noqa: E402

RUN_NAME = "pstts_yourtts"
LANGUAGES = ("en", "ko")
D_VECTOR_DIM = 512                  # 공개 YourTTS 화자 인코더 출력(config_se.json proj_dim)
LANGUAGE_DIM = 4                    # YourTTS 언어 임베딩
# 논문 4.1절 'Training and Inference'
PAPER = {"optimizer": "AdamW", "betas": [0.8, 0.99], "weight_decay": 0.01, "lr": 2e-4, "epochs": 200, "batch_size": 48}
# 해석: 논문은 TTS 소리 설정을 적지 않았다. 3.1절 원문 음성(SA) 프레임(22.05 kHz, 창 1024, hop 256)과 같게 골라
# TTS 길이 프레임 = SA 프레임이 되게 한다. mel 80 은 VITS·YourTTS 값(공개 YourTTS 는 16 kHz, hop 256)
AUDIO = {"sample_rate": 22050, "hop_length": 256, "win_length": 1024, "fft_size": 1024, "num_mels": 80,
         "mel_fmin": 0.0, "mel_fmax": None}
# 공개 YourTTS(tts_models--multilingual--multi-dataset--your_tts) config.json model_args 의 구조 값
YOURTTS_ARCH: dict[str, Any] = {
    "hidden_channels": 192, "hidden_channels_ffn_text_encoder": 768, "num_heads_text_encoder": 2,
    "num_layers_text_encoder": 10, "kernel_size_text_encoder": 3, "dropout_p_text_encoder": 0.1,
    "dropout_p_duration_predictor": 0.5, "kernel_size_posterior_encoder": 5, "dilation_rate_posterior_encoder": 1,
    "num_layers_posterior_encoder": 16, "kernel_size_flow": 5, "dilation_rate_flow": 1, "num_layers_flow": 4,
    "resblock_type_decoder": "2", "resblock_kernel_sizes_decoder": [3, 7, 11],
    "resblock_dilation_sizes_decoder": [[1, 3, 5], [1, 3, 5], [1, 3, 5]], "upsample_rates_decoder": [8, 8, 2, 2],
    "upsample_initial_channel_decoder": 512, "upsample_kernel_sizes_decoder": [16, 16, 4, 4],
    "periods_multi_period_discriminator": [2, 3, 5, 7, 11], "spec_segment_size": 62, "out_channels": 513,
}
# 점검용(smoke_train.py) 아주 작은 구조. 진짜 학습에 쓰지 않는다.
TINY_ARCH: dict[str, Any] = {
    **YOURTTS_ARCH, "hidden_channels": 16, "hidden_channels_ffn_text_encoder": 32, "num_layers_text_encoder": 1,
    "num_layers_posterior_encoder": 1, "num_layers_flow": 1, "resblock_kernel_sizes_decoder": [3],
    "resblock_dilation_sizes_decoder": [[1, 3]], "upsample_initial_channel_decoder": 32,
    "periods_multi_period_discriminator": [2], "spec_segment_size": 16,
}
ARCHES = {"yourtts": YOURTTS_ARCH, "tiny": TINY_ARCH}
# 논문에 없는 학습 값(공개 YourTTS 설정 또는 Coqui VITS 기본값)
TRAINING_DEFAULTS: dict[str, Any] = {
    "optimizer_eps": 1e-9, "lr_scheduler": "ExponentialLR", "lr_gamma": 0.999875, "grad_clip": [5.0, 5.0],
    "seed": 54321, "save_step": 10000, "save_n_checkpoints": 5, "save_best_after": 10000, "num_workers": 4,
}


# ---------------------------------------------------------------------------- 자료
def load_wav_tensor(path: str, sample_rate: int) -> torch.Tensor:
    """[1, T] float32(-1~1). soundfile 로 읽고 여러 채널이면 평균, 샘플레이트가 다르면 librosa.resample(soxr_hq)로 바꾼다.

    librosa.load(sr=...) 와 같은 방식이다. Coqui numpy_transforms.load_wav 는 sf.read 의 (T, 채널) 꼴에 librosa.to_mono 를
    불러 여러 채널 파일을 잘못 줄여서 쓰지 않는다. 바꾼 소리가 +-1 을 조금 넘을 수 있어 자른다(torchaudio.load 는 -1~1).
    """
    import librosa
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if int(sr) != int(sample_rate):
        x = librosa.resample(x, orig_sr=int(sr), target_sr=int(sample_rate))
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    return torch.from_numpy(x).unsqueeze(0)


class PsVitsDataset(VitsDataset):
    """VitsDataset 과 같고 소리 읽기만 soundfile 로 바꾼 것(Windows 에서 torchcodec·FFmpeg 없이 읽음).

    ps_sample_rate(config.audio.sample_rate)는 PsVits.get_data_loader 가 넣는다(Vits 는 데이터셋에 ap 를 주지 않음).
    """

    ps_sample_rate: int

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # coqui-tts 0.28.0 VitsDataset.__getitem__ 과 같은 순서. load_audio(torchaudio) 자리만 load_wav_tensor.
        item = self.samples[idx]
        raw_text = item["text"]
        wav = load_wav_tensor(item["audio_file"], self.ps_sample_rate)
        esr = self.model_args.encoder_sample_rate
        if esr is not None and wav.size(1) % esr != 0:
            wav = wav[:, : -int(wav.size(1) % esr)]
        token_ids = self.get_token_ids(idx, item["text"], item["language"])
        if len(token_ids) > self.max_text_len or wav.shape[1] < self.min_audio_len:
            self.rescue_item_idx += 1
            return self.__getitem__(self.rescue_item_idx)
        return {
            "raw_text": raw_text,
            "token_ids": token_ids,
            "token_len": len(token_ids),
            "wav": wav,
            "wav_file": os.path.basename(item["audio_file"]),
            "speaker_name": item["speaker_name"],
            "language_name": item["language"],
            "audio_unique_name": item["audio_unique_name"],
        }


def _file_size(sample: dict[str, Any]) -> int:
    """길이 묶음 배치의 정렬 열쇠(Coqui Vits.get_sampler 와 같은 소리 파일 크기)."""
    return os.path.getsize(sample["audio_file"])


class PsVits(Vits):
    """Vits 와 같다. 데이터 읽기를 PsVitsDataset 으로, 표본 추출기를 get_sampler 로 바꾼다(저장되는 가중치는 Vits 그대로)."""

    _ps_loading_eval = False        # get_data_loader 가 지금 검증 데이터를 만드는 중인지(get_sampler 에 넘기려고)

    def get_sampler(self, config, dataset, is_eval: bool = False):
        """표본 추출기(모듈 머리말의 '표본 추출'). None 이면 Coqui 가 차례대로 읽는 DataLoader 를 만든다."""
        if is_eval or self._ps_loading_eval:
            return None
        if getattr(config, "use_weighted_sampler", False):
            return super().get_sampler(config, dataset, is_eval=False)
        if config.start_by_longest:
            return None
        return BucketBatchSampler(RandomSampler(dataset), data=dataset.samples, batch_size=int(config.batch_size),
                                  drop_last=True, sort_key=_file_size)

    def get_data_loader(self, config, *, is_eval: bool = False, samples=None, verbose: bool = True):
        # coqui-tts 0.28.0 Vits.get_data_loader 는 get_sampler(config, dataset) 로 불러 is_eval 이 빠진다. 여기서 들고 간다.
        self._ps_loading_eval = bool(is_eval)
        try:
            loader = super().get_data_loader(config, is_eval=is_eval, samples=samples, verbose=verbose)
        finally:
            self._ps_loading_eval = False
        # coqui-tts 0.28.0 Vits.get_data_loader 는 VitsDataset 을 바로 만들어 클래스를 바꿀 자리가 없다.
        # 만든 데이터셋(이미 정렬·거르기를 마친 것)의 클래스만 바꿔 __getitem__ 을 PsVitsDataset 것으로 쓴다.
        if type(loader.dataset) is not VitsDataset:
            raise TypeError(f"예상한 VitsDataset 이 아닙니다: {type(loader.dataset)} (coqui-tts 판을 확인하세요)")
        loader.dataset.__class__ = PsVitsDataset
        loader.dataset.ps_sample_rate = int(config.audio.sample_rate)
        return loader


def load_samples(manifest: Path, max_audio_sec: float | None = None) -> tuple[list[dict[str, Any]], dict]:
    """매니페스트(phonemes 칸 포함)를 Coqui 표본 꼴로 바꾼다. text = phonemes, audio_unique_name = sample_key."""
    import soundfile as sf

    rows = read_manifest(manifest)
    if rows and PHONEMES not in rows[0]:
        raise ValueError(f"{manifest} 에 phonemes 칸이 없습니다. phonemize_manifest.py 를 먼저 실행하세요.")
    out: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    for row in rows:
        if not row[PHONEMES]:
            skipped["empty_phonemes"] += 1
            continue
        audio = audio_path(row, manifest.parent)
        if max_audio_sec is not None:
            info = sf.info(str(audio))
            if info.frames / info.samplerate > max_audio_sec:
                skipped["too_long"] += 1
                continue
        out.append({"text": row[PHONEMES], "audio_file": str(audio), "speaker_name": row["speaker"],
                    "language": row["lang"], "audio_unique_name": sample_key(row), "root_path": str(audio.parent)})
    return out, {"rows": len(rows), "used": len(out), "skipped": dict(skipped)}


# ---------------------------------------------------------------------------- 설정
def build_config(*, characters: Sequence[str], d_vector_file: str, output_path: str, run_name: str = RUN_NAME,
                 arch: str = "yourtts", epochs: int = PAPER["epochs"], batch_size: int = PAPER["batch_size"],
                 eval_batch_size: int | None = None, lr: float = PAPER["lr"],
                 num_workers: int = TRAINING_DEFAULTS["num_workers"], language_balance: bool = True,
                 print_step: int = 25, save_step: int = TRAINING_DEFAULTS["save_step"], mixed_precision: bool = False,
                 start_by_longest: bool = False, seed: int = TRAINING_DEFAULTS["seed"]) -> VitsConfig:
    """논문 4.1절 값과 Your-TTS 구조로 VitsConfig 를 만든다."""
    if arch not in ARCHES:
        raise ValueError(f"구조 이름은 {sorted(ARCHES)} 가운데 하나여야 합니다: {arch}")
    chars = list(characters)
    if not chars or any(len(c) != 1 for c in chars) or len(set(chars)) != len(chars):
        raise ValueError("글자 목록은 겹치지 않는 한 글자 토큰들이어야 합니다.")
    model_args = VitsArgs(
        **ARCHES[arch],
        use_sdp=True,                       # Your-TTS 기본 확률 길이 예측기(논문의 PT2 길이 모델 [39] 은 없음)
        condition_dp_on_speaker=True,
        use_language_embedding=True,
        embedded_language_dim=LANGUAGE_DIM,
        use_speaker_embedding=False,
        use_d_vector_file=True,
        d_vector_dim=D_VECTOR_DIM,
        d_vector_file=str(d_vector_file),
        use_speaker_encoder_as_loss=False,
    )
    audio = BaseAudioConfig(**AUDIO)
    chars_cfg = CharactersConfig(characters_class="TTS.tts.models.vits.VitsCharacters", pad="<PAD>", eos=None,
                                 bos=None, blank="<BLNK>", characters="".join(chars), punctuations="", phonemes=None,
                                 is_unique=False, is_sorted=False)
    sched = {"gamma": TRAINING_DEFAULTS["lr_gamma"], "last_epoch": -1}
    return VitsConfig(
        model_args=model_args,
        audio=audio,
        run_name=run_name,
        run_description="PS-TTS 논문 판 Your-TTS(한·영, 미리 음소화한 글, d-vector, 언어 임베딩)",
        output_path=str(output_path),
        epochs=int(epochs),
        batch_size=int(batch_size),
        eval_batch_size=int(eval_batch_size or batch_size),
        optimizer=PAPER["optimizer"],
        optimizer_params={"betas": list(PAPER["betas"]), "eps": TRAINING_DEFAULTS["optimizer_eps"],
                          "weight_decay": PAPER["weight_decay"]},
        lr_gen=float(lr),
        lr_disc=float(lr),
        lr_scheduler_gen=TRAINING_DEFAULTS["lr_scheduler"],
        lr_scheduler_gen_params=dict(sched),
        lr_scheduler_disc=TRAINING_DEFAULTS["lr_scheduler"],
        lr_scheduler_disc_params=dict(sched),
        grad_clip=list(TRAINING_DEFAULTS["grad_clip"]),
        mixed_precision=bool(mixed_precision),
        num_loader_workers=int(num_workers),
        num_eval_loader_workers=int(num_workers),
        use_phonemes=False,
        text_cleaner=None,
        add_blank=True,
        enable_eos_bos_chars=False,
        characters=chars_cfg,
        languages=list(LANGUAGES),
        datasets=[],
        test_sentences=[],
        use_weighted_sampler=bool(language_balance),
        weighted_sampler_attrs={"language": 1.0} if language_balance else {},
        print_step=int(print_step),
        plot_step=100,
        save_step=int(save_step),
        save_n_checkpoints=TRAINING_DEFAULTS["save_n_checkpoints"],
        save_best_after=TRAINING_DEFAULTS["save_best_after"],
        run_eval=True,
        print_eval=False,
        test_delay_epochs=-1,
        start_by_longest=bool(start_by_longest),
        training_seed=int(seed),
        dashboard_logger="tensorboard",
    )


def init_from_checkpoint(model: torch.nn.Module, path: str | os.PathLike) -> dict[str, Any]:
    """공개 YourTTS 등 다른 체크포인트에서 모양이 같은 층만 가져온다(글자 임베딩, 언어 임베딩처럼 다른 층은 뺀다).

    공개 YourTTS 는 옛 weight_norm 이름(weight_g, weight_v)으로 저장돼 있다. torch 의 weight_norm 호환 훅이 load_state_dict
    안에서 새 이름(parametrizations.weight.original0/1)으로 바꾸므로 이름으로 거르지 않고 strict=False 로 넘긴다.
    화자 인코더 층(speaker_encoder.*)은 Coqui Vits.load_checkpoint 처럼 뺀다.
    """
    from trainer.io import load_fsspec

    state = load_fsspec(str(path), map_location="cpu")["model"]
    own = model.state_dict()
    mismatch = sorted(k for k, v in state.items() if k in own and tuple(own[k].shape) != tuple(v.shape))
    take = {k: v for k, v in state.items() if k not in mismatch and not k.startswith("speaker_encoder.")}
    result = model.load_state_dict(take, strict=False)
    unexpected = list(result.unexpected_keys)
    missing = sorted(result.missing_keys)
    return {"path": str(path), "given": len(state), "loaded": len(own) - len(missing), "shape_mismatch": mismatch,
            "unexpected": len(unexpected) + sum(1 for k in state if k.startswith("speaker_encoder.")),
            "kept_init": missing}


# ---------------------------------------------------------------------------- 학습기
class StepLimitedLoader:
    """학습 데이터 읽기를 감싸 학습기의 걸음 수가 max_steps 에 닿으면 멈춘다."""

    def __init__(self, loader: Any, trainer: "PsTrainer") -> None:
        self.loader = loader
        self.trainer = trainer

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            if self.trainer.max_steps is not None and self.trainer.total_steps_done >= self.trainer.max_steps:
                return
            yield batch


class BatchLimitedLoader:
    """검증 데이터 읽기를 감싸 한 번 돌 때 최대 limit 배치만 낸다(--max-steps 점검에서 검증도 짧게)."""

    def __init__(self, loader: Any, limit: int) -> None:
        self.loader = loader
        self.limit = int(limit)

    def __len__(self) -> int:
        return min(len(self.loader), self.limit)

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.limit:
                return
            yield batch


class PsTrainer(Trainer):
    """Coqui Trainer 에 장치 고정(cpu 일 때 GPU 를 건드리지 않음)과 최대 걸음 수, 마지막 체크포인트 저장을 더한 것.

    max_steps 를 주면 학습은 그 걸음에서 멈추고, 검증은 한 번에 최대 max_steps 배치만 읽는다."""

    def __init__(self, *args: Any, device: str, max_steps: int | None = None, **kwargs: Any) -> None:
        self.ps_device = device
        self.max_steps = max_steps
        super().__init__(*args, **kwargs)

    def setup_training_environment(self, args: TrainerArgs, config: Any, gpu: int | None) -> tuple[bool, int]:
        if self.ps_device != "cpu":
            return super().setup_training_environment(args=args, config=config, gpu=gpu)
        # Coqui setup_torch_training_env 의 시드 부분만. CUDA 함수(empty_cache, current_device 등)는 부르지 않는다.
        seed = int(config.training_seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"학습 장치: cpu (스레드 {torch.get_num_threads()}, 시드 {seed})", file=sys.stderr, flush=True)
        return False, 0

    def format_batch(self, batch: Any) -> dict[str, Any]:
        if self.use_cuda:
            return super().format_batch(batch)
        model = self._get_model()
        batch = model.format_batch(batch)
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.contiguous()
        return model.format_batch_on_device(batch)

    def fit(self) -> None:
        epochs = self.config.epochs
        if self.max_steps is not None:
            if self.train_loader is None:
                self.train_loader = self.get_train_dataloader(self.train_samples, verbose=True)
            per_epoch = max(1, len(self.train_loader))
            remaining = max(0, self.max_steps - self.total_steps_done)
            self.config.epochs = min(epochs, self.epochs_done + math.ceil(remaining / per_epoch))
            self.train_loader = StepLimitedLoader(self.train_loader, self)
            if self.config.run_eval:
                if self.eval_loader is None:
                    self.eval_loader = self.get_eval_dataloader(self.eval_samples, verbose=True)
                self.eval_loader = BatchLimitedLoader(self.eval_loader, self.max_steps)
        try:
            super().fit()
        finally:
            self.config.epochs = epochs
        self.save_checkpoint()          # 마지막 걸음 체크포인트(Coqui 는 끝에 따로 저장하지 않음)


# ---------------------------------------------------------------------------- 실행
def train(args: argparse.Namespace) -> dict[str, Any]:
    from pstts.scorers import resolve_device

    device = resolve_device(args.device, args.cpu_threads)
    man = args.manifest_dir.resolve()
    train_samples, train_info = load_samples(manifest_path(man, "train"), args.max_audio_sec)
    eval_samples, eval_info = load_samples(manifest_path(man, "valid"), args.max_audio_sec)
    if not train_samples or not eval_samples:
        raise ValueError(f"학습({len(train_samples)})·검증({len(eval_samples)}) 표본이 있어야 합니다.")
    chars_path = args.characters or man / CHARACTERS_NAME
    d_vec = (args.d_vector_file or man / "d_vectors.pth").resolve()
    if not d_vec.is_file():
        raise FileNotFoundError(f"d-vector 파일이 없습니다: {d_vec} (compute_dvectors.py 를 먼저 실행하세요)")
    g2p_settings = man / G2P_SETTINGS_NAME
    if not g2p_settings.is_file():
        raise FileNotFoundError(f"{g2p_settings} 가 없습니다. phonemize_manifest.py 를 다시 실행하세요(G2P 설정을 적음).")

    if args.continue_path:
        config = VitsConfig()
        config.load_json(str(Path(args.continue_path) / "config.json"))
    else:
        config = build_config(characters=read_characters(chars_path), d_vector_file=str(d_vec),
                              output_path=str(args.out), run_name=args.run_name, arch=args.arch, epochs=args.epochs,
                              batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, lr=args.lr,
                              num_workers=args.num_workers, language_balance=not args.no_language_balance,
                              print_step=args.print_step, save_step=args.save_step,
                              mixed_precision=args.mixed_precision, start_by_longest=args.start_by_longest,
                              seed=args.seed)
    model = PsVits(config)
    init_info = init_from_checkpoint(model, args.init_from) if args.init_from else None
    gpu = int(device.split(":", 1)[1]) if device.startswith("cuda:") else None
    trainer = PsTrainer(TrainerArgs(continue_path=str(args.continue_path or "")), config, str(args.out),
                        model=model, train_samples=train_samples, eval_samples=eval_samples,
                        parse_command_line_args=False, gpu=gpu, device=device, max_steps=args.max_steps)
    run_dir = Path(trainer.output_path)
    shutil.copyfile(chars_path, run_dir / CHARACTERS_NAME)
    shutil.copyfile(g2p_settings, run_dir / G2P_SETTINGS_NAME)
    # 추론용 화자 평균 d-vector(pstts.paper.yourtts 가 발화별 파일 대신 올림). 학습 중 체크포인트에도 바로 쓸 수 있게 먼저 쓴다.
    means = write_speaker_means(d_vec, run_dir / SPEAKER_MEANS_NAME)
    start = time.perf_counter()
    trainer.fit()
    result: dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "final_checkpoint": str((run_dir / f"checkpoint_{trainer.total_steps_done}.pth").resolve()),
        "steps": trainer.total_steps_done,
        "epochs_done": trainer.epochs_done,
        "device": device,
        "arch": args.arch,
        "batch_size": int(config.batch_size),
        "eval_batch_size": int(config.eval_batch_size),
        "language_balance": bool(config.use_weighted_sampler),
        "max_steps": args.max_steps,
        "seconds": round(time.perf_counter() - start, 2),
        "train_samples": train_info,
        "eval_samples": eval_info,
        "init_from": init_info,
        "speaker_means": means,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }
    if device.startswith("cuda"):
        result["max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        result["max_memory_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)
    text = json.dumps(result, ensure_ascii=False, indent=1)
    (Path(trainer.output_path) / "train_result.json").write_text(text, encoding="utf-8")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "last_run.json").write_text(text, encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="논문 판 Your-TTS(한·영 음소, d-vector, 언어 임베딩)를 학습한다.")
    ap.add_argument("--manifest-dir", type=Path, required=True,
                    help="manifest_{train,valid}.csv(phonemes 칸), characters.txt, d_vectors.pth 가 있는 폴더")
    ap.add_argument("--characters", type=Path, help="글자 목록(기본 <manifest-dir>/characters.txt)")
    ap.add_argument("--d-vector-file", type=Path, help="d-vector 파일(기본 <manifest-dir>/d_vectors.pth)")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "yourtts_runs", help="학습 출력 뿌리 폴더")
    ap.add_argument("--run-name", default=RUN_NAME, help=f"run 이름(기본 {RUN_NAME})")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:N, cpu (기본 cuda. 못 쓰면 멈춤)")
    ap.add_argument("--cpu-threads", type=int, default=None, help="CPU 스레드 수(기본 기존 Config 값)")
    ap.add_argument("--arch", default="yourtts", choices=sorted(ARCHES), help="구조(tiny 는 점검용)")
    ap.add_argument("--epochs", type=int, default=PAPER["epochs"], help=f"epoch 수(논문 {PAPER['epochs']})")
    ap.add_argument("--batch-size", type=int, default=PAPER["batch_size"],
                    help=f"배치 크기(논문 {PAPER['batch_size']}, A100 4장. 12GB 카드는 줄일 것)")
    ap.add_argument("--eval-batch-size", type=int, default=None,
                    help="검증 배치 크기(기본 배치 크기와 같음). 검증은 언어 균형과 상관없이 자료를 한 번씩 차례로 읽는다")
    ap.add_argument("--lr", type=float, default=PAPER["lr"], help=f"학습률(논문 {PAPER['lr']})")
    ap.add_argument("--num-workers", type=int, default=TRAINING_DEFAULTS["num_workers"],
                    help=f"데이터 읽기 프로세스 수(기본 {TRAINING_DEFAULTS['num_workers']}, 0 이면 같은 프로세스)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="이 걸음 수에서 멈춘다. 검증도 한 번에 이 수만큼의 배치만 읽는다(점검, 메모리 재기)")
    ap.add_argument("--max-audio-sec", type=float, default=None, help="이보다 긴 발화는 뺀다(기본 모두 씀)")
    ap.add_argument("--start-by-longest", action="store_true",
                    help="첫 배치를 가장 긴 발화로(메모리 재기). --no-language-balance 와 함께 줄 때만 되고, 그때는 "
                         "epoch 마다 같은 길이 순서로 읽는다")
    ap.add_argument("--no-language-balance", action="store_true",
                    help="언어 균형 표본 추출을 끈다. 그래도 epoch 마다 섞는다(--start-by-longest 를 함께 주면 안 섞음)")
    ap.add_argument("--mixed-precision", action="store_true", help="혼합 정밀도(fp16). 논문에 없음, 기본 끔")
    ap.add_argument("--print-step", type=int, default=25, help="이만큼마다 손실을 찍는다")
    ap.add_argument("--save-step", type=int, default=TRAINING_DEFAULTS["save_step"], help="체크포인트 간격(걸음)")
    ap.add_argument("--seed", type=int, default=TRAINING_DEFAULTS["seed"], help="학습 시드")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="모양이 같은 층을 가져올 체크포인트(기본 끔). 공개 YourTTS 는 16 kHz 모델이라 22.05 kHz 학습에는 "
                         "거친 출발점일 뿐이다")
    ap.add_argument("--continue-path", type=Path, default=None, help="이어 학습할 run 폴더(그 config.json 을 씀)")
    args = ap.parse_args(argv)
    if args.batch_size < 1 or (args.max_steps is not None and args.max_steps < 1):
        ap.error("배치 크기와 최대 걸음 수는 1 이상이어야 합니다.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device.strip().lower() == "cpu":
        # CPU 학습: GPU 를 가린다. 무엇이 CUDA 를 부르더라도 사용자의 GPU 작업에 닿지 않고 오류로 드러난다.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    result = train(args)
    print(json.dumps({k: result[k] for k in ("run_dir", "final_checkpoint", "steps", "seconds", "device")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
