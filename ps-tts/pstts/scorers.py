"""뜻 점수기와 장치 고르기.

- LabseScorer: 영어 원문과 한국어 후보의 LaBSE 코사인 유사도. 거르기 조건(논문 식 5, 7: 0.75 이상)에 쓴다.
- CometScorer: COMET 점수. PS-Comet 선택(논문 식 20)의 Comet(S, T) 에 쓴다.
  기본 모델 Unbabel/wmt20-comet-qe-da 는 참조 없는 모델이고 점수가 0~1 범위가 아니다(DA z 점수에 맞춘 회귀값).
큰 라이브러리(sentence-transformers, comet)는 처음 쓸 때 불러온다.
"""
from __future__ import annotations

import gc
import logging
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Config

_DEVICE = re.compile(r"^(?:cpu|cuda(?::(\d+))?)$")

# 불러올 때 쏟아지는 기록 줄이기
_QUIET_LOGGERS = {
    "pytorch_lightning": logging.ERROR,    # GPU available…, LOCAL_RANK…, Tensor Cores 안내
    "lightning_fabric": logging.ERROR,
    "lightning": logging.ERROR,
    "comet": logging.WARNING,              # Encoder model frozen.
    "sentence_transformers": logging.WARNING,
    "transformers": logging.ERROR,
    "torch.utils.flop_counter": logging.ERROR,   # lightning 을 불러올 때 'triton not found' (Windows 에 triton 없음)
}
# COMET 체크포인트에만 있는 위치 번호 버퍼. 지금 transformers 는 이 버퍼를 저장하지 않아서 나오는 안내라 무해하다.
# 다른 키가 어긋나면 그대로 경고가 보이도록 이 키 하나만 거른다.
_COMET_EXTRA_KEY = (r"Found keys that are not in the model state dict but in the checkpoint: "
                    r"\['encoder\.model\.embeddings\.position_ids'\]$")


def resolve_device(requested: str, cpu_threads: int | None = None) -> str:
    """장치 이름을 확인해 돌려준다('cuda', 'cuda:N', 'cpu').

    cuda 를 요청했는데 쓸 수 없으면 조용히 cpu 로 바꾸지 않고 멈춘다.
    cpu 면 torch 연산 스레드 수를 cpu_threads(없으면 Config 기본값)로 맞춘다.
    """
    name = str(requested).strip().lower()
    m = _DEVICE.match(name)
    if m is None:
        raise ValueError(f"장치 이름이 잘못됐습니다: {requested} (cuda, cuda:0, cpu 중 하나)")
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU(CUDA)를 쓸 수 없습니다. CPU 로 돌리려면 --device cpu 를 명시하세요.")
        # CUDA_VISIBLE_DEVICES="" 처럼 GPU 를 가리면 is_available() 은 참인데 보이는 GPU 가 0개일 수 있다
        # (2026-11-22 검토에서 확인). 그대로 두면 모델 1 이 번역 도중 'Invalid device id' 로 죽는다.
        count = torch.cuda.device_count()
        if count < 1:
            raise RuntimeError("보이는 GPU 가 없습니다(CUDA_VISIBLE_DEVICES 를 확인하세요). "
                               "CPU 로 돌리려면 --device cpu 를 명시하세요.")
        if m.group(1) is not None and int(m.group(1)) >= count:
            raise RuntimeError(f"GPU {m.group(1)}번이 없습니다(보이는 GPU {count}개).")
        return name
    threads = Config().cpu_threads if cpu_threads is None else int(cpu_threads)
    if threads < 1:
        raise ValueError(f"CPU 스레드 수는 1 이상이어야 합니다: {threads}")
    torch.set_num_threads(threads)
    return "cpu"


def _quiet_libraries() -> None:
    for name, level in _QUIET_LOGGERS.items():
        logging.getLogger(name).setLevel(level)
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()


def _import_comet() -> Any:
    """comet 을 불러온다. comet 2.2.7 의 __init__ 이 logging.basicConfig(level=INFO) 로 루트 기록기를
    바꿔서 모든 라이브러리의 INFO 기록이 찍히게 되므로, 그때 붙은 처리기와 수준을 되돌린다."""
    _quiet_libraries()
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    import comet
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
    root.setLevel(level)
    _quiet_libraries()
    return comet


def _comet_checkpoint(comet: Any, model_name: str) -> str:
    """COMET 체크포인트 경로. 내려받아 둔 것이 있으면 Hub 에 묻지 않고 쓴다(오프라인에서도 됨).

    없을 때만 comet.download_model 로 받는다. comet 과 같은 꼴(스냅숏/checkpoints/model.ckpt)이다.
    """
    from huggingface_hub import snapshot_download
    try:
        folder = snapshot_download(repo_id=model_name, local_files_only=True)
    except Exception:    # 캐시에 없음, 또는 Hub 이름 꼴이 아님(옛 이름·경로): comet 의 원래 방법으로
        return comet.download_model(model_name)
    path = Path(folder) / "checkpoints" / "model.ckpt"
    return str(path) if path.is_file() else comet.download_model(model_name)


def free_memory(device: str) -> None:
    """내린 모델의 메모리를 돌려준다(GPU 면 캐시도 비움)."""
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


class LabseScorer:
    """LaBSE 문장 임베딩의 코사인 유사도(짝마다)."""

    def __init__(self, model_name: str = Config.labse_model, device: str = "cuda", batch_size: int = 64,
                 cpu_threads: int | None = None) -> None:
        self.model_name = model_name
        self.device = resolve_device(device, cpu_threads)
        self.batch_size = batch_size
        self._model: Any = None

    def load(self) -> Any:
        if self._model is None:
            _quiet_libraries()
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.eval()
        return self._model

    def similarity(self, srcs: Sequence[str], tgts: Sequence[str]) -> np.ndarray:
        """srcs[i] 와 tgts[i] 의 코사인 유사도 배열. 같은 문장은 한 번만 임베딩한다."""
        if len(srcs) != len(tgts):
            raise ValueError(f"원문 {len(srcs)}개와 후보 {len(tgts)}개의 수가 다릅니다.")
        if not srcs:
            return np.zeros(0, dtype=np.float64)
        texts = list(dict.fromkeys([*srcs, *tgts]))
        pos = {t: i for i, t in enumerate(texts)}
        emb = self.load().encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
        a = emb[[pos[s] for s in srcs]].astype(np.float64)
        b = emb[[pos[t] for t in tgts]].astype(np.float64)
        return np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0)

    def close(self) -> None:
        """모델을 내리고 메모리를 비운다. 다시 쓰면 다시 불러온다."""
        self._model = None
        free_memory(self.device)


class CometScorer:
    """COMET 점수(문장마다). 참조 없는 모델이면 (원문, 후보) 만, 참조형이면 refs 도 넣는다."""

    def __init__(self, model_name: str = Config.comet_model, device: str = "cuda", batch_size: int = 16,
                 cpu_threads: int | None = None) -> None:
        self.model_name = model_name
        self.device = resolve_device(device, cpu_threads)
        self.batch_size = batch_size
        self.checkpoint_path: str | None = None
        self._model: Any = None

    def load(self) -> Any:
        if self._model is None:
            comet = _import_comet()
            path = _comet_checkpoint(comet, self.model_name)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=_COMET_EXTRA_KEY)
                model = comet.load_from_checkpoint(path)
            model.eval()
            self._model, self.checkpoint_path = model, path
        return self._model

    @property
    def referenceless(self) -> bool:
        """참조 문장 없이 점수를 내는 모델인지(comet 의 requires_references() 로 판단).

        ReferencelessRegression(wmt20-comet-qe-da 등)은 참, RegressionMetric(wmt22-comet-da 등)은 거짓.
        UnifiedMetric 은 [mt, ref] 로만 학습한 모델만 거짓(xCOMET, CometKiwi 는 참).
        """
        return not self.load().requires_references()

    @property
    def uses_references(self) -> bool:
        """refs 를 주면 실제로 입력에 넣는 모델인지."""
        model = self.load()
        segments = getattr(model.hparams, "input_segments", None)
        if segments is not None:
            return "ref" in segments
        return bool(model.requires_references())

    def score(self, srcs: Sequence[str], mts: Sequence[str], refs: Sequence[str] | None = None) -> np.ndarray:
        """mts[i](후보)의 점수 배열. 참조형 모델인데 refs 가 없으면 예외."""
        n = len(mts)
        if len(srcs) != n or (refs is not None and len(refs) != n):
            n_refs = len(refs) if refs is not None else 0
            raise ValueError(f"원문 {len(srcs)}개, 후보 {n}개, 참조 {n_refs}개의 수가 맞지 않습니다.")
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        model = self.load()
        if refs is None and model.requires_references():
            raise ValueError(f"{self.model_name} 은 참조 문장이 필요한 COMET 모델입니다. "
                             "--comet-ref draft 로 초벌을 참조로 쓰거나 참조 없는 모델을 고르세요.")
        use_refs = refs is not None and self.uses_references
        samples = [{"src": s, "mt": m} for s, m in zip(srcs, mts)]
        if use_refs:
            for sample, r in zip(samples, refs):
                sample["ref"] = r
        if self.device == "cpu":
            gpus, devices, accelerator = 0, None, "cpu"
        else:
            index = self.device.partition(":")[2]
            gpus, devices, accelerator = 1, ([int(index)] if index else None), "cuda"
        with warnings.catch_warnings():
            # 데이터 로더 작업자 수 안내(PossibleUserWarning). Windows 에서는 작업자 0개가 맞다.
            warnings.filterwarnings("ignore", message=".*does not have many workers.*")
            warnings.filterwarnings("ignore", message=".*available but not used.*")
            # pytorch-lightning 2.6.6 이 torch 2.11 에서 낡은 LeafSpec 을 써서 나오는 안내
            warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")
            out = model.predict(samples, batch_size=self.batch_size, gpus=gpus, devices=devices,
                                accelerator=accelerator, num_workers=0, progress_bar=False)
        return np.asarray(out.scores, dtype=np.float64)

    def close(self) -> None:
        """모델을 내리고 메모리를 비운다. 다시 쓰면 다시 불러온다."""
        self._model = None
        free_memory(self.device)
