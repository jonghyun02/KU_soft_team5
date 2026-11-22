"""모델 1: 사용자가 학습한 Transformer-base 영→한 번역기(enko-nmt)로 초벌을 만든다.

enko-nmt 의 모듈(common, data, model, decode)은 최상위 이름이라 다른 패키지와 섞일 수 있다.
그래서 잠깐 sys.path 앞에 넣고 불러온 뒤 sys.modules 를 원래대로 되돌린다.
enko-nmt 폴더는 읽기만 한다(.pyc 도 쓰지 않음). 지금 학습 중이라 파일을 고치지 않는다.
"""
from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import torch

from .config import Config
from .scorers import free_memory, resolve_device

NMT_MODULES = ("common", "data", "model", "decode")


def isolated_import(directory: str | Path, names: Sequence[str] = NMT_MODULES,
                    entry: str = "decode") -> dict[str, ModuleType]:
    """directory 의 최상위 모듈 entry 를 다른 모듈과 섞이지 않게 불러온다.

    names 와 같은 이름이 sys.modules 에 이미 있으면 잠깐 치워 두었다가 끝나면 되돌린다.
    불러온 모듈은 sys.modules 에서 다시 빼서 뒤에 오는 import 를 가리지 않게 한다.
    돌려주는 값은 이번에 불러온 names 모듈들(이름 -> 모듈)이다.
    """
    directory = Path(directory).resolve()
    if entry not in names:
        raise ValueError(f"entry({entry})는 names 안에 있어야 합니다.")
    if not (directory / f"{entry}.py").is_file():
        raise FileNotFoundError(f"{directory} 에 {entry}.py 가 없습니다.")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    path_entry = str(directory)
    sys.path.insert(0, path_entry)
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        importlib.invalidate_caches()
        importlib.import_module(entry)
        return {name: sys.modules[name] for name in names if name in sys.modules}
    finally:
        sys.dont_write_bytecode = dont_write
        if path_entry in sys.path:
            sys.path.remove(path_entry)
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


class Model1Drafter:
    """enko-nmt 체크포인트로 영어 문장 목록을 한국어 초벌로 번역한다(beam search)."""

    def __init__(self, nmt_dir: str | Path = Config.nmt_dir, ckpt: str | Path = Config.nmt_ckpt,
                 device: str = "cuda", beam: int = Config.nmt_beam, batch_size: int = 64,
                 cpu_threads: int | None = None) -> None:
        self.nmt_dir = Path(nmt_dir).resolve()
        ckpt_path = Path(ckpt)
        self.ckpt = ckpt_path if ckpt_path.is_absolute() else self.nmt_dir / ckpt_path
        if not self.ckpt.is_file():
            raise FileNotFoundError(f"모델 1 체크포인트가 없습니다: {self.ckpt}")
        if beam < 1:
            raise ValueError(f"beam 은 1 이상이어야 합니다: {beam}")
        self.device = resolve_device(device, cpu_threads)
        self.beam = beam
        self.batch_size = batch_size

        modules = isolated_import(self.nmt_dir)
        self._decode = modules["decode"]
        build_model = modules["model"].build_model
        import sentencepiece as spm

        # decode.load_checkpoint 는 체크포인트 속 spm 상대 경로(data/spm_v2/en.model)를 현재 폴더 기준으로 풀어서
        # 여기서는 같은 일을 nmt_dir 기준으로 직접 한다.
        ck = torch.load(self.ckpt, map_location="cpu", weights_only=False)
        try:
            spm_paths, config, state = ck["spm"], ck["config"], ck["model"]
        except KeyError as e:
            raise ValueError(f"enko-nmt 체크포인트 형식이 아닙니다({e.args[0]} 칸 없음): {self.ckpt}") from None
        self.sp_en = spm.SentencePieceProcessor(model_file=str(self._under_nmt(spm_paths["en"])))
        self.sp_ko = spm.SentencePieceProcessor(model_file=str(self._under_nmt(spm_paths["ko"])))
        model = build_model(config, self.sp_en.get_piece_size(), self.sp_ko.get_piece_size(), dropout=0.0)
        model.load_state_dict(state)
        self.info = {"epoch": ck.get("epoch"), "step": ck.get("step"), "averaged": ck.get("averaged"),
                     "config": dict(config)}
        del ck, state
        self.model: torch.nn.Module | None = model.to(self.device).eval()

    def _under_nmt(self, path: str) -> Path:
        p = Path(path)
        p = p if p.is_absolute() else self.nmt_dir / p
        if not p.is_file():
            raise FileNotFoundError(f"SentencePiece 모델이 없습니다: {p}")
        return p

    def translate(self, sentences: Sequence[str]) -> list[str]:
        """영어 문장 목록 -> 한국어 초벌 목록(같은 순서). 빈 문장은 빈 문자열."""
        if self.model is None:
            raise RuntimeError("모델 1 을 이미 내렸습니다(close 뒤에는 새로 만드세요).")
        texts = [" ".join(s.split()) for s in sentences]
        out = [""] * len(texts)
        idx = [i for i, t in enumerate(texts) if t]
        if idx:
            hyps = self._decode.translate(self.model, self.sp_en, self.sp_ko, [texts[i] for i in idx],
                                          batch_size=self.batch_size, method="beam", beam=self.beam,
                                          device=self.device)
            for i, h in zip(idx, hyps):
                out[i] = h
        return out

    def close(self) -> None:
        """모델을 내리고 메모리(GPU 포함)를 비운다."""
        self.model = None
        free_memory(self.device)

    def __enter__(self) -> Model1Drafter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
