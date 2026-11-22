"""가상환경 자동 구축, 디바이스 초기화, 원자적 파일 저장 및 공통 CLI 인자 유틸리티 모듈."""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

PROJECT_DIR = Path(__file__).resolve().parent.parent
REQUIREMENTS_FILE = PROJECT_DIR / "requirements.txt"
DEFAULT_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
DEFAULT_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"


def get_default_venv_dir() -> Path:
    """OneDrive 동기화 충돌을 피하기 위한 기본 가상환경 디렉터리 경로를 반환합니다."""
    env_override = os.environ.get("EDARCH_VENV_DIR")
    if env_override:
        return Path(env_override)
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "EDarch_venv"
    return Path.home() / ".EDarch_venv"


def get_default_checkpoint_path() -> Path:
    """Stage 1 지도학습 체크포인트의 기본 파일 경로를 반환합니다."""
    env_override = os.environ.get("EDARCH_CKPT_DIR")
    if env_override:
        return Path(env_override) / "EDarch_ckpt.pt"
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "EDarch_checkpoints" / "EDarch_ckpt.pt"
    return PROJECT_DIR / "checkpoints" / "EDarch_ckpt.pt"


def get_default_grpo_checkpoint_path() -> Path:
    """Stage 2 GRPO 강화학습 체크포인트의 기본 파일 경로를 반환합니다."""
    return get_default_checkpoint_path().with_name("EDarch_grpo_ckpt.pt")


def get_venv_python(venv_dir: Path) -> Path:
    """지정된 가상환경 디렉터리 내부의 파이썬 실행 파일 경로를 반환합니다."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def is_running_in_venv(venv_dir: Path) -> bool:
    """현재 프로세스가 목표 가상환경 내부에서 실행 중인지 여부를 확인합니다."""
    try:
        return Path(sys.prefix).resolve() == venv_dir.resolve()
    except OSError:
        return False


def _probe_torch_backend(python_exe: Path) -> str | None:
    """대상 파이썬 실행 파일에서 필수 오디오·토크나이저·딥러닝 패키지 설치 여부와 백엔드를 점검합니다."""
    check_imports = (
        "import torch, torchaudio, soundfile, sentencepiece"
        if REQUIREMENTS_FILE.exists()
        else "import torch, torchaudio, sentencepiece"
    )
    probe = subprocess.run(
        [
            str(python_exe),
            "-c",
            f"{check_imports}; print('cuda' if torch.version.cuda is not None else 'cpu')",
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return None
    return probe.stdout.strip()


def ensure_venv_and_reexec(
    script_path: Path,
    venv_dir: Path,
    torch_index_url: str | None = DEFAULT_CUDA_INDEX_URL,
    cpu_only: bool = False,
    reinstall: bool = False,
    no_venv: bool = False,
) -> None:
    """가상환경 및 필수 패키지를 자동으로 구성한 뒤 가상환경 인터프리터로 스크립트를 재실행합니다."""
    if no_venv or os.environ.get("EDARCH_REEXEC") == "1":
        return

    in_target_venv = is_running_in_venv(venv_dir)
    venv_python = Path(sys.executable) if in_target_venv else get_venv_python(venv_dir)

    if not venv_python.exists():
        print(f"[Setup] 가상환경 생성 중: {venv_dir}")
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    installed_backend = _probe_torch_backend(venv_python)
    backend_mismatch = (
        installed_backend is not None and not cpu_only and installed_backend != "cuda"
    )
    need_install = reinstall or (installed_backend is None) or backend_mismatch
    force_reinstall = reinstall or backend_mismatch

    if in_target_venv and not need_install:
        return

    if need_install:
        target_label = (
            f"CPU ({DEFAULT_CPU_INDEX_URL})"
            if cpu_only
            else f"CUDA 12.8 ({torch_index_url})"
        )
        print(f"[Setup] pip 업그레이드 및 필수 패키지 설치 중 ({target_label})...")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
        )
        pip_cmd = [str(venv_python), "-m", "pip", "install"]
        if force_reinstall:
            pip_cmd.extend(["--upgrade", "--force-reinstall"])
        index_url = DEFAULT_CPU_INDEX_URL if cpu_only else torch_index_url
        if index_url:
            pip_cmd.extend(
                [
                    "--index-url",
                    index_url,
                    "--extra-index-url",
                    "https://pypi.org/simple",
                ]
            )
        if REQUIREMENTS_FILE.exists():
            pip_cmd.extend(["-r", str(REQUIREMENTS_FILE)])
        else:
            pip_cmd.extend(
                [
                    "torch>=2.7.0",
                    "torchaudio>=2.7.0,<2.9.0",
                    "soundfile>=0.12.1",
                    "sentencepiece>=0.2.0",
                ]
            )
        subprocess.run(pip_cmd, check=True)

    print(f"[Setup] 가상환경({venv_python})으로 스크립트 실행 전환...\n")
    child_env = {**os.environ, "EDARCH_REEXEC": "1"}
    try:
        result = subprocess.run(
            [str(venv_python), str(script_path.resolve()), *sys.argv[1:]],
            env=child_env,
        )
    except KeyboardInterrupt:
        sys.exit(130)
    sys.exit(result.returncode)


def setup_device(cpu_only: bool, seed: int, tag: str = "EDarch"):
    """난수 시드를 고정하고 CUDA 가용성에 따라 TF32 가속이 설정된 PyTorch 디바이스를 반환합니다."""
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = torch.cuda.is_available()
    use_cuda = (not cpu_only) and cuda_available
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        gpu_name = torch.cuda.get_device_name(0)
        print(
            f"[{tag}] Device: {device} ({gpu_name}) | "
            f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}"
        )
    else:
        if not cpu_only:
            print(
                f"[{tag}] WARNING: CUDA를 사용할 수 없어 CPU로 폴백합니다. "
                f"(PyTorch={torch.__version__}, built_cuda={torch.version.cuda}) "
                "NVIDIA 드라이버가 CUDA 12.8을 지원하는지 확인하세요."
            )
        else:
            print(f"[{tag}] Device: cpu | PyTorch: {torch.__version__}")
    return device


def atomic_torch_save(
    payload: Mapping[str, Any],
    save_path: Path,
    max_retries: int = 5,
) -> None:
    """임시 파일에 먼저 기록한 뒤 원자적 교체와 재시도를 통해 안전하게 PyTorch 객체를 저장합니다."""
    import torch

    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    try:
        torch.save(dict(payload), tmp_path)
        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, save_path)
                return
            except PermissionError:
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.1 * (2**attempt))
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def add_env_cli_args(
    parser: argparse.ArgumentParser,
    include_seed: bool = True,
) -> None:
    """실행 스크립트들이 공통으로 사용하는 가상환경·디바이스·시드 CLI 인자를 파서에 등록합니다."""
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=get_default_venv_dir(),
        help="가상환경 경로 (기본값: OneDrive 밖 %%LOCALAPPDATA%%\\EDarch_venv)",
    )
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help="가상환경 자동 생성을 건너뛰고 현재 활성 파이썬 환경에서 바로 실행",
    )
    parser.add_argument(
        "--torch-index-url",
        type=str,
        default=DEFAULT_CUDA_INDEX_URL,
        help=f"PyTorch CUDA 인덱스 URL (기본값: {DEFAULT_CUDA_INDEX_URL})",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="CUDA 대신 CPU 전용으로 설치 및 실행",
    )
    parser.add_argument(
        "--reinstall",
        action="store_true",
        help="필수 패키지 강제 재설치",
    )
    if include_seed:
        parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
