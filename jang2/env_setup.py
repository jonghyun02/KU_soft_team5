from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PROJECT_DIR / "requirements.txt"
DEFAULT_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
DEFAULT_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"


def get_default_venv_dir() -> Path:
    env_override = os.environ.get("EDARCH_VENV_DIR")
    if env_override:
        return Path(env_override)
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "EDarch_venv"
    return Path.home() / ".EDarch_venv"


def get_default_checkpoint_path() -> Path:
    env_override = os.environ.get("EDARCH_CKPT_DIR")
    if env_override:
        return Path(env_override) / "EDarch_ckpt.pt"
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "EDarch_checkpoints" / "EDarch_ckpt.pt"
    return PROJECT_DIR / "checkpoints" / "EDarch_ckpt.pt"


def get_venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def is_running_in_venv(venv_dir: Path) -> bool:
    try:
        return Path(sys.prefix).resolve() == venv_dir.resolve()
    except OSError:
        return False


def _probe_torch_backend(python_exe: Path) -> str | None:
    check_imports = (
        "import torch, torchaudio, soundfile"
        if REQUIREMENTS_FILE.exists()
        else "import torch, torchaudio"
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
        target_label = f"CPU ({DEFAULT_CPU_INDEX_URL})" if cpu_only else f"CUDA 12.8 ({torch_index_url})"
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
            pip_cmd.extend(["torch>=2.7.0", "torchaudio>=2.7.0,<2.9.0"])
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
    import torch

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
