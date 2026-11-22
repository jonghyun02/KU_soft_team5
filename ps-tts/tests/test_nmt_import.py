"""모델 1 모듈 격리 불러오기 시험. 진짜 체크포인트는 올리지 않고 tmp_path 의 가짜 모듈로만 본다."""
import sys
from pathlib import Path
from types import ModuleType

import pytest

from pstts.nmt import NMT_MODULES, isolated_import


def _write_fake_nmt(folder: Path, decode_body: str) -> Path:
    files = {
        "common.py": 'PAD = 0\nNAME = "가짜 common"\n',
        "data.py": "from common import NAME\nDATA = NAME + ' / data'\n",
        "model.py": "from common import PAD\n\ndef build_model():\n    return PAD + 1\n",
        "decode.py": decode_body,
    }
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    return folder


GOOD_DECODE = (
    "from common import NAME\nfrom data import DATA\nfrom model import build_model\n\n"
    "def translate():\n    return NAME, DATA, build_model()\n"
)


@pytest.fixture
def clean_modules(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """이미 다른 'common' 모듈이 있는 상황을 만든다. data/model/decode 는 없는 상태."""
    sentinel = ModuleType("common")
    sentinel.NAME = "원래 common"
    monkeypatch.setitem(sys.modules, "common", sentinel)
    for name in ("data", "model", "decode"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return sentinel


def test_loads_and_restores(tmp_path: Path, clean_modules: ModuleType) -> None:
    folder = _write_fake_nmt(tmp_path, GOOD_DECODE)
    path_before = list(sys.path)
    bytecode_before = sys.dont_write_bytecode

    modules = isolated_import(folder)

    assert set(modules) == set(NMT_MODULES)
    assert modules["decode"].translate() == ("가짜 common", "가짜 common / data", 1)
    assert sys.modules["common"] is clean_modules            # 원래 모듈이 되돌아옴
    assert all(name not in sys.modules for name in ("data", "model", "decode"))
    assert sys.path == path_before
    assert not (folder / "__pycache__").exists()             # 그 폴더에 .pyc 를 쓰지 않음
    assert sys.dont_write_bytecode == bytecode_before


def test_restores_after_import_error(tmp_path: Path, clean_modules: ModuleType) -> None:
    folder = _write_fake_nmt(tmp_path, "import common\nimport data\nraise RuntimeError('깨진 decode')\n")
    path_before = list(sys.path)

    with pytest.raises(RuntimeError, match="깨진 decode"):
        isolated_import(folder)

    assert sys.modules["common"] is clean_modules
    assert all(name not in sys.modules for name in ("data", "model", "decode"))
    assert sys.path == path_before


def test_each_call_gives_fresh_modules(tmp_path: Path, clean_modules: ModuleType) -> None:
    folder = _write_fake_nmt(tmp_path, GOOD_DECODE)
    first = isolated_import(folder)
    second = isolated_import(folder)
    assert first["decode"] is not second["decode"]
    assert first["decode"].translate() == second["decode"].translate()


def test_missing_entry_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        isolated_import(tmp_path)
