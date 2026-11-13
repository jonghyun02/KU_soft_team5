"""전체 실행: catalog → convert → ocr → lines → parse → review → validate."""
from __future__ import annotations

import importlib
import time

STAGES = ["catalog", "convert", "ocr", "lines", "parse", "review", "validate"]


def run(args) -> int:
    from .__main__ import COMMANDS

    for name in STAGES:
        t = time.time()
        print(f"== {name} ==", flush=True)
        rc = importlib.import_module(COMMANDS[name][0]).run(args)
        print(f"== {name} 끝 ({time.time() - t:.0f}초, rc={rc or 0}) ==", flush=True)
        if rc and name != "validate":
            return int(rc)
        if name == "validate":
            return int(rc or 0)
    return 0
