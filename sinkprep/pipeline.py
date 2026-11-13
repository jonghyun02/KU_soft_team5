"""전체 실행: catalog → convert → ocr → lines → parse → llm → review → validate.

llm 단계는 claude 가 없거나 실패해도 전체를 멈추지 않는다(그 문서는 규칙 판 유지 + 문제 기록).
all 에서 llm 단계는 캐시만 쓴다(--llm 을 주면 캐시에 없는 것도 부름).
"""
from __future__ import annotations

import importlib
import time

STAGES = ["catalog", "convert", "ocr", "lines", "parse", "llm", "review", "validate"]
SOFT = {"llm"}          # 실패해도 다음 단계로


def run(args) -> int:
    from .__main__ import COMMANDS

    for name in STAGES:
        t = time.time()
        print(f"== {name} ==", flush=True)
        try:
            rc = importlib.import_module(COMMANDS[name][0]).run(args)
        except Exception as e:  # noqa: BLE001
            if name not in SOFT:
                raise
            print(f"  ! {name} 단계 예외(건너뜀): {type(e).__name__}: {e}", flush=True)
            rc = 1
        print(f"== {name} 끝 ({time.time() - t:.0f}초, rc={rc or 0}) ==", flush=True)
        if rc and name not in SOFT and name != "validate":
            return int(rc)
        if name == "validate":
            return int(rc or 0)
    return 0
