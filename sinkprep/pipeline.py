"""전체 실행: catalog → convert → ocr → lines → parse → llm → asr → align → final → review → validate.

llm·asr·align·final 단계는 실패해도 전체를 멈추지 않는다(그 문서·영화는 문제 기록만 남기고 다음 단계로).
all 에서 llm 단계는 캐시만 쓴다(--llm 을 주면 캐시에 없는 것도 부름).
all 에서 asr 단계도 캐시만 쓴다: 받아쓰기 모델을 올리지 않는다. 받아쓰기는 몇 시간이 걸리고 옵션(--passes 등)이
영화마다 다를 수 있어 따로 돌린다(python -I -m sinkprep asr). 그래서 asr 모듈이 '`args.cmd == "all"` 이면 모델을 부르지
않고 이미 있는 asr.json 을 다른 옵션으로 덮어쓰지도 않는다'고 밝힌 경우(모듈 상수 ALL_CACHE_ONLY = True)에만 부르고,
아니면 건너뛴다. align·final 은 work/asr.json 이 이미 있는 영화만 다룬다.
"""
from __future__ import annotations

import importlib
import time

STAGES = ["catalog", "convert", "ocr", "lines", "parse", "llm", "asr", "align", "final", "review", "validate"]
SOFT = {"llm", "asr", "align", "final"}     # 실패해도 다음 단계로
CACHE_ONLY_FLAG = {"asr": "ALL_CACHE_ONLY"}  # all 안에서는 이 모듈 상수가 참일 때만 부르는 단계


def run(args) -> int:
    from .__main__ import COMMANDS

    for name in STAGES:
        t = time.time()
        print(f"== {name} ==", flush=True)
        try:
            mod = importlib.import_module(COMMANDS[name][0])
            flag = CACHE_ONLY_FLAG.get(name)
            if flag and not getattr(mod, flag, False):
                print(f"  {name}: all 에서는 건너뜀(모델을 부르지 않는 캐시 전용 실행을 모듈이 밝히지 않음: {flag}). "
                      f"따로 돌릴 것: python -I -m sinkprep {name}", flush=True)
                rc = 0
            else:
                rc = mod.run(args)
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
