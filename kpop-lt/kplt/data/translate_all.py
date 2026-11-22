"""논문이 구글 번역을 쓰는 곳을 모두 미리 번역해 캐시에 넣는다 (막히면 기다렸다 이어감).

1. sts 용 한→영 (3.1절): 논문 데이터셋 한국어 줄 전부, 평가 K-pop 원문 줄 전부
2. Non-singable 영→한 (4.2절): build_nonsingable 이 번역하고 학습 데이터까지 만든다

다 끝나면 data\\translate_done.json. 실행: translate.cmd (세션과 떨어진 창)
"""
from __future__ import annotations

import json
import time

from ..config import DATA
from ..gtranslate import GoogleTranslator
from .build import build_nonsingable


def main():
    t0 = time.time()
    gt = GoogleTranslator()
    lyrics = json.loads((DATA / "lyrics.json").read_text(encoding="utf-8"))
    kr_lines = [k for song in lyrics.values() for sec in song for k, _ in sec]
    songs = json.loads((DATA / "eval_kpop.json").read_text(encoding="utf-8"))
    eval_lines = [l for s in songs for sec in s["sections"] for l in sec]
    print(f"한→영: 데이터셋 {len(kr_lines)}줄, 평가 {len(eval_lines)}줄", flush=True)
    gt.translate_lines(kr_lines, "ko", "en", progress=True)
    gt.translate_lines(eval_lines, "ko", "en", progress=True)
    print("한→영 끝, 영→한 시작", flush=True)
    build_nonsingable()
    info = {"seconds": time.time() - t0, "requests": gt.n_requests,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (DATA / "translate_done.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print("모두 끝", info, flush=True)


if __name__ == "__main__":
    main()
