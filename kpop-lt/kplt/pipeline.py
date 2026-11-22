"""전체 순서를 한 번에 돌리는 실행기. 단계마다 끝 표시 파일이 있으면 건너뛴다.

python -m kplt.pipeline data       # 가사 정렬(lyrics.json), 덤프 추출, 학습 데이터 3단계
python -m kplt.pipeline analysis   # 3절 표 2·3, 그림 3, 표 8
python -m kplt.pipeline chain line_syl   # 한 설정의 general -> nonsingable -> singable
python -m kplt.pipeline nsp        # NSP 모델
python -m kplt.pipeline infer      # 기준 모델 + 8개 모델 번역
python -m kplt.pipeline evaluate   # 표 5·6
"""
from __future__ import annotations

import sys

from .config import CONFIGS, DATA, OUT, RUNS, STAGES


def data():
    from .data import build, dump, match
    if not (DATA / "lyrics.json").exists():
        match.build_lyrics_json()
    if not (DATA / "eval_kpop.json").exists():
        dump.main()
    st = DATA / "stages"
    if not (st / "general" / "section" / "train.jsonl").exists():
        build.build_general()
    if not (st / "nonsingable" / "section" / "train.jsonl").exists():
        build.build_nonsingable()
    if not (st / "singable" / "section" / "train.jsonl").exists():
        build.build_singable()


def chain(config: str):
    """general -> nonsingable -> singable. 다음 단계 데이터가 아직 없으면 (구글 번역 중) 생길 때까지 기다린다."""
    import time

    from .model.train import train
    form = CONFIGS[config][0]
    for stage in STAGES:
        if (RUNS / config / stage / "done.json").exists():
            print(config, stage, "이미 끝남", flush=True)
            continue
        need = [DATA / "stages" / stage / form / f"{s}.jsonl" for s in ("train", "valid")]
        if not all(p.exists() for p in need):
            print(time.strftime("%H:%M:%S"), config, stage, "데이터를 기다림 (translate.cmd 가 만듦)", flush=True)
            while not all(p.exists() for p in need):
                time.sleep(300)
            time.sleep(30)   # 파일 쓰기가 끝나도록
        train(config, stage)


def infer():
    import json

    from .model.infer import all_model_names, translate_songs
    songs = json.loads((DATA / "eval_kpop.json").read_text(encoding="utf-8"))
    d = OUT / "translations"
    d.mkdir(parents=True, exist_ok=True)
    for name in all_model_names():
        f = d / f"{name}.json"
        if f.exists():
            continue
        res = translate_songs(name, songs)
        f.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        print("번역 끝:", name)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    what = sys.argv[1]
    if what == "data":
        data()
    elif what == "analysis":
        from . import analysis
        analysis.main()
    elif what == "chain":
        assert sys.argv[2] in CONFIGS
        chain(sys.argv[2])
    elif what == "nsp":
        from .model import nsp
        if not (RUNS / "nsp" / "done.json").exists():
            nsp.main()
    elif what == "infer":
        infer()
    elif what == "evaluate":
        sys.argv = sys.argv[:1]
        from . import evaluate
        evaluate.main()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
