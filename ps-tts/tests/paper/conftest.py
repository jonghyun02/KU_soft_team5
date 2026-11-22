"""tests/paper 공통: 시험 모듈 이름 겹침 풀기와 가짜 모델.

이름 겹침: tests/ 에도 test_io.py, test_pipeline.py 가 있다. __init__.py 가 없는 폴더의 시험 모듈은 pytest 기본
가져오기(prepend)에서 파일 이름(test_io)이 모듈 이름이라 둘째 파일을 모을 때 'import file mismatch' 로 멈춘다.
그래서 이 폴더의 시험 모듈을 모을 때만 가져오기 방식을 importlib 로 바꾼다(모듈 이름이 tests.paper.test_io 가 됨).
tests/ 의 기존 시험은 그대로 prepend 로 가져온다. pytest.ini 는 건드리지 않는다.
"""
from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import pytest

from pstts.llm import GenReply, LLMUnavailable


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector: pytest.Collector):
    if not isinstance(collector, pytest.Module):
        return (yield)
    option = collector.config.option
    before = option.importmode
    option.importmode = "importlib"
    try:
        return (yield)
    finally:
        option.importmode = before


# ---------------------------------------------------------------- 가짜 모델

class FakeDrafter:
    """모델 1 대신. 영어 -> 정해 둔 초벌(없으면 default). 번역한 문장과 닫힘을 적는다."""

    def __init__(self, table: dict[str, str] | None = None, default: str = "") -> None:
        self.table = dict(table or {})
        self.default = default
        self.seen: list[list[str]] = []
        self.closed = False

    def translate(self, sentences: Sequence[str]) -> list[str]:
        self.seen.append(list(sentences))
        return [self.table.get(s, self.default) for s in sentences]

    def close(self) -> None:
        self.closed = True


class FakeLabse:
    """LaBSE 대신. 후보 글마다 정해 둔 값, 없으면 default."""

    def __init__(self, table: dict[str, float] | None = None, default: float = 0.9) -> None:
        self.table = dict(table or {})
        self.default = default
        self.calls: list[tuple[list[str], list[str]]] = []
        self.closed = False

    def similarity(self, srcs: Sequence[str], tgts: Sequence[str]) -> np.ndarray:
        assert len(srcs) == len(tgts)
        self.calls.append((list(srcs), list(tgts)))
        return np.array([self.table.get(t, self.default) for t in tgts], dtype=float)

    def close(self) -> None:
        self.closed = True


class FakeComet:
    """COMET 대신. 후보 글마다 정해 둔 값, 없으면 글 길이로 정하는 값(결정적)."""

    checkpoint_path = "fake-comet.ckpt"

    def __init__(self, table: dict[str, float] | None = None) -> None:
        self.table = dict(table or {})
        self.calls: list[tuple[list[str], list[str]]] = []
        self.closed = False

    def score(self, srcs: Sequence[str], mts: Sequence[str], refs: Sequence[str] | None = None) -> np.ndarray:
        assert refs is None and len(srcs) == len(mts)
        self.calls.append((list(srcs), list(mts)))
        return np.array([self.table.get(m, (len(m) % 7) / 10.0) for m in mts], dtype=float)

    def close(self) -> None:
        self.closed = True


class ScriptGen:
    """모델 2 대신. script(stage, round, 요청) 이 돌려주는 글 목록을 그 줄의 답으로 준다.

    pstts.llm 백엔드처럼 avoid 와 겹치는 글은 빼고 n_new 개까지 자른다. stop_stage 를 주면 그 단계에서 LLMUnavailable.
    """

    name = "script"

    def __init__(self, script, stop_stage: str | None = None) -> None:
        self.script = script
        self.stop_stage = stop_stage
        self.requests: list[tuple[str, int, str, int, list[str]]] = []   # (stage, round, id, n_new, avoid)
        self.prompts: list[tuple[str, str]] = []
        self.calls = 0

    def generate(self, reqs, prompt=None) -> GenReply:
        assert prompt is not None, "논문 판은 늘 PromptSpec 으로 부른다"
        self.prompts.append(prompt.build(list(reqs)))
        out = GenReply()
        if self.stop_stage == prompt.stage:
            err = LLMUnavailable("가짜: 모델 2 를 쓸 수 없음")
            err.partial = GenReply()
            err.partial.update({str(r.id): [] for r in reqs})
            raise err
        self.calls += 1
        out.calls = 1
        for r in reqs:
            self.requests.append((prompt.stage, int(r.round), str(r.id), int(r.n_new), list(r.avoid)))
            got = [t for t in self.script(prompt.stage, int(r.round), r) if t not in r.avoid]
            out[str(r.id)] = list(dict.fromkeys(got))[:int(r.n_new)]
            if not out[str(r.id)]:
                out.failed.append(str(r.id))
        return out


@pytest.fixture
def fakes() -> SimpleNamespace:
    """가짜 모델 클래스 묶음(시험 모듈끼리 가져오지 않고 나눠 쓰려고 픽스처로 준다)."""
    return SimpleNamespace(Drafter=FakeDrafter, Labse=FakeLabse, Comet=FakeComet, Gen=ScriptGen)
