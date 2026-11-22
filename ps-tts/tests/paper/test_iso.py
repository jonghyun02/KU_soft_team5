"""논문 판 ISO 시험: K', R, K(T)(식 2~4), 식 6 경계(정확히 26 은 통과), 식 7, 첫 통과에서 멈춤, 60번 상한,
코사인 대안, seconds 칸, 두 차수 연달아 새 후보 없음, 모델 2 멈춤, 단계 재생.

Frontend 는 frontend.py 를 불러오지 않고 같은 꼴의 가짜(FakeFront)를, LaBSE 와 바꿔 쓰기는 가짜 함수를 쓴다.
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from pstts.llm import FileBackend, GenReply
from pstts.paper.config import PaperConfig
from pstts.paper.iso import (IsoInput, IsoRun, check_inputs, fallback_index, k_frames, k_prime_of, ok_len, ok_sem,
                             run_iso, speaking_rate)
from pstts.paper.prompts import STAGE_ISO, GenOutcome, IsoRequest, iso_generate, keep_new

B = "<b>"
EXACT = PaperConfig(frame_sec_sa=1 / 64)     # 2 의 거듭제곱 프레임이라 식 (2)~(6) 의 셈이 정확하다
REAL_F = 256 / 22050


class FakeFront:
    """frontend.Frontend 꼴의 가짜. 토큰 = 글자(사이사이 빈 토큰).

    길이: totals 에 있는 글은 그 전체 길이(초)를 첫 글자 토큰에 몰아 준다(합이 정확히 그 값).
    없는 글은 글자 토큰마다 per_char 초. 빈 토큰은 0.
    """
    name = "fake"
    frame_sec = REAL_F

    def __init__(self, totals: dict[str, float] | None = None, per_char: float = 0.5) -> None:
        self.totals = dict(totals or {})
        self.per_char = per_char
        self.calls: list[tuple[str, list[str]]] = []

    def tokens(self, text: str, lang: str) -> list[str]:
        out = [B]
        for ch in text:
            out += [ch, B]
        return out

    def durations(self, seqs: list[list[str]], lang: str) -> list[np.ndarray]:
        texts = ["".join(t for t in s if t != B) for s in seqs]
        self.calls.append((lang, texts))
        out = []
        for s, text in zip(seqs, texts):
            a = np.zeros(len(s))
            idx = [i for i, t in enumerate(s) if t != B]
            if text in self.totals:
                if idx:
                    a[idx[0]] = self.totals[text]
            else:
                a[idx] = self.per_char
            out.append(a)
        return out


class FakeSim:
    """LaBSE 대신. 후보 글마다 정해 둔 값, 없으면 default. 부른 차례를 적는다."""

    def __init__(self, table: dict[str, float] | None = None, default: float = 0.9) -> None:
        self.table = dict(table or {})
        self.default = default
        self.calls: list[list[str]] = []

    def __call__(self, srcs, tgts) -> np.ndarray:
        assert len(srcs) == len(tgts)
        self.calls.append(list(tgts))
        return np.array([self.table.get(t, self.default) for t in tgts], dtype=float)


class FakeGen:
    """바꿔 쓰기 대신. script(차수, 요청) 의 후보를 돌려주고 받은 요청을 적는다."""

    def __init__(self, script, stopped_at: int | None = None) -> None:
        self.script = script
        self.stopped_at = stopped_at
        self.requests: list[list[IsoRequest]] = []

    def __call__(self, reqs: list[IsoRequest]) -> GenOutcome:
        self.requests.append([replace(r, avoid=list(r.avoid)) for r in reqs])
        k = len(self.requests)
        out = GenOutcome(candidates={r.id: list(self.script(k, r)) for r in reqs}, calls=1)
        if self.stopped_at == k:
            out.stopped = "claude 사용량 한도에 걸렸습니다."
        return out


def _run(lines, front, cfg=EXACT, gen=None, sim=None) -> tuple[IsoRun, FakeGen, FakeSim]:
    gen = gen or FakeGen(lambda k, r: [])
    sim = sim or FakeSim()
    return run_iso(lines, front, cfg, paraphrase=gen, similarity=sim, log=lambda m: None), gen, sim


# ---------------------------------------------------------------- 식 (2)~(7)

def test_formulas_match_paper_equations() -> None:
    f = REAL_F
    kp, basis = k_prime_of(2.1, 1.7, f)
    assert basis == "seconds" and kp == pytest.approx(2.1 / f)
    r = speaking_rate(kp, 1.7, f)
    # 논문 꼴: R = K' / duration_TTS(S_b), K(T_b) = R x duration_TTS(T_b) (TTS 길이를 프레임으로 셈)
    assert r == pytest.approx(kp / (1.7 / f)) == pytest.approx(2.1 / 1.7)
    assert k_frames(r, 1.3, f) == pytest.approx(r * (1.3 / f)) == pytest.approx(kp * 1.3 / 1.7)
    kp2, basis2 = k_prime_of(None, 1.7, f)
    assert basis2 == "tts" and kp2 == pytest.approx(1.7 / f)
    assert speaking_rate(kp2, 1.7, f) == pytest.approx(1.0)            # seconds 가 없으면 R = 1(해석)
    with pytest.raises(ValueError, match="R"):
        speaking_rate(10.0, 0.0, f)
    with pytest.raises(ValueError):
        k_prime_of(None, 1.0, 0.0)


def test_eq6_eq7_boundaries() -> None:
    assert ok_len(126.0, 100.0, 26) and ok_len(74.0, 100.0, 26)          # 정확히 26 은 통과
    assert not ok_len(126.01, 100.0, 26) and not ok_len(73.99, 100.0, 26)
    assert ok_len(126.0 + 1e-11, 100.0, 26)                               # 부동소수 잡음은 견딘다
    assert ok_sem(0.75, 0.75) and not ok_sem(0.7499, 0.75)


def test_exact_26_frames_through_run_iso() -> None:
    # 원문 TTS 1.0초, seconds 없음 -> K' = 64(프레임 1/64초), R = 1. K(T) = 64 x dur_T
    front = FakeFront({"S": 1.0, "T+26": 90 / 64, "T-26": 38 / 64, "T+27": 91 / 64, "P": 1.0})
    lines = [IsoInput("L1", "S", "T+26"), IsoInput("L2", "S", "T-26"), IsoInput("L3", "S", "T+27")]
    gen = FakeGen(lambda k, r: ["P"])
    run, gen, sim = _run(lines, front, gen=gen)
    l1, l2, l3 = run.lines
    assert (l1.k_prime, l1.k_prime_basis, l1.r, l1.src_dur_sec) == (64.0, "tts", 1.0, 1.0)
    assert (l1.status, l1.chosen.k_frames, l1.iterations) == ("draft", 90.0, 0)
    assert (l2.status, l2.chosen.k_frames) == ("draft", 38.0)
    assert l3.candidates[0].k_frames == 91.0 and not l3.candidates[0].ok_len
    assert (l3.status, l3.target, l3.iterations) == ("paraphrase", "P", 1)
    assert [r.id for r in gen.requests[0]] == ["L3"]


def test_exact_26_frames_with_real_frame() -> None:
    f = REAL_F
    front = FakeFront({"S": 1.0, "edge": 1.0 + 26 * f, "over": 1.0 + 26.01 * f})
    run, _, _ = _run([IsoInput("A", "S", "edge"), IsoInput("B", "S", "over")], front, cfg=PaperConfig())
    a, b = run.lines
    assert a.k_prime == pytest.approx(1.0 / f) and a.status == "draft"
    assert a.chosen.k_frames - a.k_prime == pytest.approx(26.0)
    assert not b.candidates[0].ok_len


def test_seconds_field_sets_k_prime() -> None:
    front = FakeFront({"S": 1.0, "D": 1.0, "E": 1.5})
    lines = [IsoInput("A", "S", "D", seconds=1.5), IsoInput("B", "S", "E", seconds=1.5), IsoInput("C", "S", "E")]
    run, _, _ = _run(lines, front)
    a, b, c = run.lines
    assert (a.k_prime, a.k_prime_basis, a.r) == (96.0, "seconds", 1.5)     # 1.5초 / (1/64)
    assert a.candidates[0].k_frames == 96.0 and a.status == "draft"         # R 1.5 x 1.0초 x 64 = 96 = K'
    assert b.candidates[0].k_frames == 144.0 and not b.candidates[0].ok_len
    assert (c.k_prime_basis, c.r, c.candidates[0].k_frames) == ("tts", 1.0, 96.0)


# ---------------------------------------------------------------- 흐름

def test_draft_that_passes_needs_no_paraphrase() -> None:
    front = FakeFront({"S": 1.0, "D": 1.0})
    run, gen, sim = _run([IsoInput("A", "S", "D")], front)
    line = run.lines[0]
    assert (line.status, line.target, line.selected, line.iterations, line.rounds) == ("draft", "D", 0, 0, 0)
    assert gen.requests == [] and run.calls == 0 and run.rows == []
    c = line.candidates[0]
    assert (c.origin, c.round, c.iteration, c.checked, c.passed) == ("draft", 0, 0, True, True)


def test_first_pass_stops_in_generation_order() -> None:
    front = FakeFront({"S": 1.0, "D": 2.0, "A": 2.0, "B": 1.0, "C": 1.0})
    gen = FakeGen(lambda k, r: ["A", "B", "C"])
    run, gen, sim = _run([IsoInput("L1", "S", "D")], front, gen=gen)
    line = run.lines[0]
    assert (line.status, line.target, line.selected, line.iterations) == ("paraphrase", "B", 2, 2)
    assert [(c.text, c.iteration, c.checked) for c in line.candidates] == [
        ("D", 0, True), ("A", 1, True), ("B", 2, True), ("C", None, False)]
    assert line.candidates[3].passed                       # C 도 두 조건을 채우지만 B 가 먼저라 보지 않음
    req = gen.requests[0][0]
    assert (req.id, req.en, req.ko, req.length_ratio, req.n_new, req.round, req.avoid) == (
        "L1", "S", "D", 0.5, 10, 1, ["D"])                 # K'/K(T0) = 64/128
    assert run.rows == [{"id": "L1", "stage": STAGE_ISO, "round": 1, "candidates": ["A", "B", "C"]}]


def test_sixty_iteration_cap_and_fallback() -> None:
    front = FakeFront({"S": 1.0, "D": 2.0})                  # 바꿔 쓴 글(4글자 x 0.5초 = 2초)은 모두 길이 탈락
    cfg = replace(EXACT, iso_batch=7)
    gen = FakeGen(lambda k, r: [f"r{k}-{j}" for j in range(r.n_new)])
    sim = FakeSim({"r3-2": 0.95})
    run, gen, sim = _run([IsoInput("L1", "S", "D")], front, cfg=cfg, gen=gen, sim=sim)
    line = run.lines[0]
    assert [rq[0].n_new for rq in gen.requests] == [7] * 8 + [4]           # 7 x 8 + 4 = 60
    assert line.iterations == 60 and line.rounds == 9 and len(line.candidates) == 61
    assert all(c.checked for c in line.candidates)
    assert (line.status, line.target) == ("fallback", "r3-2") and "60" in line.message
    second = gen.requests[1][0]
    assert second.ko == "D" and second.avoid == ["D"] + [f"r1-{j}" for j in range(7)]   # 바탕은 늘 T0
    assert [c.iteration for c in line.candidates] == list(range(61))


def test_fallback_tie_prefers_first_and_draft_counts() -> None:
    front = FakeFront({"S": 1.0, "D": 2.0})
    run, gen, _ = _run([IsoInput("L1", "S", "D")], front, cfg=replace(EXACT, iso_max=3),
                       gen=FakeGen(lambda k, r: ["xxx1", "xxx2", "xxx3"]), sim=FakeSim(default=0.9))
    line = run.lines[0]
    assert (line.status, line.selected, line.target) == ("fallback", 0, "D")      # 모두 0.9 면 먼저 나온 초벌
    assert line.iterations == 3 and gen.requests[0][0].n_new == 3
    cands = line.candidates
    assert fallback_index(cands) == 0


def test_fallback_uses_all_checked_candidates() -> None:
    sim = FakeSim({"D": 0.70, "a1": 0.72, "a2": 0.74, "a3": 0.74})          # 뜻 탈락만(길이는 모두 통과)
    run, _, _ = _run([IsoInput("L1", "S", "D")], FakeFront({"S": 1.0, "D": 1.0, "a1": 1.0, "a2": 1.0, "a3": 1.0}),
                     cfg=replace(EXACT, iso_max=3), gen=FakeGen(lambda k, r: ["a1", "a2", "a3"]), sim=sim)
    line = run.lines[0]
    assert not any(c.passed for c in line.candidates)
    assert (line.status, line.target) == ("fallback", "a2")                 # 0.74 가 둘이면 먼저 나온 a2


def test_iso_max_zero_goes_straight_to_fallback() -> None:
    run, gen, _ = _run([IsoInput("L1", "S", "D")], FakeFront({"S": 1.0, "D": 2.0}), cfg=replace(EXACT, iso_max=0))
    assert run.lines[0].status == "fallback" and gen.requests == []


def test_two_empty_rounds_in_a_row_stop_the_line() -> None:
    front = FakeFront({"S": 1.0, "D": 2.0})
    run, gen, _ = _run([IsoInput("L1", "S", "D")], front, gen=FakeGen(lambda k, r: []))
    line = run.lines[0]
    assert (line.status, line.target, line.rounds, line.iterations) == ("fallback", "D", 2, 0)
    assert "두 차수 연달아" in line.message and len(gen.requests) == 2
    # 빈 차수 사이에 새 후보가 오면 다시 센다: [], [x], [], [] -> 4차에서 멈춤
    script = {1: [], 2: ["xxxx"], 3: [], 4: []}
    run, gen, _ = _run([IsoInput("L1", "S", "D")], front, gen=FakeGen(lambda k, r: script.get(k, ["yy"])))
    assert run.lines[0].rounds == 4 and run.lines[0].iterations == 1 and len(gen.requests) == 4


def test_repeated_or_extra_candidates_are_not_new() -> None:
    front = FakeFront({"S": 1.0, "D": 2.0})
    # 이미 나온 글(공백, 문장부호만 다름)과 n 보다 많은 후보는 받지 않는다
    gen = FakeGen(lambda k, r: ["D!", "xx00"] + [f"xx{j:02d}" for j in range(15)] if k == 1 else ["xx 00", "xx01."])
    run, gen, _ = _run([IsoInput("L1", "S", "D")], front, cfg=replace(EXACT, iso_batch=10, iso_max=60), gen=gen)
    line = run.lines[0]
    assert [c.text for c in line.candidates[1:11]] == [f"xx{j:02d}" for j in range(10)] and line.rounds == 3
    assert line.status == "fallback" and line.iterations == 10        # 2, 3차는 새 후보가 없어 멈춤


def test_many_lines_share_one_labse_call_per_round() -> None:
    front = FakeFront({"S1": 1.0, "S2": 1.0, "S3": 1.0, "D1": 2.0, "D2": 2.0, "D3": 1.0, "ok": 1.0})
    gen = FakeGen(lambda k, r: ["L1-a", "ok"] if r.id == "L1" else [f"{r.id}-{k}a", f"{r.id}-{k}b"])
    lines = [IsoInput("L1", "S1", "D1"), IsoInput("L2", "S2", "D2"), IsoInput("L3", "S3", "D3")]
    run, gen, sim = _run(lines, front, cfg=replace(EXACT, iso_max=4), gen=gen)
    assert sim.calls[0] == ["D1", "D2", "D3"]                         # 초벌 3개를 한 번에
    assert sim.calls[1] == ["L1-a", "ok", "L2-1a", "L2-1b"]             # 1차 새 후보를 한 번에
    assert front.calls[:3] == [("en", ["S1", "S2", "S3"]), ("ko", ["D1", "D2", "D3"]),
                               ("ko", ["L1-a", "ok", "L2-1a", "L2-1b"])]
    assert [r.id for r in gen.requests[0]] == ["L1", "L2"] and [r.id for r in gen.requests[1]] == ["L2"]
    assert [line.status for line in run.lines] == ["paraphrase", "fallback", "draft"]
    assert run.lines[1].iterations == 4 and len(run.rounds) == 2
    assert run.rounds[0]["checked"] == 4 and run.rounds[0]["passed"] == 1 and run.calls == 2


def test_llm_stopped_sends_remaining_lines_to_fallback() -> None:
    front = FakeFront({"S": 1.0, "D": 2.0, "E": 2.0})
    gen = FakeGen(lambda k, r: ["aaaa"] if r.id == "A" else [], stopped_at=1)
    run, gen, _ = _run([IsoInput("A", "S", "D"), IsoInput("B", "S", "E")], front, gen=gen)
    a, b = run.lines
    assert "사용량 한도" in run.stopped and len(gen.requests) == 1
    assert a.iterations == 1 and a.status == b.status == "fallback"
    assert "모델 2 를 쓸 수 없어" in a.message and "모델 2 를 쓸 수 없어" in b.message


def test_error_lines_and_zero_length_draft() -> None:
    front = FakeFront({"Shh!": 0.0, "S": 1.0, "D": 2.0, "...": 0.0})
    lines = [IsoInput("A", "Shh!", "쉿."), IsoInput("B", "S", "  "), IsoInput("C", "S", "..."), IsoInput("D", "S", "D")]
    run, gen, _ = _run(lines, front, gen=FakeGen(lambda k, r: []))
    a, b, c, d = run.lines
    assert (a.status, a.target, a.r) == ("error", None, None) and "R(식 3)" in a.message
    assert (b.status, b.target) == ("error", None) and "초벌" in b.message
    assert (c.status, c.target, c.length_ratio) == ("fallback", "...", None) and "길이 비율" in c.message
    assert d.status == "fallback" and [r.id for r in gen.requests[0]] == ["D"]


def test_check_inputs() -> None:
    with pytest.raises(ValueError, match="하나도"):
        check_inputs([], EXACT)
    with pytest.raises(ValueError, match="같은 id"):
        check_inputs([IsoInput("A", "x", "y"), IsoInput("A", "z", "w")], EXACT)
    with pytest.raises(ValueError, match="영어 대사"):
        check_inputs([IsoInput("A", " ", "y")], EXACT)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="seconds"):
            check_inputs([IsoInput("A", "x", "y", seconds=bad)], EXACT)
    with pytest.raises(ValueError, match="iso_batch"):
        check_inputs([IsoInput("A", "x", "y")], replace(EXACT, iso_batch=0))


# ---------------------------------------------------------------- 단계 재생(진짜 FileBackend)

class ScriptBackend:
    """generate(reqs, prompt) 만 있는 백엔드. 차수와 줄마다 정해 둔 후보(다듬기 전)를 돌려준다."""
    name = "script"

    def __init__(self, script: dict[tuple[str, int], list[str]]) -> None:
        self.script = script
        self.stages: list[str | None] = []

    def generate(self, reqs, prompt=None) -> GenReply:
        self.stages.append(prompt.stage if prompt else None)
        out = GenReply()
        for r in reqs:
            out[r.id] = keep_new(r.avoid, self.script.get((r.id, r.round), []), r.n_new)
        out.calls = 1
        return out


def test_replay_from_candidates_file_gives_same_decisions(tmp_path) -> None:
    front = FakeFront({"S": 1.0, "D": 2.0, "E": 2.0, "pass": 1.0})
    script = {("A", 1): ["aaa1", "aaa2", "aaa3", "aaa4"], ("A", 2): ["pass", "after"],
              ("B", 1): ["bbb1", "bbb2", "bbb3", "bbb4"], ("B", 2): ["bbb5", "bbb6", "bbb7"]}
    cfg = replace(EXACT, iso_batch=4, iso_max=6)
    lines = [IsoInput("A", "S", "D"), IsoInput("B", "S", "E")]
    sim = FakeSim({"bbb5": 0.97})
    backend = ScriptBackend(script)
    first = run_iso(lines, front, cfg, paraphrase=lambda reqs: iso_generate(backend, reqs), similarity=sim,
                    log=lambda m: None)
    assert backend.stages == ["iso", "iso"] and first.calls == 2
    path = tmp_path / "candidates.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in first.rows), encoding="utf-8")
    replay = FileBackend(path)
    again = run_iso(lines, front, cfg, paraphrase=lambda reqs: iso_generate(replay, reqs), similarity=sim,
                    log=lambda m: None)

    def decisions(run: IsoRun) -> list:
        return [(x.status, x.target, x.selected, x.iterations, x.rounds,
                 [(c.text, c.iteration, c.checked) for c in x.candidates]) for x in run.lines]
    assert decisions(again) == decisions(first) and again.rows == first.rows and again.calls == 0
    a, b = first.lines
    assert (a.status, a.target, a.iterations) == ("paraphrase", "pass", 5)
    assert (b.status, b.target, b.iterations) == ("fallback", "bbb5", 6)   # 4 + 2(남은 몫)
    assert [r["round"] for r in first.rows] == [1, 1, 2, 2] and first.rows[3]["candidates"] == ["bbb5", "bbb6"]
