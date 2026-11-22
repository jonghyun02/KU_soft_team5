"""pipeline.py: 가짜 LLM·가짜 모델로 차수, 멈춤, fallback, 모델 올리는 차례, 출력 파일을 본다."""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pytest

from pstts import cli, pipeline
from pstts.config import Config
from pstts.io import read_lines, write_outputs
from pstts.llm import FileBackend, GenReply, GenRequest, LLMUnavailable
from pstts.pipeline import FEEDBACK_MAX, check_config, feedback_for, fresh_candidates, run_lines
from pstts.records import Candidate, LineItem

DRAFT_A = "제가 그 비용의 절반을 부담하겠습니다."       # 16음절, 하십시오체
DRAFT_B = "당신은 어젯밤에 어디에 있었습니까?"          # 15음절
ITEM_A = LineItem("A", "I'll pay half.", register="반말", target_syllables=5)              # 3~7음절
ITEM_B = LineItem("B", "Where were you last night?", register="해요체", target_syllables=7, lip=False)  # 5~9음절
ROWS = [
    {"id": "A", "round": 1, "candidates": ["내가 반 낼게.", "반은 내가 낼게.", "절반은 내가 낼게.", "제가 반 낼게요.",
                                           "1. 내가 반 낼게!"]},
    {"id": "B", "round": 1, "candidates": ["어젯밤에 어디 있었어요?", "어제 어디 있었어요?", "어젯밤 어디 있었어?"]},
    {"id": "B", "round": 2, "candidates": ["밤에 어디 있었어요?", "어제 어디 계셨어요?", "어디 갔었어요?"]},
]
LABSE = {"어디 갔었어요?": 0.5}
COMET = {"반은 내가 낼게.": 0.8, "어제 어디 계셨어요?": 0.9}


class FakeDrafter:
    def __init__(self, table: dict[str, str], events: list[str]) -> None:
        self.table, self.events, self.seen = table, events, []
        self.info = {"epoch": 11, "averaged": [7, 11], "config": {"d_model": 512}}
        self.ckpt = Path("fake/avg_last5.pt")
        events.append("drafter:load")

    def translate(self, sentences: list[str]) -> list[str]:
        self.events.append("drafter:translate")
        self.seen.append(list(sentences))
        return [self.table.get(s, "") for s in sentences]

    def close(self) -> None:
        self.events.append("drafter:close")


class FakeLabse:
    def __init__(self, table: dict[str, float], events: list[str], default: float = 0.9) -> None:
        self.table, self.events, self.default = table, events, default
        events.append("labse:make")

    def similarity(self, srcs: list[str], tgts: list[str]) -> np.ndarray:
        assert len(srcs) == len(tgts)
        self.events.append(f"labse:{len(tgts)}")
        return np.array([self.table.get(t, self.default) for t in tgts])

    def close(self) -> None:
        self.events.append("labse:close")


class FakeComet:
    def __init__(self, table: dict[str, float], events: list[str], default: float = 0.5) -> None:
        self.table, self.events, self.default = table, events, default
        self.calls: list[tuple] = []
        self.checkpoint_path = "fake/comet.ckpt"
        events.append("comet:make")

    def score(self, srcs: list[str], mts: list[str], refs: list[str] | None = None) -> np.ndarray:
        self.events.append(f"comet:{len(mts)}")
        self.calls.append((list(srcs), list(mts), refs))
        return np.array([self.table.get(m, self.default) for m in mts])

    def close(self) -> None:
        self.events.append("comet:close")


class Recorder:
    """백엔드를 감싸 요청을 적어 둔다."""

    def __init__(self, inner, events: list[str]) -> None:
        self.inner, self.events = inner, events
        self.name = getattr(inner, "name", "fake")
        self.requests: list[list[GenRequest]] = []

    def generate(self, reqs: list[GenRequest]) -> GenReply:
        self.events.append(f"llm:{len(reqs)}")
        self.requests.append(reqs)
        return self.inner.generate(reqs)


def write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


class Harness:
    def __init__(self, tmp_path: Path, rows: list[dict] = ROWS, drafts: dict[str, str] | None = None,
                 labse: dict[str, float] | None = None, comet: dict[str, float] | None = None,
                 backend=None) -> None:
        self.events: list[str] = []
        self.logs: list[str] = []
        self.drafters: list[FakeDrafter] = []
        self.comets: list[FakeComet] = []
        inner = backend if backend is not None else FileBackend(write_rows(tmp_path / "cands.jsonl", rows))
        self.backend = Recorder(inner, self.events)
        self.drafts = drafts if drafts is not None else {ITEM_A.source_en: DRAFT_A, ITEM_B.source_en: DRAFT_B}
        self.labse = LABSE if labse is None else labse
        self.comet = COMET if comet is None else comet

    def make_drafter(self) -> FakeDrafter:
        d = FakeDrafter(self.drafts, self.events)
        self.drafters.append(d)
        return d

    def make_comet(self) -> FakeComet:
        c = FakeComet(self.comet, self.events)
        self.comets.append(c)
        return c

    def kwargs(self) -> dict:
        return dict(backend=self.backend, make_drafter=self.make_drafter,
                    make_labse=lambda: FakeLabse(self.labse, self.events), make_comet=self.make_comet,
                    log=self.logs.append)

    def run(self, items: list[LineItem], cfg: Config):
        return run_lines(items, cfg, **self.kwargs())


def cfg(**kw) -> Config:
    base = dict(device="cpu", min_pass=3, n_per_round=5, max_rounds=3)
    base.update(kw)
    return Config(**base)


# ---------------------------------------------------------------- 차수와 멈춤

def test_rounds_stop_when_enough_pass(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    results, info = h.run([ITEM_A, ITEM_B], cfg())
    a, b = results

    # 1차는 두 줄, 2차는 통과 후보가 모자란 B 만, 3차는 없음
    assert [[r.id for r in reqs] for reqs in h.backend.requests] == [["A", "B"], ["B"]]
    r1a, r1b = h.backend.requests[0]
    assert (r1a.round, r1a.n_new, r1a.n_target, r1a.tol, r1a.register) == (1, 5, 5, 2, "반말")
    assert r1a.avoid == [DRAFT_A] and r1a.draft_ko == DRAFT_A
    assert r1a.feedback == [{"text": DRAFT_A, "syllables": 16,
                             "reason": "길이 16음절(목표 5±2); 말투: 반말 요청인데 하십시오체 문장"}]
    (r2b,) = h.backend.requests[1]
    assert r2b.round == 2
    assert r2b.avoid == [DRAFT_B, "어젯밤에 어디 있었어요?", "어제 어디 있었어요?", "어젯밤 어디 있었어?"]
    # 최근 차수의 탈락 후보부터
    assert [f["text"] for f in r2b.feedback] == ["어젯밤에 어디 있었어요?", "어젯밤 어디 있었어?", DRAFT_B]
    assert r2b.feedback[1]["reason"] == "말투: 해요체 요청인데 반말 문장"

    # 후보 풀: 번호 붙은 '1. 내가 반 낼게!' 는 다듬으면 '내가 반 낼게.' 와 같아서 빠진다
    assert [c.text for c in a.candidates] == [DRAFT_A, "내가 반 낼게.", "반은 내가 낼게.", "절반은 내가 낼게.",
                                              "제가 반 낼게요."]
    assert [c.origin for c in a.candidates] == ["model1"] + ["file"] * 4
    assert [c.round for c in b.candidates] == [0, 1, 1, 1, 2, 2, 2]
    assert (a.rounds_used, b.rounds_used) == (1, 2)
    assert sum(c.passed for c in a.candidates) == 3 and sum(c.passed for c in b.candidates) == 3
    assert (a.status, b.status) == ("ok", "ok")
    assert a.draft_ko == DRAFT_A and b.draft_ko == DRAFT_B

    # B 는 입이 안 보여서 COMET 만: 가장 높은 '어제 어디 계셨어요?'
    assert b.selected.text == "어제 어디 계셨어요?"
    assert b.selected.score == 0.9 and all(c.dtw is None for c in b.candidates)
    # A 는 PS-Comet: 통과 후보 점수 = 1.6 x dtw_inv + 0.4 x COMET, 고른 것이 가장 높음
    passing = [c for c in a.candidates if c.passed]
    for c in passing:
        assert c.dtw is not None and 0.0 <= c.dtw_inv <= 1.0
        assert c.score == pytest.approx(1.6 * c.dtw_inv + 0.4 * c.comet)
    assert a.selected in passing and a.selected.score == max(c.score for c in passing)
    assert all(c.dtw is not None for c in a.candidates)      # 탈락 후보도 기록용으로 잰다

    assert info.llm_calls == 0 and info.llm_backend == "file"
    assert [r["round"] for r in info.rounds] == [1, 2]
    assert info.rounds[0]["new"] == 7 and info.rounds[1]["new"] == 3
    assert info.status_counts == {"ok": 2}
    assert info.model1["epoch"] == 11 and "config" not in info.model1
    assert info.comet_checkpoint == "fake/comet.ckpt"
    assert all(line.startswith("[") for line in h.logs)


def test_models_load_one_at_a_time(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.run([ITEM_A, ITEM_B], cfg())
    ev = h.events
    assert ev[:3] == ["drafter:load", "drafter:translate", "drafter:close"]
    assert ev.index("drafter:close") < ev.index("labse:make")
    assert ev.index("labse:close") < ev.index("comet:make")
    assert ev[-1] == "comet:close"
    assert h.drafters[0].seen == [[ITEM_A.source_en, ITEM_B.source_en]]     # 한 번에 번역
    # LaBSE 는 초벌 2개, 1차 7개, 2차 3개를 각각 한 번에
    assert [e for e in ev if e.startswith("labse:") and e[6:].isdigit()] == ["labse:2", "labse:7", "labse:3"]
    # COMET 은 통과 후보 6개를 한 번에
    assert [e for e in ev if e.startswith("comet:") and e[6:].isdigit()] == ["comet:6"]


def test_given_draft_skips_model1_and_no_draft(tmp_path: Path) -> None:
    given = LineItem("A", "I'll pay half.", register="반말", target_syllables=5, draft_ko="내가  반 낼게.")
    h = Harness(tmp_path)
    results, _ = h.run([given, ITEM_B], cfg())
    assert h.drafters[0].seen == [[ITEM_B.source_en]]
    first = results[0].candidates[0]
    assert (first.text, first.origin, first.round) == ("내가 반 낼게.", "given", 0)

    h2 = Harness(tmp_path)
    results, _ = h2.run([ITEM_A, ITEM_B], cfg(use_draft=False))
    assert h2.drafters == []
    assert all(c.round >= 1 for r in results for c in r.candidates)
    assert results[0].draft_ko == "" and h2.backend.requests[0][0].draft_ko is None


def test_fallback_when_nothing_passes(tmp_path: Path) -> None:
    item = LineItem("C", "Run!", register="반말", target_syllables=3)          # 1~5음절
    rows = [{"id": "C", "round": 1, "candidates": ["저기 저 괴물을 피해 어서 빨리 도망쳐!", "빨리 도망가야 해 지금!"]}]
    h = Harness(tmp_path, rows=rows, drafts={"Run!": "달려서 도망가십시오."},
                labse={"달려서 도망가십시오.": 0.6, "저기 저 괴물을 피해 어서 빨리 도망쳐!": 0.8,
                       "빨리 도망가야 해 지금!": 0.85})
    (r,), info = h.run([item], cfg(max_rounds=2))
    assert r.status == "fallback" and r.selected.text == "빨리 도망가야 해 지금!"
    assert not any(c.passed for c in r.candidates)
    assert r.selected.comet == 0.5 and h.comets[0].calls[0][1] == ["빨리 도망가야 해 지금!"]
    assert r.selected.score is None
    assert r.message.startswith("통과 후보 없음")
    assert "2차 새 후보 없음" in r.message
    assert r.rounds_used == 2 and len(h.backend.requests) == 2
    assert info.llm_errors == ["C: 후보 파일에 2차 후보가 없음"]
    assert info.status_counts == {"fallback": 1}


def test_error_when_no_candidate_at_all(tmp_path: Path) -> None:
    h = Harness(tmp_path, rows=[{"id": "other", "candidates": ["가"]}])
    (r,), info = h.run([ITEM_A], cfg(use_draft=False, max_rounds=1))
    assert (r.status, r.selected) == ("error", None)
    assert r.message.startswith("후보가 하나도 없음")
    assert not any(e.startswith("comet") for e in h.events)       # 잴 후보가 없으면 COMET 을 올리지 않는다
    assert info.status_counts == {"error": 1}


def test_llm_unavailable_keeps_partial_and_stops(tmp_path: Path) -> None:
    class Broken:
        name = "claude-cli"

        def generate(self, reqs: list[GenRequest]) -> GenReply:
            partial = GenReply()
            partial.update({r.id: [] for r in reqs})
            partial["A"] = ["내가 반 낼게.", "반은 내가 낼게.", "절반은 내가 낼게."]
            partial.failed.append("B")
            partial.errors.append("조각 2/2(B): 다른 조각에서 LLM 을 쓸 수 없게 되어 부르지 않음")
            partial.calls = 1
            err = LLMUnavailable("claude 사용량 한도에 걸렸습니다.")
            err.partial = partial
            raise err

    h = Harness(tmp_path, backend=Broken())
    (a, b), info = h.run([ITEM_A, ITEM_B], cfg())
    assert len(h.backend.requests) == 1                       # 멈춘 뒤에는 다시 부르지 않는다
    assert info.llm_stopped == "claude 사용량 한도에 걸렸습니다." and info.llm_calls == 1
    assert a.status == "ok" and len(a.candidates) == 4
    assert b.status == "fallback" and b.selected.text == DRAFT_B
    assert "모델 2 를 쓸 수 없어 후보 모으기를 멈춤" in b.message
    assert any("쓸 수 없어 멈춤" in line for line in h.logs)


def test_max_candidates_caps_requests(tmp_path: Path) -> None:
    rows = [{"id": "A", "round": 1, "candidates": ["제가 반 낼게요.", "저도 반 낼게요.", "반은 제가 낼게요.",
                                                   "제가 절반 낼게요.", "절반은 제가 낼게요."]}]
    h = Harness(tmp_path, rows=rows)
    (r,), _ = h.run([ITEM_A], cfg(max_candidates=4))
    assert h.backend.requests[0][0].n_new == 3                # 4 - 초벌 1
    assert len(r.candidates) == 4 and len(h.backend.requests) == 1
    assert r.status == "fallback"                             # 모두 해요체라 반말 요청에서 탈락


def test_max_rounds_zero_uses_draft_only(tmp_path: Path) -> None:
    h = Harness(tmp_path, drafts={ITEM_A.source_en: "내가 반 낼게."})
    (r,), info = h.run([ITEM_A], cfg(max_rounds=0))
    assert h.backend.requests == [] and info.llm_backend == "-"
    assert r.status == "ok" and r.selected.text == "내가 반 낼게." and r.rounds_used == 0


def test_comet_reference_draft(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.run([ITEM_A, ITEM_B], cfg(comet_ref="draft"))
    srcs, mts, refs = h.comets[0].calls[0]
    assert refs is not None and len(refs) == len(mts) == 6
    assert set(refs) == {DRAFT_A, DRAFT_B}
    with pytest.raises(ValueError, match="초벌"):
        Harness(tmp_path).run([ITEM_A], cfg(comet_ref="draft", use_draft=False))


def test_comet_reference_draft_never_scores_the_draft_against_itself(tmp_path: Path) -> None:
    # 초벌이 통과해도 COMET 에 (초벌, 참조 = 초벌) 짝을 넣지 않고 고르기에서 뺀다. 넣으면 늘 이긴다(2026-11-22 검토).
    draft = "내가 반 낼게."
    h = Harness(tmp_path, drafts={ITEM_A.source_en: draft}, comet={draft: 1.0})
    (r,), info = h.run([ITEM_A], cfg(comet_ref="draft"))
    srcs, mts, refs = h.comets[0].calls[0]
    assert draft not in mts and set(refs) == {draft}
    first = r.candidates[0]
    assert (first.text, first.passed, first.comet, first.score) == (draft, True, None, None)
    assert r.status == "ok" and r.selected is not first
    assert any("초벌은 참조라서" in line for line in h.logs)
    # 초벌만 통과한 줄은 점수 없이 초벌을 고른다
    h2 = Harness(tmp_path, rows=[{"id": "A", "candidates": ["제가 반 낼게요."]}], drafts={ITEM_A.source_en: draft})
    (r2,), _ = h2.run([ITEM_A], cfg(comet_ref="draft", max_rounds=1))
    assert (r2.status, r2.selected.text, r2.selected.comet) == ("ok", draft, None)
    assert r2.message.startswith("초벌만 남아")


def test_no_pass_falls_back_to_length_and_meaning_candidates(tmp_path: Path) -> None:
    # 말투만 못 채운 후보(식 6·7 은 채움)가 있으면 그 가운데 PS-Comet 으로 고른다. 예전처럼 LaBSE 만 보면
    # 길이도 말투도 어긋난 초벌(LaBSE 0.95)이 뽑혔다(2026-11-22 검토).
    item = LineItem("C", "I'll pay half.", register="해요체", target_syllables=5)          # 3~7음절
    rows = [{"id": "C", "candidates": ["내가 반 낼게.", "반은 내가 낼게.", "절반은 내가 낼게."]}]
    h = Harness(tmp_path, rows=rows, drafts={item.source_en: DRAFT_A}, labse={DRAFT_A: 0.95},
                comet={"반은 내가 낼게.": 0.9})
    (r,), info = h.run([item], cfg(max_rounds=1))
    assert not any(c.passed for c in r.candidates)
    assert r.status == "fallback" and r.selected.text != DRAFT_A
    assert r.selected.iso_ok and r.selected.ok_register is False and r.selected.score is not None
    assert r.message.startswith("통과 후보 없음: 길이와 뜻 조건(논문 식 6, 7)을 채운 후보 가운데 PS-Comet 으로 고름")
    assert "(말투 조건은 못 채움)" in r.message
    assert sorted(h.comets[0].calls[0][1]) == sorted(rows[0]["candidates"])    # 식 6·7 묶음만 COMET 으로 잰다


def test_number_check_failures_also_fall_back_to_iso_group(tmp_path: Path) -> None:
    item = LineItem("N", "Gate 3 is closed.", register="반말", target_syllables=6)        # 4~8음절
    rows = [{"id": "N", "candidates": ["게이트가 닫혔어.", "탑승구가 닫혔어.", "문이 닫혔어."]}]
    h = Harness(tmp_path, rows=rows, drafts={item.source_en: "삼 번 게이트는 폐쇄되었습니다."},
                labse={"삼 번 게이트는 폐쇄되었습니다.": 0.97})
    (r,), _ = h.run([item], cfg(max_rounds=1, check_numbers=True))
    assert r.status == "fallback" and r.selected.text in rows[0]["candidates"]
    assert r.selected.reasons == ["숫자: 3 빠짐"] and "(숫자 조건은 못 채움)" in r.message


def test_labse_also_scores_the_spoken_form_of_numbers(tmp_path: Path) -> None:
    # 숫자 원문('Call 911!')과 한글 수 후보는 LaBSE 가 낮게 본다. 숫자를 말하는 꼴로 푼 원문과도 재서 큰 값을 쓴다
    class SpokenLabse(FakeLabse):
        def similarity(self, srcs: list[str], tgts: list[str]) -> np.ndarray:
            self.events.append(f"labse:{len(tgts)}")
            return np.array([0.9 if "nine one one" in s else 0.6 for s in srcs])

    item = LineItem("E", "Call 911!", register="반말", target_syllables=7)
    rows = [{"id": "E", "candidates": ["구일일에 전화해!", "빨리 구일일 불러!"]}]
    h = Harness(tmp_path, rows=rows, drafts={item.source_en: "911에 전화해!"})
    kw = h.kwargs()
    kw["make_labse"] = lambda: SpokenLabse({}, h.events)
    (r,), _ = run_lines([item], cfg(max_rounds=1), **kw)
    assert [c.labse for c in r.candidates] == [0.9, 0.9, 0.9] and r.status == "ok"
    assert [e for e in h.events if e.startswith("labse:") and e[6:].isdigit()] == ["labse:2", "labse:4"]
    sims = pipeline.labse_scores(SpokenLabse({}, []), ["Hi.", "Call 911!"], ["안녕.", "구일일!"])
    assert sims.tolist() == [0.6, 0.9]                     # 숫자 없는 원문은 한 번만 잰다


def test_unspecified_register_takes_model2_majority(tmp_path: Path) -> None:
    item = LineItem("F", "Then why didn't you call?", prev_en="Sorry, Mom.", target_syllables=9)
    draft = "그럼 왜 전화 안 했어요?"
    rows = [{"id": "F", "candidates": ["그럼 왜 전화 안 했어?", "왜 전화를 안 했어?", "그럼 전화는 왜 안 했니?"]}]
    h = Harness(tmp_path, rows=rows, drafts={item.source_en: draft})
    (r,), _ = h.run([item], cfg(max_rounds=1))
    first = r.candidates[0]
    assert first.text == draft and first.ok_register is False
    assert "말투: 미지정 줄인데 모델 2 후보 다수(반말)와 다름" in first.reasons
    assert r.selected.text in rows[0]["candidates"] and "말투 미지정: 모델 2 후보 다수를 따라 반말로 봄" in r.message


def test_messages_for_lip_line_without_english_syllables(tmp_path: Path) -> None:
    # 통과 후보가 없으면 LaBSE 규칙이라 'COMET 으로 고름' 을 붙이지 않는다(서로 어긋난 메모였음)
    item = LineItem("D", "...", register="반말", target_syllables=2)
    rows = [{"id": "D", "candidates": ["아무 말도 할 수가 없네요.", "그러니까 뭐라고 해야 할지 모르겠어."]}]
    h = Harness(tmp_path, rows=rows, drafts={"...": ""})
    (r,), _ = h.run([item], cfg(max_rounds=1, use_draft=False))
    assert r.status == "fallback" and "COMET 으로 고름" not in r.message
    rows = [{"id": "D", "candidates": ["음.", "어."]}]
    h = Harness(tmp_path, rows=rows, drafts={"...": ""})
    (r,), _ = h.run([item], cfg(max_rounds=1, use_draft=False))
    assert r.status == "ok" and r.message.startswith("영어 음절이 없어 입 모양 비교 없이 COMET 으로 고름")


def test_log_shows_single_target_as_one_number(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.run([ITEM_A], cfg())
    assert h.logs[0].startswith("[준비] 대사 1줄, 장치 cpu, 목표 음절 5(허용 ±2)")
    h = Harness(tmp_path)
    h.run([ITEM_A, ITEM_B], cfg())
    assert "목표 음절 5~7(허용 ±2)" in h.logs[0]


def test_bad_settings_and_items(tmp_path: Path) -> None:
    for bad in (dict(llm="gpt"), dict(comet_ref="ref"), dict(n_per_round=0), dict(min_pass=0), dict(alpha=-1.0),
                dict(labse_min=1.5), dict(max_rounds=-1), dict(coef=0.0)):
        with pytest.raises(ValueError):
            check_config(cfg(**bad))
    h = Harness(tmp_path)
    with pytest.raises(ValueError, match="같은 id"):
        h.run([ITEM_A, ITEM_A], cfg())
    with pytest.raises(ValueError, match="대사가 하나도"):
        h.run([], cfg())
    with pytest.raises(ValueError, match="말투"):
        h.run([LineItem("x", "Hi.", register="존댓말")], cfg())


def test_cuda_request_never_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="--device cpu"):
        run_lines([ITEM_A], Config(device="cuda"), log=lambda msg: None)


@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
def test_cuda_with_no_visible_gpu_stops_before_loading(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    # CUDA_VISIBLE_DEVICES="" 이면 is_available() 은 참인데 보이는 GPU 가 0개일 수 있다(2026-11-22 검토).
    # 실제 CUDA 를 건드리지 않게 두 함수를 바꿔 끼운다.
    import torch
    from pstts.scorers import resolve_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="보이는 GPU 가 없습니다"):
        resolve_device(device)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert resolve_device("cuda") == "cuda" and resolve_device("cuda:0") == "cuda:0"
    with pytest.raises(RuntimeError, match="GPU 1번이 없습니다"):
        resolve_device("cuda:1")


def test_cli_unexpected_error_is_one_korean_line(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("Invalid device id")

    monkeypatch.setattr(cli, "run_lines", boom)
    assert cli.main(["Hi.", "--device", "cpu", "--out", str(tmp_path / "o")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("오류: 예상하지 못한 문제로 멈췄습니다. AssertionError: Invalid device id")
    assert "Traceback" not in err and len(err.strip().splitlines()) == 1


def test_feedback_and_fresh_helpers() -> None:
    pool = [Candidate(f"후보{k}", "llm", k // 3, syllables=k, ok_length=False, ok_meaning=True, ok_register=True,
                      reasons=[f"길이 {k}음절(목표 1±0)"]) for k in range(9)]
    pool.append(Candidate("통과", "llm", 3, syllables=1, ok_length=True, ok_meaning=True, ok_register=True))
    fb = feedback_for(pool)
    assert len(fb) == FEEDBACK_MAX
    # 최근 차수(2차)의 탈락 후보부터, 같은 차수 안에서는 나온 순서대로. 통과 후보는 넣지 않는다
    assert [f["text"] for f in fb] == ["후보6", "후보7", "후보8", "후보3", "후보4", "후보5"]
    assert fb[0] == {"text": "후보6", "syllables": 6, "reason": "길이 6음절(목표 1±0)"}

    # 다듬은 뒤 풀이나 서로 겹치는 것은 빼고, 남은 자리(12 - 10 = 2)만큼만 받는다
    new = fresh_candidates(pool, ["- 후보1", "\"새 후보\"", "새 후보!", "  ", "또 다른 후보", "하나 더"], "llm", 4, 12)
    assert [c.text for c in new] == ["새 후보", "또 다른 후보"]
    assert all((c.origin, c.round) == ("llm", 4) for c in new)
    assert [c.text for c in fresh_candidates(pool, ["가", "나", "다"], "llm", 4, 12)] == ["가", "나"]
    assert fresh_candidates(pool, ["가"], "llm", 4, 10) == []


# ---------------------------------------------------------------- 출력 파일과 다시 돌리기

def test_outputs_and_rerun_from_candidates_file(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    results, info = h.run([ITEM_A, ITEM_B], cfg())
    columns = ["id", "source_en", "register", "target_syllables", "lip", "split"]
    paths = write_outputs(tmp_path / "out", results, info, cfg(), columns, "시험")
    assert sorted(p.name for p in paths.values()) == ["candidates.jsonl", "report.md", "results.csv",
                                                      "results.jsonl", "run.json"]
    rows = [json.loads(line) for line in paths["candidates.jsonl"].read_text(encoding="utf-8").splitlines()]
    assert [(r["id"], r["round"], len(r["candidates"])) for r in rows] == [("A", 1, 4), ("B", 1, 3), ("B", 2, 3)]

    # candidates.jsonl 을 FileBackend 로 다시 쓰면 같은 대사를 고른다
    h2 = Harness(tmp_path, backend=FileBackend(paths["candidates.jsonl"]))
    again, _ = h2.run([ITEM_A, ITEM_B], cfg())
    assert [r.selected.text for r in again] == [r.selected.text for r in results]
    assert [[c.text for c in r.candidates] for r in again] == [[c.text for c in r.candidates] for r in results]


# ---------------------------------------------------------------- 명령줄

@pytest.fixture
def fake_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    h = Harness(tmp_path)
    monkeypatch.setattr(cli, "run_lines", functools.partial(pipeline.run_lines, **h.kwargs()))
    return h


def test_cli_file_mode(fake_cli: Harness, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "lines.csv"
    src.write_text("id,source_en,register,target_syllables,lip,scene_id\n"
                   "A,I'll pay half.,반말,5,on,s1\nB,Where were you last night?,해요체,7,off,s2\n", encoding="utf-8-sig")
    out = tmp_path / "out"
    code = cli.main(["--input", str(src), "--out", str(out), "--device", "cpu", "--min-pass", "3",
                     "--n-per-round", "5"])
    text = capsys.readouterr().out
    assert code == 0
    assert "A [고름]" in text and "B [고름] 어제 어디 계셨어요?" in text
    assert "고름 2줄, 대안 0줄, 실패 0줄" in text
    header = (out / "results.csv").read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    # 결과 칸 다음에 입력의 나머지 칸(결과 칸과 이름이 겹치지 않는 것)이 입력 순서대로 붙는다
    assert header[-3:] == ["target_syllables", "lip", "scene_id"]
    assert header[:17] == ["id", "source_en", "register", "target", "tol", "draft_ko", "final_ko",
                           "final_syllables", "final_register", "labse", "comet", "dtw", "score", "status",
                           "n_candidates", "n_pass", "message"]
    items, _ = read_lines(out / "results.csv")
    assert [it.extra["final_register"] for it in items] == ["반말", "해요체"]     # 고른 대사의 실제 말투
    assert (out / "run.json").is_file()


def test_cli_single_mode(fake_cli: Harness, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake_cli.backend.inner = FileBackend(write_rows(tmp_path / "one.jsonl", [
        {"id": "L0001", "round": 1, "candidates": ["내가 반 낼게.", "반은 내가 낼게.", "절반은 내가 낼게."]}]))
    out = tmp_path / "single"
    code = cli.main(["I'll pay half.", "--register", "반말", "--target-syllables", "5", "--device", "cpu",
                     "--out", str(out), "--min-pass", "3"])
    text = capsys.readouterr().out
    assert code == 0
    assert "고른 대사: " in text and "다음 후보:" in text
    assert len([ln for ln in text.splitlines() if ln.startswith(("  1. ", "  2. ", "  3. "))]) == 3
    assert (out / "report.md").is_file() and (out / "results.csv").is_file()
    items, _ = read_lines(out / "results.csv")
    assert items[0].register == "반말" and items[0].target_syllables == 5


def test_cli_errors(fake_cli: Harness, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for argv in ([], ["Hi.", "--input", "x.csv"], ["--input", "x.csv", "--register", "반말"]):
        with pytest.raises(SystemExit) as e:
            cli.main(argv)
        assert e.value.code == 2
    assert cli.main(["--input", str(tmp_path / "없음.csv"), "--device", "cpu"]) == 2
    assert "입력 파일이 없습니다" in capsys.readouterr().err
    assert cli.main(["Hi.", "--target-syllables", "0", "--device", "cpu"]) == 2
    assert "target_syllables" in capsys.readouterr().err


def test_cli_exit_code_when_a_line_fails(fake_cli: Harness, tmp_path: Path,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    fake_cli.backend.inner = FileBackend(write_rows(tmp_path / "none.jsonl", [{"id": "zzz", "candidates": ["가"]}]))
    code = cli.main(["I'll pay half.", "--no-draft", "--max-rounds", "1", "--device", "cpu",
                     "--out", str(tmp_path / "o")])
    assert code == 1
    src = tmp_path / "one.csv"
    src.write_text("id,source_en,target_syllables\nA,I'll pay half.,5\n", encoding="utf-8")
    capsys.readouterr()
    assert cli.main(["--input", str(src), "--no-draft", "--max-rounds", "1", "--device", "cpu",
                     "--out", str(tmp_path / "o2")]) == 1
    # 고른 대사가 없으면 빈 음절 칸 없이 적는다
    assert "A [실패] (없음) (목표 5±2)" in capsys.readouterr().out


def test_config_from_args_maps_names() -> None:
    args = cli.build_parser().parse_args(["x", "--tol", "3", "--beam", "2", "--out", "o", "--no-llm-cache",
                                          "--no-draft", "--no-register-check", "--number-check", "--coef", "1.3",
                                          "--llm", "file", "--llm-file", "c.jsonl", "--device", "cpu"])
    c = cli.config_from_args(args)
    assert (c.tol_syllables, c.nmt_beam, c.out_dir, c.llm_cache, c.use_draft, c.check_register, c.check_numbers) == (
        3, 2, "o", False, False, False, True)
    assert Config().check_numbers is False                     # 숫자 조건은 논문·합의 구조에 없어서 기본 끔
    assert (c.coef, c.llm, c.llm_file, c.device) == (1.3, "file", "c.jsonl", "cpu")
    assert cli.config_from_args(cli.build_parser().parse_args(["x"])) == Config()
