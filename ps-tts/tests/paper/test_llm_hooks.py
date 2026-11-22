"""llm.py 하위 호환 갈고리 시험: 기존 판 프롬프트와 캐시 키 불변, PromptSpec 으로 바꿔 부르기, 단계 재생, DTW 반경 None.

기존 판 값은 llm.py 를 고치기 전에 잰 것이다(2026-11-22, 고치기 전 파일 sha1 e0ab63b0242d2394837f8ce3b2760385ed3b69ab).
실제 claude 와 로컬 LLM 서버는 부르지 않는다(가짜 subprocess).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading

import numpy as np
import pytest
from tslearn.metrics import dtw_path_from_metric

from pstts import llm
from pstts.config import Config
from pstts.dtw import dtw_cost
from pstts.llm import (PROMPT_VERSION, SCHEMA, SYSTEM_PROMPT, USER_HEAD, ClaudeCLIBackend, FileBackend, GenRequest,
                       OpenAICompatBackend, PromptSpec, build_prompt, cache_key)

MODEL = Config().llm_model


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 기존 판 값(고치기 전에 잼)

BASE = {
    "SYSTEM_PROMPT": "1908b4650a5a4742ae2f2d5d0d0048a58ccebf54",
    "USER_HEAD": "1ce854989406c9adb62e4f7fdd8e8fd214e21a24",
    "SCHEMA": "0381ef08dfc9a9e4241631f2c4631c20c22fee3d",
    "COMMAND": "b917f462771f08ce8fe22bf1027db5140ce1ac9e",
}
# 묶음마다 (user sha1, claude-cli 캐시 키, openai 캐시 키). system 은 모두 SYSTEM_PROMPT 와 같다.
SETS = [
    ("3d450c64573c393053f77cae7c354cf69139a443", "da88ee8a24441baad3678c5f5b2acfbcd461ca44",
     "01123866dbda39b12ba6c61fe2e76f03150967c9"),
    ("c199b7c755838cda60ab337282d5fbe2877bae45", "82edf73af295d7de21ffa344335951dc9defa88f",
     "00dd3dc0bc32e35933a4d6d0c03edbeb6711ca71"),
    ("15ba70ee33c3da05b133d833e6ae391eb4d4f659", "ecdeeaa2631d3b3fad443f4804367999a5c528fc",
     "0fa3786dd0d3673fc516973c35c22000ce55317f"),
    ("02718e4befcc6ee1cf1c28d22c122cb191a5d678", "7be010ddca87272bf6474faeaa95e8bdd27450cc",
     "8b03f54d74039dfe5fac02ce9e90c26589283de5"),
]


def _fixed_sets() -> list[list[GenRequest]]:
    a = GenRequest(id="L1", source_en="I'll pay half.", prev_en="Dinner was great.", relationship="친구 사이",
                   register="반말", n_target=6, tol=2, draft_ko="내가 반을 낼게.", meaning_notes="절반",
                   n_new=5, round=2, avoid=["내가 반을 낼게."],
                   feedback=[{"text": "내가 절반을 지불하겠다.", "syllables": 10, "reason": "길이 10음절(목표 6±2)"}])
    b = GenRequest(id="L2", source_en="Thanks.", prev_en="", relationship="", register="", n_target=3, tol=1,
                   draft_ko=None, meaning_notes="", n_new=2)
    c = GenRequest(id="ex_9", source_en="We have to leave before the storm hits.", prev_en="", relationship="",
                   register="해요체", n_target=14, tol=2, draft_ko="폭풍이 오기 전에 떠나야 해요.", meaning_notes="",
                   n_new=20, round=1, avoid=[])
    return [[a], [a, b], [c], [b, c, a]]


def test_main_prompt_constants_unchanged() -> None:
    assert PROMPT_VERSION == "pstts-gen/1"
    assert _sha(SYSTEM_PROMPT) == BASE["SYSTEM_PROMPT"]
    assert _sha(USER_HEAD) == BASE["USER_HEAD"]
    assert _sha(json.dumps(SCHEMA, ensure_ascii=False, sort_keys=True)) == BASE["SCHEMA"]
    cmd = ClaudeCLIBackend(Config(), exe="C:/fake/claude.exe").command("S")
    assert _sha(json.dumps(cmd, ensure_ascii=False)) == BASE["COMMAND"]


@pytest.mark.parametrize("k", range(len(SETS)))
def test_main_build_prompt_and_cache_key_unchanged(k: int) -> None:
    reqs = _fixed_sets()[k]
    user_sha, key_claude, key_openai = SETS[k]
    system, user = build_prompt(reqs)
    assert _sha(system) == BASE["SYSTEM_PROMPT"] and _sha(user) == user_sha
    rounds = [int(r.round) for r in reqs]
    claude = ClaudeCLIBackend(Config(), exe="C:/fake/claude.exe")
    openai = OpenAICompatBackend("http://127.0.0.1:8080", "qwen", None, Config())
    assert cache_key(claude.cache_tag, claude.model, claude.effort, system, user, rounds) == key_claude
    assert cache_key(openai.cache_tag, openai.model, openai.effort, system, user, rounds) == key_openai
    # 판을 기본값과 같게 적어 줘도 같은 키
    assert cache_key(claude.cache_tag, claude.model, claude.effort, system, user, rounds, PROMPT_VERSION) == key_claude


# ---------------------------------------------------------------- 가짜 claude

def _stream(structured: dict, model: str = MODEL) -> str:
    lines = [{"type": "system", "subtype": "init", "tools": ["StructuredOutput"], "mcp_servers": [], "model": model},
             {"type": "result", "subtype": "success", "is_error": False, "result": "", "structured_output": structured,
              "api_error_status": None, "modelUsage": {model: {"inputTokens": 1, "outputTokens": 1}}}]
    return "\n".join(json.dumps(o, ensure_ascii=False) for o in lines) + "\n"


def _items_in(stdin: str) -> list[dict]:
    text = json.loads(stdin)["message"]["content"][0]["text"]
    return json.JSONDecoder().raw_decode(text, text.index("["))[0]


class FakeRun:
    """subprocess.run 대신. 받은 명령과 입력을 적고, 줄마다 '{id}-{k}' 후보를 n 개 돌려준다."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def __call__(self, cmd, input=None, **kw):
        with self.lock:
            self.calls.append({"cmd": cmd, "input": input, **kw})
        items = _items_in(input)
        answer = {"items": [{"id": it["id"], "candidates": [f"{it['id']}-후보-{k}" for k in range(it["n"])]}
                            for it in items]}
        return subprocess.CompletedProcess(cmd, 0, _stream(answer), "")


@pytest.fixture
def claude(tmp_path, monkeypatch):
    def make(**cfg_kw) -> tuple[ClaudeCLIBackend, FakeRun]:
        fake = FakeRun()
        monkeypatch.setattr(llm.subprocess, "run", fake)
        b = ClaudeCLIBackend(Config(**cfg_kw), exe="C:/fake/claude.exe", cache_dir=tmp_path / "cache",
                             work_dir=tmp_path / "work")
        b.retry_wait = 0
        return b, fake
    return make


def test_generate_without_prompt_keeps_old_cache_file(claude, tmp_path) -> None:
    b, fake = claude()
    reqs = _fixed_sets()[2]
    out = b.generate(reqs)
    assert out["ex_9"][:2] == ["ex_9-후보-0", "ex_9-후보-1"] and len(fake.calls) == 1
    files = sorted(p.stem for p in (tmp_path / "cache").glob("*.json"))
    assert files == [SETS[2][1]]                      # 고치기 전과 같은 캐시 파일 이름
    rec = json.loads((tmp_path / "cache" / f"{SETS[2][1]}.json").read_text(encoding="utf-8"))
    assert rec["prompt_version"] == PROMPT_VERSION and _sha(rec["user"]) == SETS[2][0]
    cmd = fake.calls[0]["cmd"]
    assert cmd[cmd.index("--system-prompt") + 1] == SYSTEM_PROMPT
    with pytest.raises(ValueError, match="알 수 없는 말투"):     # prompt 가 없으면 말투 확인도 전과 같다
        b.generate([GenRequest(id="X", source_en="Hi.", prev_en="", relationship="", register="존댓말", n_target=2,
                               tol=1, draft_ko=None, meaning_notes="", n_new=1)])


# ---------------------------------------------------------------- PromptSpec 갈고리

class _Req:
    """말투가 없는 요청(논문 판 요청처럼 id, n_new, round, avoid 만)."""

    def __init__(self, id: str, n_new: int = 3, round: int = 1, avoid: list[str] | None = None) -> None:
        self.id, self.n_new, self.round, self.avoid = id, n_new, round, list(avoid or [])


def _build(reqs: list[_Req]) -> tuple[str, str]:
    items = [{"id": r.id, "n": r.n_new, "avoid": r.avoid} for r in reqs]
    return "갈고리 시스템", "갈고리 사용자\n" + json.dumps(items, ensure_ascii=False)


SPEC = PromptSpec(build=_build, version="hook-test/1", stage="iso")


def test_prompt_spec_changes_prompt_and_cache_key(claude, tmp_path) -> None:
    b, fake = claude()
    out = b.generate([_Req("A"), _Req("B", n_new=2)], prompt=SPEC)
    assert out == {"A": ["A-후보-0", "A-후보-1", "A-후보-2"], "B": ["B-후보-0", "B-후보-1"]}
    cmd = fake.calls[0]["cmd"]
    assert cmd[cmd.index("--system-prompt") + 1] == "갈고리 시스템"
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == SCHEMA          # 출력 꼴은 기존과 같다
    system, user = _build([_Req("A"), _Req("B", n_new=2)])
    key = cache_key(b.cache_tag, b.model, b.effort, system, user, [1, 1], "hook-test/1")
    assert key != cache_key(b.cache_tag, b.model, b.effort, system, user, [1, 1])
    rec = json.loads((tmp_path / "cache" / f"{key}.json").read_text(encoding="utf-8"))
    assert (rec["key"], rec["prompt_version"], rec["system"]) == (key, "hook-test/1", "갈고리 시스템")
    again = b.generate([_Req("A"), _Req("B", n_new=2)], prompt=SPEC)
    assert again == out and again.cache_hits == 1 and len(fake.calls) == 1


def test_prompt_spec_chunk_overrides_llm_chunk(claude) -> None:
    b, fake = claude(llm_chunk=6, llm_workers=1)
    reqs = [_Req(f"L{i}") for i in range(5)]
    b.generate(reqs, prompt=PromptSpec(build=_build, version="hook-test/1", chunk=2))
    assert [len(_items_in(c["input"])) for c in fake.calls] == [2, 2, 1]
    b.generate([_Req(f"M{i}") for i in range(5)], prompt=SPEC)         # chunk 가 없으면 llm_chunk(6)
    assert [len(_items_in(c["input"])) for c in fake.calls][3:] == [5]


def test_prompt_spec_dedupes_against_avoid_and_cleans(claude, monkeypatch) -> None:
    b, fake = claude()

    def answer(cmd, input=None, **kw):
        fake.calls.append({"cmd": cmd, "input": input})
        cands = ["1. 지금 가자!", "\"지금 가자!\"", "이미 나온 말.", "어서 가자."]
        reply = {"items": [{"id": "A", "candidates": cands}]}
        return subprocess.CompletedProcess(cmd, 0, _stream(reply), "")
    monkeypatch.setattr(llm.subprocess, "run", answer)
    out = b.generate([_Req("A", n_new=5, avoid=["이미 나온 말"])], prompt=SPEC)
    assert out["A"] == ["지금 가자!", "어서 가자."]                    # 기존 다듬기와 겹침 빼기를 그대로 씀


# ---------------------------------------------------------------- FileBackend 단계 재생

def _write(path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def test_file_backend_staged_rows_are_separate(tmp_path) -> None:
    p = tmp_path / "c.jsonl"
    _write(p, [{"id": "L1", "round": 1, "candidates": ["기존 판 후보."]},
               {"id": "L1", "stage": "iso", "round": 1, "candidates": ["ISO 후보 하나.", "ISO 후보 둘."]},
               {"id": "L1", "stage": "ps", "round": 1, "candidates": ["PS 후보."]},
               {"id": "L1", "stage": "iso", "round": 2, "candidates": ["ISO 이차."]}])
    fb = FileBackend(p)
    assert fb.table == {("L1", 1): ["기존 판 후보."]}                    # stage 없는 줄만 기존 표에
    assert fb.staged[("L1", "iso", 1)] == ["ISO 후보 하나.", "ISO 후보 둘."]
    old = GenRequest(id="L1", source_en="x", prev_en="", relationship="", register="반말", n_target=3, tol=1,
                     draft_ko=None, meaning_notes="", n_new=5)
    assert fb.generate([old]) == {"L1": ["기존 판 후보."]}
    iso = PromptSpec(build=_build, version="v", stage="iso")
    ps = PromptSpec(build=_build, version="v", stage="ps")
    assert fb.generate([_Req("L1", n_new=5)], prompt=iso) == {"L1": ["ISO 후보 하나.", "ISO 후보 둘."]}
    assert fb.generate([_Req("L1", n_new=1)], prompt=iso) == {"L1": ["ISO 후보 하나."]}
    assert fb.generate([_Req("L1", n_new=5, round=2)], prompt=iso) == {"L1": ["ISO 이차."]}
    assert fb.generate([_Req("L1", n_new=5)], prompt=ps) == {"L1": ["PS 후보."]}
    out = fb.generate([_Req("L1", n_new=5, round=3)], prompt=ps)
    assert out == {"L1": []} and out.failed == ["L1"] and "ps 3차 후보가 없음" in out.errors[0]
    out = fb.generate([_Req("L1", n_new=5, avoid=["ISO 후보 하나", "ISO 후보 둘!"])], prompt=iso)
    assert out == {"L1": []} and "모두 이미 나온 문장" in out.errors[0]


def test_staged_replay_returns_text_as_recorded(tmp_path) -> None:
    # 처음 받을 때 다듬은 글이 '1. 가자' 였다면(따옴표 안 번호), 재생은 다시 다듬지 않고 그 글을 그대로 돌려준다
    p = tmp_path / "c.jsonl"
    _write(p, [{"id": "L1", "round": 1, "candidates": ["1. 가자"]},
               {"id": "L1", "stage": "iso", "round": 1, "candidates": ["1. 가자", "1. 가자!", "가자"]}])
    fb = FileBackend(p)
    assert llm.clean_candidate("\"1. 가자\"") == "1. 가자" and llm.clean_candidate("1. 가자") == "가자"
    assert fb.generate([_Req("L1", n_new=5)], prompt=PromptSpec(build=_build, version="v", stage="iso")) == \
        {"L1": ["1. 가자", "가자"]}
    old = GenRequest(id="L1", source_en="x", prev_en="", relationship="", register="반말", n_target=3, tol=1,
                     draft_ko=None, meaning_notes="", n_new=5)
    assert fb.generate([old]) == {"L1": ["가자"]}                          # 기존 판 줄은 전처럼 다듬는다


def test_staged_row_bad_stage(tmp_path) -> None:
    p = tmp_path / "c.jsonl"
    _write(p, [{"id": "L1", "stage": "", "round": 1, "candidates": ["가."]}])
    with pytest.raises(ValueError, match="stage"):
        FileBackend(p)
    _write(p, [{"id": "L1", "stage": 3, "round": 1, "candidates": ["가."]}])
    with pytest.raises(ValueError, match="1행: stage"):
        FileBackend(p)


# ---------------------------------------------------------------- DTW 반경 None

def test_dtw_cost_accepts_none_radius() -> None:
    rng = np.random.default_rng(5)
    for _ in range(10):
        n, m = int(rng.integers(1, 30)), int(rng.integers(1, 30))
        cost = rng.random((n, m))
        _, default = dtw_path_from_metric(cost, metric="precomputed", global_constraint="sakoe_chiba")
        assert dtw_cost(cost, None) == float(default) == dtw_cost(cost, 1)
    with pytest.raises(ValueError):
        dtw_cost(np.ones((2, 2)), -1)
