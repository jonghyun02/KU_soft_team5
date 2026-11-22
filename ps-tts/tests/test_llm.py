"""모델 2(LLM) 시험: 답 읽기, 프롬프트, claude 명령·환경 변수(가짜 subprocess), 캐시, 조각·부분 실패,
FileBackend, OpenAI 호환 백엔드(시험 안에서 띄운 가짜 서버). 실제 claude·로컬 LLM 서버는 부르지 않는다."""
from __future__ import annotations

import http.server
import json
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

from pstts import llm
from pstts.config import Config
from pstts.llm import (PROMPT_VERSION, SCHEMA, SYSTEM_PROMPT, USER_HEAD, BadOutput, ClaudeCLIBackend, FileBackend,
                       GenReply, GenRequest, LLMLimit, LLMUnavailable, OpenAICompatBackend, Transient, build_prompt,
                       cache_key, make_backend, parse_reply, read_claude_output)

MODEL = Config().llm_model
FIELDS = {"id", "en", "prev_en", "relationship", "register", "target_syllables", "tolerance", "draft", "must_keep",
          "n", "avoid", "feedback"}


def _req(i: str, en: str = "I'll pay half.", reg: str = "반말", n: int = 6, tol: int = 2, n_new: int = 3,
         round_: int = 1, avoid: list[str] | None = None) -> GenRequest:
    return GenRequest(id=i, source_en=en, prev_en="", relationship="", register=reg, n_target=n, tol=tol,
                      draft_ko=None, meaning_notes="", n_new=n_new, round=round_, avoid=list(avoid or []))


def _stream(structured=None, result: str = "", *, model: str = MODEL, is_error: bool = False,
            subtype: str = "success", status: int | None = None, tools=("StructuredOutput",), mcp=(),
            init: bool = True) -> str:
    """가짜 `claude -p --output-format stream-json` 출력(시작 메시지 + 결과 봉투)."""
    lines = []
    if init:
        lines.append({"type": "system", "subtype": "init", "tools": list(tools), "mcp_servers": list(mcp),
                      "model": model})
    lines.append({"type": "result", "subtype": subtype, "is_error": is_error, "result": result,
                  "structured_output": structured, "api_error_status": status,
                  "modelUsage": {model: {"inputTokens": 10, "outputTokens": 20}}})
    return "\n".join(json.dumps(o, ensure_ascii=False) for o in lines) + "\n"


def _items_in(text: str) -> list[dict]:
    """프롬프트 글(또는 stream-json 메시지)에서 대사 JSON 배열을 꺼낸다."""
    if text.lstrip().startswith('{"type"'):
        text = json.loads(text)["message"]["content"][0]["text"]
    return json.JSONDecoder().raw_decode(text, text.index("["))[0]


def _answer(items: list[dict]) -> dict:
    return {"items": [{"id": it["id"], "candidates": [f"{it['id']}-후보-{k}" for k in range(it["n"])]}
                      for it in items]}


class FakeRun:
    """subprocess.run 대신. 호출을 적고 handler(호출 번호, 대사 목록, 입력) 의 (stdout, stderr, code) 를 낸다."""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def __call__(self, cmd, input=None, **kw):
        with self.lock:
            self.calls.append({"cmd": cmd, "input": input, **kw})
            n = len(self.calls)
        out = self.handler(n, _items_in(input), input)
        if isinstance(out, str):
            out = (out, "", 0)
        return subprocess.CompletedProcess(cmd, out[2], out[0], out[1])


@pytest.fixture
def make_claude(tmp_path, monkeypatch):
    def make(handler, **cfg_kw):
        fake = FakeRun(handler)
        monkeypatch.setattr(llm.subprocess, "run", fake)
        b = ClaudeCLIBackend(Config(**cfg_kw), exe="C:/fake/claude.exe", cache_dir=tmp_path / "cache",
                             work_dir=tmp_path / "work")
        b.retry_wait = 0
        return b, fake
    return make


# ---------------------------------------------------------------- 답 읽기

def test_parse_fenced_json():
    text = '```json\n{"items": [{"id": "L1", "candidates": ["반반 하자.", "내가 반 낼게."]}]}\n```'
    assert parse_reply(text, ["L1"]) == {"L1": ["반반 하자.", "내가 반 낼게."]}


def test_parse_prose_around_json():
    text = '네, 후보입니다.\n{"items": [{"id": "L1", "candidates": ["반반 하자."]}]}\n도움이 되길 바랍니다.'
    assert parse_reply(text, ["L1"]) == {"L1": ["반반 하자."]}


def test_parse_bare_list():
    text = '[{"id": "L1", "candidates": ["가요."]}, {"id": "L2", "candidates": ["나요.", "다요."]}]'
    assert parse_reply(text, ["L1", "L2"]) == {"L1": ["가요."], "L2": ["나요.", "다요."]}


def test_parse_missing_ids_get_empty_list():
    obj = {"items": [{"id": "L1", "candidates": ["좋아."]}]}
    assert parse_reply(obj, ["L1", "L2"]) == {"L1": ["좋아."], "L2": []}


def test_parse_non_string_candidates_dropped():
    obj = {"items": [{"id": "L1", "candidates": [1, None, "  좋아.  ", {"text": "그래."}, ["x"], "", {"s": 1}]}]}
    assert parse_reply(obj, ["L1"]) == {"L1": ["좋아.", "그래."]}


def test_parse_numeric_ids_and_id_mapping():
    assert parse_reply({"items": [{"id": 1, "candidates": ["가."]}]}, ["1"]) == {"1": ["가."]}
    assert parse_reply('{"L1": ["가."], "L2": ["나."]}', ["L1", "L2"]) == {"L1": ["가."], "L2": ["나."]}


def test_parse_prefers_answer_over_echoed_input():
    echo = json.dumps([{"id": "L1", "en": "Hi."}])
    ans = json.dumps({"items": [{"id": "L1", "candidates": ["안녕."]}]}, ensure_ascii=False)
    assert parse_reply(f"입력은 {echo} 이고 답은 아래와 같다.\n{ans}", ["L1"]) == {"L1": ["안녕."]}


@pytest.mark.parametrize("bad", [
    "JSON 이 없는 답", '{"items": []}', '{"items": [{"id": "X9", "candidates": ["가."]}]}', None,
    '{"items": [{"id": "L1", "candidates": [1, 2]}]}',
])
def test_parse_unusable_reply_raises(bad):
    with pytest.raises(BadOutput):
        parse_reply(bad, ["L1"])


# ---------------------------------------------------------------- 프롬프트

class _Num:
    """넘파이 수처럼 item() 이 있는 값."""

    def item(self) -> float:
        return 0.61


def test_build_prompt_fields_and_rules():
    fb = {"text": "내가 절반을 지불하겠다.", "reason": "길이 10음절(목표 6±2)", "labse": _Num()}
    reqs = [GenRequest(id="L1", source_en="I'll pay half.", prev_en="Dinner was great.", relationship="친구 사이",
                       register="반말", n_target=6, tol=2, draft_ko="내가 반을 낼게.", meaning_notes="절반",
                       n_new=5, round=2, avoid=["내가 반을 낼게."], feedback=[fb]),
            GenRequest(id="L2", source_en="Thanks.", prev_en="", relationship="", register="", n_target=3, tol=1,
                       draft_ko=None, meaning_notes="", n_new=2)]
    system, user = build_prompt(reqs)
    assert system == SYSTEM_PROMPT and user.startswith(USER_HEAD)
    items = json.loads(user[len(USER_HEAD):])
    assert [set(it) for it in items] == [FIELDS, FIELDS]
    assert items[0] == {"id": "L1", "en": "I'll pay half.", "prev_en": "Dinner was great.",
                        "relationship": "친구 사이", "register": "반말", "target_syllables": 6, "tolerance": 2,
                        "draft": "내가 반을 낼게.", "must_keep": "절반", "n": 5, "avoid": ["내가 반을 낼게."],
                        "feedback": [{"text": "내가 절반을 지불하겠다.", "reason": "길이 10음절(목표 6±2)",
                                      "labse": 0.61}]}
    assert (items[1]["register"], items[1]["draft"], items[1]["avoid"], items[1]["feedback"]) == ("미지정", "", [], [])
    for phrase in ("한글 음절 글자(가~힣) 하나가 1음절", "띄어쓰기와 문장부호", "소리 나는 대로 한글로",
                   "정확히 target_syllables", "정확히 n 개", "서로 다르게", "avoid", "feedback", "JSON 하나만"):
        assert phrase in system
    # 지시문 예시의 음절 수가 실제와 맞아야 한다(한글 음절 글자만 셈)
    assert sum("가" <= ch <= "힣" for ch in "내일 아침 일곱 시에 봐요.") == 10
    assert isinstance(PROMPT_VERSION, str) and PROMPT_VERSION


def test_build_prompt_rejects_unknown_register(make_claude):
    with pytest.raises(ValueError, match="알 수 없는 말투"):
        build_prompt([_req("L1", reg="존댓말")])
    b, fake = make_claude(lambda n, items, raw: _stream(_answer(items)))
    with pytest.raises(ValueError, match="알 수 없는 말투"):
        b.generate([_req("L1"), _req("L2", reg="존댓말")])
    assert fake.calls == []


# ---------------------------------------------------------------- claude -p 명령·환경

def test_claude_command_env_cwd_and_stdin(make_claude, monkeypatch, tmp_path):
    parent = {"CLAUDECODE": "1", "CLAUDE_CODE_ENTRYPOINT": "cli", "CLAUDE_CODE_SESSION_ID": "s",
              "CLAUDE_PID": "123", "CLAUDE_EFFORT": "max", "CLAUDE_CONFIG_DIR": "C:/cfg", "PSTTS_TEST_KEEP": "ok"}
    for k, v in parent.items():
        monkeypatch.setenv(k, v)
    b, fake = make_claude(lambda n, items, raw: _stream(_answer(items)))
    reqs = [_req("L1"), _req("L2", en="Could you tell me where the station is?", reg="해요체", n=12)]
    out = b.generate(reqs)
    assert isinstance(out, GenReply) and isinstance(out, dict)
    assert out == {"L1": ["L1-후보-0", "L1-후보-1", "L1-후보-2"], "L2": ["L2-후보-0", "L2-후보-1", "L2-후보-2"]}
    assert (out.failed, out.errors, out.calls, out.cache_hits) == ([], [], 1, 0)

    call = fake.calls[0]
    cmd = call["cmd"]
    system, user = build_prompt(reqs)

    def arg(flag: str) -> str:
        return cmd[cmd.index(flag) + 1]

    assert cmd[:2] == ["C:/fake/claude.exe", "-p"]
    assert (arg("--model"), arg("--effort")) == ("claude-sonnet-5-5", "low")
    assert (arg("--tools"), arg("--setting-sources")) == ("", "")
    for flag in ("--strict-mcp-config", "--safe-mode", "--disable-slash-commands", "--no-session-persistence",
                 "--verbose"):
        assert flag in cmd
    assert "--mcp-config" not in cmd
    assert arg("--system-prompt") == system
    assert (arg("--input-format"), arg("--output-format")) == ("stream-json", "stream-json")
    assert json.loads(arg("--json-schema")) == SCHEMA

    env = call["env"]
    for k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID", "CLAUDE_PID", "CLAUDE_EFFORT"):
        assert k not in env
    assert env["CLAUDE_CONFIG_DIR"] == "C:/cfg" and env["PSTTS_TEST_KEEP"] == "ok" and "PATH" in env
    assert Path(call["cwd"]) == tmp_path / "work" and list((tmp_path / "work").iterdir()) == []
    assert call["timeout"] == Config().llm_timeout and call["encoding"] == "utf-8"
    msg = json.loads(call["input"])
    assert msg["type"] == "user" and msg["message"]["content"] == [{"type": "text", "text": user}]


def test_claude_uses_result_text_when_no_structured_output(make_claude):
    b, fake = make_claude(lambda n, items, raw: _stream(
        None, "```json\n" + json.dumps(_answer(items), ensure_ascii=False) + "\n```"))
    out = b.generate([_req("L1")])
    assert out["L1"] == ["L1-후보-0", "L1-후보-1", "L1-후보-2"] and len(fake.calls) == 1


# ---------------------------------------------------------------- claude 오류 가르기

_REJECTED = (json.dumps({"type": "system", "subtype": "init", "tools": ["StructuredOutput"], "mcp_servers": []})
             + "\n" + json.dumps({"type": "rate_limit_event", "rate_limit_info": {
                 "status": "rejected", "resetsAt": 1763012345, "rateLimitType": "five_hour"}}) + "\n")


@pytest.mark.parametrize("stdout, stderr, exc, words", [
    (_stream(None, "Invalid API key", is_error=True, status=401), "", LLMUnavailable, "로그인"),
    ("", "Not logged in · Please run /login", LLMUnavailable, "로그인"),
    (_stream(None, "model not found", is_error=True, status=404), "", LLMUnavailable, "--llm-model"),
    (_stream(None, "Claude AI usage limit reached|1763012345", is_error=True, status=429), "", LLMLimit,
     "풀리는 시각"),
    (_REJECTED, "", LLMLimit, "사용량 한도"),
    (_stream(None, "API Error: 529 Overloaded", is_error=True, status=529), "", Transient, "과부하"),
    (_stream(None, "Failed to provide valid structured output", is_error=True,
             subtype="error_max_structured_output_retries"), "", BadOutput, "구조화 출력"),
    ("", "something odd", Transient, "결과 없음"),
    (_stream({"items": []}, tools=("StructuredOutput", "Bash")), "", LLMUnavailable, "도구"),
    (_stream({"items": []}, mcp=({"name": "x", "status": "connected"},)), "", LLMUnavailable, "MCP"),
    (_stream({"items": []}, init=False), "", LLMUnavailable, "init"),
    (_stream({"items": []}, model="claude-haiku-5-5"), "", BadOutput, "아닌 모델"),
])
def test_read_claude_output_classifies_errors(stdout, stderr, exc, words):
    with pytest.raises(exc, match=re.escape(words)) as ei:
        read_claude_output(stdout, stderr, 1, MODEL)
    if exc is LLMUnavailable:
        assert not isinstance(ei.value, LLMLimit)


def test_read_claude_output_ok():
    env = read_claude_output(_stream({"items": [{"id": "L1", "candidates": ["가."]}]}), "", 0, MODEL)
    assert env["structured_output"]["items"][0]["id"] == "L1" and list(env["modelUsage"]) == [MODEL]


def test_auth_error_is_not_retried(make_claude):
    b, fake = make_claude(lambda n, items, raw: ("", "Not logged in · Please run /login", 1))
    with pytest.raises(LLMUnavailable, match="로그인"):
        b.generate([_req("L1")])
    assert len(fake.calls) == 1


def test_usage_limit_raises_korean_error(make_claude):
    b, fake = make_claude(lambda n, items, raw: (_REJECTED, "", 1))
    with pytest.raises(LLMLimit, match="사용량 한도에 걸렸습니다"):
        b.generate([_req("L1")])
    assert len(fake.calls) == 1


def test_missing_claude_exe(tmp_path, monkeypatch):
    def never(*a, **k):
        raise AssertionError("실행 파일이 없으면 subprocess 를 부르면 안 됨")
    monkeypatch.setattr(llm.shutil, "which", lambda name: None)
    monkeypatch.setattr(llm.subprocess, "run", never)
    b = ClaudeCLIBackend(Config(), cache_dir=tmp_path / "cache", work_dir=tmp_path / "work")
    with pytest.raises(LLMUnavailable, match="claude 실행 파일을 찾을 수 없습니다"):
        b.generate([_req("L1")])


def test_launch_oserror_is_unavailable(make_claude):
    def handler(n, items, raw):
        raise FileNotFoundError("claude.exe")
    b, fake = make_claude(handler)
    with pytest.raises(LLMUnavailable, match="실행하지 못했습니다"):
        b.generate([_req("L1")])


# ---------------------------------------------------------------- 재시도

def test_malformed_reply_retried_once_with_note(make_claude):
    b, fake = make_claude(lambda n, items, raw: _stream(None, "죄송해요, 지금은 못 하겠어요.") if n == 1
                          else _stream(_answer(items)))
    out = b.generate([_req("L1")])
    assert len(fake.calls) == 2 and out.calls == 2 and out["L1"] and out.failed == []
    assert "앞 답을 쓸 수 없었다" in json.loads(fake.calls[1]["input"])["message"]["content"][0]["text"]


@pytest.mark.parametrize("first", ["timeout", "overloaded"])
def test_transient_error_retried_once(make_claude, first):
    def handler(n, items, raw):
        if n == 1 and first == "timeout":
            raise subprocess.TimeoutExpired("claude", 900)
        if n == 1:
            return "", "API Error: 529 overloaded", 1
        return _stream(_answer(items))
    b, fake = make_claude(handler)
    out = b.generate([_req("L1")])
    assert len(fake.calls) == 2 and out["L1"]
    # 일시 오류는 지시문을 바꾸지 않고 같은 입력으로 다시 부른다
    assert fake.calls[0]["input"] == fake.calls[1]["input"]


def test_chunk_failing_twice_is_recorded(make_claude):
    b, fake = make_claude(lambda n, items, raw: _stream(_answer(items), model="claude-haiku-5-5"))
    out = b.generate([_req("L1")])
    assert out["L1"] == [] and out.failed == ["L1"] and len(fake.calls) == 2
    assert "두 번 모두 실패" in out.errors[0] and "아닌 모델" in out.errors[0]


# ---------------------------------------------------------------- 캐시

def test_cache_hit_means_no_second_call(make_claude, tmp_path):
    b, fake = make_claude(lambda n, items, raw: _stream(_answer(items)))
    reqs = [_req("L1")]
    first = b.generate(reqs)
    second = b.generate(reqs)
    assert len(fake.calls) == 1
    assert second == first and (second.calls, second.cache_hits) == (0, 1)
    files = list((tmp_path / "cache").glob("*.json"))
    assert len(files) == 1 and not list((tmp_path / "cache").glob("*.tmp"))
    rec = json.loads(files[0].read_text(encoding="utf-8"))
    system, user = build_prompt(reqs)
    assert rec["key"] == files[0].stem == cache_key("claude-cli", MODEL, "low", system, user, [1])
    assert (rec["prompt_version"], rec["model"], rec["effort"]) == (PROMPT_VERSION, MODEL, "low")
    assert list(rec["meta"]["modelUsage"]) == [MODEL]
    # 차수가 다르면 같은 프롬프트라도 새로 묻는다
    b.generate([_req("L1", round_=2)])
    assert len(fake.calls) == 2


def test_cache_key_parts():
    base = cache_key("claude-cli", MODEL, "low", "s", "u", [1])
    assert base != cache_key("claude-cli", MODEL, "medium", "s", "u", [1])
    assert base != cache_key("claude-cli", "claude-opus-5-5", "low", "s", "u", [1])
    assert base != cache_key("openai", MODEL, "low", "s", "u", [1])
    assert base != cache_key("claude-cli", MODEL, "low", "s2", "u", [1])
    assert base != cache_key("claude-cli", MODEL, "low", "s", "u2", [1])
    assert base == cache_key("claude-cli", MODEL, "low", "s", "u", [1])


def test_corrupt_cache_file_is_ignored(make_claude, tmp_path):
    b, fake = make_claude(lambda n, items, raw: _stream(_answer(items)))
    reqs = [_req("L1")]
    b.generate(reqs)
    path = next((tmp_path / "cache").glob("*.json"))
    path.write_text("{깨진", encoding="utf-8")
    out = b.generate(reqs)
    assert len(fake.calls) == 2 and out["L1"] and json.loads(path.read_text(encoding="utf-8"))["key"] == path.stem


def test_no_cache_calls_every_time(make_claude, tmp_path):
    b, fake = make_claude(lambda n, items, raw: _stream(_answer(items)), llm_cache=False)
    b.generate([_req("L1")])
    b.generate([_req("L1")])
    assert len(fake.calls) == 2 and not (tmp_path / "cache").exists()


# ---------------------------------------------------------------- 조각·동시 호출·부분 실패

def test_chunks_run_in_parallel_and_partial_failure(make_claude):
    state = {"now": 0, "max": 0}
    lock = threading.Lock()

    def handler(n, items, raw):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.3)
        with lock:
            state["now"] -= 1
        if any(it["id"] == "L3" for it in items):
            return _stream(None, "형식 없는 답")
        return _stream(_answer(items))

    b, fake = make_claude(handler, llm_chunk=2, llm_workers=2)
    out = b.generate([_req(f"L{i}") for i in range(1, 6)])
    assert sorted(len(_items_in(c["input"])) for c in fake.calls) == [1, 2, 2, 2]   # [L1 L2] [L3 L4]x2 [L5]
    assert list(out) == ["L1", "L2", "L3", "L4", "L5"]
    assert out["L1"] and out["L2"] and out["L5"] and out["L3"] == [] and out["L4"] == []
    assert out.failed == ["L3", "L4"] and out.calls == 4 and len(out.errors) == 1
    assert "조각 2/3" in out.errors[0]
    assert state["max"] == 2


def test_unavailable_stops_remaining_chunks_and_keeps_partial(make_claude):
    b, fake = make_claude(lambda n, items, raw: _stream(_answer(items)) if n == 1
                          else ("", "Not logged in · Please run /login", 1), llm_chunk=1, llm_workers=1)
    with pytest.raises(LLMUnavailable, match="로그인") as ei:
        b.generate([_req("L1"), _req("L2"), _req("L3")])
    part = ei.value.partial
    assert part["L1"] == ["L1-후보-0", "L1-후보-1", "L1-후보-2"] and part["L2"] == [] and part["L3"] == []
    assert part.failed == ["L2", "L3"] and len(fake.calls) == 2


def test_reply_missing_one_id_is_recorded(make_claude):
    b, fake = make_claude(lambda n, items, raw: _stream(_answer(items[:1])))
    out = b.generate([_req("L1"), _req("L2")])
    assert out["L1"] and out["L2"] == [] and out.failed == ["L2"] and len(fake.calls) == 1
    assert "답에 이 줄이 없음" in out.errors[0]


def test_dedupe_avoid_and_cap(make_claude):
    reply = {"items": [{"id": "L1", "candidates": ["내가 반 낼게.", "내가 반 낼게!", "반반 하자.", "  반반  내자. ",
                                                   "절반은 내가 낼게."]}]}
    b, fake = make_claude(lambda n, items, raw: _stream(reply))
    out = b.generate([_req("L1", n_new=2, avoid=["반반 하자"])])
    assert out["L1"] == ["내가 반 낼게.", "반반 내자."]


def test_cleaning_happens_before_the_cap(tmp_path):
    # 번호·따옴표만 다른 겹침이 자리를 먹지 않게, 다듬은 뒤에 겹침을 빼고 n_new 개로 자른다(2026-11-22 검토)
    reply = ["1. 지금 가자!", "\"지금 가자!\"", "- 이제 가자.", "(1) 이제 가자.", "어서 가자!", "당장 가자!", "빨리 가자!",
             "가자, 지금!"]
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"id": "L1", "candidates": reply}, ensure_ascii=False) + "\n", encoding="utf-8")
    out = FileBackend(p).generate([_req("L1", n_new=4)])
    assert out["L1"] == ["지금 가자!", "이제 가자.", "어서 가자!", "당장 가자!"]


def test_zero_n_new_is_not_sent_and_duplicate_ids_rejected(make_claude):
    b, fake = make_claude(lambda n, items, raw: _stream(_answer(items)))
    out = b.generate([_req("L1", n_new=0), _req("L2")])
    assert [it["id"] for it in _items_in(fake.calls[0]["input"])] == ["L2"]
    assert out["L1"] == [] and out.failed == []
    assert b.generate([_req("L1", n_new=0)]) == {"L1": []} and len(fake.calls) == 1
    with pytest.raises(ValueError, match="같은 id"):
        b.generate([_req("L1"), _req("L1")])


# ---------------------------------------------------------------- FileBackend

def test_file_backend_rounds_and_missing(tmp_path):
    rows = [{"id": "L1", "candidates": ["반반 하자.", "내가 반 낼게."]},
            {"id": "L1", "round": 1, "candidates": ["반은 내가 낼게.", 3]},
            {"id": "L1", "round": 2, "candidates": ["절반은 내가."]},
            {"id": "L2", "round": 0, "candidates": ["초벌."]}]
    p = tmp_path / "cands.jsonl"
    p.write_text("\ufeff" + "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n\n", encoding="utf-8")
    fb = FileBackend(p)
    out = fb.generate([_req("L1", n_new=5, avoid=["반반 하자"]), _req("L2")])
    assert out == {"L1": ["내가 반 낼게.", "반은 내가 낼게."], "L2": []}
    assert out.failed == ["L2"] and "1차 후보가 없음" in out.errors[0]
    assert fb.generate([_req("L1", round_=2)]) == {"L1": ["절반은 내가."]}
    assert fb.generate([_req("L1", n_new=1)]) == {"L1": ["반반 하자."]}


def test_file_backend_bad_input(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "L1", "candidates": ["가."]}\n{"id": "L2"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="2행"):
        FileBackend(p)
    p.write_text('{"id": "L1", "candidates": ["가."], "round": "둘"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="round"):
        FileBackend(p)
    with pytest.raises(FileNotFoundError):
        FileBackend(tmp_path / "none.jsonl")


# ---------------------------------------------------------------- OpenAI 호환(가짜 서버)

class _Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        self.server.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        status, payload = self.server.reply(len(self.server.seen), body)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        return None


@pytest.fixture
def fake_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.seen = []
    srv.reply = lambda n, body: (500, {})
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _chat_reply(n: int, body: dict) -> tuple[int, dict]:
    items = _items_in(body["messages"][1]["content"])
    text = "결과입니다.\n```json\n" + json.dumps(_answer(items), ensure_ascii=False) + "\n```\n끝."
    return 200, {"model": "fake-model", "choices": [{"message": {"role": "assistant", "content": text}}],
                 "usage": {"total_tokens": 1}}


def test_openai_backend_against_fake_server(fake_server, tmp_path, monkeypatch):
    fake_server.reply = _chat_reply
    monkeypatch.setenv("PSTTS_FAKE_KEY", "sk-test")
    port = fake_server.server_address[1]
    b = OpenAICompatBackend(f"http://127.0.0.1:{port}/v1/", "fake-model", "PSTTS_FAKE_KEY", Config(),
                            cache_dir=tmp_path / "cache")
    assert b.url == f"http://127.0.0.1:{port}/v1/chat/completions"
    reqs = [_req("L1"), _req("L2")]
    out = b.generate(reqs)
    assert out == {"L1": ["L1-후보-0", "L1-후보-1", "L1-후보-2"], "L2": ["L2-후보-0", "L2-후보-1", "L2-후보-2"]}
    seen = fake_server.seen[0]
    assert seen["path"] == "/v1/chat/completions" and seen["auth"] == "Bearer sk-test"
    assert seen["body"]["temperature"] == 0.9 and seen["body"]["model"] == "fake-model"
    system, user = build_prompt(reqs)
    assert seen["body"]["messages"] == [{"role": "system", "content": system}, {"role": "user", "content": user}]
    again = b.generate(reqs)
    assert len(fake_server.seen) == 1 and again.cache_hits == 1 and again == out


def test_openai_server_error_retried(fake_server, tmp_path):
    fake_server.reply = lambda n, body: (503, {"error": "busy"}) if n == 1 else _chat_reply(n, body)
    port = fake_server.server_address[1]
    b = OpenAICompatBackend(f"http://127.0.0.1:{port}", "m", None, Config(llm_cache=False))
    b.retry_wait = 0
    out = b.generate([_req("L1")])
    assert out["L1"] and len(fake_server.seen) == 2 and fake_server.seen[0]["auth"] is None


def test_openai_auth_error(fake_server):
    fake_server.reply = lambda n, body: (401, {"error": {"message": "bad key"}})
    port = fake_server.server_address[1]
    b = OpenAICompatBackend(f"http://127.0.0.1:{port}", "m", None, Config(llm_cache=False))
    with pytest.raises(LLMUnavailable, match="인증 실패"):
        b.generate([_req("L1")])
    assert len(fake_server.seen) == 1


def test_openai_missing_key_env_and_refused_connection(monkeypatch):
    monkeypatch.delenv("PSTTS_NO_SUCH_KEY", raising=False)
    b = OpenAICompatBackend("http://127.0.0.1:9", "m", "PSTTS_NO_SUCH_KEY", Config(llm_cache=False))
    with pytest.raises(LLMUnavailable, match="API 키가 없습니다"):
        b.generate([_req("L1")])
    with socket.socket() as s:          # 빈 포트를 잡았다가 닫아 아무도 듣지 않는 주소를 만든다
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    b = OpenAICompatBackend(f"http://127.0.0.1:{port}", "m", None, Config(llm_cache=False, llm_timeout=10))
    with pytest.raises(LLMUnavailable, match="연결할 수 없습니다"):
        b.generate([_req("L1")])


# ---------------------------------------------------------------- make_backend

def test_make_backend(tmp_path):
    assert isinstance(make_backend(Config()), ClaudeCLIBackend)
    ob = make_backend(Config(llm="openai", llm_model="qwen", llm_base_url="http://127.0.0.1:1"))   # 만들기만 함
    assert isinstance(ob, OpenAICompatBackend) and ob.url == "http://127.0.0.1:1/v1/chat/completions"
    assert (ob.model, ob.effort) == ("qwen", "-")
    p = tmp_path / "c.jsonl"
    p.write_text('{"id": "L1", "candidates": ["가."]}\n', encoding="utf-8")
    assert isinstance(make_backend(Config(llm="file", llm_file=str(p))), FileBackend)
    with pytest.raises(ValueError, match="--llm-file"):
        make_backend(Config(llm="file"))
    with pytest.raises(ValueError, match="알 수 없는 LLM 백엔드"):
        make_backend(Config(llm="gpt"))
