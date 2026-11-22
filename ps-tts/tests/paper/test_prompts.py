"""논문 판 프롬프트 시험: 논문 문장만 옮긴 지시(더한 조건 없음), 줄 자료 칸, 캐시 키, 조각 나누기,
PS 60개 모으기(모자라면 더 묻기), 단계 재생, OpenAI 호환 경로(시험 안의 가짜 서버). 실제 claude 는 부르지 않는다."""
from __future__ import annotations

import http.server
import json
import subprocess
import threading

import pytest

from pstts import llm
from pstts.config import Config
from pstts.llm import (PROMPT_VERSION, SCHEMA, SYSTEM_PROMPT, ClaudeCLIBackend, FileBackend, GenReply, GenRequest,
                       LLMUnavailable, OpenAICompatBackend, build_prompt, cache_key)
from pstts.paper.prompts import (ISO_SYSTEM, ISO_USER_HEAD, PROMPT_VERSION_PAPER, PS_SYSTEM, PS_USER_HEAD, STAGE_ISO,
                                 STAGE_PS, GenOutcome, IsoRequest, PsRequest, build_iso_prompt, build_ps_prompt,
                                 iso_generate, iso_spec, keep_new, ps_generate, ps_spec)

MODEL = Config().llm_model
EN = "We have to leave before the storm hits."
T0 = "우리는 폭풍이 치기 전에 떠나야 합니다."


def _iso(i: str = "L1", n: int = 3, ratio: float = 0.8049, avoid: list[str] | None = None, rnd: int = 1) -> IsoRequest:
    return IsoRequest(id=i, en=EN, ko=T0, length_ratio=ratio, n_new=n, round=rnd, avoid=list(avoid or [T0]))


def _items(user: str, head: str) -> list[dict]:
    assert user.startswith(head)
    return json.loads(user[len(head):])


# ---------------------------------------------------------------- 프롬프트 글

def test_iso_prompt_fields() -> None:
    system, user = build_iso_prompt([_iso(), _iso("L2", n=10, ratio=1.234, avoid=[T0, "가."])])
    assert system == ISO_SYSTEM
    items = _items(user, ISO_USER_HEAD)
    assert [set(it) for it in items] == [{"id", "en", "ko", "length_ratio", "n", "avoid"}] * 2
    assert items[0] == {"id": "L1", "en": EN, "ko": T0, "length_ratio": 0.8, "n": 3, "avoid": [T0]}
    assert items[1]["length_ratio"] == 1.23 and items[1]["avoid"] == [T0, "가."]


def test_ps_prompt_fields_without_english() -> None:
    system, user = build_ps_prompt([PsRequest(id="L1", ko="폭풍이 치기 전에 떠나야 해요.", n_new=60)])
    assert system == PS_SYSTEM
    items = _items(user, PS_USER_HEAD)
    assert items == [{"id": "L1", "ko": "폭풍이 치기 전에 떠나야 해요.", "n": 60, "avoid": []}]
    assert "en" not in user and EN not in user                       # 논문은 ISO 문장만 기준


# 기존 판이 더한 조건(말투, 음절 수, 숫자, 입 모양, 말 다듬기)과 그 입력 칸 이름
FORBIDDEN = ("말투", "반말", "해요체", "하십시오체", "존댓말", "음절", "숫자", "글자", "한글로",
             "register", "syllable", "target", "tolerance", "feedback", "must_keep", "prev_en", "relationship", "draft",
             "입 모양", "모음", "더빙", "성우", "구어", "자연스러운", "번역 투", "감정", "강조")


@pytest.mark.parametrize("text", [ISO_SYSTEM, ISO_USER_HEAD, PS_SYSTEM, PS_USER_HEAD])
def test_no_added_conditions(text: str) -> None:
    for word in FORBIDDEN:
        assert word not in text, f"논문에 없는 조건 '{word}' 가 프롬프트에 있다"


def test_prompts_state_what_the_paper_says() -> None:
    for phrase in ("ko 를 바꿔 써서", "en 과 뜻이 같게 유지", "소리 내어 읽는 시간이 ko 의 length_ratio 배",
                   "1 보다 작으면 짧게", "1 보다 크면 길게", "1 에 가까우면 길이는 그대로 두고 뜻을 en 에 더 가깝게",
                   "avoid", "JSON 하나만"):
        assert phrase in ISO_SYSTEM, phrase
    for phrase in ("ko 와 소리 내어 읽는 길이가 비슷하고 뜻이 비슷한 한국어 문장을 n 개", "서로 달라야", "avoid",
                   "JSON 하나만"):
        assert phrase in PS_SYSTEM, phrase
    assert '{"items": [{"id": "입력 id", "candidates": ["문장", "문장"]}]}' in ISO_SYSTEM


def test_version_and_cache_keys_differ_from_main() -> None:
    assert PROMPT_VERSION_PAPER == "pstts-paper/1" != PROMPT_VERSION
    assert (iso_spec().version, iso_spec().stage, iso_spec().chunk) == (PROMPT_VERSION_PAPER, STAGE_ISO, None)
    assert (ps_spec(2).version, ps_spec(2).stage, ps_spec(2).chunk) == (PROMPT_VERSION_PAPER, STAGE_PS, 2)
    system, user = build_iso_prompt([_iso()])
    paper = cache_key("claude-cli", MODEL, "low", system, user, [1], PROMPT_VERSION_PAPER)
    assert paper != cache_key("claude-cli", MODEL, "low", system, user, [1])
    # 기존 판 프롬프트는 그대로(논문 판 지시가 섞이지 않음)
    main_system, _ = build_prompt([GenRequest(id="L1", source_en=EN, prev_en="", relationship="", register="해요체",
                                              n_target=14, tol=2, draft_ko=T0, meaning_notes="", n_new=5)])
    assert main_system == SYSTEM_PROMPT and ISO_SYSTEM != SYSTEM_PROMPT and PS_SYSTEM != SYSTEM_PROMPT


def test_keep_new_is_idempotent() -> None:
    cands = ["가자.", "가자!", "이미 나온 말", "  ", "...", "새 말."]
    once = keep_new(["이미 나온 말."], cands, 5)
    assert once == ["가자.", "새 말."]
    assert keep_new(["이미 나온 말."], once, 5) == once and keep_new([], once, 1) == ["가자."]


# ---------------------------------------------------------------- 가짜 claude(subprocess)

def _stream(structured: dict) -> str:
    lines = [{"type": "system", "subtype": "init", "tools": ["StructuredOutput"], "mcp_servers": [], "model": MODEL},
             {"type": "result", "subtype": "success", "is_error": False, "result": "", "structured_output": structured,
              "api_error_status": None, "modelUsage": {MODEL: {"inputTokens": 1, "outputTokens": 1}}}]
    return "\n".join(json.dumps(o, ensure_ascii=False) for o in lines) + "\n"


def _items_of_stdin(stdin: str) -> tuple[str, list[dict]]:
    text = json.loads(stdin)["message"]["content"][0]["text"]
    return text, json.JSONDecoder().raw_decode(text, text.index("["))[0]


class FakeRun:
    """subprocess.run 대신. 줄마다 '{id}-{k}.' 후보를 n 개(앞의 하나는 번호를 붙여) 돌려준다."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def __call__(self, cmd, input=None, **kw):
        with self.lock:
            self.calls.append({"cmd": cmd, "input": input})
        _, items = _items_of_stdin(input)
        answer = {"items": [{"id": it["id"], "candidates": [f"1. {it['id']} 후보 0."] +
                             [f"{it['id']} 후보 {k}." for k in range(1, it["n"])]} for it in items]}
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


def test_iso_generate_with_claude_backend(claude, tmp_path) -> None:
    b, fake = claude(llm_chunk=2, llm_workers=1)
    reqs = [_iso("L1", n=2), _iso("L2", n=3), _iso("L3", n=1)]
    out = iso_generate(b, reqs)
    assert isinstance(out, GenOutcome) and out.stopped == "" and out.failed == []
    assert out.candidates == {"L1": ["L1 후보 0.", "L1 후보 1."], "L2": ["L2 후보 0.", "L2 후보 1.", "L2 후보 2."],
                              "L3": ["L3 후보 0."]}                    # 번호는 기존 다듬기로 지워짐
    assert [len(_items_of_stdin(c["input"])[1]) for c in fake.calls] == [2, 1]       # llm_chunk 2 로 나눔
    assert (out.calls, out.cache_hits) == (2, 0)
    cmd = fake.calls[0]["cmd"]
    assert cmd[cmd.index("--system-prompt") + 1] == ISO_SYSTEM
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == SCHEMA
    assert out.rows == [{"id": i, "stage": "iso", "round": 1, "candidates": out.candidates[i]}
                        for i in ("L1", "L2", "L3")]
    records = [json.loads(p.read_text(encoding="utf-8")) for p in (tmp_path / "cache").glob("*.json")]
    assert len(records) == 2 and {r["prompt_version"] for r in records} == {PROMPT_VERSION_PAPER}
    for rec in records:
        assert rec["key"] == cache_key(b.cache_tag, b.model, b.effort, rec["system"], rec["user"], rec["rounds"],
                                       PROMPT_VERSION_PAPER)
    again = iso_generate(b, reqs)
    assert again.candidates == out.candidates and (again.calls, again.cache_hits) == (0, 2) and len(fake.calls) == 2


def test_ps_generate_chunks_by_ps_chunk(claude) -> None:
    b, fake = claude(llm_chunk=6, llm_workers=1)
    reqs = [PsRequest(id=f"L{i}", ko="폭풍이 치기 전에 떠나야 해요.", n_new=4) for i in range(3)]
    out = ps_generate(b, reqs, extra_rounds=2, chunk=2, log=lambda m: None)
    assert [len(_items_of_stdin(c["input"])[1]) for c in fake.calls] == [2, 1]
    text, items = _items_of_stdin(fake.calls[0]["input"])
    assert text.startswith(PS_USER_HEAD) and items[0] == {"id": "L0", "ko": "폭풍이 치기 전에 떠나야 해요.", "n": 4,
                                                          "avoid": []}
    assert all(len(v) == 4 for v in out.candidates.values()) and out.calls == 2
    assert [r["stage"] for r in out.rows] == ["ps"] * 3 and out.rounds[0]["new"] == 12 and len(out.rounds) == 1


# ---------------------------------------------------------------- 가짜 백엔드(generate 만)

class FakeBackend:
    """generate(reqs, prompt) 를 적고 handler(차례, reqs) 가 준 id -> 후보 목록을 GenReply 로 돌려준다."""
    name = "fake"

    def __init__(self, handler) -> None:
        self.handler = handler
        self.seen: list[tuple[list, object]] = []

    def generate(self, reqs, prompt=None) -> GenReply:
        self.seen.append(([vars(r).copy() for r in reqs], prompt))
        got = self.handler(len(self.seen), reqs)
        out = GenReply()
        for r in reqs:
            out[r.id] = keep_new(r.avoid, got.get(r.id, []), r.n_new)
            if not out[r.id]:
                out.failed.append(r.id)
                out.errors.append(f"{r.id}: 없음")
        out.calls = 1
        return out


def test_ps_generate_tops_up_missing_candidates() -> None:
    answers = {1: {"A": ["하나.", "하나!", "둘.", "셋."], "B": []},
               2: {"A": ["둘.", "넷.", "다섯."], "B": ["가.", "나."]},
               3: {"A": ["여섯."], "B": ["다.", "라.", "마.", "바."]}}
    fb = FakeBackend(lambda k, reqs: answers[k])
    reqs = [PsRequest(id="A", ko="기준.", n_new=5), PsRequest(id="B", ko="기준 둘.", n_new=5)]
    out = ps_generate(fb, reqs, extra_rounds=2, chunk=2, log=lambda m: None)
    first, second, third = (s[0] for s in fb.seen)
    assert [(r["id"], r["n_new"], r["round"], r["avoid"]) for r in first] == [("A", 5, 1, []), ("B", 5, 1, [])]
    # 2차: 모자란 만큼, 받은 후보를 avoid 로
    assert [(r["id"], r["n_new"], r["round"], r["avoid"]) for r in second] == [
        ("A", 2, 2, ["하나.", "둘.", "셋."]), ("B", 5, 2, [])]
    assert [(r["id"], r["n_new"], r["round"]) for r in third] == [("B", 3, 3)]
    assert all(p.stage == STAGE_PS and p.chunk == 2 for _, p in fb.seen)
    assert out.candidates == {"A": ["하나.", "둘.", "셋.", "넷.", "다섯."], "B": ["가.", "나.", "다.", "라.", "마."]}
    assert [r["round"] for r in out.rows] == [1, 2, 2, 3] and out.failed == []
    assert [rd["new"] for rd in out.rounds] == [3, 4, 3] and out.calls == 3
    assert out.errors[0].startswith("PS 1차: B")


def test_ps_generate_stops_after_extra_rounds() -> None:
    fb = FakeBackend(lambda k, reqs: {"A": [f"{k}차 문장."]})
    out = ps_generate(fb, [PsRequest(id="A", ko="기준.", n_new=60), PsRequest(id="B", ko="기준.", n_new=60)],
                      extra_rounds=2, chunk=2, log=lambda m: None)
    assert len(fb.seen) == 3 and out.candidates["A"] == ["1차 문장.", "2차 문장.", "3차 문장."]
    assert out.failed == ["B"]
    out = ps_generate(fb, [PsRequest(id="A", ko="기준.", n_new=60)], extra_rounds=0, chunk=2, log=lambda m: None)
    assert len(fb.seen) == 4                                           # 더 묻기 0번이면 한 번만


def test_ps_iso_sentence_itself_is_a_candidate() -> None:
    fb = FakeBackend(lambda k, reqs: {"A": ["기준 문장.", "다른 문장."]})
    out = ps_generate(fb, [PsRequest(id="A", ko="기준 문장.", n_new=2)], extra_rounds=0, chunk=2, log=lambda m: None)
    assert out.candidates["A"] == ["기준 문장.", "다른 문장."]       # ISO 문장을 avoid 로 넣지 않는다


def test_unavailable_keeps_partial_and_stops() -> None:
    def limited(k: int, reqs: list) -> dict:
        if k == 2:                      # 2차에서 사용량 한도: 그때까지 받은 결과(partial)를 담아 알린다
            e = LLMUnavailable("claude 사용량 한도에 걸렸습니다.")
            part = GenReply()
            part.update({"A": ["넷."], "B": []})
            part.failed, part.calls = ["B"], 1
            e.partial = part
            raise e
        return {"A": ["하나.", "둘.", "셋."], "B": []}

    fb = FakeBackend(limited)
    out = ps_generate(fb, [PsRequest(id="A", ko="기준.", n_new=5), PsRequest(id="B", ko="기준.", n_new=5)],
                      extra_rounds=5, chunk=2, log=lambda m: None)
    assert "사용량 한도" in out.stopped and out.candidates["A"] == ["하나.", "둘.", "셋.", "넷."]
    assert out.failed == ["B"] and len(out.rounds) == 2 and out.calls == 2 and len(fb.seen) == 2

    def no_login(k: int, reqs: list) -> dict:
        raise LLMUnavailable("로그인이 필요합니다.")         # partial 없이 나는 경우

    iso = iso_generate(FakeBackend(no_login), [_iso("A")])
    assert iso.stopped == "로그인이 필요합니다." and iso.candidates == {"A": []} and iso.failed == ["A"]


def test_request_validation() -> None:
    fb = FakeBackend(lambda k, reqs: {})
    with pytest.raises(ValueError, match="같은 id"):
        iso_generate(fb, [_iso("A"), _iso("A")])
    with pytest.raises(ValueError, match="ko"):
        iso_generate(fb, [IsoRequest(id="A", en=EN, ko=" ", length_ratio=1.0, n_new=1)])
    with pytest.raises(ValueError, match="length_ratio"):
        iso_generate(fb, [IsoRequest(id="A", en=EN, ko=T0, length_ratio=0.0, n_new=1)])
    with pytest.raises(ValueError, match="extra_rounds"):
        ps_generate(fb, [PsRequest(id="A", ko="가.", n_new=1)], extra_rounds=-1, chunk=2)
    assert fb.seen == []


# ---------------------------------------------------------------- 재생

def test_ps_replay_from_file_gives_same_candidates(claude, tmp_path, monkeypatch) -> None:
    b, _ = claude(llm_workers=1)
    answers = {1: ["1. 가자.", "\"2. 가자\"", "가자!", "어서 가자."],
               2: ["어서 가자.", "빨리 가자.", "지금 가자.", "또 가자."]}
    stdin: list[str] = []

    def run(cmd, input=None, **kw):
        stdin.append(input)
        _, items = _items_of_stdin(input)
        reply = {"items": [{"id": it["id"], "candidates": answers[len(stdin)]} for it in items]}
        return subprocess.CompletedProcess(cmd, 0, _stream(reply), "")
    monkeypatch.setattr(llm.subprocess, "run", run)
    reqs = [PsRequest(id="A", ko="가자.", n_new=5)]
    first = ps_generate(b, reqs, extra_rounds=1, chunk=2, log=lambda m: None)
    # '"2. 가자"' 는 다듬으면 '2. 가자' 가 된다. 재생이 다시 다듬으면 '가자' 가 되어 '가자.' 와 겹쳐 결과가 달라진다
    assert first.candidates["A"] == ["가자.", "2. 가자", "어서 가자.", "빨리 가자.", "지금 가자."] and len(stdin) == 2
    path = tmp_path / "candidates.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in first.rows), encoding="utf-8")
    again = ps_generate(FileBackend(path), reqs, extra_rounds=1, chunk=2, log=lambda m: None)
    assert again.candidates == first.candidates and again.rows == first.rows and again.calls == 0


# ---------------------------------------------------------------- OpenAI 호환(시험 안의 가짜 서버)

class _Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        self.server.seen.append(body)
        user = body["messages"][1]["content"]
        items = json.JSONDecoder().raw_decode(user, user.index("["))[0]
        answer = {"items": [{"id": it["id"], "candidates": [f"{it['id']} 문장 {k}." for k in range(it["n"])]}
                            for it in items]}
        message = {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}
        data = json.dumps({"model": "fake", "choices": [{"message": message}]}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
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
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def test_openai_compat_path_with_paper_prompts(fake_server, tmp_path) -> None:
    port = fake_server.server_address[1]
    b = OpenAICompatBackend(f"http://127.0.0.1:{port}", "gpt-4o", None, Config(), cache_dir=tmp_path / "cache")
    iso = iso_generate(b, [_iso("L1", n=2)])
    assert iso.candidates == {"L1": ["L1 문장 0.", "L1 문장 1."]}
    body = fake_server.seen[0]
    assert body["model"] == "gpt-4o" and body["messages"][0] == {"role": "system", "content": ISO_SYSTEM}
    assert body["messages"][1]["content"] == build_iso_prompt([_iso("L1", n=2)])[1]
    ps = ps_generate(b, [PsRequest(id="L1", ko="떠나야 해요.", n_new=3)], extra_rounds=0, chunk=2, log=lambda m: None)
    assert ps.candidates == {"L1": ["L1 문장 0.", "L1 문장 1.", "L1 문장 2."]}
    assert fake_server.seen[1]["messages"][0]["content"] == PS_SYSTEM
