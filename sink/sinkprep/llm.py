"""LLM 백엔드: 이 PC 의 Claude Code 구독 `claude -p` 를 subprocess 로 부른다(설계 notes/llm_parse.md).

재현성
- 모델 ID 를 정확히 고정한다(MODEL, 별칭 금지). 응답이 실제로 그 모델에서 왔는지 modelUsage 로 확인하고, 다른 모델이
  섞였으면(대체 모델 등) 실패로 본다.
- 응답 전문과 메타를 cache/llm/{key}.json 에 원자적으로 저장한다. key = sha1(캐시 형식, 모델 ID, effort, 프롬프트 버전,
  입력 해시). 같은 키가 있으면 부르지 않는다. 그래서 같은 입력으로 다시 돌리면 같은 응답, 같은 결과(바이트 동일)가 나온다.
  다시 묻고 싶으면 그 파일을 지운다.

안전(대본 글은 인터넷에서 받은 신뢰할 수 없는 입력, 프롬프트 주입 위험)
- `--tools ""`: 내장 도구 전부 끔(파일·셸·웹 없음, 구조화 출력 도구만 남음).
- `--strict-mcp-config`(+ --mcp-config 없음): MCP 서버 없음. `--safe-mode`: CLAUDE.md·스킬·플러그인·훅·사용자 명령 끔.
  `--setting-sources ""`: 사용자·프로젝트·로컬 설정 파일을 읽지 않음. `--disable-slash-commands`, `--no-session-persistence`.
- `--system-prompt` 로 기본 시스템 프롬프트를 통째로 바꾼다. 현재 폴더는 빈 작업 폴더 cache/llm_work.
- 부모 Claude Code 세션의 환경 변수(CLAUDECODE, CLAUDE_CODE_* 등)는 넘기지 않는다.

시험에서는 set_backend() 로 가짜 백엔드를 끼운다.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import paths, util

# 모델 ID 는 정확한 ID 로 고정한다(별칭 'sonnet'·'opus' 는 가리키는 모델이 바뀔 수 있어 쓰지 않음).
MODEL_SONNET = "claude-sonnet-5-5"
MODEL_OPUS = "claude-opus-5-5"
MODEL = MODEL_OPUS            # 정답 dev·hard-dev 창 비교로 고름(notes/llm_parse.md '모델 고르기': 오류 1/4, 시간 1.5배)
EFFORT = "medium"             # 생각 깊이도 결과에 영향을 주므로 고정하고 캐시 키에 넣는다
CACHE_SCHEMA = "sink.llmcache/1"
TIMEOUT = 900                 # 호출 하나 최대 초
RETRIES = 2                   # 구조 검증 실패·일시 오류 때 더 부르는 횟수(모두 3번)
BACKOFF = (30, 90, 270)       # 사용량 한도·과부하 때 기다리는 초(차례로). 다 써도 안 되면 멈춤
LONG_LIMIT = 600              # 한도 풀리는 시각이 이보다 멀면 기다리지 않고 멈춤
DEFAULT_WORKERS = 3


def cache_dir() -> Path:
    return paths.CACHE / "llm"


def work_dir() -> Path:
    return paths.CACHE / "llm_work"


# ---------------------------------------------------------------- 오류

class LLMError(Exception):
    """LLM 호출 실패(그 요청만)."""


class LLMUnavailable(LLMError):
    """claude 를 쓸 수 없음(설치·로그인·모델 접근). 남은 호출도 모두 실패하므로 단계를 멈춘다."""


class LLMLimit(LLMUnavailable):
    """사용량 한도. 기다려도 곧 풀리지 않으면 단계를 멈춘다."""


class LLMOffline(LLMError):
    """캐시만 쓰는 실행(sinkprep all 기본)에서 캐시에 없는 요청. 부르지 않는다."""


class BadOutput(LLMError):
    """응답 구조가 틀림(재시도 대상)."""


class Transient(LLMError):
    """일시 오류(과부하·네트워크·시간 초과, 재시도 대상)."""


class RateLimited(Transient):
    """429·529·한도 문구(백오프 뒤 재시도)."""

    def __init__(self, msg: str, reset_at: float | None = None):
        super().__init__(msg)
        self.reset_at = reset_at


# ---------------------------------------------------------------- 요청·응답

@dataclass
class Request:
    """LLM 요청 하나. 키에 들어가는 것: task, system, user, schema, image_ids (+ 모델·effort·prompt_version)."""
    task: str                                   # 'profile' | 'label' …
    prompt_version: str
    system: str
    user: str
    schema: dict
    image_ids: list[str] = field(default_factory=list)          # 쪽 그림 식별자(원본 sha1·쪽·dpi·렌더러 버전)
    render: Callable[[str], bytes] | None = None                # 식별자 -> PNG 바이트(캐시에 없을 때만 부름)
    check: Callable[[Any], str | None] | None = None            # 구조 검증: 틀리면 까닭 문자열
    meta: dict = field(default_factory=dict)                    # 기록용(doc_id, 조각 범위). 키에 들어가지 않음

    def input_sha1(self) -> str:
        blob = json.dumps({"task": self.task, "system": self.system, "user": self.user, "schema": self.schema,
                           "images": list(self.image_ids)}, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def cache_key(req: Request, model: str, effort: str | None = None) -> str:
    effort = EFFORT if effort is None else effort
    s = "\n".join([CACHE_SCHEMA, model, effort, req.prompt_version, req.input_sha1()])
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


@dataclass
class Reply:
    key: str
    response: Any
    cached: bool
    seconds: float
    retries: int
    path: Path


# ---------------------------------------------------------------- 백엔드

def _clean_env() -> dict:
    """부모 Claude Code 세션 표시·소켓·effort 를 넘기지 않는다(설정 위치 CLAUDE_CONFIG_DIR 만 남김)."""
    env = {}
    for k, v in os.environ.items():
        ku = k.upper()
        if ku.startswith(("CLAUDECODE", "CLAUDE_CODE_", "CLAUDE_PID", "CLAUDE_EFFORT")) or ku == "CLAUDE":
            continue
        env[k] = v
    return env


_RX_LIMIT = re.compile(r"usage limit|rate.?limit|hit your limit|limit reached|too many requests|overloaded|"
                       r"try again later|quota", re.I)
_RX_AUTH = re.compile(r"not logged in|please run /login|invalid api key|authenticat|unauthori[sz]ed|"
                      r"oauth token|credential", re.I)


def _reset_at(text: str) -> float | None:
    """'…usage limit reached|1763012345' 같은 꼴에서 풀리는 시각(epoch 초)."""
    m = re.search(r"\|(\d{10})\b", text or "")
    return float(m.group(1)) if m else None


class ClaudeCLI:
    """`claude -p` 백엔드. 호출 하나 = 프로세스 하나(세션 저장 없음)."""

    name = "claude-cli"

    def __init__(self, exe: str | None = None):
        self.exe = exe or shutil.which("claude")

    def unavailable(self) -> str | None:
        return None if self.exe else "claude 실행 파일을 찾을 수 없음(PATH)"

    def command(self, req: Request, model: str, effort: str) -> list[str]:
        return [
            self.exe, "-p",
            "--model", model,
            "--effort", effort,
            "--tools", "",                      # 내장 도구 모두 끔
            "--strict-mcp-config",              # MCP 서버 없음(--mcp-config 를 주지 않음)
            "--setting-sources", "",            # 사용자·프로젝트·로컬 설정 파일 안 읽음
            "--safe-mode",                      # CLAUDE.md·스킬·플러그인·훅·사용자 명령·에이전트 끔
            "--disable-slash-commands",
            "--no-session-persistence",
            "--system-prompt", req.system,
            "--input-format", "stream-json",    # 그림 블록을 넣으려고 stream-json(글만일 때도 같은 길)
            "--output-format", "stream-json", "--verbose",
            "--json-schema", json.dumps(req.schema, ensure_ascii=False, separators=(",", ":")),
        ]

    def __call__(self, req: Request, model: str, effort: str, images: list[bytes], user: str) -> dict:
        if not self.exe:
            raise LLMUnavailable(self.unavailable())
        content: list[dict] = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                            "data": base64.b64encode(b).decode("ascii")}}
                               for b in images]
        content.append({"type": "text", "text": user})
        msg = json.dumps({"type": "user", "message": {"role": "user", "content": content}}, ensure_ascii=False)
        wd = work_dir()
        paths.assert_not_raw(wd)
        wd.mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.run(self.command(req, model, effort), input=msg + "\n", capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", env=_clean_env(), cwd=str(wd),
                                  timeout=TIMEOUT)
        except subprocess.TimeoutExpired as e:
            raise Transient(f"claude 시간 초과({TIMEOUT}초)") from e
        except OSError as e:
            raise LLMUnavailable(f"claude 실행 실패: {e}") from e
        env = None
        init = None
        plain: list[str] = []               # JSON 이 아닌 줄(오류 문구). JSON 줄의 'rate_limit' 같은 낱말로 잘못 분류하지 않게
        limited = None                      # rate_limit_event 중 허용(allowed)이 아닌 것
        for ln in proc.stdout.splitlines():
            ln = ln.strip()
            if not ln.startswith("{"):
                if ln:
                    plain.append(ln)
                continue
            try:
                o = json.loads(ln)
            except json.JSONDecodeError:
                plain.append(ln)
                continue
            if o.get("type") == "result":
                env = o
            elif o.get("type") == "system" and o.get("subtype") == "init":
                init = o
            elif o.get("type") == "rate_limit_event":
                info = o.get("rate_limit_info") or {}
                if info.get("status") == "rejected":           # allowed·allowed_warning 은 한도가 아님
                    limited = info
        if env is None:
            if limited is not None:
                ra = limited.get("resetsAt")
                raise RateLimited(f"claude 한도: {json.dumps(limited, ensure_ascii=False)[:300]}",
                                  float(ra) if isinstance(ra, (int, float)) else None)
            tail = ((proc.stderr or "")[-800:] + "\n" + "\n".join(plain)[-800:]).strip()
            if _RX_AUTH.search(tail):
                raise LLMUnavailable(f"claude 로그인 필요: {tail}")
            if _RX_LIMIT.search(tail):
                raise RateLimited(f"claude 한도: {tail}", _reset_at(tail))
            raise Transient(f"claude 결과 없음(exit {proc.returncode}): {tail}")
        # 안전 확인: 시작 메시지(init)로 도구가 구조화 출력 하나뿐이고 MCP 가 없음을 확인한다
        if init is not None:
            extra = [t for t in init.get("tools") or [] if t != "StructuredOutput"]
            if extra or init.get("mcp_servers"):
                raise LLMUnavailable(f"도구·MCP 가 꺼지지 않음: tools={extra} mcp={init.get('mcp_servers')}")
        if env.get("is_error") or env.get("subtype") != "success":
            text = str(env.get("result") or env.get("subtype") or "")
            st = env.get("api_error_status")
            if st in (401, 403) or _RX_AUTH.search(text):
                raise LLMUnavailable(f"claude 인증 오류 {st}: {text[:300]}")
            if st == 404:
                raise LLMUnavailable(f"모델 접근 불가 {model}: {text[:300]}")
            if st in (429, 529) or _RX_LIMIT.search(text):
                raise RateLimited(f"claude 한도·과부하 {st}: {text[:300]}", _reset_at(text))
            if "structured" in text.lower() or "schema" in text.lower():
                raise BadOutput(f"구조화 출력 실패: {text[:300]}")
            raise Transient(f"claude 오류 {st}: {text[:300]}")
        if init is None or not isinstance(init.get("tools"), list):
            # 성공 응답도 init 의 도구 목록으로 안전을 확인하지 못했으면 쓰지 않는다(fail-closed)
            raise LLMUnavailable("claude 시작 메시지(init)의 도구 목록이 없어 도구가 꺼졌는지 확인할 수 없음")
        used = sorted((env.get("modelUsage") or {}).keys())
        if used != [model]:
            raise BadOutput(f"고정한 모델 {model} 이 아닌 모델이 응답함: {used}")
        return env


class _State:
    lock = threading.Lock()
    sem = threading.BoundedSemaphore(DEFAULT_WORKERS)
    backend: Any = None
    pause_until = 0.0
    stopped: str | None = None          # LLMUnavailable 이 한 번 나면 이 실행 동안 더 부르지 않음
    offline = False                     # True 면 캐시만 쓰고 부르지 않음(LLMOffline)
    calls = 0                           # 실제로 부른 횟수(캐시 적중 제외)
    key_locks: dict[str, threading.Lock] = {}


def set_backend(backend) -> None:
    """가짜 백엔드 끼우기(시험). None 이면 claude CLI."""
    with _State.lock:
        _State.backend = backend
        _State.stopped = None
        _State.pause_until = 0.0


def get_backend():
    if _State.backend is None:
        _State.backend = ClaudeCLI()
    return _State.backend


def set_workers(n: int) -> None:
    """동시에 띄우는 claude 프로세스 수 상한."""
    _State.sem = threading.BoundedSemaphore(max(1, int(n)))


def reset_run(offline: bool = False) -> None:
    """새 실행 시작: 멈춤 표시·호출 수를 지운다. offline 이면 이번 실행은 캐시만 쓴다."""
    with _State.lock:
        _State.stopped = None
        _State.pause_until = 0.0
        _State.calls = 0
        _State.offline = bool(offline)


def calls_made() -> int:
    return _State.calls


def stopped() -> str | None:
    return _State.stopped


def cache_path(key: str) -> Path:
    return cache_dir() / f"{key}.json"


def load_cached(req: Request, model: str | None = None) -> dict | None:
    model = model or MODEL
    key = cache_key(req, model)
    p = cache_path(key)
    if not p.exists():
        return None
    try:
        rec = util.read_json(p)
    except (OSError, ValueError):
        return None
    if rec.get("schema") != CACHE_SCHEMA or rec.get("key") != key:
        return None
    if req.check is not None and req.check(rec.get("response")) is not None:
        return None
    return rec


def _wait_pause() -> None:
    while True:
        with _State.lock:
            left = _State.pause_until - time.time()
            if _State.stopped:
                raise LLMUnavailable(_State.stopped)
        if left <= 0:
            return
        time.sleep(min(left, 5.0))


def _key_lock(key: str) -> threading.Lock:
    with _State.lock:
        return _State.key_locks.setdefault(key, threading.Lock())


def call(req: Request, model: str | None = None) -> Reply:
    """캐시에 있으면 그대로, 없으면 백엔드를 불러 검증한 뒤 캐시에 쓴다.

    실패: BadOutput/Transient(재시도 다 씀) -> LLMError, 쓸 수 없음·한도 -> LLMUnavailable(이후 호출도 바로 실패)."""
    model = model or MODEL
    key = cache_key(req, model)
    p = cache_path(key)
    with _key_lock(key):
        rec = load_cached(req, model)
        if rec is not None:
            return Reply(key, rec["response"], True, float(rec.get("seconds") or 0), int(rec.get("retries") or 0), p)
        if _State.offline:
            raise LLMOffline(f"캐시에 없음({req.task} {key[:12]})")
        backend = get_backend()
        images: list[bytes] | None = None
        errors: list[str] = []
        limit_hits = 0
        attempt = 0
        t0 = time.time()
        while True:
            _wait_pause()
            if images is None:
                images = [req.render(i) for i in req.image_ids] if req.image_ids and req.render else []
            user = req.user
            if errors and not errors[-1].startswith("[한도]"):
                # 구조가 틀렸던 까닭을 알려 주고 다시 묻는다(키는 처음 요청 그대로)
                user = req.user + "\n\n[이전 답이 거부된 까닭: " + errors[-1][:300] + " — 형식을 지켜 다시 답하라]"
            ts = time.time()
            try:
                with _State.sem:
                    _wait_pause()
                    with _State.lock:
                        _State.calls += 1
                    env = backend(req, model, EFFORT, images, user)
                resp = env.get("structured_output")
                if resp is None:
                    raise BadOutput("structured_output 없음")
                why = req.check(resp) if req.check else None
                if why:
                    raise BadOutput(why)
            except LLMUnavailable as e:
                with _State.lock:
                    _State.stopped = f"{type(e).__name__}: {e}"
                raise
            except RateLimited as e:
                limit_hits += 1
                errors.append(f"[한도] {e}")
                now = time.time()
                wait = BACKOFF[min(limit_hits, len(BACKOFF)) - 1]
                if e.reset_at and e.reset_at > now:
                    wait = max(wait, e.reset_at - now + 5)
                if limit_hits > len(BACKOFF) or wait > LONG_LIMIT:
                    msg = f"사용량 한도·과부하가 풀리지 않음({limit_hits}번, 다음 대기 {wait:.0f}초): {e}"
                    with _State.lock:
                        _State.stopped = msg
                    raise LLMLimit(msg) from e
                with _State.lock:
                    _State.pause_until = max(_State.pause_until, now + wait)
                continue
            except (BadOutput, Transient) as e:
                errors.append(f"{type(e).__name__}: {e}")
                attempt += 1
                if attempt > RETRIES:
                    raise LLMError(f"{RETRIES + 1}번 모두 실패: {errors[-1]}") from e
                time.sleep(min(5.0 * attempt, 15.0) if isinstance(e, Transient) else 0)
                continue
            secs = round(time.time() - ts, 2)
            rec = {
                "schema": CACHE_SCHEMA,
                "key": key,
                "model": model,
                "effort": EFFORT,
                "prompt_version": req.prompt_version,
                "task": req.task,
                "input_sha1": req.input_sha1(),
                "image_ids": list(req.image_ids),
                "meta": req.meta,
                "backend": getattr(backend, "name", type(backend).__name__),
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seconds": secs,
                "seconds_total": round(time.time() - t0, 2),
                "retries": attempt,
                "errors": errors,
                "response": resp,
                "envelope": env,
            }
            util.write_json(p, rec)
            return Reply(key, resp, False, secs, attempt, p)
