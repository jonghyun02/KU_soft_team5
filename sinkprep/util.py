"""공통 도구: 원자적 쓰기, NFC, 해시, 문제 기록, 단계 지문."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from . import paths


def nfc(s: str | None) -> str | None:
    return None if s is None else unicodedata.normalize("NFC", s)


def sha1_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    paths.assert_not_raw(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_bytes(path: Path, data: bytes) -> None:
    _atomic_write_bytes(path, data)


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    _atomic_write_bytes(path, text.encode(encoding))


def write_json(path: Path, obj: Any) -> None:
    _atomic_write_bytes(path, (json.dumps(obj, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    _atomic_write_bytes(path, body.encode("utf-8"))


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[Any]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


# ---- 단계별 문제 기록: cache/problems/{stage}/{doc_id}.json ----

def set_problems(stage: str, key: str, problems: list[dict]) -> None:
    """한 단계가 한 문서(또는 영화)에 대해 찾은 문제 목록을 통째로 기록(빈 목록이면 파일 삭제).

    problem dict 칸: level(doc|film|item), doc_id, film_id, item_id, problem, score, detail, path
    """
    p = paths.PROBLEMS / stage / f"{key}.json"
    if problems:
        write_json(p, problems)
    elif p.exists():
        p.unlink()


def all_problems(with_stage: bool = False) -> list:
    """모든 단계의 문제 목록. with_stage 면 (단계, 문제) 짝 목록."""
    out: list = []
    if not paths.PROBLEMS.exists():
        return out
    for p in sorted(paths.PROBLEMS.glob("*/*.json")):
        rows = read_json(p)
        out.extend([(p.parent.name, x) for x in rows] if with_stage else rows)
    return out


# ---- 단계 지문: 입력이 같으면 건너뛰기 ----

def stage_fp(stage: str, key: str) -> dict | None:
    p = paths.STATE / stage / f"{key}.json"
    return read_json(p) if p.exists() else None


def set_stage_fp(stage: str, key: str, fp: dict) -> None:
    write_json(paths.STATE / stage / f"{key}.json", fp)
