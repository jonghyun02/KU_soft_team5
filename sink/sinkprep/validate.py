"""검사 도구: schema/*.schema.json(JSON Schema draft 2020-12) + 의미 규칙.

명세 `.omc/specs/deep-interview-sink-format.md` 4장. 수락 기준 코드를 오류 앞에 붙인다.
  [schema] 구조·칸·열거값 (JSON Schema)       [E1] schema 칸 없음·모르는 값
  [E2] UTF-8·NFC·BOM                          [B2] 줄 덮기·n 연속   [ref] 인물·씬 참조
  [C1] asr 낱말 번호·참조   [C2] align↔final 1:1, film.id = 폴더 이름   [D2]~[D9] final 의미 규칙
  [T] 시각 소수 셋째 자리   [Q] quality 집계      [A1][A2] catalog   [catalog]/[review] csv
  [D1] 영상이 있는 영화마다 final.json (인자 없이 돌릴 때)   [B1] 문서마다 lines·script (같음)

final 의미 규칙 가운데 비교가 필요한 것
  [D2] start < end, 낱말이 줄 안, heard = words, 시각이 영상 길이 안(DURATION_TOL), 줄 안 낱말 시작 시각이 줄지 않음
  [D4] changed 는 diffs 있음·same 은 없음, 그리고 글자로 맞춘 줄(match=text)은 same 이면 heard 와 script_ref.text 의
       열쇠(textnorm.key: 공백·문장부호 뺀 글자열)가 같고 changed 면 다름
  [D5] ko·ko_src·lang 이 맞물림, 그리고 시각 있는 대사의 들린 글이 영상 언어의 글자면 lang 은 영상 언어
       (textnorm.spoken_lang: final 이 줄 언어를 정하는 바로 그 함수)
  [D8] 참조가 실제로 있음, 그리고 script.json 의 scene·action·dialogue·onscreen·transition 요소가 빠짐없이 한 번씩 나옴
  [D9] CBoE 가 final.json 만 읽고 쓰는 칸이 믿을 만함: script.json 을 찾았을 때 대본에서 복사한 칸(대사의 speakers·how·
       lang·paren·sung·overlap·scene·script_ref.text, 대사 아닌 항목의 text·scene·kind, 씬의 number·heading·place·time·
       flags, 인물의 name·aliases)이 대본과 같고(lang 은 시각 있는 줄이면 textnorm.spoken_lang 으로 정한 값),
       씬 구간이 그 씬 항목들의 시각과 어긋나지 않음
align 의 [C2]
       같은 pass 의 같은 낱말을 두 op 이 갖지 않음(timeline.shared_words), added op 에 asr 범위가 있음,
       final·asr 을 함께 볼 때 대사 항목의 words(w, s, e, p)가 그 op 의 asr 범위 낱말과 같음(시각은 timeline.op_words 가
       바로잡은 값)

함수는 오류 문자열 목록을 돌려준다(빈 목록 = 통과). CLI: python -I -m sinkprep validate [경로 ...]
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import paths, textnorm, timeline

MAX_ERRORS = 1000      # 파일 하나에서 모으는 오류 수 상한
SHOW = 20              # 파일마다 화면에 찍는 오류 수
EPS = 1e-6
DURATION_TOL = timeline.DURATION_TOL   # final 의 시각이 영상 길이(film.video.duration)를 넘어도 되는 허용 오차(초, 0.5)
SHOW_CLAIMS = 20       # 같은 낱말을 두 op 이 가진 곳을 오류로 적는 수(나머지는 수만)
FINAL_ITEM_TYPES = ("scene", "action", "dialogue", "onscreen", "transition")   # final 에 빠짐없이 나와야 하는 대본 요소 종류

SCHEMA_KINDS = {
    "sink.lines/1": "lines",
    "sink.script/1": "script",
    "sink.asr/1": "asr",
    "sink.align/1": "align",
    "sink.final/1": "final",
}

# ---- catalog.csv (명세 4.3 + DESIGN 의 file_formats) ----
CATALOG_COLUMNS = (
    "doc_id", "post_id", "category", "files", "sha1", "real_format", "file_formats",
    "title_raw", "title_norm", "script_lang", "is_translation", "version_label", "version_raw",
    "dup_group", "film_id", "film_title", "film_title_orig", "year",
    "video_path", "video_sha1", "video_duration", "video_lang",
    "is_primary", "match", "extract_quality", "status", "note",
)
REAL_FORMATS = ("hwp3", "hwp5", "hwpx", "doc", "docx", "rtf", "pdf_text", "pdf_scan", "pdf_broken",
                "txt", "xls", "archive", "other", "mixed")
CATEGORIES = ("ko_film", "foreign_film")
VERSION_LABELS = ("draft", "revised", "bound", "shooting", "final", "translation", "unknown")
STATUSES = ("pending", "parsed", "aligned", "done", "excluded")
FILE_ROOTS = ("filmmakers_scripts/", "extracted/")
MANIFEST_NAME = "manifest.csv"

# ---- qa/needs_review.csv (명세 4.8) ----
NEEDS_REVIEW_COLUMNS = ("level", "doc_id", "film_id", "item_id", "problem", "score", "detail", "path")
LEVELS = ("doc", "film", "item")
PROBLEMS = ("extract_broken", "ocr_low", "parse_low_dialogue", "speaker_missing", "version_mismatch",
            "translation_unmatched", "asr_low_conf", "absent_many", "other")

RE_DOC_ID = re.compile(r"^fm[0-9]+(_[0-9]+)?$")
RE_FILM_ID = re.compile(r"^(kmdb|tmdb|local)_[A-Za-z0-9]+$")
RE_ITEM_ID = re.compile(r"^I[0-9]{4,}$")
RE_SHA1 = re.compile(r"^[0-9a-f]{40}$")
RE_LANG = re.compile(r"^([a-z]{2,3}(-[A-Za-z0-9]{2,8})?|mixed|unknown)$")
RE_GROUP = re.compile(r"^G[0-9]{3,}$")
RE_ASR_REF = re.compile(r"^([a-z][a-z0-9_]*):(0|[1-9][0-9]*)$")
RE_SPACE = re.compile(r"\s+")


class _Errs(list):
    """오류 목록. MAX_ERRORS 를 넘으면 더 담지 않고 수만 센다."""

    def __init__(self) -> None:
        super().__init__()
        self.over = 0

    def add(self, msg: str) -> None:
        if len(self) < MAX_ERRORS:
            self.append(msg)
        else:
            self.over += 1

    def done(self) -> list[str]:
        out = list(self)
        if self.over:
            out.append(f"[limit] 오류가 너무 많아 {self.over}개를 더 세지 않고 생략")
        return out


# =====================================================================
# 스키마
# =====================================================================

@lru_cache(maxsize=None)
def load_schema(kind: str) -> dict:
    with open(paths.SCHEMA / f"{kind}.schema.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def schema_validator(kind: str, defn: str | None = None):
    """kind 스키마의 검사기. defn 을 주면 $defs/defn 부분 검사기(참조는 원래 스키마 기준)."""
    from jsonschema import Draft202012Validator

    schema = load_schema(kind)
    Draft202012Validator.check_schema(schema)
    root = Draft202012Validator(schema)
    return root if defn is None else root.evolve(schema=schema["$defs"][defn])


def _short(s: str, n: int = 200) -> str:
    return s if len(s) <= n else s[:n] + "…"


def _schema_errors(v, inst: Any, errs: _Errs, prefix: str = "", base: tuple = ()) -> int:
    """jsonschema 오류를 errs 에. base 는 inst 가 파일 안에서 있는 자리(부분 검사일 때).

    최상위 schema 칸(E1)은 _check_kind 가 따로 알리므로 뺀다. 같은 글은 한 번만.
    """
    k = 0
    seen: set[str] = set()
    for e in v.iter_errors(inst):
        p = list(e.absolute_path)
        if not base and (p == ["schema"] or (not p and e.validator == "required" and "'schema'" in e.message)):
            continue
        loc = "/".join(str(x) for x in (*base, *p)) or "(최상위)"
        msg = f"[schema] {prefix}{loc}: {_short(e.message)}"
        if msg in seen:
            continue
        seen.add(msg)
        errs.add(msg)
        k += 1
    return k


# =====================================================================
# 읽기 (UTF-8, BOM, NFC, NaN)
# =====================================================================

def _reject_const(name: str):
    raise ValueError(f"{name} 는 JSON 값이 아님")


def _json_loads(s: str) -> Any:
    return json.loads(s, parse_constant=_reject_const)


def _read_text(path: Path, errs: _Errs, bom_ok: bool = False) -> str | None:
    try:
        raw = Path(path).read_bytes()
    except OSError as e:
        errs.add(f"[io] 읽기 실패: {e}")
        return None
    if raw.startswith(b"\xef\xbb\xbf"):
        if not bom_ok:
            errs.add("[E2] 파일 앞에 UTF-8 BOM 이 있음(JSON·JSONL 은 BOM 없이)")
        raw = raw[3:]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as e:
        errs.add(f"[E2] UTF-8 이 아님: {e}")
        return None


def _iter_strings(o: Any, where: str):
    if isinstance(o, str):
        yield where, o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield f"{where}/{k}(키)", k
            yield from _iter_strings(v, f"{where}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _iter_strings(v, f"{where}/{i}")


def _text_is_nfc(text: str, data: Any) -> bool:
    if not unicodedata.is_normalized("NFC", text):
        return False
    if "\\u" in text:   # \uXXXX 이스케이프 속에 숨은 글자까지 본다
        return unicodedata.is_normalized("NFC", json.dumps(data, ensure_ascii=False))
    return True


def _nfc_errors(data: Any, errs: _Errs, prefix: str = "") -> None:
    for where, s in _iter_strings(data, ""):
        if not unicodedata.is_normalized("NFC", s):
            errs.add(f"[E2] {prefix}{where.lstrip('/') or '(최상위)'}: NFC 가 아님 {_short(s, 40)!r}")


def _load_json(path: Path, errs: _Errs) -> Any:
    text = _read_text(path, errs)
    if text is None:
        return None
    try:
        data = _json_loads(text)
    except ValueError as e:
        errs.add(f"[json] JSON 으로 읽을 수 없음: {e}")
        return None
    if not _text_is_nfc(text, data):
        _nfc_errors(data, errs)
    return data


def _read_jsonl(path: Path, errs: _Errs) -> list[tuple[int, Any]] | None:
    """[(파일 줄 번호 1부터, 값)]. 빈 줄은 건너뛴다. U+2028 같은 글자는 줄 나눔으로 보지 않는다."""
    text = _read_text(path, errs)
    if text is None:
        return None
    rows: list[tuple[int, Any]] = []
    for ln, s in enumerate(text.split("\n"), 1):
        s = s.rstrip("\r")
        if not s.strip():
            continue
        try:
            rows.append((ln, _json_loads(s)))
        except ValueError as e:
            errs.add(f"[json] {ln}행: JSON 으로 읽을 수 없음: {e}")
    if not unicodedata.is_normalized("NFC", text) or "\\u" in text:
        for ln, r in rows:
            _nfc_errors(r, errs, prefix=f"{ln}행 ")
    return rows


def _check_kind(data: Any, want: str, errs: _Errs) -> bool:
    """E1: schema 칸이 있고 이 파일 종류의 값인지. 최상위가 객체가 아니면 False."""
    if not isinstance(data, dict):
        errs.add(f"[schema] (최상위): 객체가 아님 ({type(data).__name__})")
        return False
    got = data.get("schema")
    if got is None:
        errs.add("[E1] (최상위): schema 칸이 없음")
    elif got not in SCHEMA_KINDS:
        errs.add(f"[E1] schema: 모르는 값 {got!r}")
    elif SCHEMA_KINDS[got] != want:
        errs.add(f"[E1] schema: {got!r} 는 {want} 형식이 아님")
    return True


def _is_num(v: Any) -> bool:
    return type(v) in (int, float)


def _ms_ok(v: Any) -> bool:
    """시각은 초, 소수 셋째 자리까지."""
    return not _is_num(v) or abs(v * 1000 - round(v * 1000)) < 1e-4


def _in_docs(path: Path) -> bool:
    try:
        return Path(path).resolve().parent.parent == paths.DOCS.resolve()
    except OSError:
        return False


def _in_films(path: Path) -> bool:
    """films/{film_id}/ 바로 아래 파일인가."""
    try:
        return Path(path).resolve().parent.parent == paths.FILMS.resolve()
    except OSError:
        return False


# =====================================================================
# lines.jsonl
# =====================================================================

_LINE_KEYS = frozenset(("n", "file", "page", "x", "y", "text", "bold", "cell", "src", "conf"))
_LINE_SRC = frozenset(("hwp", "doc", "docx", "rtf", "pdf_text", "ocr", "txt", "xls"))
_CELL_KEYS = frozenset(("table", "row", "col"))


def _line_fast_ok(r: Any) -> bool:
    """lines.schema.json $defs/line 의 충분조건(참이면 스키마도 통과). 거짓이면 jsonschema 로 다시 본다.

    줄 행은 문서 700개에 수백만 개라 jsonschema(행마다 약 70µs)만으로는 느리다.
    시험(test_validate.py)에서 스키마와 결과가 같은지 확인한다.
    """
    if type(r) is not dict or r.keys() != _LINE_KEYS:
        return False
    n, f, pg = r["n"], r["file"], r["page"]
    if type(n) is not int or n < 0 or type(f) is not int or f < 0:
        return False
    if pg is not None and (type(pg) is not int or pg < 1):
        return False
    for k in ("x", "y"):
        v = r[k]
        if v is not None and (not _is_num(v) or not 0 <= v <= 1):
            return False
    if type(r["text"]) is not str:
        return False
    b = r["bold"]
    if b is not None and type(b) is not bool:
        return False
    c = r["cell"]
    if c is not None:
        if type(c) is not dict or c.keys() != _CELL_KEYS:
            return False
        if any(type(c[k]) is not int or c[k] < 0 for k in _CELL_KEYS):
            return False
    src, conf = r["src"], r["conf"]
    if type(src) is not str or src not in _LINE_SRC:
        return False
    if conf is not None:
        if src != "ocr" or not _is_num(conf) or not 0 <= conf <= 1:
            return False
    return True


def validate_lines(path) -> list[str]:
    """lines.jsonl: 머리 행 + 줄 행 스키마, n 이 0부터 빠짐없이 차례로, file 이 머리 행 files 범위 안, NFC."""
    path = Path(path)
    errs = _Errs()
    rows = _read_jsonl(path, errs)
    if rows is None:
        return errs.done()
    if not rows:
        errs.add("[E1] 빈 파일: 머리 행({\"schema\":\"sink.lines/1\",…})이 없음")
        return errs.done()
    v_head = schema_validator("lines", "header")
    v_line = schema_validator("lines", "line")

    h_ln, head = rows[0]
    start = 1
    nfiles = None
    if isinstance(head, dict) and "schema" in head:
        if head["schema"] != "sink.lines/1":
            errs.add(f"[E1] {h_ln}행(머리 행) schema: {head['schema']!r} 는 sink.lines/1 이 아님")
        else:
            _schema_errors(v_head, head, errs, prefix=f"{h_ln}행(머리 행) ")
        if isinstance(head.get("files"), list):
            nfiles = len(head["files"])
        if _in_docs(path) and head.get("doc_id") != path.parent.name:
            errs.add(f"[ref] 머리 행 doc_id {head.get('doc_id')!r} 가 폴더 이름 {path.parent.name!r} 와 다름")
    else:
        errs.add(f"[E1] {h_ln}행: 첫 행이 머리 행이 아님(schema 칸 없음)")
        if isinstance(head, dict) and "n" in head:
            start = 0     # 머리 행 없이 줄 행부터 시작한 파일: 줄 행으로 계속 검사

    n_bad = 0
    for idx, (ln, r) in enumerate(rows[start:]):
        if not _line_fast_ok(r):
            _schema_errors(v_line, r, errs, prefix=f"{ln}행 ")
        if not isinstance(r, dict):
            continue
        n = r.get("n")
        if n != idx:
            n_bad += 1
            if n_bad <= 5:
                errs.add(f"[B2] {ln}행: n={n!r}, 기대값 {idx} (n 은 0부터 빠짐없이 차례로)")
        f = r.get("file")
        if nfiles is not None and type(f) is int and f >= nfiles:
            errs.add(f"[ref] {ln}행: file={f} 가 머리 행 files 개수({nfiles}) 밖")
    if n_bad > 5:
        errs.add(f"[B2] n 차례가 어긋난 행 모두 {n_bad}개")
    return errs.done()


def _lines_summary(path: Path) -> tuple[str | None, int]:
    """(머리 행 doc_id, 줄 행 수). 줄 행은 JSON 으로 풀지 않고 센다(n 검사는 validate_lines 몫)."""
    doc_id = None
    count = 0
    first = True
    with open(path, encoding="utf-8-sig", newline="") as f:
        for s in f.read().split("\n"):
            if not s.strip():
                continue
            if first:
                first = False
                try:
                    h = json.loads(s)
                except ValueError:
                    h = None
                if isinstance(h, dict) and "schema" in h:
                    doc_id = h.get("doc_id")
                    continue
            count += 1
    return doc_id, count


# =====================================================================
# script.json
# =====================================================================

def _ids(lst: Any, what: str, errs: _Errs) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not isinstance(lst, list):
        return out
    for i, o in enumerate(lst):
        if not isinstance(o, dict):
            continue
        k = o.get("id")
        if not isinstance(k, str):
            continue
        if k in out:
            errs.add(f"[ref] {what}/{i}: id {k!r} 가 겹침")
        else:
            out[k] = o
    return out


def _fmt_nums(ns: list[int], k: int = 10) -> str:
    s = ", ".join(str(n) for n in ns[:k])
    return s + (f" … (모두 {len(ns)}개)" if len(ns) > k else "")


_RE_EL_F = re.compile(r"E[0-9]{4,}")
_RE_SC_F = re.compile(r"SC[0-9]{3,}")
_RE_P_F = re.compile(r"P[0-9]{2,}")
_RE_LANG_F = re.compile(r"[a-z]{2,3}(-[A-Za-z0-9]{2,8})?|mixed|unknown")
_EL_PLAIN = frozenset(("front", "action", "transition", "unknown"))
_EL_PLAIN_KEYS = frozenset(("id", "type", "scene", "text", "lines"))
_EL_ONSCREEN_KEYS = _EL_PLAIN_KEYS | {"kind"}
_EL_DIALOGUE_KEYS = _EL_PLAIN_KEYS | {"speakers", "speaker_raw", "how", "lang", "raw", "paren", "gloss",
                                      "sung", "overlap"}
_EL_DIALOGUE_KEYS_INF = _EL_DIALOGUE_KEYS | {"speaker_inferred"}       # 선택 칸(파서 0.2.0)
_HOW = frozenset(("on", "off", "narration", "phone", "radio", "broadcast", "video", "inner", "unknown"))
_ONSCREEN_KIND = frozenset(("subtitle", "insert", "title", "other"))
_PAREN_KEYS = frozenset(("text", "at"))


def _opt_str(v: Any) -> bool:
    return v is None or type(v) is str


def _line_refs_ok(v: Any) -> bool:
    return (type(v) is list and len(v) > 0 and all(type(x) is int and x >= 0 for x in v)
            and len(set(v)) == len(v))


def _element_fast_ok(e: Any) -> bool:
    """script.schema.json $defs/element 의 충분조건(참이면 스키마도 통과). 거짓이면 jsonschema 로 다시 본다."""
    if type(e) is not dict:
        return False
    t = e.get("type")
    keys = e.keys()
    if type(t) is not str:
        return False
    if t in _EL_PLAIN:
        if keys != _EL_PLAIN_KEYS:
            return False
    elif t == "scene":
        if keys != _EL_PLAIN_KEYS or type(e["scene"]) is not str:
            return False
    elif t == "onscreen":
        k = e.get("kind")
        if keys != _EL_ONSCREEN_KEYS or type(k) is not str or k not in _ONSCREEN_KIND:
            return False
    elif t == "dialogue":
        if keys != _EL_DIALOGUE_KEYS and not (keys == _EL_DIALOGUE_KEYS_INF and type(e["speaker_inferred"]) is bool):
            return False
        sp = e["speakers"]
        if type(sp) is not list or not all(type(x) is str and _RE_P_F.fullmatch(x) for x in sp) \
                or len(set(sp)) != len(sp):
            return False
        how, lang = e["how"], e["lang"]
        if type(how) is not str or how not in _HOW or type(lang) is not str or not _RE_LANG_F.fullmatch(lang):
            return False
        if not (_opt_str(e["speaker_raw"]) and type(e["raw"]) is str and _opt_str(e["gloss"])
                and type(e["sung"]) is bool):
            return False
        ov = e["overlap"]
        if ov is not None and (type(ov) is not int or ov < 0):
            return False
        pa = e["paren"]
        if type(pa) is not list:
            return False
        for x in pa:
            if type(x) is not dict or x.keys() != _PAREN_KEYS or type(x["text"]) is not str \
                    or type(x["at"]) is not int or x["at"] < 0:
                return False
    else:
        return False
    i, sc = e["id"], e["scene"]
    if type(i) is not str or not _RE_EL_F.fullmatch(i):
        return False
    if sc is not None and (type(sc) is not str or not _RE_SC_F.fullmatch(sc)):
        return False
    return type(e["text"]) is str and _line_refs_ok(e["lines"])


def validate_script(script_path, lines_path=None) -> list[str]:
    """script.json: 스키마(B3 요소 7종 포함), id 유일, 씬·인물 참조, paren 위치, 줄 덮기(B2), NFC.

    lines_path 를 주면 elements.lines 와 dropped.lines 가 lines.jsonl 의 모든 n 을 정확히 한 번씩 덮는지 본다.
    """
    path = Path(script_path)
    errs = _Errs()
    data = _load_json(path, errs)
    if data is None or not _check_kind(data, "script", errs):
        return errs.done()
    els = data.get("elements") if isinstance(data.get("elements"), list) else []
    # 요소는 많아서 하나씩: 빠른 검사로 통과하지 못한 요소만 jsonschema 로(같은 스키마 $defs/element).
    _schema_errors(schema_validator("script"), dict(data, elements=[]) if els else data, errs)
    v_el = schema_validator("script", "element")
    for i, el in enumerate(els):
        if not _element_fast_ok(el):
            _schema_errors(v_el, el, errs, base=("elements", i))

    people = _ids(data.get("people"), "people", errs)
    scenes = _ids(data.get("scenes"), "scenes", errs)
    _ids(els, "elements", errs)

    owner: dict[int, str] = {}
    dup: list[str] = []

    def own(n: Any, who: str) -> None:
        if type(n) is not int:
            return
        if n in owner:
            dup.append(f"{n}({owner[n]}, {who})")
        else:
            owner[n] = who

    for i, el in enumerate(els):
        if not isinstance(el, dict):
            continue
        who = f"elements/{i}({el.get('id')})"
        sc = el.get("scene")
        if isinstance(sc, str) and sc not in scenes:
            errs.add(f"[ref] {who}: scene {sc!r} 가 scenes 에 없음")
        if el.get("type") == "dialogue":
            for sp in el.get("speakers") or []:
                if isinstance(sp, str) and sp not in people:
                    errs.add(f"[ref] {who}: speakers 의 {sp!r} 가 people 에 없음")
            text = el.get("text")
            for j, pa in enumerate(el.get("paren") or []):
                if isinstance(pa, dict) and type(pa.get("at")) is int and isinstance(text, str) \
                        and pa["at"] > len(text):
                    errs.add(f"[ref] {who}/paren/{j}: at={pa['at']} 가 text 길이({len(text)}) 밖")
        for n in el.get("lines") or []:
            own(n, who)
    drops = data.get("dropped") if isinstance(data.get("dropped"), list) else []
    for i, d in enumerate(drops):
        if isinstance(d, dict):
            for n in d.get("lines") or []:
                own(n, f"dropped/{i}")
    for s in dup[:10]:
        errs.add(f"[B2] 줄 {s} 가 두 번 이상 쓰임(줄은 요소 또는 dropped 중 정확히 하나에 속해야 함)")
    if len(dup) > 10:
        errs.add(f"[B2] 두 번 이상 쓰인 줄 모두 {len(dup)}개")

    if _in_docs(path) and data.get("doc_id") != path.parent.name:
        errs.add(f"[ref] doc_id {data.get('doc_id')!r} 가 폴더 이름 {path.parent.name!r} 와 다름")

    if lines_path is not None:
        lp = Path(lines_path)
        try:
            ldoc, nlines = _lines_summary(lp)
        except (OSError, UnicodeDecodeError) as e:
            errs.add(f"[B2] lines.jsonl 을 읽을 수 없음: {e}")
        else:
            if ldoc is not None and ldoc != data.get("doc_id"):
                errs.add(f"[ref] doc_id {data.get('doc_id')!r} 가 lines.jsonl 머리 행 {ldoc!r} 와 다름")
            missing = [n for n in range(nlines) if n not in owner]
            outside = sorted(n for n in owner if n >= nlines)
            if missing:
                errs.add(f"[B2] 요소에도 dropped 에도 없는 줄: {_fmt_nums(missing)}")
            if outside:
                errs.add(f"[B2] lines.jsonl 에 없는 줄 번호(줄 {nlines}개): {_fmt_nums(outside)}")
    return errs.done()


# =====================================================================
# asr.json
# =====================================================================

_ASR_WORD_KEYS = frozenset(("i", "w", "s", "e", "p"))
_ASR_WORD_KEYS_L = _ASR_WORD_KEYS | {"lang"}
_FIN_WORD_KEYS = frozenset(("w", "s", "e", "p"))
_FIN_WORD_KEYS_L = _FIN_WORD_KEYS | {"lang"}


def _word_common_ok(w: dict) -> bool:
    s, e, p = w["s"], w["e"], w["p"]
    if not (_is_num(s) and s >= 0 and _is_num(e) and e >= 0):
        return False
    if p is not None and (not _is_num(p) or not 0 <= p <= 1):
        return False
    if "lang" in w and (type(w["lang"]) is not str or not _RE_LANG_F.fullmatch(w["lang"])):
        return False
    return True


def _asr_word_fast_ok(w: Any) -> bool:
    """asr.schema.json $defs/word 의 충분조건."""
    if type(w) is not dict or (w.keys() != _ASR_WORD_KEYS and w.keys() != _ASR_WORD_KEYS_L):
        return False
    i = w["i"]
    return type(i) is int and i >= 0 and type(w["w"]) is str and _word_common_ok(w)


def _final_word_fast_ok(w: Any) -> bool:
    """final.schema.json $defs/word 의 충분조건."""
    if type(w) is not dict or (w.keys() != _FIN_WORD_KEYS and w.keys() != _FIN_WORD_KEYS_L):
        return False
    return type(w["w"]) is str and len(w["w"]) > 0 and _word_common_ok(w)


_RE_I_F = re.compile(r"I[0-9]{4,}")
_TIMING = frozenset(("aligned", "estimated", "none"))
_FIN_PLAIN_TYPES = frozenset(("scene", "action", "onscreen", "transition"))
_FIN_PLAIN_KEYS = frozenset(("id", "type", "scene", "text", "start", "end", "timing", "src_el"))
_FIN_PLAIN_KEYS_K = _FIN_PLAIN_KEYS | {"kind"}
_FIN_DIA_KEYS = frozenset(("id", "type", "scene", "speakers", "speaker_from", "how", "lang", "heard", "ko",
                           "ko_src", "start", "end", "timing", "words", "status", "match", "script_ref", "diffs",
                           "paren", "sung", "overlap", "conf", "src_el"))
_SPEAKER_FROM = frozenset(("script", "translation_match", "unknown"))
_KO_SRC = frozenset(("same", "mt", "human"))
_STATUS = frozenset(("same", "changed", "added", "absent"))
_MATCH = frozenset(("text", "translation"))
_DIFF_KEYS = frozenset(("kind", "script", "heard"))
_DIFF_KIND = frozenset(("filler", "repeat", "word", "number", "spelling"))
_REF_KEYS = frozenset(("el", "text"))


def _time_ok(v: Any) -> bool:
    return v is None or (_is_num(v) and v >= 0)


def _unit_or_null(v: Any) -> bool:
    return v is None or (_is_num(v) and 0 <= v <= 1)


def _opt_id(v: Any, rx: re.Pattern) -> bool:
    return v is None or (type(v) is str and rx.fullmatch(v) is not None)


def _paren_ok(pa: Any) -> bool:
    return type(pa) is list and all(
        type(x) is dict and x.keys() == _PAREN_KEYS and type(x["text"]) is str
        and type(x["at"]) is int and x["at"] >= 0 for x in pa)


def _item_fast_ok(it: Any) -> bool:
    """final.schema.json $defs/item 의 충분조건(낱말까지 모두 봄). 참이면 스키마도 통과."""
    if type(it) is not dict:
        return False
    t = it.get("type")
    if type(t) is not str:
        return False
    keys = it.keys()
    if t in _FIN_PLAIN_TYPES:
        if t == "onscreen":
            if keys != _FIN_PLAIN_KEYS and keys != _FIN_PLAIN_KEYS_K:
                return False
            if "kind" in it and (type(it["kind"]) is not str or it["kind"] not in _ONSCREEN_KIND):
                return False
        elif keys != _FIN_PLAIN_KEYS:
            return False
        if type(it["text"]) is not str:
            return False
    elif t == "dialogue":
        if keys != _FIN_DIA_KEYS:
            return False
        sp = it["speakers"]
        if type(sp) is not list or not all(type(x) is str and _RE_P_F.fullmatch(x) for x in sp) \
                or len(set(sp)) != len(sp):
            return False
        sf, how, lang, ks = it["speaker_from"], it["how"], it["lang"], it["ko_src"]
        if type(sf) is not str or sf not in _SPEAKER_FROM or type(how) is not str or how not in _HOW:
            return False
        if type(lang) is not str or not _RE_LANG_F.fullmatch(lang):
            return False
        if ks is not None and (type(ks) is not str or ks not in _KO_SRC):
            return False
        heard = it["heard"]
        if not (_opt_str(heard) and _opt_str(it["ko"])):
            return False
        ws = it["words"]
        if type(ws) is not list or not all(_final_word_fast_ok(w) for w in ws):
            return False
        status, match, ref = it["status"], it["match"], it["script_ref"]
        if type(status) is not str or status not in _STATUS:
            return False
        if match is not None and (type(match) is not str or match not in _MATCH):
            return False
        if ref is not None and (type(ref) is not dict or ref.keys() != _REF_KEYS or type(ref["text"]) is not str
                                or type(ref["el"]) is not str or not _RE_EL_F.fullmatch(ref["el"])):
            return False
        diffs = it["diffs"]
        if type(diffs) is not list or not all(
                type(x) is dict and x.keys() == _DIFF_KEYS and type(x["kind"]) is str and x["kind"] in _DIFF_KIND
                and type(x["script"]) is str and type(x["heard"]) is str for x in diffs):
            return False
        if not (_paren_ok(it["paren"]) and type(it["sung"]) is bool and _unit_or_null(it["conf"])):
            return False
        ov = it["overlap"]
        if ov is not None and (type(ov) is not int or ov < 0):
            return False
        # 스키마의 조건 규칙(D3·D4·D6·D5 일부)
        st, en = it["start"], it["end"]
        if status == "absent" and not (st is None and en is None and it["timing"] == "none" and not ws
                                       and heard is None and ref is not None and match is not None):
            return False
        if status == "added" and not (ref is None and match is None and it["src_el"] is None):
            return False
        if status != "absent" and not (st is not None and en is not None and ws
                                       and type(heard) is str and heard):
            return False
        if status in ("same", "changed") and not (ref is not None and match is not None):
            return False
        if (status == "changed") != bool(diffs):
            return False
        if match == "translation" and diffs:
            return False
        if ks == "same" and lang != "ko":
            return False
    else:
        return False
    st, en, tm = it["start"], it["end"], it["timing"]
    if not (_time_ok(st) and _time_ok(en)) or type(tm) is not str or tm not in _TIMING:
        return False
    if tm == "none" and not (st is None and en is None):
        return False
    if tm != "none" and st is None:
        return False
    return (type(it["id"]) is str and _RE_I_F.fullmatch(it["id"]) is not None
            and _opt_id(it["scene"], _RE_SC_F) and _opt_id(it["src_el"], _RE_EL_F))


def _schema_with_words(kind: str, data: dict, errs: _Errs) -> None:
    """낱말이 많은 asr·final 을 빠르게 검사한다(결과는 전체 스키마와 같음).

    asr: 낱말 배열을 첫 낱말만 남긴 사본은 전체 스키마로, 나머지 낱말은 하나씩(빠른 검사 → 실패하면 $defs/word).
    final: 항목 배열을 비운 사본은 전체 스키마로, 항목은 하나씩(빠른 검사 → 실패하면 $defs/item, 낱말은 위와 같이).
    첫 낱말을 남기므로 낱말 배열이 비었는지(minItems·maxItems 조건)는 그대로 검사된다.
    """
    v_word = schema_validator(kind, "word")
    if kind == "asr":
        lst, fast_word = data.get("passes"), _asr_word_fast_ok
    else:
        lst, fast_word = data.get("items"), _final_word_fast_ok
    if not isinstance(lst, list):
        _schema_errors(schema_validator(kind), data, errs)
        return

    def slim(o: Any, base: tuple) -> tuple[Any, list]:
        ws = o.get("words") if isinstance(o, dict) else None
        if isinstance(ws, list) and len(ws) > 1:
            return dict(o, words=ws[:1]), [((*base, "words", j), w) for j, w in enumerate(ws[1:], 1)]
        return o, []

    rest: list[tuple[tuple, Any]] = []
    if kind == "asr":
        passes = []
        for i, o in enumerate(lst):
            o, r = slim(o, ("passes", i))
            passes.append(o)
            rest.extend(r)
        _schema_errors(schema_validator(kind), dict(data, passes=passes), errs)
    else:
        _schema_errors(schema_validator(kind), dict(data, items=[]), errs)
        v_item = schema_validator(kind, "item")
        for i, it in enumerate(lst):
            if _item_fast_ok(it):
                continue
            it, r = slim(it, ("items", i))
            _schema_errors(v_item, it, errs, base=("items", i))
            rest.extend(r)
    for base, w in rest:
        if not fast_word(w):
            _schema_errors(v_word, w, errs, base=base)


_OP_REQ = frozenset(("item", "op", "script_el", "asr", "match", "decided_by"))
_OP_ALL = _OP_REQ | {"text_score", "candidates", "chosen"}
_DECIDED_BY = frozenset(("rule", "acoustic", "llm", "translation"))
_CAND_KEYS = frozenset(("text", "src", "score"))
_RE_ASR_REF_F = re.compile(r"[a-z][a-z0-9_]*:(0|[1-9][0-9]*)")


def _op_fast_ok(o: Any) -> bool:
    """align.schema.json $defs/op 의 충분조건."""
    if type(o) is not dict:
        return False
    keys = o.keys()
    if not (_OP_REQ <= keys and keys <= _OP_ALL):
        return False
    op, sel, rng, match, db = o["op"], o["script_el"], o["asr"], o["match"], o["decided_by"]
    if type(o["item"]) is not str or not _RE_I_F.fullmatch(o["item"]):
        return False
    if type(op) is not str or op not in _STATUS or type(db) is not str or db not in _DECIDED_BY:
        return False
    if not _opt_id(sel, _RE_EL_F):
        return False
    if match is not None and (type(match) is not str or match not in _MATCH):
        return False
    if rng is not None and (type(rng) is not list or len(rng) != 2
                            or not all(type(x) is str and _RE_ASR_REF_F.fullmatch(x) for x in rng)):
        return False
    if "text_score" in o and o["text_score"] is not None and not _is_num(o["text_score"]):
        return False
    if "chosen" in o and o["chosen"] is not None and (type(o["chosen"]) is not int or o["chosen"] < 0):
        return False
    if "candidates" in o:
        c = o["candidates"]
        if type(c) is not list or not all(
                type(x) is dict and x.keys() == _CAND_KEYS and type(x["text"]) is str and type(x["src"]) is str
                and x["src"] and (x["score"] is None or _is_num(x["score"])) for x in c):
            return False
    if op == "added":
        return sel is None and match is None
    if op == "absent":
        return rng is None and sel is not None and match is not None
    return rng is not None and sel is not None and match is not None


def validate_asr(path) -> list[str]:
    """asr.json: 스키마(정렬 결과 칸 금지), pass id 유일, 낱말 i 가 pass 안에서 0..n-1 유일·연속(C1), s≤e, 시각."""
    path = Path(path)
    errs = _Errs()
    data = _load_json(path, errs)
    if data is None or not _check_kind(data, "asr", errs):
        return errs.done()
    _schema_with_words("asr", data, errs)
    passes = data.get("passes") if isinstance(data.get("passes"), list) else []
    seen: set[str] = set()
    for pi, ps in enumerate(passes):
        if not isinstance(ps, dict):
            continue
        pid = ps.get("id")
        if pid in seen:
            errs.add(f"[C1] passes/{pi}: pass id {pid!r} 가 겹침")
        seen.add(pid)
        words = ps.get("words") if isinstance(ps.get("words"), list) else []
        nums = [w.get("i") for w in words if isinstance(w, dict)]
        cnt: dict[Any, int] = {}
        for i in nums:
            cnt[i] = cnt.get(i, 0) + 1
        dups = sorted(i for i, c in cnt.items() if c > 1 and type(i) is int)
        gaps = [i for i in range(len(nums)) if i not in cnt]
        if dups:
            errs.add(f"[C1] passes/{pi}({pid}): 낱말 번호 i 가 겹침: {_fmt_nums(dups)}")
        if gaps:
            errs.add(f"[C1] passes/{pi}({pid}): 낱말 번호 i 가 0..{len(nums) - 1} 에서 빠짐: {_fmt_nums(gaps)}")
        bad_t = 0
        for wi, w in enumerate(words):
            if not isinstance(w, dict):
                continue
            s, e = w.get("s"), w.get("e")
            if _is_num(s) and _is_num(e) and s > e + EPS:
                errs.add(f"[C1] passes/{pi}/words/{wi}: s={s} > e={e}")
            if not (_ms_ok(s) and _ms_ok(e)):
                bad_t += 1
        if bad_t:
            errs.add(f"[T] passes/{pi}({pid}): 소수 셋째 자리를 넘는 낱말 시각 {bad_t}개")
    for i, se in enumerate(data.get("vad") or []):
        if isinstance(se, list) and len(se) == 2 and all(_is_num(x) for x in se) and se[0] > se[1] + EPS:
            errs.add(f"[schema] vad/{i}: 시작 {se[0]} > 끝 {se[1]}")
    for i, ch in enumerate(data.get("chunks") or []):
        if isinstance(ch, dict) and _is_num(ch.get("start")) and _is_num(ch.get("end")) \
                and ch["start"] > ch["end"] + EPS:
            errs.add(f"[schema] chunks/{i}: start {ch['start']} > end {ch['end']}")
    return errs.done()


# =====================================================================
# final.json
# =====================================================================

def _script_elements(script_path: Path | None, errs: _Errs, who: str
                     ) -> tuple[str | None, dict[str, dict] | None, dict | None]:
    """(script doc_id, {요소 id: 요소}, script.json 전체). 읽을 수 없으면 (None, None, None)."""
    if script_path is None:
        return None, None, None
    try:
        with open(script_path, encoding="utf-8") as f:
            s = json.load(f)
        els = {e["id"]: e for e in s.get("elements", []) if isinstance(e, dict) and "id" in e}
        return s.get("doc_id"), els, s
    except (OSError, ValueError, AttributeError, TypeError) as e:
        errs.add(f"[D8] {who}: script.json 을 읽을 수 없음({script_path}): {e}")
        return None, None, None


def _copied_diffs(it: dict, e: dict, video_lang: str | None = None) -> list[str]:
    """final 항목 it 이 대본 요소 e 에서 복사한 칸 가운데 대본과 다른 칸 이름(명세 4.7: CBoE 가 script.json 없이 쓰는 칸).
    시각 있는 대사의 lang 은 대본 lang 을 textnorm.spoken_lang(들린 글, 영상 언어 video_lang)으로 정한 값과 견준다."""
    bad = []
    if it.get("scene") != e.get("scene"):
        bad.append("scene")
    text = str(e.get("text") or "")
    if it.get("type") != "dialogue":
        if it.get("text") != text:
            bad.append("text")
        if it.get("kind") != (e.get("kind") or None if it.get("type") == "onscreen" else None):
            bad.append("kind")
        return bad
    ref = it.get("script_ref")
    if isinstance(ref, dict) and ref.get("text") != text:
        bad.append("script_ref.text")
    speakers = list(e.get("speakers") or [])
    if it.get("speakers") != speakers:
        bad.append("speakers")
    if it.get("speaker_from") == "script" and (not speakers or e.get("speaker_inferred")):
        bad.append("speaker_from")                       # 대본에 적힌 화자가 아닌데(없거나 이어받은 추정) script
    if it.get("how") != (e.get("how") or "unknown"):
        bad.append("how")
    # lang 은 글자로 맞춘 줄만 대본에서 온다(뜻으로 맞춘 줄은 들린 말의 언어). 시각 있는 줄은 글자의 기준이 영상이라
    # 대본 lang 이 영상 언어와 달라도 들린 글이 영상 언어의 글자면 영상 언어(final 과 같은 함수)
    if it.get("match") == "text" and isinstance(e.get("lang"), str):
        want = textnorm.spoken_lang(e["lang"], it.get("heard"), video_lang) if _is_num(it.get("start")) else e["lang"]
        if it.get("lang") != want:
            bad.append("lang")
    if it.get("paren") != [{"text": p.get("text"), "at": p.get("at")} for p in e.get("paren") or [] if isinstance(p, dict)]:
        bad.append("paren")
    if it.get("sung") != bool(e.get("sung")):
        bad.append("sung")
    if it.get("overlap") != e.get("overlap"):
        bad.append("overlap")
    return bad


def validate_final(path, script_path=None) -> list[str]:
    """final.json: 스키마 + D2~D9 + id 유일·차례 + 시각 셋째 자리 + quality 집계.

    script_path 를 주면 script_ref.el·src_el 이 그 script.json 요소에 있는지(D8)와 종류가 맞는지, 그리고
    scene·action·dialogue·onscreen·transition 요소가 빠짐없이 정확히 한 번씩 items 에 나오는지, 대본에서 복사한 칸이
    대본과 같은지(D9) 본다. 씬 구간이 그 씬 항목들의 시각과 어긋나지 않는지(D9)는 script 없이도 본다.
    파일이 films/{film_id}/final.json 자리에 있으면 film.id 가 폴더 이름과 같아야 한다(C2).
    """
    path = Path(path)
    errs = _Errs()
    data = _load_json(path, errs)
    if data is None or not _check_kind(data, "final", errs):
        return errs.done()
    _schema_with_words("final", data, errs)

    people = _ids(data.get("people"), "people", errs)
    scenes = _ids(data.get("scenes"), "scenes", errs)
    items = data.get("items") if isinstance(data.get("items"), list) else []
    _ids(items, "items", errs)

    film = data.get("film") if isinstance(data.get("film"), dict) else {}
    fscript = film.get("script") if isinstance(film.get("script"), dict) else {}
    s_doc, s_els, s_all = _script_elements(Path(script_path) if script_path else None, errs, "final")
    if s_doc is not None and fscript.get("doc_id") != s_doc:
        errs.add(f"[D8] film.script.doc_id {fscript.get('doc_id')!r} 가 script.json doc_id {s_doc!r} 와 다름")
    if path.name == "final.json" and _in_films(path) and film.get("id") != path.parent.name:
        errs.add(f"[C2] film.id {film.get('id')!r} 가 폴더 이름 {path.parent.name!r} 와 다름")

    # D2: 모든 시각은 영상 길이 안(허용 오차 DURATION_TOL)
    video = film.get("video") if isinstance(film.get("video"), dict) else {}
    dur = video.get("duration") if _is_num(video.get("duration")) and video["duration"] > 0 else None
    video_lang = video.get("lang") if isinstance(video.get("lang"), str) else None
    from_heard = 0
    late: list[str] = []

    def too_late(*ts) -> bool:
        return dur is not None and any(_is_num(t) and t > dur + DURATION_TOL for t in ts)

    for i, sc in enumerate(data.get("scenes") or []):
        if isinstance(sc, dict) and _is_num(sc.get("start")) and _is_num(sc.get("end")) \
                and sc["start"] > sc["end"] + EPS:
            errs.add(f"[D7] scenes/{i}({sc.get('id')}): start {sc['start']} > end {sc['end']}")
        if isinstance(sc, dict) and too_late(sc.get("start"), sc.get("end")):
            late.append(f"scenes/{i}({sc.get('id')})")

    prev_id = -1
    prev_t: tuple[float, str] | None = None
    counts = {"dialogue": 0, "same": 0, "changed": 0, "added": 0, "absent": 0}
    bad_t = 0
    src_count: dict[str, int] = {}
    scene_t: dict[str, list[float]] = {}          # 씬마다 그 씬 항목들의 [가장 이른 시각, 가장 늦은 시각]
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        iid = it.get("id")
        w = f"items/{idx}({iid})"
        if isinstance(it.get("scene"), str) and _is_num(it.get("start")):
            ts = [t for t in (it.get("start"), it.get("end")) if _is_num(t)]
            span = scene_t.setdefault(it["scene"], [min(ts), max(ts)])
            span[0], span[1] = min(span[0], min(ts)), max(span[1], max(ts))
        if isinstance(iid, str) and RE_ITEM_ID.match(iid):
            num = int(iid[1:])
            if num <= prev_id:
                errs.add(f"[D7] {w}: 항목 id 가 items 차례대로 커지지 않음(앞 항목 I{prev_id:04d})")
            prev_id = num
        typ = it.get("type")
        st, en = it.get("start"), it.get("end")
        if not (_ms_ok(st) and _ms_ok(en)):
            bad_t += 1
        if too_late(st, en):
            late.append(w)
        # D7: 시각 있는 항목끼리 start 비감소. 시각 없는 항목은 어디든 끼어 있어도 됨.
        if _is_num(st):
            if prev_t is not None and st < prev_t[0] - EPS:
                errs.add(f"[D7] {w}: start {st} 가 앞의 시각 있는 항목 {prev_t[1]} 의 start {prev_t[0]} 보다 앞")
            prev_t = (st, str(iid))
            if _is_num(en) and en < st - EPS:
                errs.add(f"[D7] {w}: end {en} < start {st}")
        # D8 참조
        sc = it.get("scene")
        if isinstance(sc, str) and sc not in scenes:
            errs.add(f"[D8] {w}: scene {sc!r} 가 scenes 에 없음")
        src_el = it.get("src_el")
        if isinstance(src_el, str):
            src_count[src_el] = src_count.get(src_el, 0) + 1
        if s_els is not None and isinstance(src_el, str):
            if src_el not in s_els:
                errs.add(f"[D8] {w}: src_el {src_el!r} 가 script.json 요소에 없음")
            elif typ is not None and s_els[src_el].get("type") != typ:
                errs.add(f"[D8] {w}: 항목 종류 {typ!r} 와 src_el {src_el} 의 종류 {s_els[src_el].get('type')!r} 가 다름")
            else:                                  # D9: 대본에서 복사한 칸(글은 적지 않고 칸 이름만 알린다)
                bad = _copied_diffs(it, s_els[src_el], video_lang)
                if bad:
                    errs.add(f"[D9] {w}: 대본 요소 {src_el} 에서 복사한 칸이 대본과 다름: {', '.join(bad)}")
                if "lang" not in bad and typ == "dialogue" and it.get("match") == "text" and _is_num(it.get("start")) \
                        and isinstance(s_els[src_el].get("lang"), str) and it.get("lang") != s_els[src_el]["lang"]:
                    from_heard += 1                # 대본 lang 대신 들린 글자로 정한 줄(quality.lang_from_heard)
        if typ != "dialogue":
            continue

        counts["dialogue"] += 1
        status = it.get("status")
        if status in counts:
            counts[status] += 1
        words = it.get("words") if isinstance(it.get("words"), list) else []
        diffs = it.get("diffs") if isinstance(it.get("diffs"), list) else []
        ref = it.get("script_ref")
        heard, ko, ko_src, lang = it.get("heard"), it.get("ko"), it.get("ko_src"), it.get("lang")

        for sp in it.get("speakers") or []:
            if isinstance(sp, str) and sp not in people:
                errs.add(f"[D8] {w}: speakers 의 {sp!r} 가 people 에 없음")
        if isinstance(ref, dict):
            el = ref.get("el")
            if s_els is not None and isinstance(el, str) and el not in s_els:
                errs.add(f"[D8] {w}: script_ref.el {el!r} 가 script.json 요소에 없음")
            if src_el != el:
                errs.add(f"[D8] {w}: src_el {src_el!r} 와 script_ref.el {el!r} 가 다름")

        # D2
        if status in ("same", "changed", "added"):
            if not (_is_num(st) and _is_num(en)):
                errs.add(f"[D2] {w}: status={status} 인데 start/end 가 없음")
            elif not st < en:
                errs.add(f"[D2] {w}: start {st} < end {en} 이 아님")
            if not words:
                errs.add(f"[D2] {w}: status={status} 인데 words 가 비어 있음")
            out = back = 0
            prev_s = None
            late_word = False
            for wd in words:
                if not isinstance(wd, dict):
                    continue
                s, e = wd.get("s"), wd.get("e")
                if not (_ms_ok(s) and _ms_ok(e)):
                    bad_t += 1
                if not (_is_num(s) and _is_num(e)):
                    continue
                if s > e + EPS or (_is_num(st) and s < st - EPS) or (_is_num(en) and e > en + EPS):
                    out += 1
                if prev_s is not None and s < prev_s - EPS:
                    back += 1
                prev_s = s
                late_word = late_word or too_late(s, e)
            if out:
                errs.add(f"[D2] {w}: 낱말 {out}개의 시각이 [start {st}, end {en}] 밖이거나 s>e")
            if back:
                errs.add(f"[D2] {w}: 낱말 시작 시각이 앞 낱말보다 앞선 곳 {back}군데(낱말은 시각 차례여야 함)")
            if late_word and not too_late(st, en):
                late.append(w)
            if isinstance(heard, str) and words:
                joined = "".join(wd.get("w", "") for wd in words if isinstance(wd, dict) and isinstance(wd.get("w"), str))
                if RE_SPACE.sub("", joined) != RE_SPACE.sub("", heard):
                    errs.add(f"[D2] {w}: heard 가 words 의 w 를 이은 글과 다름 ({_short(heard, 40)!r} / {_short(joined, 40)!r})")
        # D3
        if status == "absent":
            if st is not None or en is not None:
                errs.add(f"[D3] {w}: status=absent 인데 start/end 가 null 이 아님")
            if not isinstance(ref, dict):
                errs.add(f"[D3] {w}: status=absent 인데 script_ref 가 없음")
        if status == "added" and ref is not None:
            errs.add(f"[D3] {w}: status=added 인데 script_ref 가 null 이 아님")
        if status in ("same", "changed") and not isinstance(ref, dict):
            errs.add(f"[D3] {w}: status={status} 인데 script_ref 가 없음")
        # D4
        if status == "changed" and not diffs:
            errs.add(f"[D4] {w}: status=changed 인데 diffs 가 비어 있음")
        if status in ("same", "added", "absent") and diffs:
            errs.add(f"[D4] {w}: status={status} 인데 diffs 가 {len(diffs)}개 있음")
        # D4: 글자로 맞춘 줄은 same 이면 열쇠(공백·문장부호 뺀 글자열)가 같고 changed 면 다르다
        if status in ("same", "changed") and it.get("match") == "text" and isinstance(heard, str) \
                and isinstance(ref, dict) and isinstance(ref.get("text"), str):
            alike = textnorm.key(heard) == textnorm.key(ref["text"])
            if status == "same" and not alike:
                errs.add(f"[D4] {w}: status=same 인데 heard 와 script_ref.text 가 띄어쓰기·문장부호를 빼도 다름 "
                         f"({_short(heard, 40)!r} / {_short(ref['text'], 40)!r})")
            elif status == "changed" and alike:
                errs.add(f"[D4] {w}: status=changed 인데 heard 와 script_ref.text 가 띄어쓰기·문장부호만 다름")
        # D5
        if _is_num(st) and not (isinstance(ko, str) and ko.strip()):
            errs.add(f"[D5] {w}: 시각 있는 대사인데 ko 가 없음")
        foreign = [wd.get("lang") for wd in words if isinstance(wd, dict)
                   and wd.get("lang") not in (None, "ko")]
        if foreign:
            if ko_src == "same":
                errs.add(f"[D5] {w}: 외국어 낱말({foreign[0]})이 있는데 ko_src=same (mt 또는 human 이어야 함)")
        elif lang == "ko" and (_is_num(st) or heard is not None or ko is not None):
            if ko_src != "same":
                errs.add(f"[D5] {w}: lang=ko 인데 ko_src={ko_src!r} (same 이어야 함)")
            if ko != heard:
                errs.add(f"[D5] {w}: lang=ko 인데 ko 와 heard 가 다름")
        if ko_src == "same" and lang != "ko":
            errs.add(f"[D5] {w}: ko_src=same 인데 lang={lang!r}")
        # 글자의 기준은 영상: 시각 있는 줄의 들린 글이 영상 언어의 글자면 영상 언어로 말한 줄이다(final 과 같은 함수)
        if _is_num(st) and isinstance(lang, str) and textnorm.spoken_lang(lang, heard, video_lang) != lang:
            errs.add(f"[D5] {w}: 들린 글이 영상 언어 {video_lang} 의 글자인데 lang={lang!r} (영상 언어여야 함)")
        # D6
        if it.get("match") == "translation" and diffs:
            errs.add(f"[D6] {w}: match=translation 인데 diffs 가 {len(diffs)}개 있음")
    if bad_t:
        errs.add(f"[T] 소수 셋째 자리를 넘는 시각 {bad_t}개")
    if late:
        errs.add(f"[D2] 영상 길이 {dur}초를 {DURATION_TOL}초 넘게 지난 시각이 {len(late)}곳: {', '.join(late[:5])}"
                 f"{' …' if len(late) > 5 else ''}")
    _check_untimed_order(items, errs)

    # D8: 대본 요소는 빠짐없이 정확히 한 번씩
    def el_order(k: str) -> tuple[int, str]:
        return len(k), k

    twice = sorted((k for k, c in src_count.items() if c > 1), key=el_order)
    if twice:
        errs.add(f"[D8] 같은 대본 요소를 가리키는 항목이 둘 이상인 요소 {len(twice)}개: {', '.join(twice[:10])}")
    if s_els is not None:
        miss = sorted((k for k, e in s_els.items() if e.get("type") in FINAL_ITEM_TYPES and k not in src_count),
                      key=el_order)
        if miss:
            errs.add(f"[D8] items 에 없는 대본 요소 {len(miss)}개(scene·action·dialogue·onscreen·transition 요소는 "
                     f"빠짐없이 한 번씩): {', '.join(miss[:10])}")

    # D9: 씬 구간은 그 씬 항목들의 시각과 어긋나지 않는다(구간의 start 가 그 씬 항목들의 시각 범위 안. 시각 있는 항목이
    # 있는 씬은 구간이 있고, 없는 씬은 구간이 없다). end 는 start 이상이기만 하면 된다(다음 씬 시작까지로 잡을 수도 있다).
    for i, sc in enumerate(data.get("scenes") or []):
        if not isinstance(sc, dict) or not isinstance(sc.get("id"), str):
            continue
        span = scene_t.get(sc["id"])
        st = sc.get("start")
        if span is None and _is_num(st):
            errs.add(f"[D9] scenes/{i}({sc['id']}): 시각 있는 항목이 없는 씬인데 구간이 있음(start {st})")
        elif span is not None and not _is_num(st):
            errs.add(f"[D9] scenes/{i}({sc['id']}): 시각 있는 항목이 있는 씬인데 구간이 없음")
        elif span is not None and not (span[0] - EPS <= st <= span[1] + EPS):
            errs.add(f"[D9] scenes/{i}({sc['id']}): 구간 start {st} 가 그 씬 항목들의 시각 범위 "
                     f"[{span[0]}, {span[1]}] 밖")
    # D9: 대본에서 복사한 씬·인물 칸
    if s_all is not None:
        def copied(k: str, e: dict):
            if k in ("flags", "aliases"):
                return list(e.get(k) or [])
            return (e.get(k) or "") if k == "heading" else e.get(k)

        for what, keys in (("scenes", ("number", "heading", "place", "time", "flags")), ("people", ("name", "aliases"))):
            src = {x["id"]: x for x in s_all.get(what) or [] if isinstance(x, dict) and "id" in x}
            for i, x in enumerate(data.get(what) or []):
                if not isinstance(x, dict):
                    continue
                e = src.get(x.get("id"))
                if e is None:
                    errs.add(f"[D9] {what}/{i}({x.get('id')}): script.json 의 {what} 에 없음")
                    continue
                bad = [k for k in keys if x.get(k) != copied(k, e)]
                if bad:
                    errs.add(f"[D9] {what}/{i}({x.get('id')}): 대본에서 복사한 칸이 대본과 다름: {', '.join(bad)}")

    q = data.get("quality")
    if isinstance(q, dict):
        for k, v in counts.items():
            if type(q.get(k)) is int and q[k] != v:
                errs.add(f"[Q] quality.{k}={q[k]} 인데 items 로 센 값은 {v}")
        if s_els is not None and type(q.get("lang_from_heard")) is int and q["lang_from_heard"] != from_heard:
            errs.add(f"[Q] quality.lang_from_heard={q['lang_from_heard']} 인데 대본 lang 과 다른 시각 있는 대사는 {from_heard}줄")
    return errs.done()


def _src_pos(it) -> int | None:
    s = it.get("src_el") if isinstance(it, dict) else None
    return int(s[1:]) if isinstance(s, str) and re.fullmatch(r"E\d+", s) else None


def _check_untimed_order(items: list, errs) -> None:
    """D7 뒤 절반: 시각 없는 항목은 대본 순서로 앞뒤 항목 사이에 놓인다. 대본 요소(src_el)가 있는 가장 가까운
    앞·뒤 항목이 서로 대본 순서일 때, 시각 없는 항목의 대본 위치가 그 둘 사이인지 본다(앞뒤가 서로 뒤바뀐
    곳은 배우가 대본 순서와 다르게 말한 곳이라 판단하지 않음)."""
    pos = [_src_pos(it) for it in items]
    for i, it in enumerate(items):
        if not isinstance(it, dict) or _is_num(it.get("start")) or pos[i] is None:
            continue
        prv = next((pos[j] for j in range(i - 1, -1, -1) if pos[j] is not None), None)
        nxt = next((pos[j] for j in range(i + 1, len(items)) if pos[j] is not None), None)
        if prv is not None and nxt is not None and prv < nxt and not (prv < pos[i] < nxt):
            errs.add(f"[D7] items/{i}({it.get('id')}): 시각 없는 항목의 대본 위치 {it.get('src_el')} 가 "
                     f"앞뒤 항목(E{prv:04d}, E{nxt:04d}) 사이가 아님")


# =====================================================================
# align.json
# =====================================================================

def _asr_index(asr_path: Path, errs: _Errs) -> tuple[str | None, dict[str, int] | None, str | None]:
    """(asr 파일 sha1, {pass id: 낱말 수}, video_sha1)."""
    import hashlib

    try:
        raw = Path(asr_path).read_bytes()
        a = json.loads(raw.decode("utf-8"))
        idx = {p["id"]: len(p.get("words", [])) for p in a.get("passes", []) if isinstance(p, dict) and "id" in p}
        return hashlib.sha1(raw).hexdigest(), idx, a.get("video_sha1")
    except (OSError, ValueError, AttributeError, TypeError) as e:
        errs.add(f"[C1] asr.json 을 읽을 수 없음({asr_path}): {e}")
        return None, None, None


def _asr_words(asr_path: Path) -> dict[str, list[dict]] | None:
    """{pass id: 낱말 목록(번호 i 순)} (timeline.asr_index). 읽을 수 없거나 꼴이 깨졌으면 None(다른 검사가 알린다)."""
    try:
        with open(asr_path, encoding="utf-8") as f:
            index = timeline.asr_index(json.load(f))
        for ws in index.values():
            for k, w in enumerate(ws):
                if w["i"] != k or not (_is_num(w["s"]) and _is_num(w["e"])):
                    return None
        return index
    except (OSError, ValueError, AttributeError, TypeError, KeyError):
        return None


def _word_key(w: Any) -> tuple:
    return (w.get("w"), w.get("s"), w.get("e"), w.get("p")) if isinstance(w, dict) else (None,) * 4


def validate_align(path, final_path=None, asr_path=None, script_path=None) -> list[str]:
    """align.json: 스키마 + op 1개 = final 대사 1줄(C2: item 1:1, op==status, script_el==script_ref.el, match 같음)
    + asr 낱말 참조가 asr.json 에 있음(C1) + asr_sha1 이 asr.json 파일 sha1 + script_el 이 script.json 에 있음.

    같은 pass 의 같은 낱말을 두 op 이 갖지 않는지도 본다(C2, timeline.shared_words: 한 번 난 소리는 한 줄에만).
    서로 다른 pass 에서 온 줄끼리 시간이 겹친 곳은 시각만으로 잘못이라 단정할 수 없어 오류로 보지 않는다(final 이
    quality.pass_overlaps 와 항목 수준 문제로 남긴다). final.json 과 asr.json 을 함께 주면 대사 항목의 words(w, s, e, p)가
    그 op 의 asr 범위 낱말과 같은지 본다(C2). 견주는 값은 timeline.op_words 가 영상 길이(film.video.duration)를 받아
    시각을 바로잡은 낱말이다: final 이 낱말을 만드는 바로 그 함수라서, item 이나 asr 범위가 어긋난 사본과 받아쓰기가
    바뀐 뒤의 낡은 final.json 이 걸린다."""
    path = Path(path)
    errs = _Errs()
    data = _load_json(path, errs)
    if data is None or not _check_kind(data, "align", errs):
        return errs.done()
    ops = data.get("ops") if isinstance(data.get("ops"), list) else []
    _schema_errors(schema_validator("align"), dict(data, ops=[]) if ops else data, errs)
    v_op = schema_validator("align", "op")
    for i, op in enumerate(ops):
        if not _op_fast_ok(op):
            _schema_errors(v_op, op, errs, base=("ops", i))

    a_sha, a_idx, a_video = _asr_index(Path(asr_path), errs) if asr_path else (None, None, None)
    a_words = _asr_words(Path(asr_path)) if a_idx is not None else None
    if a_sha is not None and data.get("asr_sha1") != a_sha:
        errs.add(f"[C1] asr_sha1 {data.get('asr_sha1')!r} 가 asr.json 파일 sha1 {a_sha} 와 다름")
    s_doc, s_els, _ = _script_elements(Path(script_path) if script_path else None, errs, "align")
    if s_doc is not None and data.get("script_doc") != s_doc:
        errs.add(f"[D8] script_doc {data.get('script_doc')!r} 가 script.json doc_id {s_doc!r} 와 다름")

    fitems: dict[str, dict] | None = None
    fscenes: set[str] | None = None
    f_dur: float | None = None
    if final_path:
        try:
            with open(final_path, encoding="utf-8") as f:
                fin = json.load(f)
            fitems = {it["id"]: it for it in fin.get("items", []) if isinstance(it, dict) and "id" in it}
            fscenes = {s["id"] for s in fin.get("scenes", []) if isinstance(s, dict) and "id" in s}
            fdoc = (fin.get("film") or {}).get("script", {}).get("doc_id")
            if fdoc is not None and data.get("script_doc") != fdoc:
                errs.add(f"[C2] script_doc {data.get('script_doc')!r} 가 final film.script.doc_id {fdoc!r} 와 다름")
            fvideo = (fin.get("film") or {}).get("video") or {}
            fvid = fvideo.get("sha1")
            if a_video is not None and fvid is not None and a_video != fvid:
                errs.add(f"[C2] asr.json video_sha1 {a_video} 가 final film.video.sha1 {fvid} 와 다름(다른 영상)")
            if _is_num(fvideo.get("duration")) and fvideo["duration"] > 0:
                f_dur = float(fvideo["duration"])
        except (OSError, ValueError, AttributeError, TypeError, KeyError) as e:
            errs.add(f"[C2] final.json 을 읽을 수 없음({final_path}): {e}")
            fitems = None

    seen: dict[str, int] = {}
    sound: list[dict] = []                         # 낱말 참조가 멀쩡한 op (한 소리를 두 op 이 가졌는지 볼 대상)
    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            continue
        item = op.get("item")
        w = f"ops/{i}({item})"
        if item in seen:
            errs.add(f"[C2] {w}: item 이 ops/{seen[item]} 와 겹침")
        elif isinstance(item, str):
            seen[item] = i
        cands = op.get("candidates")
        ch = op.get("chosen")
        if type(ch) is int and (not isinstance(cands, list) or ch >= len(cands)):
            errs.add(f"[schema] {w}: chosen={ch} 가 candidates 범위 밖")
        rng = op.get("asr")
        rng_ok = False
        if rng is None and op.get("op") == "added":
            errs.add(f"[C2] {w}: op=added 인데 asr 낱말 범위가 없음(대본에 없는 말은 들린 낱말이 있어야 함)")
        if isinstance(rng, list) and len(rng) == 2 and all(isinstance(x, str) for x in rng):
            m = [RE_ASR_REF.match(x) for x in rng]
            if all(m):
                (p0, i0), (p1, i1) = [(x.group(1), int(x.group(2))) for x in m]
                rng_ok = p0 == p1 and i0 <= i1
                if p0 != p1:
                    errs.add(f"[C1] {w}: asr 범위의 처음·끝 pass 가 다름 ({rng[0]}, {rng[1]})")
                elif i0 > i1:
                    errs.add(f"[C1] {w}: asr 범위 처음 {i0} > 끝 {i1}")
                if a_idx is not None:
                    for pid, k, ref in ((p0, i0, rng[0]), (p1, i1, rng[1])):
                        if pid not in a_idx:
                            errs.add(f"[C1] {w}: asr {ref!r} 의 pass {pid!r} 가 asr.json 에 없음")
                            rng_ok = False
                        elif k >= a_idx[pid]:
                            errs.add(f"[C1] {w}: asr {ref!r} 의 낱말 번호가 asr.json 범위(0..{a_idx[pid] - 1}) 밖")
                            rng_ok = False
        if rng_ok:
            sound.append(op)
        if rng_ok and a_words is not None:
            it = fitems.get(item) if fitems is not None and isinstance(item, str) else None
            if isinstance(it, dict) and it.get("type") == "dialogue":
                # C2: final 의 낱말은 이 op 의 asr 범위 낱말 그대로다(시각만 timeline 이 바로잡은 값)
                want = [_word_key(x) for x in timeline.op_words(a_words, op, f_dur)]
                got = [_word_key(x) for x in it.get("words") or []] if isinstance(it.get("words"), list) else []
                if got != want:
                    at = next((k for k, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
                    errs.add(f"[C2] {w}: final 항목의 words 가 asr 범위 {rng[0]}~{rng[1]} 의 낱말과 다름"
                             f"(final {len(got)}개, asr {len(want)}개, 처음 다른 곳 {at}번째)")
        sel = op.get("script_el")
        if s_els is not None and isinstance(sel, str) and sel not in s_els:
            errs.add(f"[D8] {w}: script_el {sel!r} 가 script.json 요소에 없음")
        if fitems is not None and isinstance(item, str):
            it = fitems.get(item)
            if it is None:
                errs.add(f"[C2] {w}: final.json items 에 {item!r} 가 없음")
            elif it.get("type") != "dialogue":
                errs.add(f"[C2] {w}: final 항목 {item} 이 대사가 아님(type={it.get('type')!r})")
            else:
                if op.get("op") != it.get("status"):
                    errs.add(f"[C2] {w}: op={op.get('op')!r} 가 final status={it.get('status')!r} 와 다름")
                ref = it.get("script_ref")
                fel = ref.get("el") if isinstance(ref, dict) else None
                if sel != fel:
                    errs.add(f"[C2] {w}: script_el {sel!r} 가 final script_ref.el {fel!r} 와 다름")
                if op.get("match") != it.get("match"):
                    errs.add(f"[C2] {w}: match {op.get('match')!r} 가 final match {it.get('match')!r} 와 다름")
    claims = timeline.shared_words(sound)
    for c in claims[:SHOW_CLAIMS]:
        errs.add(f"[C2] {c}")
    if len(claims) > SHOW_CLAIMS:
        errs.add(f"[C2] 같은 낱말을 두 op 이 가진 곳이 {len(claims) - SHOW_CLAIMS}곳 더 있음")
    if fitems is not None:
        miss = [k for k, it in fitems.items() if it.get("type") == "dialogue" and k not in seen]
        if miss:
            errs.add(f"[C2] ops 에 없는 final 대사 항목 {len(miss)}개: {', '.join(miss[:10])}")
    if fscenes is not None:
        for i, sc in enumerate(data.get("scenes") or []):
            if isinstance(sc, dict) and isinstance(sc.get("scene"), str) and sc["scene"] not in fscenes:
                errs.add(f"[C2] scenes/{i}: scene {sc['scene']!r} 가 final scenes 에 없음")
    return errs.done()


# =====================================================================
# catalog.csv / needs_review.csv
# =====================================================================

def _read_csv(path: Path, errs: _Errs, tag: str, need_bom: bool) -> tuple[list[str], list[tuple[int, list[str]]]] | None:
    try:
        raw = Path(path).read_bytes()
    except OSError as e:
        errs.add(f"[{tag}] 읽기 실패: {e}")
        return None
    if need_bom and not raw.startswith(b"\xef\xbb\xbf"):
        errs.add(f"[{tag}] 파일 앞에 UTF-8 BOM 이 없음(명세 4.3)")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        errs.add(f"[E2] UTF-8 이 아님: {e}")
        return None
    if not unicodedata.is_normalized("NFC", text):
        errs.add("[E2] NFC 가 아닌 글자가 있음")
    rd = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(rd)
    except StopIteration:
        errs.add(f"[{tag}] 빈 파일(헤더 없음)")
        return None
    except csv.Error as e:
        errs.add(f"[{tag}] CSV 를 읽을 수 없음: {e}")
        return None
    rows: list[tuple[int, list[str]]] = []
    try:
        for r in rd:
            if r:
                rows.append((rd.line_num, r))
    except csv.Error as e:
        errs.add(f"[{tag}] CSV 를 읽을 수 없음({rd.line_num}행): {e}")
    return header, rows


def _split(v: str) -> list[str]:
    return v.split("|") if v else []


def _float_in(v: str, lo: float, hi: float | None) -> bool:
    try:
        x = float(v)
    except ValueError:
        return False
    return x == x and lo <= x and (hi is None or x <= hi)


def _raw_target_files() -> list[str]:
    """A1 대상: 원본 폴더 아래 모든 파일(manifest.csv 제외) + extracted/ 아래 모든 파일. ROOT 기준 상대 경로."""
    out: list[str] = []
    for base in (paths.RAW, paths.EXTRACTED):
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if base == paths.RAW and p.parent == paths.RAW and p.name == MANIFEST_NAME:
                continue
            out.append(p.relative_to(paths.ROOT).as_posix())
    return sorted(out)


def validate_catalog(check_files: bool = True, path=None, check_sha1: bool = False) -> list[str]:
    """catalog.csv: 헤더, 열거값, 칸 모양, files·sha1(·file_formats) 개수, A2.

    check_files: 목록의 파일이 있는지 + A1(원본·extracted 의 모든 파일이 정확히 한 행에 들어 있음).
    check_sha1: 파일 sha1 을 다시 계산해 비교(느림).
    """
    path = Path(path) if path else paths.CATALOG
    errs = _Errs()
    got = _read_csv(path, errs, "catalog", need_bom=True)
    if got is None:
        return errs.done()
    header, rows = got
    hs = set(header)
    if len(hs) != len(header):
        errs.add(f"[catalog] 헤더에 겹친 칸: {sorted(c for c in hs if header.count(c) > 1)}")
    miss = [c for c in CATALOG_COLUMNS if c not in hs]
    extra = [c for c in header if c not in CATALOG_COLUMNS]
    if miss:
        errs.add(f"[catalog] 헤더에 없는 칸: {miss}")
    if extra:
        errs.add(f"[catalog] 헤더에 모르는 칸: {extra}")

    seen_doc: dict[str, int] = {}
    where_file: dict[str, list[str]] = {}
    primaries: dict[str, list[str]] = {}
    for ln, r in rows:
        if len(r) != len(header):
            errs.add(f"[catalog] {ln}행: 칸 수 {len(r)} 가 헤더 {len(header)} 와 다름")
            continue
        d = dict(zip(header, r))
        did = d.get("doc_id", "")
        w = f"{ln}행({did or '?'})"
        st = d.get("status", "")
        excl = st == "excluded"

        def bad(col: str, why: str) -> None:
            errs.add(f"[catalog] {w}: {col}={d.get(col, '')!r} {why}")

        if not did:
            bad("doc_id", "비어 있음")
        elif did in seen_doc:
            bad("doc_id", f"가 {seen_doc[did]}행과 겹침")
        else:
            seen_doc[did] = ln
        pid = d.get("post_id", "")
        if not pid.isdigit():
            bad("post_id", "가 숫자가 아님")
        if not excl and did and not (RE_DOC_ID.match(did) and re.match(rf"^fm{re.escape(pid)}(_[0-9]+)?$", did)):
            bad("doc_id", "가 fm{post_id} 또는 fm{post_id}_{k} 모양이 아님")
        for col, allowed in (("category", CATEGORIES), ("real_format", REAL_FORMATS), ("status", STATUSES)):
            if d.get(col, "") not in allowed:
                bad(col, f"가 {'/'.join(allowed)} 가 아님")
        vl = d.get("version_label", "")
        if vl not in VERSION_LABELS and not (excl and vl == ""):
            bad("version_label", f"가 {'/'.join(VERSION_LABELS)} 가 아님")
        for col in ("is_translation", "is_primary"):
            v = d.get(col, "")
            if v not in ("true", "false") and not (excl and v == ""):
                bad(col, "가 true/false 가 아님")
        sl = d.get("script_lang", "")
        if not RE_LANG.match(sl) and not (excl and sl == ""):
            bad("script_lang", "가 언어 코드(ko, en, …, mixed, unknown)가 아님")
        if excl and d.get("is_primary") == "true":
            bad("is_primary", "인데 status=excluded")
        if excl and not d.get("note"):
            bad("note", "가 비어 있음(excluded 행은 사유를 note 에)")
        fid = d.get("film_id", "")
        if fid and not RE_FILM_ID.match(fid):
            bad("film_id", "가 kmdb_/tmdb_/local_ 모양이 아님")
        if not excl and not fid:
            bad("film_id", "가 비어 있음")
        if not excl and not d.get("title_raw"):
            bad("title_raw", "가 비어 있음")
        dg = d.get("dup_group", "")
        if dg and not RE_GROUP.match(dg):
            bad("dup_group", "가 G001 모양이 아님")
        y = d.get("year", "")
        if y and not (y.isdigit() and 1880 <= int(y) <= 2100):
            bad("year", "가 연도가 아님")
        for col in ("match", "extract_quality"):
            v = d.get(col, "")
            if v and not _float_in(v, 0.0, 1.0):
                bad(col, "가 0~1 수가 아님")
        vd = d.get("video_duration", "")
        if vd and not (_float_in(vd, 0.0, None) and float(vd) > 0):
            bad("video_duration", "가 양수가 아님")
        vs = d.get("video_sha1", "")
        if vs and not RE_SHA1.match(vs):
            bad("video_sha1", "가 sha1(40자리 16진수)이 아님")
        vlang = d.get("video_lang", "")
        if vlang and not RE_LANG.match(vlang):
            bad("video_lang", "가 언어 코드가 아님")

        files = _split(d.get("files", ""))
        sha1s = _split(d.get("sha1", ""))
        fmts = _split(d.get("file_formats", ""))
        if not files:
            bad("files", "가 비어 있음")
        if len(sha1s) != len(files):
            errs.add(f"[catalog] {w}: files {len(files)}개와 sha1 {len(sha1s)}개 개수가 다름")
        for h in sha1s:
            if not RE_SHA1.match(h):
                bad("sha1", f"의 {h!r} 가 sha1 이 아님")
                break
        if fmts:
            if len(fmts) != len(files):
                errs.add(f"[catalog] {w}: files {len(files)}개와 file_formats {len(fmts)}개 개수가 다름")
            for x in fmts:
                if x not in REAL_FORMATS or x == "mixed":
                    bad("file_formats", f"의 {x!r} 가 형식 값이 아님")
                    break
            kinds = set(fmts)
            rf = d.get("real_format", "")
            if len(kinds) > 1 and rf != "mixed":
                bad("real_format", f"인데 파일 형식이 여럿({sorted(kinds)}) → mixed 여야 함")
            if len(kinds) == 1 and rf not in (fmts[0], "mixed"):
                bad("real_format", f"가 file_formats {fmts[0]!r} 와 다름")
        elif d.get("real_format") == "mixed":
            bad("file_formats", "가 비어 있음(real_format=mixed 면 파일별 형식 필요)")
        if len(set(files)) != len(files):
            bad("files", "에 같은 파일이 두 번")
        for fp in files:
            if "\\" in fp or fp.startswith("/") or ":" in fp or not fp.startswith(FILE_ROOTS):
                errs.add(f"[catalog] {w}: files 의 {fp!r} 가 ROOT 기준 상대 경로(filmmakers_scripts/… 또는 extracted/…)가 아님")
            where_file.setdefault(fp, []).append(did or f"{ln}행")
        if check_files and check_sha1 and len(sha1s) == len(files):
            from .util import sha1_file
            for fp, h in zip(files, sha1s):
                p = paths.absp(fp)
                if p.is_file() and sha1_file(p) != h:
                    errs.add(f"[catalog] {w}: {fp} 의 sha1 이 실제 파일과 다름")
        if d.get("is_primary") == "true" and fid:
            primaries.setdefault(fid, []).append(did)

    # A2
    for fid, ds in sorted(primaries.items()):
        if len(ds) > 1:
            errs.add(f"[A2] film_id {fid}: is_primary=true 가 {len(ds)}개 ({', '.join(ds)})")

    # A1
    for fp, ds in sorted(where_file.items()):
        if len(ds) > 1:
            errs.add(f"[A1] {fp} 가 여러 행에 들어 있음: {', '.join(ds)}")
    if check_files:
        for fp in sorted(where_file):
            if not paths.absp(fp).is_file():
                errs.add(f"[A1] 없는 파일을 가리킴: {fp}")
        missing = [fp for fp in _raw_target_files() if fp not in where_file]
        for fp in missing:
            errs.add(f"[A1] catalog 어느 행에도 없는 파일: {fp}")
    return errs.done()


def validate_needs_review(path=None) -> list[str]:
    """qa/needs_review.csv: 헤더(차례까지), level·problem 열거값, level 에 맞는 id, score 수."""
    path = Path(path) if path else paths.NEEDS_REVIEW
    errs = _Errs()
    got = _read_csv(path, errs, "review", need_bom=False)
    if got is None:
        return errs.done()
    header, rows = got
    if tuple(header) != NEEDS_REVIEW_COLUMNS:
        errs.add(f"[review] 헤더가 {','.join(NEEDS_REVIEW_COLUMNS)} 가 아님: {','.join(header)}")
        if set(header) != set(NEEDS_REVIEW_COLUMNS):
            return errs.done()
    for ln, r in rows:
        if len(r) != len(header):
            errs.add(f"[review] {ln}행: 칸 수 {len(r)} 가 헤더 {len(header)} 와 다름")
            continue
        d = dict(zip(header, r))
        w = f"{ln}행"
        lv, pb = d["level"], d["problem"]
        if lv not in LEVELS:
            errs.add(f"[review] {w}: level={lv!r} 가 {'/'.join(LEVELS)} 가 아님")
        if pb not in PROBLEMS:
            errs.add(f"[review] {w}: problem={pb!r} 가 명세 4.8 의 값이 아님")
        need = {"doc": "doc_id", "film": "film_id", "item": "item_id"}.get(lv)
        if need and not d[need]:
            errs.add(f"[review] {w}: level={lv} 인데 {need} 가 비어 있음")
        if d["doc_id"] and not RE_DOC_ID.match(d["doc_id"]):
            errs.add(f"[review] {w}: doc_id={d['doc_id']!r} 모양이 아님")
        if d["film_id"] and not RE_FILM_ID.match(d["film_id"]):
            errs.add(f"[review] {w}: film_id={d['film_id']!r} 모양이 아님")
        if d["item_id"] and not RE_ITEM_ID.match(d["item_id"]):
            errs.add(f"[review] {w}: item_id={d['item_id']!r} 모양이 아님")
        if d["score"]:
            try:
                float(d["score"])
            except ValueError:
                errs.add(f"[review] {w}: score={d['score']!r} 가 수가 아님")
    return errs.done()


# =====================================================================
# 파일 하나 검사(종류·짝 파일 찾기) + run
# =====================================================================

def _find_script(doc_id: Any, near: Path) -> Path | None:
    """doc_id 의 script.json: near(검사하는 파일 옆) 아래 폴더에서 doc_id 가 같은 script.json, 없으면 docs/{doc_id}/script.json.

    옆 폴더를 먼저 본다: examples/final.json 의 doc_id(fm55574)가 실제 문서와 같아도 예시끼리 짝을 맞추기 위함.
    """
    if not isinstance(doc_id, str):
        return None
    for base in (near, near.parent):
        cand = base / doc_id / "script.json"
        if cand.is_file():
            return cand
        for c in sorted(base.glob("*/script.json")):
            try:
                with open(c, encoding="utf-8") as f:
                    head = f.read(4096)
                m = re.search(r'"doc_id"\s*:\s*"([^"]+)"', head)
                if m and m.group(1) == doc_id:
                    return c
            except OSError:
                continue
    p = paths.DOCS / doc_id / "script.json"
    return p if p.is_file() else None


def _peek_schema(path: Path) -> str | None:
    try:
        with open(path, encoding="utf-8-sig") as f:
            head = f.read(4096)
    except (OSError, UnicodeDecodeError):
        return None
    m = re.search(r'"schema"\s*:\s*"([^"]+)"', head)
    if m:
        return m.group(1)
    try:
        with open(path, encoding="utf-8-sig") as f:
            d = json.load(f)
        return d.get("schema") if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def check_file(path) -> tuple[str, list[str]]:
    """파일 하나를 종류에 맞게 검사. (종류, 오류 목록). 짝 파일(lines·script·asr·final)은 근처에서 찾는다."""
    p = Path(path)
    name = p.name
    if name == "catalog.csv":
        return "catalog", validate_catalog(check_files=p.resolve() == paths.CATALOG.resolve(), path=p)
    if name == "needs_review.csv":
        return "needs_review", validate_needs_review(p)
    if p.suffix == ".jsonl":
        return "lines", validate_lines(p)
    if p.suffix != ".json":
        return "skip", []
    kind = SCHEMA_KINDS.get(_peek_schema(p) or "")
    if kind is None:
        errs = _Errs()
        data = _load_json(p, errs)
        if data is not None:
            _check_kind(data, "?", errs)
        return "json", errs.done() or ["[E1] schema 값을 알 수 없음"]
    if kind == "lines":
        return kind, ["[E1] sink.lines/1 은 .jsonl 파일이어야 함"]
    if kind == "script":
        lp = p.parent / "lines.jsonl"
        return kind, validate_script(p, lp if lp.is_file() else None)
    if kind == "asr":
        return kind, validate_asr(p)
    if kind == "final":
        try:
            with open(p, encoding="utf-8") as f:
                doc = ((json.load(f).get("film") or {}).get("script") or {}).get("doc_id")
        except (OSError, ValueError, AttributeError):
            doc = None
        return kind, validate_final(p, _find_script(doc, p.parent))
    # align
    film_dir = p.parent.parent if p.parent.name == "work" else p.parent
    fin = next((c for c in (p.parent / "final.json", film_dir / "final.json") if c.is_file()), None)
    asr = p.parent / "asr.json"
    try:
        with open(p, encoding="utf-8") as f:
            doc = json.load(f).get("script_doc")
    except (OSError, ValueError, AttributeError):
        doc = None
    return kind, validate_align(p, fin, asr if asr.is_file() else None, _find_script(doc, film_dir))


def _check_file_worker(path_str: str) -> tuple[str, str, list[str]]:
    try:
        kind, errs = check_file(path_str)
    except Exception as e:  # 검사기 자체 오류도 그 파일의 오류로 남기고 계속
        kind, errs = "error", [f"[crash] 검사 중 예외: {type(e).__name__}: {e}"]
    return path_str, kind, errs


def _walk(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    out = []
    for c in sorted(p.rglob("*")):
        if c.is_file() and (c.suffix in (".json", ".jsonl") or c.name in ("catalog.csv", "needs_review.csv")):
            out.append(c)
    return out


def _coverage_errors(doc_ids: list[str]) -> list[str]:
    """B1: catalog 의 문서(excluded 제외)마다 docs/{doc_id}/lines.jsonl·script.json 이 있는지, 모르는 docs 폴더가 없는지."""
    errs: list[str] = []
    want: set[str] = set()
    if paths.CATALOG.is_file():
        try:
            with open(paths.CATALOG, encoding="utf-8-sig", newline="") as f:
                for d in csv.DictReader(f):
                    if d.get("status") != "excluded" and d.get("doc_id"):
                        want.add(d["doc_id"])
        except (OSError, csv.Error, UnicodeDecodeError) as e:
            return [f"[B1] catalog.csv 를 읽을 수 없음: {e}"]
    have = set(doc_ids)
    for did in sorted(want):
        for fn in ("lines.jsonl", "script.json"):
            if not (paths.DOCS / did / fn).is_file():
                errs.append(f"[B1] {did}: docs/{did}/{fn} 이 없음")
    if want:
        for did in sorted(have - want):
            errs.append(f"[B1] docs/{did} 가 catalog 의 문서(excluded 제외)가 아님")
    return errs


def _film_coverage_errors() -> list[str]:
    """D1: catalog 에서 영상이 있고(video_path) 대표 대본인(is_primary=true) 영화마다 films/{film_id}/final.json 이 있는지.
    있으면 catalog 행과 같은 대본·같은 영상의 것인지(대표 문서, 영상 sha1, 영상 길이, 일치율)도 본다."""
    errs: list[str] = []
    if not paths.CATALOG.is_file():
        return errs
    try:
        with open(paths.CATALOG, encoding="utf-8-sig", newline="") as f:
            rows = [d for d in csv.DictReader(f)
                    if d.get("status") != "excluded" and d.get("video_path") and d.get("is_primary") == "true"
                    and d.get("film_id")]
    except (OSError, csv.Error, UnicodeDecodeError) as e:
        return [f"[D1] catalog.csv 를 읽을 수 없음: {e}"]

    def num(v) -> float | None:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    for d in sorted(rows, key=lambda d: d["film_id"]):
        fid, did = d["film_id"], d.get("doc_id")
        p = paths.FILMS / fid / "final.json"
        if not p.is_file():
            errs.append(f"[D1] {fid}: films/{fid}/final.json 이 없음(영상이 있는 영화, 대표 문서 {did})")
            continue
        try:
            with open(p, encoding="utf-8") as f:
                film = json.load(f).get("film") or {}
            fscript, fvideo = film.get("script") or {}, film.get("video") or {}
        except (OSError, ValueError, AttributeError):
            continue                       # 읽을 수 없는 파일은 그 파일의 검사가 알린다
        if fscript.get("doc_id") != did:
            errs.append(f"[D1] {fid}: final 의 film.script.doc_id {fscript.get('doc_id')!r} 가 catalog 의 대표 문서 "
                        f"{did!r} 와 다름")
        if d.get("video_sha1") and fvideo.get("sha1") != d["video_sha1"]:
            errs.append(f"[D1] {fid}: final 의 film.video.sha1 이 catalog 의 video_sha1 과 다름")
        c_dur, f_dur = num(d.get("video_duration")), num(fvideo.get("duration"))
        if c_dur is not None and f_dur is not None and abs(c_dur - f_dur) > 0.0015:
            errs.append(f"[D1] {fid}: final 의 film.video.duration {f_dur} 가 catalog 의 video_duration {c_dur} 와 다름")
        c_m, f_m = num(d.get("match")), num(fscript.get("match"))
        if f_m is not None and (c_m is None or abs(c_m - f_m) > 0.0005):
            errs.append(f"[D1] {fid}: final 의 film.script.match {f_m} 가 catalog 의 match {d.get('match')!r} 와 다름")
    return errs


def _collect(args) -> tuple[list[Path], list[tuple[str, list[str]]]]:
    """검사할 파일 목록과 덮기 오류 [(이름, 오류 목록)]: B1(문서마다 lines·script), D1(영상 있는 영화마다 final)."""
    given = list(getattr(args, "paths", None) or [])
    docs_filter = getattr(args, "docs", None)
    limit = getattr(args, "limit", None)
    if given:
        out: list[Path] = []
        for g in given:
            p = Path(g)
            if not p.is_absolute() and not p.exists():
                p = paths.ROOT / g
            out.extend(_walk(p) if p.exists() else [p])
        return out, []

    out = []
    wanted = set(docs_filter.split(",")) if docs_filter else None
    doc_dirs = sorted(d for d in paths.DOCS.iterdir() if d.is_dir()) if paths.DOCS.is_dir() else []
    all_ids = [d.name for d in doc_dirs]
    if wanted is not None:
        doc_dirs = [d for d in doc_dirs if d.name in wanted]
    if limit:
        doc_dirs = doc_dirs[:limit]
    if wanted is None:
        for f in (paths.CATALOG, paths.NEEDS_REVIEW):
            if f.is_file():
                out.append(f)
    for d in doc_dirs:
        for fn in ("lines.jsonl", "script.json"):
            if (d / fn).is_file():
                out.append(d / fn)
    if wanted is None:
        if paths.FILMS.is_dir():
            for fd in sorted(x for x in paths.FILMS.iterdir() if x.is_dir()):
                if (fd / "final.json").is_file():
                    out.append(fd / "final.json")
                if (fd / "work").is_dir():
                    out.extend(sorted((fd / "work").glob("*.json")))
        if paths.EXAMPLES.is_dir():
            out.extend(_walk(paths.EXAMPLES))
    cover: list[tuple[str, list[str]]] = []
    if wanted is None and not limit:
        cover = [("docs (B1 덮기)", _coverage_errors(all_ids)), ("films (D1 덮기)", _film_coverage_errors())]
    return out, [(name, errs) for name, errs in cover if errs]


def run(args) -> int:
    t0 = time.time()
    files, cover = _collect(args)
    workers = getattr(args, "workers", None) or min(os.cpu_count() or 1, 12)
    jobs = [str(p) for p in files]
    # 큰 파일 먼저(병렬일 때 끝이 고르게)
    jobs.sort(key=lambda s: -(Path(s).stat().st_size if Path(s).exists() else 0))
    results: list[tuple[str, str, list[str]]] = []
    if workers > 1 and len(jobs) >= 16:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_check_file_worker, jobs, chunksize=max(1, len(jobs) // (workers * 8))))
    else:
        results = [_check_file_worker(j) for j in jobs]
    for name, errs in cover:
        results.append((name, "coverage", errs))

    def rel(s: str) -> str:
        try:
            return Path(s).resolve().relative_to(paths.ROOT).as_posix()
        except ValueError:
            return s

    results.sort(key=lambda r: rel(r[0]))
    by_kind: dict[str, list[int]] = {}
    n_fail = n_err = 0
    for ps, kind, errs in results:
        if not Path(ps).exists() and kind != "coverage":
            errs = errs or ["[io] 파일이 없음"]
        k = by_kind.setdefault(kind, [0, 0])
        k[0] += 1
        if errs:
            k[1] += 1
            n_fail += 1
            n_err += len(errs)
            print(f"FAIL {rel(ps)} ({kind}): 오류 {len(errs)}개", flush=True)
            for e in errs[:SHOW]:
                print(f"  {e}")
            if len(errs) > SHOW:
                print(f"  … {len(errs) - SHOW}개 더")
    for kind, (n, f) in sorted(by_kind.items()):
        print(f"  {kind}: {n}개 검사, {f}개 실패")
    print(f"검사 {len(results)}개, 실패 {n_fail}개, 오류 {n_err}개, {time.time() - t0:.1f}초"
          f"{'' if files or cover else ' (검사할 파일 없음)'}", flush=True)
    return 1 if n_fail else 0
