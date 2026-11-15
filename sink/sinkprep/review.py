"""확인 필요 목록과 카탈로그 마무리.

1. 문서마다 글을 읽어 정해지는 칸을 catalog.csv 에 반영한다(catalog.finalize):
   script_lang, is_translation, extract_quality, status, 그리고 영화마다 is_primary 를 다시 고른다.
   status 가 aligned·done(맞추기·최종본 단계가 올린 값)이면 그대로 두고, 대표 대본은 match 가 있으면 match 가 높은 것부터 본다.
2. 단계마다 남긴 문제(cache/problems/**)를 모아 qa/needs_review.csv 를 쓴다(심한 것부터).
   영화 ID 가 local_ 인 것(KMDb/TMDB 키 없음)은 '못 읽은 대본'이 아니므로 목록에 넣지 않는다(catalog 의 film_id 로 보임).
3. llm 단계가 LLM 판(parser sink-parse-llm)을 script.json 으로 쓴 문서는 catalog note 에 'parser:sink-parse-llm' 을 붙이고,
   품질 문제는 지금 쓰는 판의 것만 올린다(LLM 판이면 parse 단계=규칙 판 문제를 빼고 llm 단계 문제를, 아니면 그 반대).
"""
from __future__ import annotations

import csv
import io
import re
import sys
from collections import Counter, defaultdict

from . import paths, util

STAGE = "review"
VERSION = "0.1.0"

COLUMNS = ("level", "doc_id", "film_id", "item_id", "problem", "score", "detail", "path")
SEVERITY = {"extract_broken": 0, "ocr_low": 1, "parse_low_dialogue": 2, "speaker_missing": 3,
            "translation_unmatched": 4, "version_mismatch": 5, "asr_low_conf": 6, "absent_many": 7, "other": 8}
VERSION_RANK = {"shooting": 5, "final": 4, "bound": 3, "revised": 2, "unknown": 1, "draft": 0, "translation": -1}

_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-zÀ-ÿ]")
_WORD = re.compile(r"[a-zà-ÿ¿¡ñ]+")
STOP = {
    "en": {"the", "and", "you", "to", "of", "is", "it", "that", "what", "in", "he", "she", "his", "her"},
    "fr": {"le", "la", "les", "et", "vous", "est", "je", "pas", "une", "des", "il", "elle", "dans", "qui"},
    "es": {"el", "los", "las", "que", "y", "de", "no", "es", "por", "una", "con", "para", "se", "lo"},
}


def detect_lang(texts: list[str]) -> str:
    """ko | en | fr | es | mixed | unknown (줄 글 목록에서)."""
    joined = "\n".join(texts)
    h = len(_HANGUL.findall(joined))
    lat = len(_LATIN.findall(joined))
    if h + lat < 200:
        return "unknown"
    ratio = h / (h + lat)
    if ratio >= 0.6:
        return "ko"
    if ratio > 0.15:
        return "mixed"
    words = _WORD.findall(joined.lower())
    if not words:
        return "unknown"
    cnt = Counter(words)
    score = {lang: sum(cnt[w] for w in sw) / len(words) for lang, sw in STOP.items()}
    best = max(score, key=score.get)
    return best if score[best] > 0.02 else "unknown"


def doc_info(doc: dict) -> dict:
    doc_id = doc["doc_id"]
    lp = paths.DOCS / doc_id / "lines.jsonl"
    sp = paths.DOCS / doc_id / "script.json"
    rows = util.read_jsonl(lp)[1:] if lp.exists() else []
    info = {"n_lines": len(rows), "lang": detect_lang([r["text"] for r in rows]) if rows else "unknown",
            "quality": 0.0, "has_script": False, "ocr_conf": None, "parser": None}
    if sp.exists() and rows:
        s = util.read_json(sp)
        q = s.get("quality") or {}
        info["quality"] = float(q.get("score") or 0.0)
        info["has_script"] = True
        info["parser"] = (s.get("parser") or {}).get("name")
    confs = [r["conf"] for r in rows if r.get("src") == "ocr" and r.get("conf") is not None]
    if confs:
        info["ocr_conf"] = sum(confs) / len(confs)
        info["quality"] = min(info["quality"], info["ocr_conf"]) if info["has_script"] else info["ocr_conf"]
    return info


def note_with_parser(note: str, parser: str | None) -> str:
    """catalog note 에 LLM 판을 쓰는 문서면 'parser:sink-parse-llm' 을 붙인다(규칙 판이면 뗌, 여러 번 돌려도 같음)."""
    toks = [t for t in note.split(";") if t and not t.startswith("parser:")]
    if parser and parser != "sink-parse":
        toks.append(f"parser:{parser}")
    return ";".join(toks)


def current_parser(doc_id: str) -> str | None:
    sp = paths.DOCS / doc_id / "script.json"
    if not sp.exists():
        return None
    try:
        return (util.read_json(sp).get("parser") or {}).get("name")
    except (OSError, ValueError):
        return None


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _match(d: dict) -> float:
    """catalog match 칸(영상과의 일치율). 비었거나 수가 아니면 -1(맞춰 본 문서보다 뒤)."""
    try:
        return float(d.get("match") or -1.0)
    except ValueError:
        return -1.0


def finalize_catalog(rows: list[dict]) -> dict[str, dict]:
    from . import catalog

    infos = {}
    updates: dict[str, dict] = {}
    for d in rows:
        if d.get("status") == "excluded":
            continue
        info = doc_info(d)
        infos[d["doc_id"]] = info
        lang = info["lang"] if info["lang"] != "unknown" else (d.get("script_lang") or "unknown")
        # 외국영화의 한국어 대본, 한국영화의 외국어 대본(예: 박쥐 영문 시나리오 fm416831)은 번역 대본.
        # 외국영화의 원어는 KMDb/TMDB 없이는 모르므로(스페인어 영화의 영문 대본 등) 판정하지 않는다.
        # catalog 의 is_translation 은 지난 review 값을 이어받은 것이라 쓰지 않고(한번 참이면 못 돌아옴),
        # catalog 와 같은 제목·파일 이름 단서(translation_cue)를 다시 본다.
        cue = catalog.translation_cue([d.get("title_raw") or ""] + list(d.get("files_list") or []))
        is_tr = (d.get("category") == "foreign_film" and cue and lang != "mixed") \
            or (d.get("category") == "foreign_film" and lang == "ko") \
            or (d.get("category") == "ko_film" and (lang not in ("ko", "mixed", "unknown")
                                                     or "영문" in (d.get("title_raw") or ""))) \
            or d.get("version_label") == "translation"
        updates[d["doc_id"]] = {
            "script_lang": lang,
            "is_translation": is_tr,
            "extract_quality": f"{info['quality']:.3f}",
            # 맞추기(aligned)·최종본(done) 단계가 올린 상태는 대본이 그대로 있는 한 되돌리지 않는다
            "status": (d["status"] if d.get("status") in ("aligned", "done") else "parsed")
            if info["has_script"] else "pending",
            "note": note_with_parser(d.get("note") or "", info["parser"]),
        }
    # 영화마다 대표 대본 다시 고르기
    films: dict[str, list[dict]] = defaultdict(list)
    for d in rows:
        if d.get("status") != "excluded" and d.get("film_id"):
            films[d["film_id"]].append(d)

    def key(d):
        u, i = updates[d["doc_id"]], infos[d["doc_id"]]
        # 두 언어가 한 줄씩 번갈아 섞인 대본(mixed)은 깨끗한 한 언어 판이 있으면 뒤로(fm56461_2 vs fm56461_1)
        # 영상과 맞춰 본 문서가 있으면 일치율(match)이 가장 높은 것이 먼저다(영화 1편 = 대본 1개, 명세 Constraints)
        return (i["has_script"], _match(d), not u["is_translation"], u["script_lang"] != "mixed", round(i["quality"], 1),
                VERSION_RANK.get(d.get("version_label") or "unknown", 1), i["quality"], i["n_lines"],
                -int(re.sub(r"\D", "", d["doc_id"]) or 0))

    for fid, ds in films.items():
        best = max(ds, key=key)
        for d in ds:
            updates[d["doc_id"]]["is_primary"] = d is best
    for d in rows:
        if d.get("status") == "excluded" and _truthy(d.get("is_primary")):
            updates.setdefault(d["doc_id"], {})["is_primary"] = False
    catalog.finalize(updates=updates, rows=rows)
    return infos


def _ocr_replaced(p: dict, files: list[str]) -> bool:
    """catalog 가 '열거나 글자를 꺼낼 수 없음'(pdf_broken)으로 적은 파일이라도 OCR 결과가 있으면 그 판정은 뺀다.
    실제로 얼마나 읽었는지는 OCR 단계 문제(쪽 그림 실패·낮은 신뢰도)가 따로 알려 준다."""
    if p.get("problem") != "extract_broken" or "catalog 판별" not in str(p.get("detail") or ""):
        return False
    path = p.get("path") or ""
    return any((not path or f == path) and (paths.OCR_CACHE / p["doc_id"] / f"{k}.json").exists()
               for k, f in enumerate(files))


def collect_problems(rows: list[dict]) -> list[dict]:
    film_of = {d["doc_id"]: d.get("film_id") or "" for d in rows}
    files_of = {d["doc_id"]: d.get("files_list") or [] for d in rows}
    excluded = {d["doc_id"] for d in rows if d.get("status") == "excluded"}
    seen = set()
    out = []
    parser_of: dict[str, str | None] = {}
    for stage, p in util.all_problems(with_stage=True):
        if p.get("level") == "film" and str(p.get("detail") or "").startswith("KMDb/TMDB"):
            continue
        doc_id = p.get("doc_id") or ""
        if doc_id in excluded or (doc_id and doc_id not in film_of):
            continue                       # 제외 문서, 또는 catalog 에 더는 없는 문서의 낡은 기록
        if stage in ("parse", "llm") and doc_id:
            # 품질 문제는 지금 docs/{doc}/script.json 으로 쓰는 판의 것만: LLM 판이면 parse 단계(규칙 판) 문제를 빼고,
            # 규칙 판이면 llm 단계의 LLM 판 품질 문제를 뺀다(LLM 실패 기록 kind=llm_fail 은 늘 남김).
            if doc_id not in parser_of:
                parser_of[doc_id] = current_parser(doc_id)
            is_llm = parser_of[doc_id] == "sink-parse-llm"
            if (stage == "parse" and is_llm) or (stage == "llm" and not is_llm and p.get("kind") != "llm_fail"):
                continue
        if _ocr_replaced(p, files_of.get(doc_id, [])):
            continue
        problem = p.get("problem") if p.get("problem") in SEVERITY else "other"
        row = {
            "level": p.get("level") if p.get("level") in ("doc", "film", "item") else "doc",
            "doc_id": doc_id,
            "film_id": p.get("film_id") or film_of.get(doc_id, ""),
            "item_id": p.get("item_id") or "",
            "problem": problem,
            "score": "" if p.get("score") is None else str(p.get("score")),
            "detail": util.nfc(str(p.get("detail") or "")).replace("\n", " "),
            "path": p.get("path") or "",
        }
        k = (row["level"], row["doc_id"], row["film_id"], row["item_id"], row["problem"], row["detail"])
        if k in seen:
            continue
        seen.add(k)
        out.append(row)
    out.sort(key=lambda r: (SEVERITY[r["problem"]], r["doc_id"], r["film_id"], r["detail"]))
    return out


def write_needs_review(problems: list[dict]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in problems:
        w.writerow(r)
    util.write_text(paths.NEEDS_REVIEW, "﻿" + buf.getvalue())


def run(args) -> int:
    from . import catalog

    rows = catalog.load()
    infos = finalize_catalog(rows)
    rows = catalog.load()
    problems = collect_problems(rows)
    write_needs_review(problems)
    c = Counter(p["problem"] for p in problems)
    docs_with = len({p["doc_id"] for p in problems if p["doc_id"]})
    langs = Counter(r.get("script_lang") for r in rows if r.get("status") != "excluded")
    print(f"review: 문서 {len(infos)}개, 확인 필요 {len(problems)}건(문서 {docs_with}개) {dict(c)}", flush=True)
    print(f"review: script_lang {dict(langs)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(run(None))
