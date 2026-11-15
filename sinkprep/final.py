"""최종 영상 기준 대본 -> films/{film_id}/final.json  (명령 final, 명세 4.7, schema/final.schema.json)

입력은 docs/{doc_id}/script.json, films/{film_id}/work/asr.json, work/align.json, catalog 행이다.
대상은 catalog 에서 영상이 있고(video_path) 대표 대본이며(is_primary) asr.json·align.json 이 있는 영화. 열쇠는 film_id.

하는 일
 1. 차례·시각·id: ``timeline.build_timeline`` 을 align 과 같은 입력으로 다시 불러 같은 항목 뼈대를 얻는다.
    align.json ops 의 item 과 어긋나면 멈춘다(대본·받아쓰기·영상 길이가 align 뒤에 바뀐 것: align 을 다시 돌린다).
 2. 대사 항목: 들린 글(heard)과 낱말은 align 이 고른 받아쓰기 pass 범위 그대로(w, s, e, p. 어느 pass 든 같다: plain,
    lv3, rescue ...). 낱말 시각만 ``timeline.fix_words`` 로 바로잡는다(한 줄 안에서 거꾸로 간 시작 시각, 영상 길이 밖의
    시각). 바로잡은 줄 수는 quality.time_fixed 에, 바로잡아도 믿기 어려운 줄은 quality.time_suspect 와 항목 수준 문제로
    남기고 영화는 그대로 만든다. 화자·how·lang·paren·sung·
    overlap 은 대본 요소에서 복사, same/changed 와 차이 조각은 ``textnorm.verdict``, conf 는 ``align.op_conf``.
    줄 언어(lang)는 글자의 기준인 영상을 따른다(``textnorm.spoken_lang``): 시각 있는 대본 대사의 lang 이 영상 언어와
    달라도 들린 글이 영상 언어의 글자로 적혔으면(영상 언어가 ko 면 글자의 절반 이상이 한글) 영상 언어로 말한 줄로 두고,
    그런 줄 수를 quality.lang_from_heard 와 항목 수준 문제(other)로 남긴다. 시각 없는 줄(absent)은 대본 lang 그대로.
    한국어 줄은 ko = heard, ko_src = same. 번역은 이 단계의 범위 밖이라, 시각 있는 대사 가운데 그래도 한국어가 아닌
    줄(들린 글이 한글이 아닌 외국어 줄)이 있으면 그 영화는 실패로 두고 translation_unmatched 를 기록한다.
 3. 대사 아닌 항목(scene, action, onscreen, transition)은 대본 글 그대로에 추정 시각.
 4. 쓰기 전에 검사: 임시 파일로 쓴 뒤 validate_final·validate_align·validate_asr 가 모두 통과해야 final.json 이 된다.
    통과하면 catalog status 를 done 으로, match 를 final 의 일치율로 맞춘다.

같은 입력이면 final.json 바이트가 같다(시각·난수·걸린 시간을 넣지 않는다).
"""
from __future__ import annotations

import math
import os
import re
import sys
import time
from collections import Counter

from . import __version__, align, paths, textnorm, timeline, util

STAGE = "final"
VERSION = "0.3.0"          # 0.2.0: 낱말 시각 바로잡기 집계, added 의 씬·씬 구간 규칙(timeline), 문제 기록에 항목 id
#                            0.3.0: 줄 언어를 들린 글자로(quality.lang_from_heard), made.timing 에 시각 바로잡기 규칙 넷 모두
SCHEMA = "sink.final/1"
KO = "ko"
VERSION_LABELS = ("draft", "revised", "bound", "shooting", "final", "translation", "unknown")
ROW_FIELDS = ("doc_id", "film_id", "film_title", "film_title_orig", "title_norm", "title_raw", "year", "video_sha1",
              "video_duration", "video_lang", "version_label", "script_lang", "is_translation")   # final 에 들어가는 catalog 칸
SHOW_ERRORS = 10           # 검사 실패 때 화면에 보여 주는 오류 수

TIMING_TEXT = ("aligned: 받아쓰기 낱말 시각 그대로(받아쓴 글자를 Whisper cross-attention DTW 로 소리에 맞춘 시각. "
               "강제 정렬을 따로 돌리지 않는다). 다만 한 줄의 낱말 시각을 넷만 바로잡는다: 영상 길이를 넘은 시각이 허용 오차"
               f"({timeline.DURATION_TOL}초)에 닿으면 영상 끝으로 당기고, 시작 시각이 앞 낱말의 시작보다 이른 낱말은 앞 낱말의 "
               "시작으로 올리고, 끝이 시작보다 이른 낱말은 끝을 시작에 맞추고, 줄 전체가 한 순간(가장 이른 시작 = 가장 늦은 끝)"
               f"이면 가장 늦게 끝나는 낱말의 끝을 {timeline.MIN_DUR}초 늘린다(그런 줄 수가 quality.time_fixed). "
               "estimated: 대본 순서로 앞뒤 대사 시각 사이를 글 길이에 비례해 나눈 추정"
               "(씬 머리·지문·화면 글자·전환, end 는 null. 한 씬의 요소는 그 씬의 대사 덩어리 곁에 둔다). "
               "씬 구간도 추정: 그 씬의 시각 있는 대본 대사와 그 씬으로 분류한 대본에 없는 말의 범위(대본 순서와 다른 자리에서 "
               "들린 대사는 뺀다), 그런 대사가 없는 씬은 씬 머리·지문의 추정 시각 범위. none: 시각 없음(absent).")
SCENE_TEXT = ("대본 요소의 scene 은 대본 그대로. 대본에 없는 말(added)의 scene 은 시각으로 앞뒤에 있는 대본 대사의 씬 가운데 "
              "하나: 둘이 같은 씬이면 그 씬, 다르면 말이 없던 시간(쉼)의 합으로 잰 거리가 가까운 쪽(같으면 앞 대사), "
              "앞이나 뒤에 대본 대사가 없으면 있는 쪽, 둘 다 없으면 null. 씬 머리·지문의 추정 시각은 보지 않는다.")
MATCH_TEXT = ("일치율 = same·changed 줄에서 대본 글자와 정확히 맞은 글자 수(공백·문장부호를 뺀 글자열의 최장 공통 부분열)의 합 "
              "÷ 말 있는 대본 대사 전체의 글자 수. 0~1, 소수 셋째 자리. catalog 의 match 와 같은 값.")
SPEAKER_TEXT = ("speaker_from script: 대본에 적힌 화자. unknown: 화자를 모르거나(speakers 가 빈 목록: 대본에 없는 말, 화자 없는 "
                "대본 줄) 대본에 적힌 이름이 아니라 앞 대사에서 이어받은 추정 화자(speakers 가 있음. 확정값으로 세지 않는다). "
                "quality.speaker_unknown 은 speaker_from 이 unknown 인 대사 수, speaker_inferred 는 그 가운데 추정 화자가 붙은 수.")
QUALITY_TEXT = ("spoken: 말 있는 대본 대사 수. silent_turns: 말 없는 대본 대사 수(글자 없이 문장부호·지시뿐인 요소. absent 로 두되 "
                "일치율과 잘린 대사 통계에서 뺀다). absent_spoken: 말 있는 대본 대사 가운데 absent. line_coverage: (same + changed) "
                "÷ spoken. match: 일치율. mean_conf: same·changed·added 줄의 conf 평균. "
                f"low_conf: 그 가운데 conf 가 {align.ITEM_LOW_CONF} 미만인 줄 수. moved: 대본 순서와 다른 자리에서 들린 대본 대사 수. "
                "words: 낱말 수. by_pass: 받아쓰기 pass 별로 그 pass 에서 온 줄 수. diff_kinds: 차이 조각 종류별 수. "
                "time_fixed: 낱말 시각을 바로잡은 줄 수(made.timing 의 규칙), time_fixed_words: 그 낱말 수. "
                "time_suspect: 바로잡았지만 시각을 믿기 어려운 줄의 항목 id(영상 길이 밖의 낱말이 있었거나 시작 시각이 "
                f"{timeline.FAR_BACK_S}초 넘게 뒤집혀 있었음). pass_overlaps: 서로 다른 받아쓰기 pass 에서 온 두 줄의 시간 "
                f"구간이 {timeline.SAME_SOUND_S}초 넘게 겹친 곳 [앞 항목 id, 뒤 항목 id, 겹친 초](같은 소리가 두 줄에 "
                "들어갔을 수 있는 곳. 시각만으로는 단정할 수 없어 막지 않고 적어 둔다). lang_from_heard: 대본 lang 이 영상 "
                "언어와 다른데 들린 글이 영상 언어의 글자로 적혀(글자의 "
                f"{textnorm.LANG_SCRIPT_SHARE:.0%} 이상) lang 을 영상 언어로 둔 시각 있는 대본 대사 수.")
LANG_FROM_HEARD_TEXT = "대본 lang {script}, 들린 글은 영상 언어 {video} 글자: lang 을 {video} 로 둠"
TIME_SUSPECT_TEXT = (f"낱말 시각을 바로잡았지만 믿기 어려움(영상 길이 밖의 낱말이 있었거나 시작 시각이 "
                     f"{timeline.FAR_BACK_S}초 넘게 뒤집혀 있었음)")


class FinalError(Exception):
    """이 영화의 final.json 을 만들 수 없다. 글은 사람이 읽을 설명이고 대사 글을 담지 않는다."""


class Stale(FinalError):
    """align.json 이 지금의 대본·받아쓰기·영상 길이와 맞지 않는다(align 을 다시 돌려야 한다)."""

    def __init__(self, why: str):
        super().__init__(f"align.json 이 지금 입력과 맞지 않음: {why}. python -I -m sinkprep align --force 로 다시 맞출 것")


class NeedsTranslation(FinalError):
    """시각 있는 대사 가운데 한국어가 아닌 줄이 있다(번역 단계가 아직 없다)."""

    def __init__(self, count: int, timed: int, langs: Counter):
        self.count = count
        by = ", ".join(f"{k} {v}" for k, v in sorted(langs.items()))
        super().__init__(f"시각 있는 대사 {timed}줄 가운데 한국어가 아닌 줄 {count}개({by}): 번역 단계가 없어 final.json 을 만들지 않음")


# ------------------------------------------------------------------ 조립
def _duration(doc: dict) -> float | None:
    try:
        d = float(doc.get("video_duration") or 0)
    except ValueError:
        return None
    return d if math.isfinite(d) and d > 0 else None


def _year(v) -> int | None:
    s = str(v or "").strip()
    return int(s) if s.isdigit() and 1880 <= int(s) <= 2100 else None


def asr_summary(asr: dict) -> str | None:
    """made.asr: pass 마다 'id: 모델 문자열 [context]' 를 '; ' 로 이은 글."""
    parts = []
    for p in asr.get("passes") or []:
        ctx = p.get("context")
        parts.append(f"{p.get('id')}: {p.get('model')}" + (f" [{ctx}]" if ctx else ""))
    return "; ".join(parts) or None


def match_rate(script: dict, heard_of: dict[str, str]) -> float:
    """일치율(MATCH_TEXT). heard_of: {대본 요소 id: 들린 글} (same·changed 줄만)."""
    hit = total = 0
    for e in script.get("elements") or []:
        if e.get("type") != "dialogue":
            continue
        k = textnorm.key(e.get("text"))
        total += len(k)
        if k and e["id"] in heard_of:
            hit += textnorm.lcs(k, textnorm.key(heard_of[e["id"]]))
    return round(hit / total, 3) if total else 0.0


def build(doc: dict, script: dict, asr: dict, al: dict) -> dict:
    """final.json 의 내용을 만든다. 파일을 읽거나 쓰지 않는다.

    doc: catalog 행(dict), script: script.json, asr: asr.json, al: align.json 을 읽은 dict.
    만들 수 없으면 FinalError(Stale: align 을 다시 돌려야 함, NeedsTranslation: 번역이 필요한 줄이 있음).
    """
    doc_id, film_id = doc["doc_id"], doc["film_id"]
    duration = _duration(doc)
    if duration is None:
        raise FinalError("catalog 의 video_duration 이 비었거나 양수가 아님(python -I -m sinkprep video 로 영상을 다시 등록)")
    video_sha1 = doc.get("video_sha1") or ""
    if asr.get("video_sha1") != video_sha1:
        raise FinalError("asr.json 의 video_sha1 이 catalog 의 video_sha1 과 다름(다른 영상의 받아쓰기: asr 를 다시 돌릴 것)")
    if al.get("script_doc") != doc_id:
        raise Stale(f"script_doc {al.get('script_doc')!r} 가 대표 문서 {doc_id!r} 가 아님")
    ops = al.get("ops") or []
    method = al.get("method") if isinstance(al.get("method"), dict) else {}

    # 1. 차례·시각·id 를 align 과 같은 함수로 다시 얻는다
    try:
        skel = timeline.build_timeline(script, ops, asr, duration)
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise Stale(f"ops 로 항목을 다시 만들 수 없음({type(e).__name__}: {e})") from e
    for it in skel:
        if it["op"] is not None and ops[it["op"]].get("item") != it["id"]:
            raise Stale(f"ops[{it['op']}] 의 item 이 {ops[it['op']].get('item')} 인데 다시 계산하면 {it['id']}")
    confs = align.op_conf(script, asr, ops, method)       # method: LLM 짝 판정의 헛소리 낱말(llm_pair.noise)
    index = timeline.asr_index(asr)
    els = {e["id"]: e for e in script.get("elements") or []}
    film_lang = doc.get("video_lang") or asr.get("lang") or "unknown"
    script_lang = doc.get("script_lang") or script.get("lang") or "unknown"

    # 2. 항목
    items: list[dict] = []
    heard_of: dict[str, str] = {}
    foreign: Counter = Counter()
    cnt: Counter = Counter()
    by_pass: Counter = Counter()
    kinds: Counter = Counter()
    word_ps: list[float] = []
    timed_conf: list[float] = []
    n_words = 0
    suspects: list[str] = []
    for it in skel:
        if it["type"] != "dialogue":
            e = els[it["src_el"]]
            row = {"id": it["id"], "type": it["type"], "scene": it["scene"]}
            if it["type"] == "onscreen" and e.get("kind"):
                row["kind"] = e["kind"]
            row.update(text=str(e.get("text") or ""), start=it["start"], end=it["end"], timing=it["timing"],
                       src_el=it["src_el"])
            items.append(row)
            continue
        op = ops[it["op"]]
        e = els[it["src_el"]] if it["src_el"] is not None else None
        status, heard, words = it["status"], it["heard"], it["words"]
        timed = it["start"] is not None
        if e is None:                                        # 대본에 없는 말
            speakers, inferred, how, lang = [], False, "unknown", film_lang
            paren, sung, overlap, ref = [], False, None, None
        else:
            speakers = list(e.get("speakers") or [])
            inferred = bool(e.get("speaker_inferred")) and bool(speakers)
            how, lang = e.get("how") or "unknown", e.get("lang") or script_lang
            paren = [{"text": p["text"], "at": p["at"]} for p in e.get("paren") or []]
            sung, overlap = bool(e.get("sung")), e.get("overlap")
            ref = {"el": e["id"], "text": str(e.get("text") or "")}
        diffs: list[dict] = []
        if timed and e is not None:
            said = textnorm.spoken_lang(lang, heard, film_lang)       # 글자의 기준은 영상: 들린 글자로 줄 언어
            if said != lang:
                lang = said
                cnt["lang_from_heard"] += 1
            want, diffs = textnorm.verdict(ref["text"], heard)
            if want != status:
                raise Stale(f"{it['id']}({e['id']}) 의 op 은 {status} 인데 지금 대본 글로는 {want}")
            if status == "changed" and not diffs:
                raise FinalError(f"{it['id']}({e['id']}): changed 인데 차이 조각을 만들지 못함")
            heard_of[e["id"]] = heard
        if timed:
            pid, i0, i1 = timeline.op_range(op)
            by_pass[pid] += 1
            word_lang = any(w.get("lang") not in (None, KO) for w in index[pid][i0:i1 + 1])
            if op.get("match") == "translation":
                foreign["match=translation"] += 1
            elif lang != KO or word_lang:
                foreign[lang if lang != KO else "낱말 lang"] += 1
            n_words += len(words)
            word_ps.extend(float(w["p"]) for w in words if w["p"] is not None)
            fix = it.get("time_fix") or {}
            cnt["time_fixed"] += bool(fix.get("fixed"))
            cnt["time_fixed_words"] += int(fix.get("fixed") or 0)
            if timeline.suspect(fix):
                suspects.append(it["id"])
        conf = confs[it["op"]]
        if timed and conf is not None:
            timed_conf.append(conf)
        cnt[status] += 1
        cnt["dialogue"] += 1
        cnt["speaker_unknown"] += not (speakers and not inferred)
        cnt["speaker_inferred"] += inferred
        cnt["moved"] += bool(it.get("moved"))
        cnt["silent_turns"] += e is not None and not textnorm.sayable(ref["text"])
        cnt["low_conf"] += timed and conf is not None and conf < align.ITEM_LOW_CONF
        kinds.update(d["kind"] for d in diffs)
        items.append({
            "id": it["id"], "type": "dialogue", "scene": it["scene"],
            "speakers": speakers, "speaker_from": "script" if speakers and not inferred else "unknown",
            "how": how, "lang": lang,
            "heard": heard, "ko": heard, "ko_src": "same" if timed else None,
            "start": it["start"], "end": it["end"], "timing": it["timing"], "words": words,
            "status": status, "match": op.get("match"), "script_ref": ref, "diffs": diffs,
            "paren": paren, "sung": sung, "overlap": overlap, "conf": conf, "src_el": it["src_el"]})
    if foreign:
        raise NeedsTranslation(sum(foreign.values()), cnt["same"] + cnt["changed"] + cnt["added"], foreign)

    # 3. 일치율: align 이 적은 값과 같아야 한다
    match = match_rate(script, heard_of)
    stated = (method.get("stats") or {}).get("match") if isinstance(method.get("stats"), dict) else None
    if isinstance(stated, (int, float)) and abs(float(stated) - match) > 1e-9:
        raise Stale(f"method.stats.match 가 {stated} 인데 다시 계산하면 {match}")

    spans = {s["id"]: s for s in timeline.scene_spans(script, skel, duration)}
    scenes = [{"id": sc["id"], "number": sc.get("number"), "heading": sc.get("heading") or "", "place": sc.get("place"),
               "time": sc.get("time"), "flags": list(sc.get("flags") or []),
               "start": spans[sc["id"]]["start"], "end": spans[sc["id"]]["end"], "timing": spans[sc["id"]]["timing"]}
              for sc in script.get("scenes") or []]
    spoken = cnt["same"] + cnt["changed"] + cnt["absent"] - cnt["silent_turns"]
    version = doc.get("version_label")
    aligner = method.get("aligner") or "sinkprep.align"
    return {
        "schema": SCHEMA,
        "film": {
            "id": film_id,
            "title": doc.get("film_title") or doc.get("title_norm") or doc.get("title_raw") or doc_id,
            "title_orig": doc.get("film_title_orig") or None,
            "year": _year(doc.get("year")),
            "video": {"sha1": video_sha1, "duration": duration, "lang": film_lang},
            "script": {"doc_id": doc_id, "version": version if version in VERSION_LABELS else "unknown",
                       "lang": script_lang, "translation": str(doc.get("is_translation")).lower() == "true",
                       "match": match},
        },
        "made": {
            "pipeline": __version__,
            "asr": asr_summary(asr),
            "aligner": f"{aligner} | {method['anchor']}" if method.get("anchor") else aligner,
            "translator": None,
            "final": f"sinkprep.final {VERSION}",
            "timing": TIMING_TEXT,
            "scene": SCENE_TEXT,
            "forced": method.get("forced"),
            "conf": align.CONF_TEXT,
            "match": MATCH_TEXT,
            "speaker": SPEAKER_TEXT,
            "quality": QUALITY_TEXT,
        },
        "people": [{"id": p["id"], "name": p["name"], "aliases": list(p.get("aliases") or [])}
                   for p in script.get("people") or []],
        "scenes": scenes,
        "items": items,
        "quality": {
            "dialogue": cnt["dialogue"], "same": cnt["same"], "changed": cnt["changed"], "added": cnt["added"],
            "absent": cnt["absent"], "speaker_unknown": cnt["speaker_unknown"],
            "mean_word_p": round(sum(word_ps) / len(word_ps), 3) if word_ps else None,
            "spoken": spoken, "silent_turns": cnt["silent_turns"],
            "absent_spoken": cnt["absent"] - cnt["silent_turns"],
            "line_coverage": round((cnt["same"] + cnt["changed"]) / spoken, 3) if spoken else None,
            "match": match,
            "speaker_inferred": cnt["speaker_inferred"],
            "mean_conf": round(sum(timed_conf) / len(timed_conf), 3) if timed_conf else None,
            "low_conf": cnt["low_conf"], "moved": cnt["moved"], "words": n_words,
            "by_pass": dict(sorted(by_pass.items())), "diff_kinds": dict(sorted(kinds.items())),
            "time_fixed": cnt["time_fixed"], "time_fixed_words": cnt["time_fixed_words"], "time_suspect": suspects,
            "pass_overlaps": [[a, b, sec] for a, b, sec in timeline.cross_pass_overlaps(ops, index)],
            "lang_from_heard": cnt["lang_from_heard"],
        },
    }


def problems(doc_id: str, film_id: str, data: dict, script: dict | None = None) -> list[dict]:
    """final.json 내용에서 영화·항목 수준 문제(align 단계와 같은 기준, 같은 글). detail 에는 수치·코드만 적는다.

    낱말 시각을 바로잡았지만 믿기 어려운 줄(quality.time_suspect)은 그 줄만 항목 수준 other 로 남긴다(영화는 그대로 만든다).
    서로 다른 받아쓰기 pass 에서 온 줄끼리 시간이 겹친 곳(quality.pass_overlaps)도 뒤 줄에 항목 수준 other 로 남긴다.
    script(script.json)를 주면 대본 lang 과 다른 lang 을 들린 글자로 정한 줄(quality.lang_from_heard)도 항목 수준 other 로
    남긴다. 항목 수준 문제는 종류마다 영화당 ``align.ITEM_PROBLEMS_MAX`` 개까지.
    """
    q = data["quality"]
    path = f"films/{film_id}/final.json"
    rows = [(it["conf"], it["id"], it["status"]) for it in data["items"]
            if it["type"] == "dialogue" and it["status"] != "absent"]
    out = align.film_problems(doc_id, film_id, spoken=q["spoken"], absent_spoken=q["absent_spoken"],
                              match=data["film"]["script"]["match"], conf_rows=rows, path=path)
    for iid in (q.get("time_suspect") or [])[:align.ITEM_PROBLEMS_MAX]:
        out.append({"level": "item", "doc_id": doc_id, "film_id": film_id, "item_id": iid, "problem": "other",
                    "score": None, "detail": TIME_SUSPECT_TEXT, "path": path})
    for a, b, sec in (q.get("pass_overlaps") or [])[:align.ITEM_PROBLEMS_MAX]:
        out.append({"level": "item", "doc_id": doc_id, "film_id": film_id, "item_id": b, "problem": "other",
                    "score": sec, "detail": f"다른 받아쓰기 pass 에서 온 줄 {a} 과 시간 구간이 {sec:.3f}초 겹침"
                                            "(같은 소리가 두 줄에 들어갔을 수 있음)", "path": path})
    if script is not None:
        els = {e["id"]: e for e in script.get("elements") or []}
        video, doc_lang = data["film"]["video"]["lang"], data["film"]["script"]["lang"]
        moved = [(it["id"], els[it["src_el"]].get("lang") or doc_lang) for it in data["items"]
                 if it["type"] == "dialogue" and it["start"] is not None and it["src_el"] in els
                 and it["lang"] != (els[it["src_el"]].get("lang") or doc_lang)]
        for iid, lang in moved[:align.ITEM_PROBLEMS_MAX]:
            out.append({"level": "item", "doc_id": doc_id, "film_id": film_id, "item_id": iid, "problem": "other",
                        "score": None, "detail": LANG_FROM_HEARD_TEXT.format(script=lang, video=video), "path": path})
    return out


# ------------------------------------------------------------------ 실행
def out_path(film_id: str):
    return paths.FILMS / film_id / "final.json"


def select_docs(args) -> list[dict]:
    """대상: catalog 에서 영상이 있고(video_path) 대표 대본이며(is_primary) asr.json·align.json 이 있는 영화의 문서."""
    from . import catalog

    def ready(d: dict) -> bool:
        work = paths.FILMS / d["film_id"] / "work"
        return (work / "asr.json").exists() and (work / "align.json").exists()

    docs = [d for d in catalog.load()
            if d.get("status") != "excluded" and d.get("video_path") and d.get("is_primary") == "true"
            and d.get("film_id") and ready(d)]
    if getattr(args, "docs", None):
        want = set(args.docs.split(","))
        docs = [d for d in docs if d["doc_id"] in want]
    if getattr(args, "limit", None):
        docs = docs[: args.limit]
    return docs


_CODE = re.compile(r"^\[([^\]]+)\]")
_ITEM = re.compile(r"\bI[0-9]{4,}\b")
SHOW_ITEMS = 5             # 검사 실패 때 문제 기록에 적는 항목 id 수


def _codes(errs: list[str]) -> str:
    """검사 오류를 코드별 수로 줄인 글('[D2] 3, [D7] 1'). 오류 글에는 대사가 섞일 수 있어 문제 기록에는 이것만 적는다."""
    c = Counter(m.group(0) if (m := _CODE.match(e)) else "[?]" for e in errs)
    return ", ".join(f"{k} {v}" for k, v in sorted(c.items()))


def _where(errs: list[str]) -> str:
    """검사 오류가 가리키는 항목 id 몇 개('; 항목 I0004, I0010 …'). 없으면 빈 글. 어느 줄인지 다시 돌리지 않고 알 수 있게."""
    ids = list(dict.fromkeys(i for e in errs for i in _ITEM.findall(e)))
    if not ids:
        return ""
    return "; 항목 " + ", ".join(ids[:SHOW_ITEMS]) + (" …" if len(ids) > SHOW_ITEMS else "")


def process(doc: dict, *, force: bool = False) -> tuple[str, str, float | None]:
    """문서 하나(= 영화 하나)의 final.json 을 만든다. (상태 낱말 ok|skip|fail, 설명, 일치율)."""
    from . import validate

    doc_id, film_id = doc["doc_id"], doc["film_id"]
    sp = paths.DOCS / doc_id / "script.json"
    work = paths.FILMS / film_id / "work"
    ap, lp = work / "asr.json", work / "align.json"
    out = out_path(film_id)
    for p, what in ((sp, "script.json"), (ap, "work/asr.json"), (lp, "work/align.json")):
        if not p.exists():
            return "fail", f"{what} 없음", None
    fp = {"version": VERSION, "align_version": align.VERSION, "timeline_version": timeline.VERSION, "pipeline": __version__,
          "script_sha1": util.sha1_file(sp), "asr_sha1": util.sha1_file(ap), "align_sha1": util.sha1_file(lp),
          "row": {k: doc.get(k) or "" for k in ROW_FIELDS}}
    old = util.stage_fp(STAGE, film_id) or {}
    if not force and {k: old.get(k) for k in fp} == fp and out.exists() and util.sha1_file(out) == old.get("out_sha1"):
        return "skip", "", old.get("match")

    base = {"doc_id": doc_id, "film_id": film_id, "item_id": "", "path": f"films/{film_id}/final.json"}
    script = util.read_json(sp)
    try:
        data = build(doc, script, util.read_json(ap), util.read_json(lp))
    except NeedsTranslation as e:
        util.set_problems(STAGE, film_id, [dict(base, level="film", problem="translation_unmatched", score=e.count,
                                                detail=str(e), path=f"films/{film_id}/work/align.json")])
        return "fail", str(e), None
    except FinalError as e:
        util.set_problems(STAGE, film_id, [dict(base, level="film", problem="other", score=None, detail=str(e),
                                                path=f"films/{film_id}/work/align.json")])
        return "fail", str(e), None

    # 쓰기 전에 스스로 검사: 임시 파일이 세 검사를 모두 통과해야 final.json 으로 바꿔 놓는다
    tmp = out.with_name(out.name + ".check")
    util.write_json(tmp, data)
    try:
        errs = validate.validate_final(tmp, sp)
        errs += validate.validate_align(lp, final_path=tmp, asr_path=ap, script_path=sp)
        errs += validate.validate_asr(ap)
        if not errs:
            os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    if errs:
        for e in errs[:SHOW_ERRORS]:
            print(f"  ! {film_id}: {e}", file=sys.stderr, flush=True)
        msg = f"검사 실패 {len(errs)}건({_codes(errs)}{_where(errs)}): final.json 을 쓰지 않음"
        util.set_problems(STAGE, film_id, [dict(base, level="film", problem="other", score=len(errs), detail=msg)])
        return "fail", msg, None

    util.set_problems(STAGE, film_id, problems(doc_id, film_id, data, script))
    match = data["film"]["script"]["match"]
    util.set_stage_fp(STAGE, film_id, dict(fp, out_sha1=util.sha1_file(out), match=match))
    q = data["quality"]
    return "ok", (f"항목 {len(data['items'])}개(대사 {q['dialogue']}: same {q['same']} changed {q['changed']} "
                  f"added {q['added']} absent {q['absent']}, 말 없는 대사 {q['silent_turns']}), 일치율 {match:.3f}"), match


def run(args) -> int:
    from . import catalog

    docs = select_docs(args)
    force = bool(getattr(args, "force", False))
    counts: Counter = Counter()
    updates: dict[str, dict] = {}
    t0 = time.time()
    if not docs:
        print("final: 대상 없음(영상이 있는 대표 문서 가운데 work/asr.json 과 work/align.json 이 있는 영화가 없음)", flush=True)
        return 0
    for k, d in enumerate(docs, 1):
        try:
            st, msg, match = process(d, force=force)
        except Exception as e:  # noqa: BLE001 - 한 영화가 실패해도 나머지는 계속
            st, msg, match = "error", f"{type(e).__name__}: {e}", None
        counts[st] += 1
        if st in ("ok", "skip") and match is not None:
            want = {"status": "done", "match": f"{match:.3f}"}
            if any(d.get(c) != v for c, v in want.items()):
                updates[d["doc_id"]] = want
        elif st in ("fail", "error") and d.get("status") == "done":
            updates[d["doc_id"]] = {"status": "aligned"}     # 최종본을 못 만든 영화를 done 으로 남기지 않는다
        if st != "skip":
            print(f"[{k}/{len(docs)}] {d['doc_id']} {d['film_id']} {st}: {msg}", flush=True)
            if st in ("fail", "error"):
                print(f"  ! {d['doc_id']}: {msg}", file=sys.stderr, flush=True)
    if updates:
        catalog.finalize(updates=updates)
    print(f"{STAGE}: {dict(counts)} ({time.time() - t0:.0f}초)", flush=True)
    return 1 if (counts["fail"] or counts["error"]) else 0


if __name__ == "__main__":
    sys.exit(run(None))
