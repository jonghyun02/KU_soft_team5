"""가짜 영화로 맞추기(sinkprep.align) 채점.

사용: .venv/Scripts/python -I tools/eval_align_synth.py [--docs fm55664,fm55890,...] [--seeds 1,2,3]
        [--set NAME=값 ...] [--rate NAME=값 ...] [--json 결과.json] [--quiet]
  --set   sinkprep.align 의 상수를 바꿔 돌린다(기준값 고르기). 예: --set R_LONG=0.6 --set SHORT_NEAR=5
  --rate  가짜 영화 변형 확률(tools/align_synth.RATES)을 바꾼다. 예: --rate char_err=0.12

docs/{doc_id}/script.json 을 읽기만 한다. 문서마다 씨앗마다 가짜 영화를 만들고 두 번 맞춘다.
  plain  : plain pass 만 준 것(맞추기 자체의 채점. 정답은 plain 낱말 기준)
  full   : plain + ctx + rescue 를 다 준 것(pass 고르기까지. 정답은 실제로 한 말 기준)
--alt 를 주면 대안 pass 가 든 영화(align_synth.make_movie 의 alt=True)를 따로 만들어 두 번 더 맞춘다.
  base   : 그 영화의 plain + rescue (대안 pass 없이: two 와 견줄 기준)
  two    : plain + lv3 + rescue (지금 실제 받아쓰기의 꼴. 기준과 대안 가운데 고르기, rescue 의 added)
  all    : plain + lv3 + ctx + rescue (네 pass 를 다 주었을 때도 한 소리를 두 줄에 담지 않는지)

지표
  자리      맞춘 줄의 시간 구간이 정답 구간과 겹치는가. 정밀도 = 겹친 줄 ÷ 맞췄다고 한 줄, 재현율 = 겹친 줄 ÷ 정답에 있는 줄.
            줄 글자 수 1~3, 4~7, 8+ 로 나눠 본다.
  상태      same·changed·absent 가 정답과 같은 줄의 비율(시각 있는 줄은 자리도 맞아야 함)과 혼동 행렬.
  absent    정밀도·재현율.
  added     낱말 단위 정밀도·재현율(plain 낱말 번호 기준). 재현율이 낮으면 대본에 없는 말이 대본 줄에 붙은 것이고,
            정밀도가 낮으면 대본 줄의 말이 added 로 떨어진 것이다.
  옮긴 씬   자리를 옮긴 씬의 줄 가운데 자리가 맞은 비율.
  순서 바뀐 줄  씬 안에서 이웃 줄과 순서를 바꿔 말한 줄 가운데 자리가 맞은 비율(swap_acc).
  없는 줄 짝   잘린 씬 자리에 대본에 없는 말이 있을 때, 그 씬의 줄(영상에 없음)을 맞췄다고 한 수(ghost_paired, 0 이어야 함).
  붙은 글자    맞춘 줄(plain)의 낱말 범위에서 정답 범위 밖에 있는 낱말의 글자 수(over_chars)와, 그 가운데 정답이
            '대본에 없는 말'인 글자 수(over_added). 남은 낱말 붙이기가 다른 말을 얼마나 삼키는지 본다.
  힌트 되풀이  ctx 가 말 없는 자리에 그대로 적은 줄을 받아들인 수(0 이어야 함).
  ctx 고르기   ctx 가 대본 쪽으로 잘못 끌어간 줄에서 ctx 를 고른 수(pull_taken. 같은 줄에서 ctx 가 다른 오류를 고친 것도 섞여 있다),
            그 가운데 plain 이 맞았는데 버린 수(pull_lost, 0 이어야 함), ctx 가 받아쓰기 오류를 고친 줄에서 ctx 를 고른 비율(fix_taken).
  rescue   말했지만 plain 에 없는 줄을 되찾은 비율.
  pass 고르기  plain 과 lv3 가 둘 다 그 줄을 적었고 실제 말과의 글자 닮음이 서로 다른 줄 가운데, 실제 말에 더 가까운 쪽을
            고른 비율(pick_acc, 줄 수 pick_n). pick_alt 는 lv3 를 고른 줄 수. heard_sim 은 고른 글과 실제 말의 글자 닮음 평균,
            heard_sim_plain 은 plain 만 썼을 때의 값.
  rescue added  rescue 낱말로 만든 added 의 낱말 가운데 '말 구간 검출이 버린 대본 밖 말'의 비율(radd_p)과 그런 말을 찾은
            비율(radd_r). radd_dup 은 plain 이 이미 적은 소리를 rescue 낱말로 또 담은 수(0 이어야 함), radd_junk 는 헛소리 수.
  겹친 소리    서로 다른 pass 에서 온 줄끼리 시간이 0.3초 넘게 겹친 곳의 수(cross, 0 에 가까워야 함).
  conf      conf 가 맞은 판정과 틀린 판정을 가르는 정도(conf_auc: 맞은 줄의 conf 가 틀린 줄보다 높을 확률, 0.5 면 못 가름).
            맞은 판정 = 상태가 정답과 같고 자리도 맞음. conf_auc_absent 는 absent 로 본 줄만(정말 없는 줄과 놓친 줄),
            conf_auc_timed 는 same·changed 로 본 줄만. conf_auc_heard 는 same·changed 로 본 줄에서 '자리가 맞고 들린 글이
            실제 말과 같은 줄'과 나머지를 가르는 정도(받아쓰기가 틀린 줄의 conf 가 낮은가).

주의: 가짜 받아쓰기는 실제보다 쉽다(tools/align_synth.py 머리말). 이 수치는 알고리즘·기준값을 견주는 데만 쓴다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import align_synth  # noqa: E402
from sinkprep import align, textnorm, timeline  # noqa: E402

DEFAULT_DOCS = ("fm55664", "fm55890", "fm54455", "fm55244", "fm847721")
OTHERS = ("fm55103", "fm54665", "fm55496")       # 대본에 없는 말·바꿔 말한 낱말을 뽑는 다른 대본
BUCKETS = (("1-3", 1, 3), ("4-7", 4, 7), ("8+", 8, 10 ** 9))


def load_script(doc_id: str) -> dict:
    return json.loads((ROOT / "docs" / doc_id / "script.json").read_text(encoding="utf-8"))


def other_lines(doc_ids, skip: str) -> list[str]:
    out = []
    for d in doc_ids:
        if d == skip:
            continue
        p = ROOT / "docs" / d / "script.json"
        if p.exists():
            out += [e["text"] for e in load_script(d)["elements"]
                    if e["type"] == "dialogue" and 4 <= len(textnorm.key(e["text"])) <= 40]
    return out


def _bucket(n: int) -> str:
    return next(name for name, lo, hi in BUCKETS if lo <= n <= hi)


def _auc(pos: list[float], neg: list[float]) -> float | None:
    """pos 에서 하나, neg 에서 하나를 뽑았을 때 pos 쪽 값이 더 클 확률(같으면 반). 한쪽이 비면 None."""
    if not pos or not neg:
        return None
    vals = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(vals):
        j = i
        while j < len(vals) and vals[j][0] == vals[i][0]:
            j += 1
        rank_sum += (i + j + 1) / 2 * sum(k for _, k in vals[i:j])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def score(truth: dict, res: align.AlignResult, full: bool, asr: dict | None = None) -> dict:
    """한 번 맞춘 결과를 정답과 견준다. full 이면 정답 상태는 '실제로 한 말' 기준, 아니면 plain 낱말 기준.
    asr: 맞출 때 준 asr(dict). 주면 어떤 pass 가 있었는지 보고(대안 pass 가 실제 말을 맞게 적었으면 그것이 정답 상태),
    서로 다른 pass 에서 온 줄끼리 겹친 곳도 센다."""
    item = {it["id"]: it for it in res.items}
    have = {p["id"] for p in asr["passes"]} if asr is not None else {"plain", "ctx", "rescue"}
    pred: dict[str, dict] = {}
    added_words: set[int] = set()
    rescue_words: set[int] = set()
    for op in res.ops:
        it = item[op["item"]]
        if op["script_el"] is None:
            pid, a, b = timeline.op_range(op)
            if pid == "plain":
                added_words.update(range(a, b + 1))
            elif pid == "rescue":
                rescue_words.update(range(a, b + 1))
            continue
        pred[op["script_el"]] = {"op": op["op"], "item": op["item"],
                                 "span": (it["start"], it["end"]) if it["start"] is not None else None,
                                 "pass": timeline.op_range(op)[0] if op["asr"] else None, "heard": it["heard"],
                                 "range": timeline.op_range(op)[1:] if op["asr"] else None,
                                 "words": it.get("words") or []}
    out: dict = {"conf": Counter()}
    pos = {name: Counter() for name, _, _ in BUCKETS}
    absent = Counter()
    moved = Counter()
    ok_status = 0
    heard_right = Counter()
    judged: list[tuple[float, bool, str]] = []
    heard_ok: list[tuple[float, bool]] = []
    pick = Counter()
    heard_sim: Counter = Counter()
    for el, tl in truth["lines"].items():
        p = pred[el]
        if full:
            if not tl["said"]:
                ref = "absent"
            elif tl["vad_miss"]:
                ref = tl["status_rescue"]
            elif tl["plain_right"] or ("ctx" in have and tl["ctx_right"]) or ("lv3" in have and tl.get("alt_right")):
                ref = tl["status_true"]
            else:
                ref = tl["status_plain"]
            here = tl["t"] if tl["said"] else None
        else:
            ref = tl["status_plain"]
            here = tl["t"] if tl["plain"] is not None else None
        hit = here is not None and p["span"] is not None and min(here[1], p["span"][1]) - max(here[0], p["span"][0]) > 0
        b = pos[_bucket(tl["chars"])]
        b["ref"] += here is not None
        b["pred"] += p["span"] is not None
        b["hit"] += hit
        out["conf"][(ref, p["op"] if (p["op"] == "absent" or hit or here is None) else "elsewhere")] += 1
        ok = (ref == p["op"]) and (ref == "absent" or hit)
        ok_status += ok
        c = res.conf.get(p["item"])
        if c is not None:
            judged.append((c, bool(ok), p["op"]))
            if full and p["op"] != "absent":
                heard_ok.append((c, bool(hit) and textnorm.key(p["heard"]) == tl["true_key"]))
        if full and hit and p["pass"] in ("plain", "lv3") and tl["plain"] is not None:
            sp = textnorm.similarity(tl["true_key"], tl["plain_key"]) if tl.get("plain_key") is not None else None
            sa = textnorm.similarity(tl["true_key"], tl["alt_key"]) if tl.get("alt_key") is not None else None
            if sp is not None:
                heard_sim["n"] += 1
                heard_sim["plain"] += sp
                heard_sim["chosen"] += textnorm.similarity(tl["true_key"], textnorm.key(p["heard"]))
            if sp is not None and sa is not None:
                pick["alt"] += p["pass"] == "lv3"
                if abs(sp - sa) > 1e-9:
                    pick["n"] += 1
                    pick["ok"] += (p["pass"] == "lv3") == (sa > sp)
        absent["ref"] += ref == "absent"
        absent["pred"] += p["op"] == "absent"
        absent["both"] += ref == "absent" and p["op"] == "absent"
        if tl["moved"] and here is not None:
            moved["n"] += 1
            moved["hit"] += hit
        if full and tl["plain"] is not None:
            heard_right["n"] += 1
            heard_right["plain"] += tl["plain_right"]
            heard_right["chosen"] += hit and textnorm.key(p["heard"]) == tl["true_key"]
    n = len(truth["lines"])
    out["status_acc"] = ok_status / n if n else 0.0
    ghost = [el for el, tl in truth["lines"].items() if tl.get("ghost")]
    out["ghost_n"] = len(ghost)
    out["ghost_paired"] = sum(pred[el]["op"] != "absent" for el in ghost)
    swapped = [el for el, tl in truth["lines"].items() if tl.get("swapped") and tl["t"] is not None
               and (full or tl["plain"] is not None)]
    out["swap_n"] = len(swapped)
    out["swap_acc"] = (sum(pred[el]["span"] is not None and
                           min(truth["lines"][el]["t"][1], pred[el]["span"][1]) -
                           max(truth["lines"][el]["t"][0], pred[el]["span"][0]) > 0 for el in swapped) / len(swapped)
                       if swapped else None)
    t_added_all = set(truth["added"])
    over = over_added = 0
    for el, tl in truth["lines"].items():
        p = pred[el]
        if p["pass"] != "plain" or tl["plain"] is None or p["range"] is None:
            continue
        a, b = p["range"]
        for w, word in zip(range(a, b + 1), p["words"] if len(p["words"]) == b - a + 1 else []):
            if not tl["plain"][0] <= w <= tl["plain"][1]:
                c = len(textnorm.key(word["w"]))
                over += c
                over_added += c if w in t_added_all else 0
    out["over_chars"] = over
    out["over_added"] = over_added
    for name, c in pos.items():
        out[f"pos_p_{name}"] = c["hit"] / c["pred"] if c["pred"] else None
        out[f"pos_r_{name}"] = c["hit"] / c["ref"] if c["ref"] else None
        out[f"n_{name}"] = c["ref"]
    hit_all = sum(c["hit"] for c in pos.values())
    out["pos_p"] = hit_all / max(1, sum(c["pred"] for c in pos.values()))
    out["pos_r"] = hit_all / max(1, sum(c["ref"] for c in pos.values()))
    out["absent_p"] = absent["both"] / absent["pred"] if absent["pred"] else None
    out["absent_r"] = absent["both"] / absent["ref"] if absent["ref"] else None
    t_added = set(truth["added"])
    both = len(t_added & added_words)
    out["added_p"] = both / len(added_words) if added_words else None
    out["added_r"] = both / len(t_added) if t_added else None
    # 헛소리 묶음 빼기(align.drop_noise): added 로 만들지 않은 낱말 수(drop_words)와 그 가운데 정답이 헛소리가 아닌 낱말 수
    # (drop_real, 0 이어야 함). 정답의 added 에는 plain 헛소리가 들어 있어서 뺀 만큼 added_r 이 내려간다
    junk = {"plain": set(truth.get("halluc") or []), "rescue": set(truth.get("rescue_junk") or [])}
    dropped = []
    for d in res.method.get("noise_dropped") or []:
        (p0, a), (_, b) = (x.rsplit(":", 1) for x in d["asr"])
        dropped += [(p0, i) for i in range(int(a), int(b) + 1)]
    out["drop_words"] = len(dropped)
    out["drop_real"] = sum(i not in junk.get(p0, set()) for p0, i in dropped)
    out["moved_acc"] = moved["hit"] / moved["n"] if moved["n"] else None
    out["moved_n"] = moved["n"]
    out["match"] = res.match
    out["conf_auc"] = _auc([c for c, ok, _ in judged if ok], [c for c, ok, _ in judged if not ok])
    out["conf_auc_absent"] = _auc([c for c, ok, st in judged if ok and st == "absent"],
                                  [c for c, ok, st in judged if not ok and st == "absent"])
    out["conf_auc_timed"] = _auc([c for c, ok, st in judged if ok and st != "absent"],
                                 [c for c, ok, st in judged if not ok and st != "absent"])
    if full:
        out["conf_auc_heard"] = _auc([c for c, ok in heard_ok if ok], [c for c, ok in heard_ok if not ok])
        t_radd = set(truth.get("rescue_added") or [])
        out["radd_n"] = len(t_radd)
        out["radd_words"] = len(rescue_words)
        out["radd_p"] = len(t_radd & rescue_words) / len(rescue_words) if rescue_words else None
        out["radd_r"] = len(t_radd & rescue_words) / len(t_radd) if t_radd else None
        out["radd_dup"] = len(rescue_words & set(truth.get("rescue_dup") or []))
        out["radd_junk"] = len(rescue_words & set(truth.get("rescue_junk") or []))
        out["pick_n"], out["pick_alt"] = pick["n"], pick["alt"]
        out["pick_acc"] = pick["ok"] / pick["n"] if pick["n"] else None
        out["heard_sim_plain"] = heard_sim["plain"] / heard_sim["n"] if heard_sim["n"] else None
        out["heard_sim"] = heard_sim["chosen"] / heard_sim["n"] if heard_sim["n"] else None
        out["cross"] = len(timeline.cross_pass_overlaps(res.ops, timeline.asr_index(asr), 0.3)) if asr is not None else 0
        out["echo_n"] = len(truth["echo"])
        out["echo_taken"] = sum(pred[el]["op"] != "absent" for el in truth["echo"])
        pulled = [el for el, tl in truth["lines"].items() if tl["ctx_pulled"]]
        out["pull_n"] = len(pulled)
        out["pull_taken"] = sum(pred[el]["pass"] == "ctx" and textnorm.key(pred[el]["heard"]) != truth["lines"][el]["true_key"]
                                for el in pulled)
        out["pull_lost"] = sum(pred[el]["pass"] == "ctx" and truth["lines"][el]["plain_right"]
                               and textnorm.key(pred[el]["heard"]) != truth["lines"][el]["true_key"] for el in pulled)
        fixable = [el for el, tl in truth["lines"].items() if tl["plain"] is not None and tl["ctx_right"] and not tl["plain_right"]]
        out["fix_n"] = len(fixable)
        out["fix_taken"] = sum(pred[el]["pass"] == "ctx" for el in fixable) / len(fixable) if fixable else None
        missed = [el for el, tl in truth["lines"].items() if tl["vad_miss"]]
        out["rescue_n"] = len(missed)
        out["rescue_got"] = sum(pred[el]["pass"] == "rescue" for el in missed) / len(missed) if missed else None
        out["heard_right_plain"] = heard_right["plain"] / heard_right["n"] if heard_right["n"] else None
        out["heard_right"] = heard_right["chosen"] / heard_right["n"] if heard_right["n"] else None
    return out


KEYS_PLAIN = ("pos_p", "pos_r", "pos_p_1-3", "pos_r_1-3", "pos_p_4-7", "pos_r_4-7", "pos_p_8+", "pos_r_8+", "status_acc",
              "absent_p", "absent_r", "added_p", "added_r", "moved_acc", "swap_acc", "sec")
KEYS_FULL = ("pos_p", "pos_r", "status_acc", "absent_p", "absent_r", "added_p", "added_r", "moved_acc", "swap_acc",
             "fix_taken", "rescue_got", "heard_right_plain", "heard_right", "conf_auc", "conf_auc_timed", "conf_auc_absent",
             "conf_auc_heard", "sec")
KEYS_TWO = ("pos_p", "pos_r", "status_acc", "absent_p", "absent_r", "added_p", "added_r", "moved_acc", "swap_acc",
            "rescue_got", "heard_right_plain", "heard_right", "heard_sim_plain", "heard_sim", "pick_acc", "radd_p", "radd_r",
            "conf_auc", "conf_auc_timed", "conf_auc_absent", "conf_auc_heard", "sec")
COUNTS_PLAIN = ("ghost_n", "ghost_paired", "over_chars", "over_added", "drop_words", "drop_real")
COUNTS_FULL = ("ghost_n", "ghost_paired", "echo_n", "echo_taken", "pull_n", "pull_taken", "pull_lost", "radd_words", "radd_dup",
               "radd_junk", "cross", "drop_words", "drop_real")
COUNTS_TWO = ("ghost_n", "ghost_paired", "pick_n", "pick_alt", "radd_n", "radd_words", "radd_dup", "radd_junk", "cross",
              "drop_words", "drop_real")
MODES = {"plain": (("plain",), False, KEYS_PLAIN, COUNTS_PLAIN),             # 방식: (줄 pass, 실제 말 기준인가, 지표, 수)
         "full": (("plain", "ctx", "rescue"), True, KEYS_FULL, COUNTS_FULL),
         "base": (("plain", "rescue"), True, KEYS_TWO, COUNTS_TWO),
         "two": (("plain", "lv3", "rescue"), True, KEYS_TWO, COUNTS_TWO),
         "all": (("plain", "lv3", "ctx", "rescue"), True, KEYS_TWO, COUNTS_TWO + ("echo_n", "echo_taken", "pull_n", "pull_lost"))}


def _fmt(v) -> str:
    return "  -  " if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=",".join(DEFAULT_DOCS))
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--rate", action="append", default=[])
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--alt", action="store_true", help="대안 pass 가 든 영화로 base·two 방식도 채점")
    args = ap.parse_args(argv)
    for kv in args.set:
        k, v = kv.split("=", 1)
        if not hasattr(align, k):
            raise SystemExit(f"sinkprep.align 에 없는 상수: {k}")
        setattr(align, k, type(getattr(align, k))(float(v)) if not isinstance(getattr(align, k), tuple) else tuple(json.loads(v)))
    rates = {}
    for kv in args.rate:
        k, v = kv.split("=", 1)
        if k not in align_synth.RATES:
            raise SystemExit(f"align_synth.RATES 에 없는 확률: {k}")
        rates[k] = float(v)
    docs = [d for d in args.docs.split(",") if d]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    modes = ("plain", "full") + (("base", "two", "all") if args.alt else ())
    rows = {m: [] for m in modes}
    conf = {m: Counter() for m in modes}
    for d in docs:
        if not (ROOT / "docs" / d / "script.json").exists():
            print(f"{d}: script.json 없음, 건너뜀")
            continue
        script = load_script(d)
        others = other_lines(OTHERS + tuple(docs), d)
        for seed in seeds:
            movies = {False: align_synth.make_movie(script, others, seed, rates)}
            if args.alt:
                movies[True] = align_synth.make_movie(script, others, seed, rates, alt=True)
            for mode in modes:
                ids, full, keys, counts = MODES[mode]
                mv = movies[mode in ("base", "two", "all")]
                truth, asr = mv["truth"], mv["asr"]
                a = align_synth.only_passes(asr, ids)
                t0 = time.perf_counter()
                res = align.build_ops(script, a, truth["duration"])
                sec = time.perf_counter() - t0
                r = score(truth, res, full, a)
                r.update(doc=d, seed=seed, sec=sec, lines=len(truth["lines"]), words=len(asr["passes"][0]["words"]))
                conf[mode].update(r.pop("conf"))
                rows[mode].append(r)
                if not args.quiet:
                    print(f"{d} seed={seed} {mode:5s} 줄 {r['lines']} 낱말 {r['words']} | " +
                          " ".join(f"{k}={_fmt(r.get(k))}" for k in keys + counts), flush=True)
    for mode in modes:
        keys = MODES[mode][2]
        rs = rows[mode]
        if not rs:
            continue
        print(f"\n== {mode}: 문서 {len({r['doc'] for r in rs})}개 × 씨앗 {len(seeds)}개 = {len(rs)}번 ==")
        print(f"{'지표':22s} {'평균':>7s} {'최악':>7s}")
        for k in keys:
            vals = [r[k] for r in rs if r.get(k) is not None]
            if not vals:
                continue
            worst = max(vals) if k == "sec" else min(vals)
            print(f"{k:22s} {sum(vals) / len(vals):7.3f} {worst:7.3f}")
        for k in MODES[mode][3]:
            print(f"{k:22s} 합 {sum(r[k] for r in rs)}, 최대 {max(r[k] for r in rs)}")
        print("혼동 행렬(정답 -> 결과, 줄 수 합):")
        for ref in ("same", "changed", "absent"):
            print("  " + ref.ljust(8) + " ".join(f"{p}={conf[mode][(ref, p)]}" for p in ("same", "changed", "absent", "elsewhere")))
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "set": args.set, "rate": args.rate}, ensure_ascii=False,
                                              indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
