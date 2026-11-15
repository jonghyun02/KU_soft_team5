"""항목 차례와 시각 추정. align 단계와 final 단계가 같은 함수를 불러 같은 차례·같은 id 를 얻는다.

align 은 ``build_timeline`` 으로 ops 의 ``item`` 을 정하고, final 은 같은 입력(script.json, align.json 의 ops,
asr.json, 영상 길이)으로 다시 불러 똑같은 항목 뼈대를 얻은 뒤 나머지 칸(화자, diffs, ko 등)을 채운다.
순수 함수뿐이고 파일을 읽거나 쓰지 않는다.

규칙
- 대사 시각: same·changed·added 는 낱말 범위에서 start = 가장 이른 낱말 s, end = 가장 늦은 낱말 e. absent 는 시각 없음.
- 낱말 시각 바로잡기(``op_words``): 받아쓰기 낱말은 말한 차례다. 한 줄 안에서 시작 시각이 앞 낱말의 시작보다 이르면
  앞 낱말의 시작으로 올리고, 끝이 시작보다 이르면 끝을 시작에 맞춘다. 영상 길이를 넘은 시각이 검사의 허용 오차
  (DURATION_TOL 초)에 닿으면 영상 끝으로 당긴다. 줄 전체가 한 순간이면 마지막 낱말의 끝을 MIN_DUR 초 늘린다(start < end).
  낱말 글자와 차례는 그대로다. 바로잡은 낱말 수는 항목의 ``time_fix`` 에 남는다.
- 영화 흐름에서 벗어난 대사(moved): 대본 순서와 시각 순서가 함께 늘어나는 가장 긴 사슬 밖의 대본 대사.
- added 줄의 scene: 시각으로 앞뒤에 있는 대본 대사(시각 있는 것) P, N 의 씬 가운데 하나다. 둘이 같은 씬이면 그 씬.
  다르면 쉼으로 잰 거리를 견준다: P 의 끝에서 그 줄 시작까지 말이 없던 시간의 합이, 그 줄 끝에서 N 의 시작까지 말이
  없던 시간의 합보다 길지 않으면 P 의 씬, 아니면 N 의 씬(그 사이의 다른 added 줄이 말한 시간은 세지 않는다. 그래서
  앞 대사에 잇달아 붙어 나온 말들은 앞 씬에 남는다. added 가 하나뿐이면 '앞 대사 끝과 뒤 대사 시작 가운데 가까운 쪽').
  P 가 없으면 N 의 씬, N 이 없으면 P 의 씬, 둘 다 없으면 None. 씬 머리·지문의 추정 시각은 보지 않는다.
- 대사 아닌 요소(scene, action, onscreen, transition)는 추정 시각(timing "estimated", end 는 None). 대본 순서로 앞뒤의
  시각 있는 대사 a, b 사이를 글 길이에 비례해 나눈다(첫 요소가 a 의 끝 시각. 맨 앞·맨 뒤 끝점은 0 과 영상 길이).
  그런 다음 한 씬의 요소가 그 씬의 대사 덩어리와 붙어 다니게 자리를 좁힌다.
  · a 와 같은 씬의 요소는 a 뒤에, 시각상 다음에 오는 다른 씬 대사 쪽 말(거기 붙은 added 포함)이 시작하기 전에 둔다.
  · 그 밖의 요소는 b 앞에, 시각상 앞에 있는 다른 씬 대사 쪽 말(거기 붙은 added 포함)이 끝난 뒤에 둔다. 새 씬의 첫
    요소(씬 머리)는 b 쪽 added 가 시작하기 전이다. 그래서 a 와 b 사이에 자리를 옮긴 다른 씬의 대사가 끼어 있어도 한 씬의
    씬 머리와 지문이 그 덩어리 양쪽으로 갈리지 않고, 앞 씬에 붙은 added 뒤·뒤 씬에 붙은 added 앞에 씬 머리가 놓인다.
    ('시각상 앞·뒤의 다른 씬 대사'는 시각 순으로 잇달아 있는 같은 씬 대사들을 한 덩어리로 보고 그 덩어리 밖에서 찾는다.)
  · a 나 b 가 흐름에서 벗어난 대사면(옮겨진 덩어리의 경계) 나누지 않고 같은 씬 쪽 대사에 붙인다.
- 차례: 시각 있는 항목을 start 순으로. 같은 시각이면 앞 대사에 딸린 요소, 어느 대사에도 딸리지 않은 요소, 뒤 대사에
  딸린 요소와 그 대사, added 차례다. 시각 없는 항목(absent)은 E 번호 오름차순으로 하나씩 '이미 놓인 항목 가운데 E 번호가
  자기보다 작은 것 중 가장 큰 항목 바로 뒤'에 넣는다(없으면 맨 앞). validate 의 D7 검사를 통과하는 넣기 규칙이다.
- id: 차례대로 I0001 부터.
- 씬 구간(``scene_spans``): 그 씬의 시각 있는 대본 대사와 그 씬으로 분류한 added 줄의 시각 범위.
- 한 번 난 소리는 한 줄에만: ``shared_words`` 는 같은 pass 의 같은 낱말을 두 op 이 가진 곳(확실한 잘못. validate 가 오류로
  본다), ``cross_pass_overlaps`` 는 서로 다른 pass 에서 온 두 줄의 시간 구간이 SAME_SOUND_S 초 넘게 겹친 곳(의심.
  final 이 quality 와 항목 수준 문제로 남긴다)을 알려 준다.
"""
from __future__ import annotations

import bisect
import math

VERSION = "0.2.0"         # 차례·시각·씬 규칙이 바뀌면 올린다(같은 ops 에서 다른 item id 가 나온다: 단계 지문에 넣는 값)
ITEM_TYPES = ("scene", "action", "dialogue", "onscreen", "transition")
MIN_DUR = 0.02            # 대사 start < end 를 지키려고 늘리는 길이(초)
SAME_SOUND_S = 0.6        # 서로 다른 pass 에서 온 두 줄의 시간 구간이 이보다 길게 겹치면 같은 소리를 두 번 담았을 수 있다
#                           (추격자의 turbo·large-v3 낱말 2,865짝에서 pass 사이 줄 경계는 0.5초까지 어긋났다: 그보다 커야 한다)
DURATION_TOL = 0.5        # 시각이 영상 길이를 넘어도 되는 허용 오차(초). validate 의 D2 가 같은 값을 쓴다
FAR_BACK_S = 1.0          # 낱말 시작 시각이 이보다 크게 뒤집혀 있었으면 바로잡아도 믿기 어려운 줄로 본다
INF = math.inf


def _enum(el_id: str) -> int:
    return int(el_id[1:])


def asr_index(asr: dict) -> dict[str, list[dict]]:
    """{pass id: 낱말 목록(번호 i 순)}."""
    return {p["id"]: sorted(p.get("words") or [], key=lambda w: w["i"]) for p in asr.get("passes") or []}


def op_range(op: dict) -> tuple[str, int, int] | None:
    """op 의 asr 참조 ["plain:12", "plain:15"] -> ("plain", 12, 15). 없으면 None."""
    rng = op.get("asr")
    if not rng:
        return None
    p0, i0 = rng[0].rsplit(":", 1)
    p1, i1 = rng[1].rsplit(":", 1)
    if p0 != p1 or int(i0) > int(i1):
        raise ValueError(f"asr 범위가 잘못됨: {rng}")
    return p0, int(i0), int(i1)


def fix_words(src: list[dict], duration: float | None = None) -> tuple[list[dict], dict]:
    """받아쓰기 낱말 목록(말한 차례)을 final.json 의 낱말 꼴 ``{"w", "s", "e", "p"}`` 로 바꾸고 시각을 바로잡는다.

    글자가 빈 낱말은 뺀다(final 은 w 1자 이상). 시각은 소수 셋째 자리. 바로잡는 규칙은 넷뿐이고 낱말 글자·차례는 그대로다.
      1. duration 을 주었을 때, 영상 길이를 넘은 시각이 검사의 허용 오차에 닿으면(영상 길이 + DURATION_TOL - 2 × MIN_DUR
         초과. 한 순간짜리 줄을 4번 규칙으로 늘려도 허용 오차 안에 남게 조금 일찍 자른다) 영상 길이로 당긴다.
      2. 시작 시각이 앞 낱말의 시작보다 이르면 앞 낱말의 시작으로 올린다(받아쓰기 묶음이 겹쳐 같은 말이 두 번 적히거나
         낱말 시각이 흔들린 곳. 한 군데 때문에 영화 전체의 최종본을 못 만들지 않게 한다).
      3. 끝이 시작보다 이르면 끝을 시작에 맞춘다.
      4. 줄 전체가 한 순간이면(가장 이른 s == 가장 늦은 e) 가장 늦게 끝나는 낱말의 e 를 MIN_DUR 초 늘린다.
    돌려주는 것: (낱말 목록, {"fixed": 시각이 바뀐 낱말 수, "back": 시작 시각을 올린 가장 큰 폭(초),
                              "late": 영상 길이 밖이던 낱말 수}).
    """
    limit = end_at = None
    if duration:
        end_at = round(float(duration), 3)
        limit = float(duration) + DURATION_TOL - 2 * MIN_DUR
    out: list[dict] = []
    raw: list[tuple[float, float]] = []
    floor = None
    back = 0.0
    late = 0
    for w in src:
        text = str(w.get("w") or "").strip()
        if not text:
            continue
        s0, e0 = round(float(w["s"]), 3), round(float(w["e"]), 3)
        s, e = s0, e0
        outside = limit is not None and (s > limit or e > limit)
        if outside:
            late += 1
            s, e = min(s, end_at), min(e, end_at)
        if floor is not None and s < floor:
            if not outside:                       # 영상 끝으로 당긴 낱말을 다시 올린 폭은 뒤집힘으로 세지 않는다
                back = max(back, floor - s)
            s = floor
        floor = s
        e = max(s, e)
        raw.append((s0, e0))
        out.append({"w": text, "s": s, "e": e, "p": w.get("p")})
    if out:
        start = min(w["s"] for w in out)
        end = max(w["e"] for w in out)
        if end <= start:
            last = max(range(len(out)), key=lambda k: (out[k]["e"], k))
            out[last]["e"] = round(start + MIN_DUR, 3)
    fixed = sum((w["s"], w["e"]) != r for w, r in zip(out, raw))
    return out, {"fixed": fixed, "back": round(back, 3), "late": late}


def op_words_info(index: dict[str, list[dict]], op: dict, duration: float | None = None) -> tuple[list[dict], dict]:
    """op 이 가리키는 낱말(``fix_words`` 로 바로잡은 것)과 바로잡은 기록. 범위가 없으면 ([], 기록 0)."""
    r = op_range(op)
    if r is None:
        return [], {"fixed": 0, "back": 0.0, "late": 0}
    pid, i0, i1 = r
    return fix_words(index[pid][i0:i1 + 1], duration)


def op_words(index: dict[str, list[dict]], op: dict, duration: float | None = None) -> list[dict]:
    """op 이 가리키는 낱말을 final.json 의 낱말 꼴 ``{"w", "s", "e", "p"}`` 로(번호 i 는 뗀다). 시각은 ``fix_words`` 로
    바로잡는다. duration(영상 길이)을 주면 영상 길이 밖의 시각도 당긴다(final.json 의 words 는 영상 길이를 준 결과다)."""
    return op_words_info(index, op, duration)[0]


def suspect(info: dict) -> bool:
    """시각을 바로잡았지만 믿기 어려운 줄인가: 영상 길이 밖의 낱말이 있었거나 시작 시각이 FAR_BACK_S 넘게 뒤집혀 있었다."""
    return bool(info.get("late")) or float(info.get("back") or 0.0) > FAR_BACK_S


def _op_rows(ops: list[dict]) -> list[tuple[str, int, int, str]]:
    """낱말 참조가 멀쩡한 op 마다 (pass, 처음, 끝, 항목 id 또는 'ops/번호')."""
    rows = []
    for k, op in enumerate(ops):
        try:
            r = op_range(op)
        except (ValueError, AttributeError, TypeError, IndexError):
            continue
        if r is not None:
            rows.append((r[0], r[1], r[2], str(op.get("item") or f"ops/{k}")))
    return rows


def shared_words(ops: list[dict]) -> list[str]:
    """같은 받아쓰기 낱말을 두 op 이 가진 곳의 설명 목록(없으면 빈 목록): 같은 pass 의 낱말 범위가 서로 겹친다.

    한 번 난 소리는 한 줄에만 들어간다는 약속의 가장 분명한 꼴이다. align 이 끝에서 확인하고 validate 가 오류로 본다.
    설명에는 항목 id 와 낱말 번호만 적는다(대사 글을 넣지 않는다)."""
    out = []
    last: dict[str, tuple] = {}
    for row in sorted(_op_rows(ops)):
        prev = last.get(row[0])
        if prev is not None and row[1] <= prev[2]:
            out.append(f"{prev[3]} 와 {row[3]} 가 같은 낱말을 가짐({row[0]}:{row[1]}~{min(prev[2], row[2])})")
        if prev is None or row[2] > prev[2]:
            last[row[0]] = row
    return out


def cross_pass_overlaps(ops: list[dict], index: dict[str, list[dict]],
                        min_s: float = SAME_SOUND_S) -> list[tuple[str, str, float]]:
    """서로 다른 pass 에서 온 두 op 의 시간 구간이 min_s 초 넘게 겹친 곳 [(앞 항목, 뒤 항목, 겹친 초)] (시작 시각 순).

    같은 소리를 두 pass 의 글로 두 번 담았을 수 있는 곳이다. 시각만으로는 확정할 수 없다: pass 마다 낱말 경계가 다르고
    (그래서 min_s 보다 짧은 겹침은 보지 않는다) 두 사람이 겹쳐 말한 곳을 두 pass 가 한 사람씩 받아썼을 수도 있다.
    그래서 오류로 막지 않고 final 이 quality 와 항목 수준 문제로 남긴다. 구간은 ``op_words`` 로 바로잡은 낱말의 범위다."""
    rows = []
    for pid, i0, i1, item in _op_rows(ops):
        if i1 >= len(index.get(pid) or ()):
            continue
        ws, _ = fix_words(index[pid][i0:i1 + 1])
        if ws:
            rows.append((min(w["s"] for w in ws), max(w["e"] for w in ws), pid, item))
    out = []
    if len({r[2] for r in rows}) < 2:
        return out
    live: list[tuple] = []
    for row in sorted(rows):
        live = [x for x in live if x[1] > row[0] + min_s]
        for x in live:
            ov = min(x[1], row[1]) - row[0]
            if x[2] != row[2] and ov > min_s:
                out.append((x[3], row[3], round(ov, 3)))
        live.append(row)
    return out


def double_claims(ops: list[dict], index: dict[str, list[dict]], tol: float = SAME_SOUND_S) -> list[str]:
    """한 번 난 소리를 두 줄이 가졌거나 가졌을 수 있는 곳의 설명 목록: ``shared_words`` (같은 pass 의 같은 낱말, 확실함)
    뒤에 ``cross_pass_overlaps`` (다른 pass 의 줄끼리 tol 초 넘게 겹침, 의심)를 잇는다. 대사 글은 넣지 않는다.
    운영 차례는 두 함수를 따로 부른다(align 과 validate 의 C2 가 shared_words, final 이 cross_pass_overlaps). 이것은 둘을 한 번에
    보는 조사·시험용이다."""
    return shared_words(ops) + [f"{a} 와 {b} 의 시간 구간이 {sec:.3f}초 겹침(서로 다른 pass: 같은 소리가 두 줄에 들어갔을 수 있음)"
                                for a, b, sec in cross_pass_overlaps(ops, index, tol)]


def _in_order(starts: list[float]) -> list[bool]:
    """대본 순서로 늘어놓은 시작 시각에서, 줄어들지 않는 가장 긴 사슬에 든 자리 표시."""
    tails: list[float] = []            # 길이 k+1 사슬의 가장 작은 끝값
    tail_at: list[int] = []
    prev = [-1] * len(starts)
    for k, t in enumerate(starts):
        j = bisect.bisect_right(tails, t)
        if j == len(tails):
            tails.append(t)
            tail_at.append(k)
        else:
            tails[j] = t
            tail_at[j] = k
        prev[k] = tail_at[j - 1] if j else -1
    keep = [False] * len(starts)
    k = tail_at[-1] if tail_at else -1
    while k >= 0:
        keep[k] = True
        k = prev[k]
    return keep


def _split_gap(p: dict | None, n: dict | None, xs: list[dict]) -> tuple[int, float, float]:
    """시각상 이웃한 대본 대사 p, n 사이의 added 줄 xs(시각 순)를 가른다.

    돌려주는 것: (xs 가운데 p 쪽인 앞쪽 줄 수, p 쪽 말이 끝난 시각, n 쪽 말이 시작하는 시각).
    뒤의 두 값은 씬이 바뀌는 자리의 쉼이다. p 와 n 이 같은 씬이면 가를 것이 없어 (-INF, INF), p 가 없으면 앞 끝이 -INF,
    n 이 없으면 뒤 끝이 INF 다.
    """
    if p is None and n is None:
        return 0, -INF, INF
    if p is None:
        return 0, -INF, (xs[0]["start"] if xs else n["start"])
    if n is None:
        return len(xs), max([p["end"]] + [x["end"] for x in xs]), INF
    if p["scene"] == n["scene"]:
        return len(xs), -INF, INF
    pauses = []                                  # xs[i] 앞의 쉼, 마지막은 n 앞의 쉼
    at = p["end"]
    for x in xs:
        pauses.append(max(0.0, x["start"] - at))
        at = max(at, x["end"])
    pauses.append(max(0.0, n["start"] - at))
    before, after = 0.0, sum(pauses)
    cut = 0
    for i in range(len(xs)):
        before += pauses[i]
        after -= pauses[i]
        if before > after + 1e-9:                # 뒤 대사 쪽이 더 가깝다: 여기서부터 n 의 씬
            break
        cut = i + 1
    lo = max([p["end"]] + [x["end"] for x in xs[:cut]])
    hi = xs[cut]["start"] if cut < len(xs) else n["start"]
    return cut, min(lo, hi), hi


def build_timeline(script: dict, ops: list[dict], asr: dict, video_duration: float) -> list[dict]:
    """모든 대본 요소(scene, action, dialogue, onscreen, transition)와 added 줄의 차례·시각·id 를 정한다.

    입력
      script          script.json 을 읽은 dict. ``elements`` 가운데 front·unknown 은 넣지 않는다.
      ops             align.json 의 ops 꼴 목록. 칸 ``op``(same|changed|added|absent), ``script_el``, ``asr``
                      (["pass:i", "pass:j"] 또는 None)만 읽는다. ``item`` 은 읽지 않는다(여기서 정한다).
                      대본의 대사 요소마다 op 이 정확히 하나 있어야 하고(없거나 둘이면 ValueError),
                      ``script_el`` 이 None 인 op 은 모두 added 여야 한다.
      asr             asr.json 을 읽은 dict(낱말 시각의 출처).
      video_duration  영상 길이(초). 맨 뒤 요소의 추정 시각 끝점이고, 이 길이 밖의 낱말 시각을 당기는 기준이다.

    돌려주는 것: 항목 뼈대 목록(final.json items 의 차례 그대로). 항목마다 dict:
      "id"      "I0001" 부터 차례대로
      "type"    scene | action | dialogue | onscreen | transition
      "src_el"  대본 요소 id("E0123"), added 줄은 None
      "scene"   대본 요소의 scene, added 줄은 시각으로 앞뒤 대본 대사의 씬 가운데 하나(없으면 None)
      "start"   초(소수 셋째 자리) 또는 None
      "end"     대사는 초 또는 None(absent), 대사 아닌 항목은 늘 None
      "timing"  대사 same·changed·added 는 "aligned", absent 는 "none", 대사 아닌 항목은 "estimated"
      "op"      대사 항목만: 이 항목에 해당하는 ops 안 번호(정수). 대사 아닌 항목은 None
      "near"    그 항목이 딸린 대본 대사의 요소 id: added 줄은 씬을 받아 온 대사, 대사 아닌 항목은 붙어 다니는 대사.
                대본 대사 자신과 어느 대사에도 딸리지 않은 항목은 None
      대사 항목에만 더 있는 칸:
      "status"  same | changed | added | absent (op 의 ``op`` 값)
      "words"   [{"w", "s", "e", "p"}] (final.json words 의 꼴 그대로, absent 는 빈 목록)
      "heard"   words 의 w 를 공백으로 이은 글, absent 는 None
      "moved"   참이면 영화 흐름에서 벗어난 자리에서 들린 대본 대사(대본 순서와 시각 순서가 어긋남)
      "time_fix" 낱말 시각을 바로잡은 기록 {"fixed", "back", "late"} (``fix_words``)

    같은 입력이면 늘 같은 결과다. ``ops[k]["item"] = items[..]["id"]`` 를 붙이려면 ``attach_items`` 를 쓴다.
    """
    index = asr_index(asr)
    pass_order = {p["id"]: k for k, p in enumerate(asr.get("passes") or [])}
    by_el: dict[str, int] = {}
    added: list[int] = []
    for k, op in enumerate(ops):
        el = op.get("script_el")
        if el is None:
            if op.get("op") != "added":
                raise ValueError(f"ops[{k}]: script_el 이 없는데 op={op.get('op')!r}")
            added.append(k)
        elif el in by_el:
            raise ValueError(f"ops[{k}]: 대본 요소 {el} 에 op 이 둘")
        else:
            by_el[el] = k
    els = sorted((e for e in script.get("elements") or [] if e.get("type") in ITEM_TYPES), key=lambda e: _enum(e["id"]))
    duration = max(0.0, float(video_duration or 0.0))

    def dialogue_item(k: int, src_el: str | None, scene) -> dict:
        op = ops[k]
        words, info = op_words_info(index, op, duration or None) if op.get("op") != "absent" else ([], None)
        if op.get("op") != "absent" and not words:
            raise ValueError(f"ops[{k}]: op={op.get('op')!r} 인데 낱말이 없음")
        it = {"id": None, "type": "dialogue", "src_el": src_el, "scene": scene,
              "start": None, "end": None, "timing": "none", "op": k, "near": None, "status": op.get("op"),
              "words": words, "heard": None, "moved": False,
              "time_fix": info or {"fixed": 0, "back": 0.0, "late": 0}}
        if words:
            it["start"] = min(w["s"] for w in words)
            it["end"] = max(w["e"] for w in words)
            it["timing"] = "aligned"
            it["heard"] = " ".join(w["w"] for w in words)
        return it

    items: list[dict] = []
    for e in els:
        if e["type"] == "dialogue":
            if e["id"] not in by_el:
                raise ValueError(f"대본 대사 요소 {e['id']} 에 op 이 없음")
            items.append(dialogue_item(by_el[e["id"]], e["id"], e.get("scene")))
        else:
            items.append({"id": None, "type": e["type"], "src_el": e["id"], "scene": e.get("scene"),
                          "start": None, "end": None, "timing": "estimated", "op": None, "near": None,
                          "_weight": max(1, len(str(e.get("text") or "").strip()))})
    unknown = sorted(set(by_el) - {e["id"] for e in els if e["type"] == "dialogue"})
    if unknown:
        raise ValueError(f"ops 가 가리키는 대본 대사 요소가 없음: {unknown[:5]}")

    # 영화 흐름(대본 순서와 시각이 함께 늘어나는 가장 긴 사슬)에서 벗어난 대사
    timed = [k for k, it in enumerate(items) if it["type"] == "dialogue" and it["start"] is not None]
    for k, ok in zip(timed, _in_order([items[k]["start"] for k in timed])):
        items[k]["moved"] = not ok

    # 시각 순으로 늘어놓은 대본 대사와 그 사이 틈. 틈마다 added 줄을 앞 대사 쪽과 뒤 대사 쪽으로 가른다(added 의 씬)
    said = sorted((items[k] for k in timed), key=lambda it: (it["start"], _enum(it["src_el"])))
    j = 0
    while j < len(said):                          # 시각상 잇달아 있는 같은 씬의 대사들이 한 덩어리: _at = (첫 자리, 끝 자리)
        k = j
        while k + 1 < len(said) and said[k + 1]["scene"] == said[j]["scene"]:
            k += 1
        for it in said[j:k + 1]:
            it["_at"] = (j, k)
        j = k + 1
    starts = [it["start"] for it in said]
    extra = []
    for k in added:
        it = dialogue_item(k, None, None)
        r = op_range(ops[k])
        extra.append(((it["start"], r[1], pass_order.get(r[0], len(pass_order))), it))
    extra.sort(key=lambda t: t[0])
    in_gap: list[list[dict]] = [[] for _ in range(len(said) + 1)]       # in_gap[j]: said[j-1] 과 said[j] 사이
    for _, it in extra:
        in_gap[bisect.bisect_right(starts, it["start"])].append(it)
    pause: list[tuple[float, float]] = []                               # 틈마다 씬이 바뀌는 자리의 쉼
    for j, xs in enumerate(in_gap):
        p = said[j - 1] if j else None
        n = said[j] if j < len(said) else None
        cut, lo, hi = _split_gap(p, n, xs)
        pause.append((lo, hi))
        for i, x in enumerate(xs):
            owner = p if i < cut else n
            if owner is not None:
                x["scene"], x["near"] = owner["scene"], owner["src_el"]

    # 대사 아닌 요소의 추정 시각. 같은 시각에서의 차례를 정하는 열쇠 _tie = (갈래, 딸린 대사의 E 번호)
    AFTER, LOOSE, BEFORE = 0, 1, 2

    def put(k: int, t: float, tie: int, owner: dict | None) -> None:
        items[k]["start"] = round(max(0.0, t), 3)
        items[k]["near"] = owner["src_el"] if owner is not None else None
        items[k]["_tie"] = (tie, _enum(owner["src_el"]) if owner is not None else 0)

    def place(run: list[int], a: dict | None, b: dict | None) -> None:
        if not run:
            return
        # 시각상 a 의 덩어리 바로 뒤의 틈과 b 의 덩어리 바로 앞의 틈(b 가 없으면 맨 끝 틈)에서 씬이 바뀌는 자리의 쉼
        after_a = pause[a["_at"][1] + 1] if a is not None else (-INF, INF)
        before_b = pause[b["_at"][0]] if b is not None else pause[len(said)]

        def is_a(k: int) -> bool:
            return a is not None and items[k]["scene"] is not None and items[k]["scene"] == a["scene"]

        def is_b(k: int) -> bool:
            return b is not None and items[k]["scene"] is not None and items[k]["scene"] == b["scene"]

        if (a is not None and a["moved"]) or (b is not None and b["moved"]):
            for k in run:                             # 자리를 옮긴 덩어리의 경계: 나누지 않고 같은 씬 쪽 대사에 붙인다
                if is_a(k):
                    put(k, a["end"], AFTER, a)
                elif is_b(k):
                    put(k, min(b["start"], before_b[1]), BEFORE, b)
                elif a is not None and (b is None or (b["moved"] and not a["moved"])):
                    put(k, max(a["end"], after_a[0]), LOOSE, None)
                else:
                    put(k, min(b["start"], before_b[1]), LOOSE, None)
            return
        lo = a["end"] if a is not None else 0.0
        hi = b["start"] if b is not None else max(lo, duration)
        lo = min(lo, hi)                              # 앞 대사가 뒤 대사와 겹치면 뒤 대사의 시작에 모은다
        total = sum(items[k]["_weight"] for k in run)
        acc = 0
        left_a = met_b = False
        for k in run:
            t = lo + (hi - lo) * acc / total
            acc += items[k]["_weight"]
            if not left_a and is_a(k):                # a 의 씬에 남은 요소: 다음 대사 쪽 말이 시작하기 전
                t, tie, owner = min(t, after_a[1]), AFTER, a
            else:
                left_a = True
                if is_b(k):                           # b 의 씬: 앞 대사 쪽 말이 끝난 뒤, 첫 요소는 b 쪽 말이 시작하기 전
                    t = max(t, before_b[0]) if met_b else min(max(t, before_b[0]), before_b[1])
                    met_b, tie, owner = True, BEFORE, b
                else:                                 # 시각 있는 대사가 없는 씬: 씬이 바뀌는 쉼 안
                    t = max(t, before_b[0]) if met_b else min(max(t, before_b[0]), before_b[1])
                    tie, owner = LOOSE, None
            put(k, min(hi, max(lo, t)), tie, owner)

    prev_d: dict | None = None
    run: list[int] = []
    for k, it in enumerate(items):
        if it["type"] != "dialogue":
            run.append(k)
        elif it["start"] is not None:
            place(run, prev_d, it)
            run = []
            prev_d = it
    place(run, prev_d, None)

    # 차례: 시각 있는 항목은 start 순. 같은 시각이면 앞 대사에 딸린 요소 → 딸린 데 없는 요소 → 뒤 대사에 딸린 요소와
    # 그 대사(대본 순서) → added(낱말 번호 순)
    keyed = []
    for it in items:
        if it["start"] is None:
            continue
        if it["type"] == "dialogue":
            keyed.append(((it["start"], BEFORE, _enum(it["src_el"]), 1, 0), it))
        else:
            tie, owner = it["_tie"]
            keyed.append(((it["start"], tie, owner, 0, _enum(it["src_el"])), it))
    for (start, w0, pk), it in extra:
        keyed.append(((start, 3, w0, pk, 0), it))
    keyed.sort(key=lambda t: t[0])
    ordered = [t[1] for t in keyed]

    # 시각 없는 항목(absent): E 번호가 자기보다 작은 시각 있는 대본 항목 가운데 가장 큰 것 뒤에, E 번호 순으로
    # (하나씩 넣는 규칙과 같은 결과다: 앞서 넣은 absent 는 늘 같은 뿌리 뒤에 차례로 붙는다)
    placed = sorted(_enum(it["src_el"]) for it in ordered if it["src_el"] is not None)
    after: dict[int | None, list[dict]] = {}
    for it in items:
        if it["start"] is None:
            n = _enum(it["src_el"])
            j = bisect.bisect_left(placed, n)
            after.setdefault(placed[j - 1] if j else None, []).append(it)
    out: list[dict] = list(after.get(None, []))
    for it in ordered:
        out.append(it)
        if it["src_el"] is not None:
            out.extend(after.get(_enum(it["src_el"]), []))

    for n, it in enumerate(out, 1):
        it["id"] = f"I{n:04d}"
        for key in ("_weight", "_tie", "_at"):
            it.pop(key, None)
    return out


def attach_items(ops: list[dict], items: list[dict]) -> None:
    """items 의 id 를 ops 의 ``item`` 칸에 적는다(제자리 수정). 모든 op 이 항목 하나에 대응해야 한다."""
    seen = 0
    for it in items:
        if it.get("op") is not None:
            ops[it["op"]]["item"] = it["id"]
            seen += 1
    if seen != len(ops):
        raise ValueError(f"항목에 대응하지 않는 op 이 있음: ops {len(ops)}개, 대사 항목 {seen}개")


def scene_spans(script: dict, items: list[dict], video_duration: float | None = None) -> list[dict]:
    """씬마다 구간 ``{"id", "start", "end", "timing"}`` (script.json scenes 의 차례 그대로. final.json scenes 에 넣는다).

    - 시각 있는 대본 대사가 있는 씬: 그 대사들과 그 씬으로 분류한 added 줄의 범위(첫 줄 시작 ~ 마지막 줄 끝).
      영화 흐름에서 벗어난 대사(moved)와 거기 딸린 added 줄은 흐름 안 대사가 하나라도 있으면 범위에서 뺀다.
      씬 머리·지문의 추정 시각은 넣지 않는다(들린 말이 있는 범위만).
    - 대사가 없거나 모두 absent 인 씬: 그 씬의 대사 아닌 항목들의 추정 시각 범위.
    - 시각 있는 항목이 하나도 없으면 start·end None, timing "none". 그 밖에는 모두 "estimated".
    구간은 항목 시각에서 그대로 온다(validate 의 D9: 구간의 start 는 그 씬 항목들의 시각 범위 안). video_duration 을 주면
    영상 길이 + DURATION_TOL 을 넘는 값만 자른다(``build_timeline`` 에 같은 영상 길이를 준 항목이라면 넘는 값이 없다).
    """
    spoken: dict[str, list[dict]] = {}
    extra: dict[str, list[dict]] = {}
    others: dict[str, list[float]] = {}
    for it in items:
        sc = it.get("scene")
        if sc is None or it.get("start") is None:
            continue
        if it["type"] != "dialogue":
            others.setdefault(sc, []).append(it["start"])
        elif it.get("src_el") is not None:
            spoken.setdefault(sc, []).append(it)
        else:
            extra.setdefault(sc, []).append(it)
    out = []
    for sc in script.get("scenes") or []:
        sid = sc["id"]
        start = end = None
        if sid in spoken:
            ds = [it for it in spoken[sid] if not it.get("moved")] or spoken[sid]
            keep = {it["src_el"] for it in ds}
            ds = ds + [it for it in extra.get(sid, []) if it.get("near") in keep]
            start, end = min(it["start"] for it in ds), max(it["end"] for it in ds)
        elif sid in others:
            start, end = min(others[sid]), max(others[sid])
        if start is not None and video_duration:
            limit = float(video_duration) + DURATION_TOL
            start, end = min(start, limit), min(end, limit)
        out.append({"id": sid, "start": None if start is None else round(start, 3),
                    "end": None if end is None else round(end, 3),
                    "timing": "none" if start is None else "estimated"})
    return out
