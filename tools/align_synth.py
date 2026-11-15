"""가짜 영화: 실제 대본(script.json)에서 가짜 받아쓰기(asr.json 꼴)와 정답을 만든다.

맞추기(sinkprep.align)를 영상 없이 개발·채점하려는 것이다. tools/eval_align_synth.py 와 tests 가 같이 쓴다.

주의: 가짜 받아쓰기는 실제보다 쉽다. 오류가 낱말마다 독립이고, 말이 겹치지 않고, 배우가 줄을 통째로 바꿔 말하지
않고, 받아쓰기가 문장을 통째로 지어내지 않는다. 낱말 확률도 '맞는 낱말은 높고 틀린 낱말은 낮다'는 가정으로 만든
값이다. 여기서 나온 수치는 알고리즘끼리 견주고 기준값을 고르는 데만 쓰고, 실제 영화의 정확도로 읽지 말 것.

실제 받아쓰기(추격자, large-v3-turbo)에서 잰 모양을 따른 것: 문장 끝 부호로 끝나는 낱말은 18% 뿐이고 줄 끝 낱말도
절반만 부호가 있다(punct_drop, punct_drop_end). 낱말 시각은 이어 붙어서 말 차례 사이 틈의 절반 가까이가 0 이다
(gap_zero). 대본에 없는 말과 짧은 대답도 같은 틈으로 대사에 이어진다(남은 낱말을 어느 줄에 붙일지 틈과 부호만으로는
알 수 없다). 받아쓰기는 줄 경계를 넘어 두 낱말을 한 낱말로 적기도 한다(cross_merge).

넣는 변형(확률은 RATES)
  씬 자르기(일부는 그 자리에 대본에 없는 말이 들어감), 씬 자리 옮기기, 씬 안에서 이웃한 두 줄의 순서 바꾸기,
  줄 빼기(일부는 '말했지만 말 구간 검출이 버린 줄'로 rescue pass 에만 나옴),
  줄의 일부만 말하기, 낱말 바꾸기·빼기, 짧은 대답을 같은 뜻의 다른 말로(예 -> 네), 군말·더듬기 넣기,
  대본에 없는 말 끼우기(다른 대본의 대사), 짧은 대답 끼우기, 받아쓰기 글자 오류, 숫자를 아라비아 숫자로,
  낱말 붙이기·쪼개기(줄 경계를 넘는 것 포함), 문장부호 빼기, 받아쓰기 헛소리(낮은 확률).
  ctx pass: plain 의 글자 오류 일부를 실제 말대로 고침, 배우가 바꾼 낱말 일부를 대본대로 끌어감(낮은 확률),
            말 없는 자리에 대본 줄을 그대로 적음(힌트 되풀이, 낮은 확률).
  rescue pass: 말 구간 검출이 버린 줄이 틈에서 되살아남, 가끔 헛소리.
  alt=True 일 때 더 넣는 것(실제 추격자의 turbo·large-v3·rescue 에서 본 모양. 이 변형들은 따로 둔 난수로 만들어서,
  alt 를 켜도 plain·ctx 의 글과 줄 차례는 같은 씨앗의 alt=False 영화와 거의 같다)
    대안 pass(id lv3, context null): 같은 말을 다른 모델이 대본 없이 받아쓴 것. 글자 오류가 plain 과 다른 곳에서 나고,
      낱말을 다르게 붙이고 쪼개고, 낱말 경계 시각이 흔들리고, 쉼 뒤 첫 낱말의 시작이 앞 침묵 쪽으로 늘어나고,
      욕설을 가려 적고('새끼' -> '새X'), 말 차례를 통째로 빠뜨리기도 한다.
    기준 pass 가 뭉갠 줄(plain_blur): plain 이 한 줄을 통째로 엉망으로 적는다(대안만 맞출 수 있는 줄).
    rescue 의 대본에 없는 말(rescue_foreign): 말은 했지만 말 구간 검출이 버린 대본 밖 말. rescue 에만 나온다.
    rescue 가 겹쳐 적은 낱말(rescue_dup): 쉼 뒤 첫 낱말을 rescue 가 또 적는다(시작 시각이 앞 침묵 쪽으로 늘어난 같은 소리).
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sinkprep import textnorm  # noqa: E402

RATES = {
    "scene_cut": 0.12,        # 씬을 통째로 자를 확률
    "cut_filled": 0.3,        # 잘린 씬 자리에 대본에 없는 말(그 씬의 줄 수만큼, 3~30줄)이 들어갈 확률
    "line_swap": 0.02,        # 씬 안에서 이웃한 두 줄(둘 다 6글자 이상)의 순서를 바꿔 말할 확률
    "scene_moves": 3,         # 자리를 옮기는 씬 수
    "line_drop": 0.08,        # 남은 씬에서 줄이 빠질 확률
    "vad_miss": 0.35,         # 빠진 줄 가운데 '말은 했지만 plain 이 받아쓰지 않은 줄'의 비율(rescue 에 나옴)
    "partial": 0.03,          # 긴 줄의 앞이나 뒤 절반만 말할 확률
    "word_sub": 0.04,         # 낱말을 다른 낱말로 바꿔 말할 확률(낱말마다)
    "word_del": 0.02,         # 낱말을 빼고 말할 확률(낱말마다)
    "answer_swap": 0.4,       # 짧은 대답 낱말을 같은 뜻의 다른 말로 할 확률(예 <-> 네, 응 <-> 어)
    "filler": 0.10,           # 줄에 군말을 넣을 확률
    "stutter": 0.05,          # 줄에 더듬기를 넣을 확률
    "char_err": 0.06,         # 받아쓰기 글자 오류(글자마다)
    "digits": 0.7,            # 한글 수사를 아라비아 숫자로 적을 확률
    "merge": 0.05,            # 이웃 낱말을 붙여 적을 확률
    "split": 0.05,            # 낱말을 쪼개 적을 확률
    "punct_drop": 0.3,        # 낱말 끝 문장부호를 빼고 적을 확률(말 차례의 마지막 낱말 제외)
    "punct_drop_end": 0.2,    # 말 차례의 마지막 낱말 끝 부호를 빼고 적을 확률
    "gap_zero": 0.45,         # 말 차례 사이 틈이 0 일 확률(대본에 없는 말·짧은 대답 앞뒤도 같다)
    "inner_pause": 0.04,      # 말 차례 안의 낱말 사이에 긴 쉼(0.3~1.2초)이 들 확률
    "cross_merge": 0.03,      # 틈 0 인 말 차례 경계에서 앞 차례 끝 낱말과 뒤 차례 첫 낱말을 한 낱말로 적을 확률
    "foreign": 0.05,          # 줄 사이에 대본에 없는 말(다른 대본의 대사 1~3줄)을 끼울 확률
    "short_answer": 0.06,     # 줄 사이에 짧은 대답을 끼울 확률
    "halluc": 0.10,           # 씬 사이 말 없는 곳에 받아쓰기 헛소리를 넣을 확률
    "ctx_fix": 0.6,           # ctx 가 plain 의 글자 오류를 실제 말대로 고칠 확률(낱말마다)
    "ctx_pull": 0.15,         # ctx 가 배우가 바꾼 낱말을 대본대로 끌어갈 확률(낱말마다)
    "ctx_digits": 0.7,        # ctx 가 숫자를 대본처럼 한글로 적을 확률
    "ctx_echo": 0.15,         # 말하지 않은 줄을 ctx 가 그대로 적을 확률(힌트 되풀이)
    "rescue_junk": 0.03,      # 씬 사이에 rescue 가 헛소리를 적을 확률
    "alt_err": 0.85,          # (alt) 대안 pass 의 글자 오류율 = char_err × 이 값
    "alt_drop": 0.03,         # (alt) 대안 pass 가 말 차례를 통째로 빠뜨릴 확률
    "alt_mask": 0.5,          # (alt) 대안 pass 가 욕설이 든 낱말을 가려 적을 확률
    "alt_jitter": 0.08,       # (alt) 대안 pass 낱말 경계 시각의 흔들림(초, 표준편차)
    "alt_stretch": 0.3,       # (alt) 쉼(0.5초 이상) 뒤 첫 낱말의 시작이 앞 침묵 쪽으로 늘어날 확률
    "plain_blur": 0.03,       # (alt) 기준 pass 가 한 줄을 통째로 엉망으로 적을 확률
    "rescue_foreign": 0.04,   # (alt) 줄 뒤에 '말 구간 검출이 버린 대본 밖 말'이 들어갈 확률(rescue 에만 나옴)
    "rescue_dup": 0.15,       # (alt) 쉼(0.9초 이상) 뒤 첫 낱말을 rescue 가 또 적을 확률
    "duration": 7200.0,       # 영화 길이(초). 말이 이보다 길면 늘어난다
    "sec_per_char": 0.13,     # 글자당 말하는 시간
}

FILLERS = ("어,", "음...", "아,", "저,", "그,", "어", "아이,", "에이,", "뭐,")
SHORT_ANSWERS = ("예.", "네.", "응.", "어.", "아니요.", "그래.", "왜?", "뭐?")
ANSWER_SWAP = {"예": "네", "네": "예", "응": "어", "어": "응"}
_PUNCT = ".,?!…"
HALLUC = (("감사합니다.",), ("시청해주셔서", "감사합니다."), ("한글", "자막", "제공"), ("아", "아", "아", "아", "아"),
          ("다음", "영상에서", "만나요."))
_NUM = "영일이삼사오육칠팔구"
_NUM_RUN = re.compile(r"^([공영일이삼사오육칠팔구십백천]{2,})(.*)$")
_COUNTERS = "호번층년월일시분초개명살원대위째세권장동"
_SYL = [chr(c) for c in range(0xAC00, 0xD7A4)]
_MASKS = (("씨발", "씨X"), ("시발", "시X"), ("씨팔", "씨X"), ("새끼", "새X"), ("좆", "X"), ("미친년", "미친X"), ("미친놈", "미친X"))


def _hangul_to_digits(token: str) -> str | None:
    """'삼백육호요.' -> '306호요.', '사팔팔오' -> '4885'. 바꿀 수 없으면 None."""
    m = _NUM_RUN.match(token)
    if not m:
        return None
    run, rest = m.group(1), m.group(2)
    nxt = textnorm.key(rest)[:1]
    if nxt and nxt not in _COUNTERS:
        return None
    if not any(c in run for c in "십백천"):
        if len(run) < 3:
            return None
        return "".join("0" if c in "공영" else str(_NUM.index(c)) for c in run) + rest
    if any(c in run for c in "공영"):
        return None
    total, cur = 0, 0
    for c in run:
        if c in "십백천":
            total += (cur or 1) * {"십": 10, "백": 100, "천": 1000}[c]
            cur = 0
        elif cur:
            return None
        else:
            cur = _NUM.index(c)
    return str(total + cur) + rest


def _garble(rng: random.Random, c: str) -> str:
    """한글 음절 하나를 받아쓰기 오류처럼 바꾼다(자모 하나를 바꾸거나 아예 다른 음절)."""
    j = textnorm.jamo(c)
    if j is None:
        return c
    lead, vowel, tail = j
    r = rng.random()
    if r < 0.4:
        vowel = (vowel + rng.choice((1, 2, 4, 20))) % 21
    elif r < 0.7:
        tail = rng.choice((0, 1, 4, 8, 16, 17, 21)) if tail else rng.choice((1, 4, 8, 21))
    elif r < 0.85:
        lead = (lead + rng.choice((1, 2, 3))) % 19
    else:
        return rng.choice(_SYL[:2000:7])
    out = chr(0xAC00 + lead * 588 + vowel * 28 + tail)
    return out if out != c else chr(0xAC00 + lead * 588 + ((vowel + 1) % 21) * 28 + tail)


def _split_unspaced(rng: random.Random, text: str) -> list[str]:
    """띄어쓰기 없는 대본 줄을 낱말 비슷한 조각으로(받아쓰기는 띄어 쓴다). 문장부호 뒤에서는 늘 끊는다."""
    out: list[str] = []
    for tok in _glue(text.split()):
        if len(textnorm.key(tok)) <= 6:
            out.append(tok)
            continue
        cur, n, want = "", 0, rng.randint(2, 4)
        for k, c in enumerate(tok):
            cur += c
            is_key = bool(textnorm.key(c))
            n += is_key
            nxt_key = k + 1 < len(tok) and bool(textnorm.key(tok[k + 1]))
            if nxt_key and (n >= want or not is_key):
                out.append(cur)
                cur, n, want = "", 0, rng.randint(2, 4)
        if cur:
            out.append(cur)
    return out


def _glue(tokens: list[str]) -> list[str]:
    """문장부호뿐인 조각('!', '...')은 낱말이 아니다. 앞 낱말 뒤에(맨 앞이면 다음 낱말 앞에) 붙인다."""
    out: list[str] = []
    head = ""
    for t in tokens:
        if textnorm.key(t):
            out.append(head + t)
            head = ""
        elif out:
            out[-1] += t
        else:
            head += t
    return out


def _p(rng: random.Random, mean: float, sd: float) -> float:
    return round(min(0.999, max(0.01, rng.gauss(mean, sd))), 3)


def make_movie(script: dict, others: list[str], seed: int, rates: dict | None = None, alt: bool = False) -> dict:
    """가짜 영화 하나. alt 가 참이면 대안 pass(lv3)와 그에 딸린 변형을 더 넣는다(머리말). 돌려주는 것::

        {"asr":   sink.asr/1 꼴 dict (passes: plain[, lv3], ctx, rescue),
         "truth": {"duration": 초,
                   "lines": {요소 id: {...}},       # 말 있는 대사마다(아래 칸)
                   "added": [plain 낱말 번호...],   # 어느 대본 줄에도 속하지 않는 plain 낱말
                   "echo":  [요소 id...],           # ctx 가 말 없는 자리에 그대로 적은 줄
                   "rescue_added": [rescue 낱말 번호...],   # 말 구간 검출이 버린 대본 밖 말(rescue 에만 있는 진짜 말)
                   "rescue_dup": [rescue 낱말 번호...],     # plain 낱말과 같은 소리를 rescue 가 또 적은 것
                   "rescue_junk": [rescue 낱말 번호...],    # rescue 의 헛소리
                   "halluc": [plain 낱말 번호...],          # plain 의 헛소리(HALLUC 글귀, added 에도 들어 있다)
                   "moved_scenes": [씬 id...], "cut_scenes": [씬 id...]}}

    lines 의 칸: chars(글자 수), scene, said(영상에서 말했는가), moved(옮긴 씬의 줄), vad_miss(말했지만 plain 에 없음),
    ghost(잘린 씬의 줄인데 그 자리에 대본에 없는 말이 있음), swapped(이웃 줄과 순서를 바꿔 말함),
    t([시작, 끝] 또는 None), plain([처음 낱말, 끝 낱말] 또는 None), ctx(같은 것, ctx pass), rescue(같은 것),
    true_key(실제로 한 말의 열쇠), status_true(실제 말 기준 same|changed|absent),
    status_plain(plain 낱말 그대로일 때 same|changed|absent), status_rescue(rescue 낱말 그대로일 때), ctx_pulled(ctx 가 바꾼 말을 대본대로 끌어감),
    ctx_right(ctx 글이 실제 말과 같음), plain_right(plain 글이 실제 말과 같음).
    alt 일 때 더 있는 칸: alt([처음 낱말, 끝 낱말] 또는 None, lv3 pass), alt_key·plain_key(그 pass 낱말 그대로의 열쇠),
    alt_right(대안 글이 실제 말과 같음), status_alt(대안 낱말 그대로일 때 same|changed|absent), blurred(plain 이 뭉갠 줄).
    """
    R = dict(RATES, **(rates or {}))
    rng = random.Random(seed)
    rng2 = random.Random(seed * 7919 + 104729)     # alt 변형에만 쓰는 난수(plain·ctx 를 만드는 난수 차례를 건드리지 않는다)
    vocab = sorted({t for ln in others for t in ln.split() if 2 <= len(textnorm.key(t)) <= 5})
    if not vocab:
        vocab = ["그래서", "근데", "정말", "아무튼", "그러니까"]

    text_key = {e["id"]: textnorm.key(e.get("text")) for e in script.get("elements") or []}

    # 1. 씬 단위로 묶기
    scenes: list[dict] = []
    for e in script.get("elements") or []:
        if e.get("type") != "dialogue" or not textnorm.sayable(e.get("text")):
            continue
        if not scenes or scenes[-1]["scene"] != e.get("scene"):
            scenes.append({"scene": e.get("scene"), "lines": [], "cut": False, "moved": False})
        scenes[-1]["lines"].append(e)
    for sc in scenes:
        sc["cut"] = rng.random() < R["scene_cut"]
    movable = [k for k, sc in enumerate(scenes) if not sc["cut"] and len(sc["lines"]) >= 3]
    order = list(range(len(scenes)))
    for k in rng.sample(movable, min(int(R["scene_moves"]), len(movable))):
        scenes[k]["moved"] = True
        order.remove(k)
        far = [p for p in range(len(order) + 1) if abs(p - k) >= 3] or list(range(len(order) + 1))
        order.insert(rng.choice(far), k)

    # 2. 말 차례 만들기(영화 순서)
    truth_lines: dict[str, dict] = {}
    seq: list[dict] = []          # 말 차례: {"kind", "el", "toks": [{"plain","ctx","true","pp","pc"}], "gap"}

    def atom(text: str, **kw) -> dict:
        return dict({"true": text, "plain": text, "ctx": text, "pp": _p(rng, 0.92, 0.06), "pc": None,
                     "fix": False, "pull": False, "atoms": [text]}, **kw)

    def alt_word(true: str, last: bool) -> tuple[str, float]:
        """실제 말 한 낱말을 대안 pass 가 적은 글과 확률(plain 과 따로 틀린다)."""
        chars = list(true)
        hit = False
        for i, c in enumerate(chars):
            if textnorm.jamo(c) is not None and rng2.random() < R["char_err"] * R["alt_err"]:
                chars[i] = _garble(rng2, c)
                hit = True
        text = "".join(chars)
        if not hit:
            dig = _hangul_to_digits(true)
            if dig is not None and rng2.random() < R["digits"]:
                text = dig
            for bad, shown in _MASKS:
                if bad in text and rng2.random() < R["alt_mask"]:
                    text = text.replace(bad, shown)
        if rng2.random() < (R["punct_drop_end"] if last else R["punct_drop"]):
            text = text.rstrip(_PUNCT) or text
        return text, (_p(rng2, 0.6, 0.15) if hit else _p(rng2, 0.92, 0.06))

    def blur(toks: list[dict]) -> None:
        """plain 이 줄을 통째로 엉망으로 적는다(글자의 절반 넘게 틀림, 낮은 확률)."""
        for tk in toks:
            chars = [(_garble(rng2, c) if textnorm.jamo(c) is not None and rng2.random() < 0.6 else c) for c in tk["plain"]]
            tk.update(plain="".join(chars), pp=_p(rng2, 0.4, 0.15))
            tk["pc"] = min(0.999, max(0.01, round(tk["pp"] + rng2.gauss(0, 0.02), 3)))
            if not tk["fix"]:
                tk["ctx"] = tk["plain"]

    def render(toks: list[dict]) -> None:
        """실제 말(true)에 받아쓰기 오류·숫자 표기·문장부호 누락을 입힌다(plain, ctx)."""
        for k, t in enumerate(toks):
            true = t["true"]
            plain = true
            chars = list(plain)
            hit = False
            for i, c in enumerate(chars):
                if textnorm.jamo(c) is not None and rng.random() < R["char_err"]:
                    chars[i] = _garble(rng, c)
                    hit = True
            plain = "".join(chars)
            ctx = plain
            pp = _p(rng, 0.6, 0.15) if hit else _p(rng, 0.92, 0.06)
            pc = min(0.999, max(0.01, round(pp + rng.gauss(0, 0.02), 3)))
            if hit and rng.random() < R["ctx_fix"]:
                ctx, pc, t["fix"] = true, _p(rng, 0.88, 0.07), True
            if not hit:
                dig = _hangul_to_digits(true)
                if dig is not None and rng.random() < R["digits"]:
                    plain = dig
                    ctx = true if rng.random() < R["ctx_digits"] else dig
            if t.get("script") and not hit and rng.random() < R["ctx_pull"]:
                ctx, pc, t["pull"] = t["script"], _p(rng, 0.3, 0.12), True
            if rng.random() < (R["punct_drop"] if k + 1 < len(toks) else R["punct_drop_end"]):
                plain, ctx = plain.rstrip(_PUNCT), ctx.rstrip(_PUNCT)
                plain, ctx = plain or true, ctx or true
            t.update(plain=plain, ctx=ctx, pp=pp, pc=pc)

    def segment(toks: list[dict]) -> list[dict]:
        """낱말 붙이기·쪼개기(plain 과 ctx 를 같은 자리에서)."""
        out: list[dict] = []
        for t in toks:
            if out and rng.random() < R["merge"] and not textnorm.ends_sentence(out[-1]["plain"]):
                o = out[-1]
                o.update(plain=o["plain"] + t["plain"], ctx=o["ctx"] + t["ctx"], true=o["true"] + t["true"],
                         pp=round((o["pp"] + t["pp"]) / 2, 3), pc=round((o["pc"] + t["pc"]) / 2, 3),
                         fix=o["fix"] or t["fix"], pull=o["pull"] or t["pull"], atoms=o["atoms"] + t["atoms"])
                continue
            n = len(t["plain"])
            if n >= 4 and len(t["ctx"]) == n and len(t["true"]) == n and rng.random() < R["split"] \
                    and textnorm.key(t["plain"][:2]) and textnorm.key(t["plain"][2:]) \
                    and not any(c.isdigit() for c in t["plain"] + t["ctx"]):
                cut = rng.randint(2, n - 2)
                if textnorm.key(t["plain"][:cut]) and textnorm.key(t["plain"][cut:]) \
                        and textnorm.key(t["ctx"][:cut]) and textnorm.key(t["ctx"][cut:]):
                    out.append(dict(t, plain=t["plain"][:cut], ctx=t["ctx"][:cut], true=t["true"][:cut], atoms=[t["true"][:cut]]))
                    out.append(dict(t, plain=t["plain"][cut:], ctx=t["ctx"][cut:], true=t["true"][cut:], atoms=[t["true"][cut:]]))
                    continue
            out.append(t)
        return out

    def say_line(e: dict) -> list[dict]:
        """대본 줄을 배우가 말한 낱말 열로(바꿔 말하기, 군말, 더듬기)."""
        base = _split_unspaced(rng, e["text"])
        n_chars = len(textnorm.key(e["text"]))
        if n_chars >= 12 and len(base) >= 4 and rng.random() < R["partial"]:
            keep = max(2, int(len(base) * rng.uniform(0.4, 0.6)))
            base = base[:keep] if rng.random() < 0.5 else base[-keep:]
        toks: list[dict] = []
        for w in base:
            r = rng.random()
            if r < R["word_del"] and len(base) > 1:
                continue
            if r < R["word_del"] + R["word_sub"] and textnorm.key(w):
                new = rng.choice(vocab).rstrip(_PUNCT) + w[len(w.rstrip(_PUNCT)):]
                if textnorm.key(new) != textnorm.key(w):
                    toks.append(atom(new, script=w))
                    continue
            other = ANSWER_SWAP.get(textnorm.key(w))
            if other and w.startswith(textnorm.key(w)) and rng.random() < R["answer_swap"]:
                toks.append(atom(other + w[len(textnorm.key(w)):], script=w))
                continue
            toks.append(atom(w))
        if not any(textnorm.key(t["true"]) for t in toks):
            toks = [atom(w) for w in base]
        if rng.random() < R["filler"]:
            pos = 0 if rng.random() < 0.7 else rng.randrange(len(toks))
            toks.insert(pos, atom(rng.choice(FILLERS), extra=True))
        if rng.random() < R["stutter"]:
            cand = [k for k, t in enumerate(toks) if not t.get("extra") and textnorm.jamo(t["true"][:1] or " ")]
            if cand:
                k = rng.choice(cand)
                j = textnorm.jamo(toks[k]["true"][0])
                head = chr(0xAC00 + j[0] * 588 + j[1] * 28) + ","
                rep = head if rng.random() < 0.6 else toks[k]["true"].rstrip(".,?!…")
                toks.insert(k, atom(rep, extra=True))
        return toks

    def extra_turn(kind: str, words: tuple | list, p_mean: float, p_sd: float) -> dict:
        toks = [atom(w, pp=_p(rng, p_mean, p_sd)) for w in words]
        for k, t in enumerate(toks):
            t["pc"] = min(0.999, max(0.01, round(t["pp"] + rng.gauss(0, 0.02), 3)))
            if kind in ("foreign", "short") and \
                    rng.random() < (R["punct_drop"] if k + 1 < len(toks) else R["punct_drop_end"]):
                bare = t["plain"].rstrip(_PUNCT)       # 대본에 없는 말도 받아쓰기는 부호를 빼먹는다
                if textnorm.key(bare):
                    t["plain"] = t["ctx"] = bare
        return {"kind": kind, "el": None, "toks": toks}

    def gap_between() -> float:
        if rng.random() < R["gap_zero"]:
            return 0.0
        r = rng.random()
        if r < 0.4:
            return rng.uniform(0.05, 0.4)
        if r < 0.8:
            return rng.uniform(0.4, 1.5)
        return rng.uniform(1.5, 4.0)

    scene_marks: list[int] = []                     # seq 안에서 씬이 시작하는 자리
    for k in order:
        sc = scenes[k]
        scene_marks.append(len(seq))
        if rng.random() < R["halluc"]:
            seq.append(dict(extra_turn("halluc", rng.choice(HALLUC), 0.3, 0.1), gap=rng.uniform(1.0, 3.0)))
        if rng.random() < R["rescue_junk"]:
            seq.append(dict(extra_turn("rescue_junk", rng.choice(HALLUC), 0.3, 0.1), gap=rng.uniform(1.0, 3.0)))
        ghost = sc["cut"] and bool(others) and rng.random() < R["cut_filled"]
        if ghost:                                    # 잘린 씬 자리에 대본에 없는 말
            for _ in range(min(30, max(3, len(sc["lines"])))):
                ws = _split_unspaced(rng, rng.choice(others))
                seq.append(dict(extra_turn("foreign", ws, 0.9, 0.07), gap=gap_between()))
        said_order = list(sc["lines"])
        swapped: set[str] = set()
        at = 0
        while at + 1 < len(said_order):              # 이웃한 두 줄의 순서만 바꿔 말함
            a, b = said_order[at], said_order[at + 1]
            if not sc["cut"] and min(len(textnorm.key(a["text"])), len(textnorm.key(b["text"]))) >= 6 \
                    and rng.random() < R["line_swap"]:
                said_order[at], said_order[at + 1] = b, a
                swapped.update((a["id"], b["id"]))
                at += 2
            else:
                at += 1
        for e in said_order:
            tl = {"chars": len(textnorm.key(e["text"])), "scene": e.get("scene"), "said": False, "moved": sc["moved"],
                  "vad_miss": False, "t": None, "plain": None, "ctx": None, "rescue": None, "true_key": None,
                  "status_true": "absent", "status_plain": "absent", "status_rescue": "absent", "ctx_pulled": False, "ctx_right": False,
                  "plain_right": False, "ghost": ghost, "swapped": e["id"] in swapped,
                  "alt": None, "alt_key": None, "plain_key": None, "alt_right": False, "status_alt": "absent", "blurred": False}
            truth_lines[e["id"]] = tl
            dropped = sc["cut"] or rng.random() < R["line_drop"]
            if dropped and not (not sc["cut"] and rng.random() < R["vad_miss"]):
                if rng.random() < R["ctx_echo"]:
                    toks = [atom(w, pc=_p(rng, 0.15, 0.08)) for w in _split_unspaced(rng, e["text"])]
                    seq.append({"kind": "echo", "el": e["id"], "toks": toks, "gap": rng.uniform(0.6, 2.0)})
                continue
            toks = say_line(e)
            tl["said"] = True
            tl["true_key"] = textnorm.key(" ".join(t["true"] for t in toks))
            tl["status_true"] = "same" if tl["true_key"] == textnorm.key(e["text"]) else "changed"
            render(toks)
            toks = segment(toks)
            if alt and not dropped and rng2.random() < R["plain_blur"]:
                blur(toks)
                tl["blurred"] = True
            if dropped:                              # 말했지만 말 구간 검출이 버린 줄
                tl["vad_miss"] = True
                seq.append({"kind": "vad_miss", "el": e["id"], "toks": toks, "gap": rng.uniform(0.9, 2.5)})
                continue
            seq.append({"kind": "line", "el": e["id"], "toks": toks, "gap": gap_between()})
            if rng.random() < R["foreign"] and others:
                for _ in range(rng.randint(1, 3)):
                    ws = _split_unspaced(rng, rng.choice(others))
                    turn = extra_turn("foreign", ws, 0.9, 0.07)
                    seq.append(dict(turn, gap=gap_between()))
            elif rng.random() < R["short_answer"]:
                seq.append(dict(extra_turn("short", (rng.choice(SHORT_ANSWERS),), 0.85, 0.1), gap=gap_between()))
            if alt and others and rng2.random() < R["rescue_foreign"]:      # 말 구간 검출이 버린 대본 밖 말
                ws = _split_unspaced(rng2, rng2.choice(others)) if rng2.random() < 0.7 else [rng2.choice(SHORT_ANSWERS)]
                toks2 = [dict(true=w, plain=w, ctx=w, pp=_p(rng2, 0.8, 0.12), pc=None, fix=False, pull=False, atoms=[w])
                         for w in ws]
                seq.append({"kind": "rescue_foreign", "el": None, "toks": toks2, "gap": rng2.uniform(0.9, 2.5)})

    # 3. 시각: 말 속도와 쉼, 씬 사이 침묵으로 영화 길이를 맞춘다
    def dur(tok: dict, r: random.Random) -> float:
        n = max(1, len(textnorm.key(tok["plain"])) or len(textnorm.key(tok["ctx"])) or 1)
        return max(0.08, R["sec_per_char"] * n * r.uniform(0.8, 1.25))

    speech = 0.0
    for turn in seq:
        r = rng2 if turn["kind"] == "rescue_foreign" else rng
        turn["durs"] = [dur(t, r) for t in turn["toks"]]
        turn["inner"] = [(r.uniform(0.3, 1.2) if r.random() < R["inner_pause"] else
                          r.choice((0.0, 0.0, 0.0, r.uniform(0.0, 0.06))) +
                          (r.uniform(0.05, 0.18) if t["plain"].endswith((",", "...", "…")) else 0.0))
                         for t in turn["toks"][:-1]]
        if turn["kind"] in ("halluc", "rescue_junk"):
            turn["after"] = rng.uniform(0.35, 1.5)   # 헛소리는 말 없는 곳에 따로 떨어져 있다
        elif turn["kind"] == "rescue_foreign":
            turn["after"] = rng2.uniform(0.9, 2.0)   # 말 구간 검출이 버린 말은 앞뒤 말과 떨어져 있다
        else:
            turn["after"] = 0.0                      # 대본에 없는 말 뒤의 틈은 다음 말 차례의 틈이 정한다
        speech += sum(turn["durs"]) + sum(turn["inner"]) + turn["gap"] + turn["after"]
    marks = set(scene_marks)
    weights = {k: rng.uniform(0.5, 3.0) for k in marks}
    spare = max(0.0, R["duration"] - speech - 2.0 * len(marks) - 20.0)
    wsum = sum(weights.values()) or 1.0

    plain: list[dict] = []
    ctx: list[dict] = []
    rescue: list[dict] = []
    alts: list[dict] = []
    vad: list[list[float]] = []
    added: list[int] = []
    echo: list[str] = []
    rescue_added: list[int] = []
    rescue_dup: list[int] = []
    rescue_junk: list[int] = []
    halluc: list[int] = []
    t = 5.0
    spoken = ("line", "foreign", "short")

    def alt_turn(turn: dict, spans: list[tuple[float, float]], quiet: float) -> list[int] | None:
        """말 차례 하나를 대안 pass 가 받아쓴 낱말로(alts 에 붙인다). spans: 낱말 조각(turn["toks"])마다 (시작, 끝),
        quiet: 이 차례 앞의 침묵 길이(초). 돌려주는 것: [처음 낱말, 끝 낱말] 또는 빠뜨렸으면 None."""
        if rng2.random() < R["alt_drop"]:
            return None
        pieces: list[list] = []                      # [글, 시작, 끝, 확률]
        for tok, (s, e) in zip(turn["toks"], spans):
            parts = tok["atoms"]
            sizes = [max(1, len(textnorm.key(a))) for a in parts]
            at = s
            for a, n in zip(parts, sizes):
                b = at + (e - s) * n / sum(sizes)
                pieces.append([a, at, b])
                at = b
        words: list[list] = []
        for j, (text, s, e) in enumerate(pieces):
            w, pw_ = alt_word(text, j + 1 == len(pieces))
            if words and rng2.random() < R["merge"] and not textnorm.ends_sentence(words[-1][0]):
                words[-1][0] += w
                words[-1][2] = e
                words[-1][3] = round((words[-1][3] + pw_) / 2, 3)
            else:
                words.append([w, s, e, pw_])
        for a, b in zip(words, words[1:]):           # 낱말 경계가 plain 과 다르게 흔들린다
            if b[1] - a[2] < 0.02:
                cut = min(max(a[2] + rng2.gauss(0, R["alt_jitter"]), a[1] + 0.02), b[2] - 0.02)
                a[2] = b[1] = cut
        if quiet >= 0.5 and rng2.random() < R["alt_stretch"]:
            words[0][1] -= rng2.uniform(0.2, min(quiet - 0.05, 1.2))
        first = len(alts)
        for w, s, e, pw_ in words:
            alts.append({"i": len(alts), "w": w, "s": round(s, 3), "e": round(max(e, s + 0.01), 3), "p": pw_})
        return [first, len(alts) - 1]

    last_end = 0.0                                   # 어느 pass 든 마지막 낱말이 끝난 시각
    for k, turn in enumerate(seq):
        if k in marks:
            t += 2.0 + spare * weights[k] / wsum
        t += turn["gap"]
        t0 = t
        quiet = t0 - last_end
        spans: list[tuple[float, float]] = []
        p0, c0, r0 = len(plain), len(ctx), len(rescue)
        prev = seq[k - 1] if k else None
        glued = (k not in marks and turn["gap"] == 0.0 and prev is not None and prev["kind"] in spoken
                 and turn["kind"] in spoken and len(turn["toks"]) >= 2 and bool(plain)
                 and not textnorm.ends_sentence(plain[-1]["w"]) and rng.random() < R["cross_merge"])
        for j, tok in enumerate(turn["toks"]):
            s, e = round(t, 3), round(t + turn["durs"][j], 3)
            kind = turn["kind"]
            if glued and j == 0:                     # 앞 차례의 끝 낱말에 붙여 적는다(그 낱말은 앞 차례의 것으로 친다)
                plain[-1].update(w=plain[-1]["w"] + tok["plain"], e=e)
                ctx[-1].update(w=ctx[-1]["w"] + tok["ctx"], e=e)
                p0, c0 = len(plain), len(ctx)
                if prev["el"] is not None:
                    truth_lines[prev["el"]].update(status_plain="changed", plain_right=False, ctx_right=False)
            elif kind in ("line", "foreign", "short", "halluc"):
                plain.append({"i": len(plain), "w": tok["plain"], "s": s, "e": e, "p": tok["pp"]})
                ctx.append({"i": len(ctx), "w": tok["ctx"], "s": s, "e": e, "p": tok["pc"]})
            elif kind == "echo":
                ctx.append({"i": len(ctx), "w": tok["true"], "s": s, "e": e, "p": tok["pc"]})
            else:                                    # vad_miss, rescue_junk, rescue_foreign
                rescue.append({"i": len(rescue), "w": tok["plain"], "s": s, "e": e, "p": tok["pp"]})
            spans.append((s, e))
            if alt and j == 0 and kind in spoken and quiet >= 0.9 and e - s + 0.15 < quiet - 0.05 \
                    and rng2.random() < R["rescue_dup"]:
                # 쉼 뒤 첫 낱말을 rescue 가 또 적는다: 같은 소리인데 시작 시각이 앞 침묵 쪽으로 늘어나 있다
                back = rng2.uniform(e - s + 0.15, min(quiet - 0.05, e - s + 1.5))
                rescue_dup.append(len(rescue))
                rescue.append({"i": len(rescue), "w": tok["plain"], "s": round(s - back, 3), "e": e,
                               "p": min(0.999, max(0.01, round(tok["pp"] + rng2.gauss(0, 0.05), 3)))})
            t = e + (turn["inner"][j] if j < len(turn["inner"]) else 0.0)
        t1 = round(t, 3)
        last_end = t1
        t += turn["after"]
        kind = turn["kind"]
        a_range = alt_turn(turn, spans, quiet) if alt and kind in spoken else None
        if kind == "rescue_foreign":
            rescue_added.extend(range(r0, len(rescue)))
        if kind == "rescue_junk":
            rescue_junk.extend(range(r0, len(rescue)))
        if kind in ("line", "foreign", "short"):
            if vad and t0 - vad[-1][1] < 0.5:
                vad[-1][1] = t1
            else:
                vad.append([round(t0, 3), t1])
        if kind in ("foreign", "short", "halluc"):
            added.extend(range(p0, len(plain)))
        if kind == "halluc":
            halluc.extend(range(p0, len(plain)))
        if kind == "echo":
            echo.append(turn["el"])
            truth_lines[turn["el"]]["ctx"] = [c0, len(ctx) - 1]
        if kind in ("line", "vad_miss"):
            tl = truth_lines[turn["el"]]
            tl["t"] = [round(t0, 3), t1]
            own = turn["toks"][1:] if glued else turn["toks"]
            pk = textnorm.key(" ".join(x["plain"] for x in own))
            ck = textnorm.key(" ".join(x["ctx"] for x in own))
            tl["plain_right"] = pk == tl["true_key"]
            if kind == "line":
                tl["plain"] = [p0, len(plain) - 1]
                tl["ctx"] = [c0, len(ctx) - 1]
                tl["status_plain"] = "same" if pk == text_key[turn["el"]] else "changed"
                tl["ctx_pulled"] = any(x["pull"] for x in turn["toks"])
                tl["ctx_right"] = ck == tl["true_key"]
                tl["plain_key"] = pk
                if a_range is not None:
                    ak = textnorm.key(" ".join(w["w"] for w in alts[a_range[0]:a_range[1] + 1]))
                    tl.update(alt=a_range, alt_key=ak, alt_right=ak == tl["true_key"],
                              status_alt="same" if ak == text_key[turn["el"]] else "changed")
            else:
                tl["rescue"] = [r0, len(rescue) - 1]
                tl["status_rescue"] = "same" if pk == text_key[turn["el"]] else "changed"
    duration = round(max(R["duration"], t + 10.0), 3)
    chunks = []                                      # 받아쓴 묶음: 말 구간을 0.6초 안쪽 틈은 이어 붙이고 25초에서 끊은 것
    for a, b in vad:
        if chunks and a - chunks[-1]["end"] <= 0.6 and b - chunks[-1]["start"] <= 25.0:
            chunks[-1]["end"] = b
        else:
            chunks.append({"start": a, "end": b, "lang": "ko"})
    passes = [{"id": "plain", "model": "synth", "context": None, "words": plain}]
    if alt:
        passes.append({"id": "lv3", "model": "synth", "context": None, "words": alts})
    passes += [{"id": "ctx", "model": "synth", "context": "script_hotwords", "words": ctx},
               {"id": "rescue", "model": "synth", "context": "full_window_stable", "words": rescue}]
    asr = {"schema": "sink.asr/1", "video_sha1": "0" * 40, "lang": "ko", "passes": passes, "vad": vad, "chunks": chunks}
    truth = {"duration": duration, "lines": truth_lines, "added": added, "echo": echo,
             "rescue_added": rescue_added, "rescue_dup": rescue_dup, "rescue_junk": rescue_junk, "halluc": halluc,
             "moved_scenes": [sc["scene"] for sc in scenes if sc["moved"]],
             "cut_scenes": [sc["scene"] for sc in scenes if sc["cut"]]}
    return {"asr": asr, "truth": truth}


def only_passes(asr: dict, ids: tuple[str, ...]) -> dict:
    """asr 에서 pass 몇 개만 남긴 사본(plain 만으로 맞추기를 따로 채점할 때)."""
    return dict(asr, passes=[p for p in asr["passes"] if p["id"] in ids])


def tiny_script(turns: list[tuple[str, str, str]]) -> dict:
    """시험용 작은 script.json: turns = [(씬 번호, 종류, 글)], 종류는 scene|action|dialogue|onscreen|transition."""
    els, scenes, seen = [], [], {}
    for n, (sc, typ, text) in enumerate(turns, 1):
        sid = f"SC{int(sc):03d}"
        if sid not in seen:
            seen[sid] = True
            scenes.append({"id": sid, "number": str(sc), "heading": f"씬 {sc}", "place": None, "time": None, "flags": []})
        e = {"id": f"E{n:04d}", "type": typ, "scene": sid, "text": text, "lines": [n]}
        if typ == "dialogue":
            e.update(speakers=["P01"], speaker_raw="가", how="on", lang="ko", raw=text, paren=[], gloss=None,
                     sung=False, overlap=None)
        if typ == "onscreen":
            e["kind"] = "insert"
        els.append(e)
    return {"schema": "sink.script/1", "doc_id": "fm1", "parser": {"name": "sink-parse", "version": "0"}, "lang": "ko",
            "front": {"title": None, "writer": None, "version_raw": None, "date": None},
            "people": [{"id": "P01", "name": "가", "aliases": [], "count": 1}], "scenes": scenes, "elements": els,
            "dropped": [], "quality": {}}


def say(words: list[tuple[str, float, float] | tuple[str, float, float, float]], pass_id: str = "plain",
        context: str | None = None) -> dict:
    """시험용 pass: words = [(글, 시작, 끝[, 확률])]."""
    return {"id": pass_id, "model": "synth", "context": context,
            "words": [{"i": i, "w": w[0], "s": w[1], "e": w[2], "p": w[3] if len(w) > 3 else 0.9}
                      for i, w in enumerate(words)]}


def asr_of(*passes: dict, vad: list | None = None) -> dict:
    return {"schema": "sink.asr/1", "video_sha1": "0" * 40, "lang": "ko", "passes": list(passes),
            "vad": vad or [], "chunks": []}
