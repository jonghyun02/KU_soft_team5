"""final.json -> 자막 파일(SRT). 영상 플레이어에서 자막으로 띄워 맞춘 결과를 눈으로 확인하는 용도다.

    .venv\\Scripts\\python -I tools\\final_to_srt.py <film_id 또는 final.json 경로> [--out 경로] [--absent] [--width 42]

- 시각 있는 대사(same·changed·added)만 자막으로 낸다. 글은 들린 그대로(heard)다.
- 화자 이름이 있으면 앞에 `이름: ` (여럿이면 `이름/이름: `). 앞 대사에서 이어받은 추정 화자는 `이름?: `.
- 대본에 없는 말(added)은 앞에 `[+] `. changed 는 표시 없이 들린 그대로.
- `--absent` 를 주면 영상에 없는 대본 대사(absent)를 앞뒤 자막 사이에 짧게 `[없음] 이름: 대본 글` 로 끼운다
  (시각이 없으므로 앞 대사가 끝난 자리에 놓는다. 말 없는 대사는 넣지 않는다).
- 긴 줄은 낱말 시각으로 잘라 한 자막에 두 줄(한 줄 --width 자 이하)까지만 보이게 한다. 한 줄보다 긴 낱말은 잘라 넣고
  그 낱말의 시간을 글자 수에 비례해 나눈다.
- 어느 받아쓰기 pass(plain, lv3, rescue ...)에서 온 줄이든 final.json 의 words 만 읽으므로 같은 꼴로 나온다.
기본 출력은 final.json 옆의 preview.srt (films/{film_id}/preview.srt). 다시 돌리면 덮어쓴다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sinkprep import paths, textnorm, util  # noqa: E402

WIDTH = 42                # 한 줄 글자 수 상한
LINES = 2                 # 한 자막의 줄 수 상한
MIN_SHOW = 0.7            # 자막 하나를 보여 주는 가장 짧은 시간(초). 다음 자막 시작을 넘지 않는다
ABSENT_SHOW = 2.0         # absent 자막을 보여 주는 가장 긴 시간(초)
ABSENT_MIN = 0.5


def stamp(t: float) -> str:
    """초 -> SRT 시각 'HH:MM:SS,mmm'."""
    ms = max(0, int(round(t * 1000)))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _prefix(it: dict, names: dict[str, str]) -> str:
    who = "/".join(names.get(p, p) for p in it.get("speakers") or [])
    if who and it.get("speaker_from") != "script":
        who += "?"
    mark = "[+] " if it.get("status") == "added" else ""
    return mark + (f"{who}: " if who else "")


def _pieces(word: tuple[str, float, float], width: int) -> list[tuple[str, float, float]]:
    """width 보다 긴 낱말은 width 자씩 자르고, 그 낱말의 시간을 글자 수에 비례해 조각에 나눠 준다
    (조각이 여러 자막에 걸쳐도 자막 시간이 겹치지 않게)."""
    text, s, e = word
    n = len(text)
    if n <= width:
        return [word]
    return [(text[k:k + width], s + (e - s) * k / n, s + (e - s) * min(n, k + width) / n) for k in range(0, n, width)]


def _wrap(tokens: list[tuple[str, float, float]], width: int) -> list[list[tuple[str, float, float]]]:
    """낱말 (글, 시작, 끝) 목록을 한 줄 width 자 이하의 줄 목록으로(낱말 경계에서, 앞에서부터 채움)."""
    toks = [p for t in tokens for p in _pieces(t, width)]
    total = sum(len(w) for w, _, _ in toks) + len(toks) - 1
    if width < total <= LINES * width and len(toks) > 1:      # 두 줄이면 되는 글: 길이가 비슷하게 나눈다
        best = None
        for k in range(1, len(toks)):
            a = sum(len(w) for w, _, _ in toks[:k]) + k - 1
            b = total - a - 1
            if a <= width and b <= width and (best is None or abs(a - b) < best[0]):
                best = (abs(a - b), k)
        if best is not None:
            return [toks[:best[1]], toks[best[1]:]]
    lines: list[list[tuple[str, float, float]]] = []
    used = 0
    for t in toks:
        if lines and used + 1 + len(t[0]) <= width:
            lines[-1].append(t)
            used += 1 + len(t[0])
        else:
            lines.append([t])
            used = len(t[0])
    return lines


def _short(text: str, width: int) -> str:
    return text if len(text) <= width else text[:max(1, width - 1)] + "…"


def cues(data: dict, absent: bool = False, width: int = WIDTH) -> list[tuple[float, float, str]]:
    """final.json 내용 -> 자막 목록 [(시작, 끝, 글)] (시작 순). 글의 줄바꿈은 '\\n'."""
    names = {p["id"]: p["name"] for p in data.get("people") or []}
    out: list[tuple[float, float, str]] = []
    waiting: list[dict] = []                 # 앞 대사 뒤에 끼울 absent 줄
    last_end = 0.0

    def flush(next_start: float | None) -> None:
        if not waiting:
            return
        start = last_end
        room = (next_start - start) if next_start is not None else ABSENT_SHOW
        end = start + max(ABSENT_MIN, min(ABSENT_SHOW, room))
        rows = [_short("[없음] " + _prefix(it, names) + it["script_ref"]["text"], width) for it in waiting[:LINES]]
        if len(waiting) > LINES:
            rows[-1] = _short(f"[없음] … 외 {len(waiting) - LINES + 1}줄", width)
        out.append((start, end, "\n".join(rows)))
        waiting.clear()

    for it in data.get("items") or []:
        if it.get("type") != "dialogue":
            continue
        if it.get("start") is None:
            if absent and it.get("script_ref") and textnorm.sayable(it["script_ref"].get("text")):
                waiting.append(it)
            continue
        flush(it["start"])
        words = [(w["w"], float(w["s"]), float(w["e"])) for w in it.get("words") or []]
        if not words:
            words = [(it.get("heard") or "", float(it["start"]), float(it["end"]))]
        head = _prefix(it, names)
        words[0] = (head + words[0][0], words[0][1], words[0][2])
        lines = _wrap(words, width)
        for k in range(0, len(lines), LINES):
            block = lines[k:k + LINES]
            start = min(s for ln in block for _, s, _ in ln)
            end = max(e for ln in block for _, _, e in ln)
            out.append((start, end, "\n".join(" ".join(w for w, _, _ in ln) for ln in block)))
        last_end = max(last_end, float(it["end"]))
    flush(None)

    out.sort(key=lambda c: c[0])
    fixed = []
    for k, (start, end, text) in enumerate(out):
        if end - start < MIN_SHOW:
            nxt = next((c[0] for c in out[k + 1:] if c[0] > start), None)
            end = start + MIN_SHOW if nxt is None else max(end, min(start + MIN_SHOW, nxt))
        if end <= start:
            end = start + 0.05
        fixed.append((start, end, text))
    return fixed


def to_srt(rows: list[tuple[float, float, str]]) -> str:
    return "".join(f"{n}\n{stamp(s)} --> {stamp(e)}\n{text}\n\n" for n, (s, e, text) in enumerate(rows, 1))


def main(argv: list[str] | None = None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="final.json -> 눈으로 확인하는 자막 파일(SRT)")
    ap.add_argument("film", help="film_id(films/{film_id}/final.json) 또는 final.json 경로")
    ap.add_argument("--out", default=None, help="자막 파일 경로(기본: final.json 옆 preview.srt)")
    ap.add_argument("--absent", action="store_true", help="영상에 없는 대본 대사를 [없음] 으로 끼운다")
    ap.add_argument("--width", type=int, default=WIDTH, help=f"한 줄 글자 수 상한(기본 {WIDTH})")
    args = ap.parse_args(argv)
    src = Path(args.film)
    if not src.is_file():
        src = paths.FILMS / args.film / "final.json"
    if not src.is_file():
        print(f"final.json 을 찾을 수 없음: {args.film}", file=sys.stderr)
        return 1
    data = util.read_json(src)
    rows = cues(data, absent=args.absent, width=max(8, args.width))
    out = Path(args.out) if args.out else src.with_name("preview.srt")
    util.write_text(out, to_srt(rows), encoding="utf-8-sig")       # BOM: 윈도우 플레이어가 UTF-8 로 읽게
    print(f"{out}: 자막 {len(rows)}개 (대사 항목 {sum(1 for it in data.get('items') or [] if it.get('type') == 'dialogue')}개)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
