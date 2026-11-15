"""REPORT.md 생성: 전체 실행 결과와 수락 기준(A1~A3, B1~B5, C1~C3, D1~D9, E1~E2) 증거.

사용: .venv/Scripts/python -I tools/make_report.py [--out FILE] [--root DIR] [--films DIR] [--no-eval] [--eval-dir DIR]
                                                  [--workers N]
  --out FILE      보고를 쓸 파일(기본 저장소의 REPORT.md)
  --root DIR      DIR 을 저장소로 보고 센다: sinkprep.paths 에서 ROOT 아래에 있는 경로(catalog.csv, docs/, films/, cache/,
                  qa/ 등)를 DIR 아래로 바꾼다. schema/·examples/ 와 정답 표본(tests/gold)은 그대로 이 저장소의 것
  --films DIR     films/ 만 DIR 로 바꿔 센다(다른 곳에서 만든 영화 결과를 지금 catalog·cache 와 함께 볼 때)
  --no-eval       5절 파서 정확도(tools/eval_parse.py 10번 호출)를 건너뛴다
  --eval-dir DIR  eval_parse 결과 JSON 을 DIR 에 쓰고, eval_parse 가 쓰는 캐시(규칙 판 cache/parse_rules)도 DIR/cache 로
                  돌린다(저장소의 cache/ 를 건드리지 않고 돌려 볼 때). 주지 않으면 전처럼 cache/eval_*.json
  --workers N     검사(validate) 병렬 수(기본 min(CPU, 12))
숫자는 모두 지금 디스크의 결과물에서 다시 센다(손으로 쓴 숫자 없음). 대사 글은 적지 않는다(수치·id·파일 경로만).
검사는 `sinkprep validate`(인자 없음)와 같은 파일·같은 함수로 이 도구 안에서 돌려 오류를 코드별로 모두 센다(화면에 찍는
20개 상한 없이). 영상 쪽(6b절)은 영화마다 catalog 행, cache/audio/{sha1}.timemap.json, films/{film_id}/ 의 asr·align·final,
qa/needs_review.csv, cache/problems/{asr,align,final}/{film_id}.json 만 읽는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from sinkprep import catalog, paths, util, validate  # noqa: E402

PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
KEEP_PATHS = ("SCHEMA", "EXAMPLES")          # --root 로도 바꾸지 않는 경로(검사 규칙과 예시는 이 저장소의 것)
PROBLEM_STAGES = ("asr", "align", "final")   # 영화 단위 문제 기록을 남기는 단계
_CODE = re.compile(r"^\[([^\]]+)\]")

# eval_parse 를 sinkprep.paths 를 바꾼 채로 돌리는 감싸개(환경 변수 SINK_REPORT_PATHS 의 {이름: 경로})
_EVAL_RUNNER = (
    "import json, os, runpy, sys\n"
    "from pathlib import Path\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from sinkprep import paths\n"
    "for k, v in json.loads(os.environ['SINK_REPORT_PATHS']).items():\n"
    "    setattr(paths, k, Path(v))\n"
    "sys.argv = sys.argv[2:]\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n"
)


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


# ------------------------------------------------------------------ 경로 바꾸기
def path_overrides(root: str | Path | None = None, films: str | Path | None = None) -> dict[str, str]:
    """--root·--films -> {sinkprep.paths 속성 이름: 새 경로}."""
    ov: dict[str, str] = {}
    if root:
        new, base = Path(root).resolve(), Path(paths.ROOT)
        for name, val in vars(paths).items():
            if name in KEEP_PATHS or not isinstance(val, Path):
                continue
            try:
                rel = val.relative_to(base)
            except ValueError:
                continue                     # ROOT 밖(한컴·반디집 실행 파일 등)
            ov[name] = str(new / rel)
    if films:
        ov["FILMS"] = str(Path(films).resolve())
    return ov


def apply_paths(ov: dict[str, str]) -> dict[str, Path]:
    """sinkprep.paths 속성을 바꾸고 예전 값을 돌려준다(검사 일꾼 프로세스의 initializer 로도 쓴다)."""
    old = {k: getattr(paths, k) for k in ov}
    for k, v in ov.items():
        setattr(paths, k, Path(v))
    return old


def show(p: Path) -> str:
    """보고에 적는 경로: 저장소(paths.ROOT) 안이면 상대 경로."""
    try:
        return Path(p).resolve().relative_to(Path(paths.ROOT).resolve()).as_posix()
    except ValueError:
        return Path(p).as_posix()


def read_json_or_none(p: Path):
    try:
        return util.read_json(p) if p.is_file() else None
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ 검사(validate)
def code_of(err: str) -> str:
    m = _CODE.match(err)
    return m.group(0) if m else "[?]"


def counts(c: Counter) -> str:
    return ", ".join(f"{k} {v:,}" for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))) or "없음"


class VRes:
    """검사 결과 [(경로 또는 덮기 이름, 종류, 오류 목록)] 와 걸린 시간."""

    def __init__(self, results: list[tuple[str, str, list[str]]], seconds: float) -> None:
        self.results = results
        self.seconds = seconds
        self._by_path = {}
        for ps, kind, errs in results:
            if kind != "coverage":
                self._by_path[self._key(ps)] = errs

    @staticmethod
    def _key(p) -> str:
        return os.path.normcase(str(Path(p).resolve()))

    def of(self, path: Path) -> list[str] | None:
        """그 파일의 오류 목록(검사하지 않은 파일이면 None)."""
        return self._by_path.get(self._key(path))

    def group(self, ps: str, kind: str) -> str:
        """결과가 속한 곳: coverage, catalog, needs_review, docs, films, examples, 그 밖."""
        if kind == "coverage":
            return "coverage"
        if kind in ("catalog", "needs_review"):
            return kind
        p = Path(ps).resolve()
        for name, base in (("docs", paths.DOCS), ("films", paths.FILMS), ("examples", paths.EXAMPLES)):
            try:
                p.relative_to(Path(base).resolve())
                return name
            except ValueError:
                continue
        return "other"

    def codes(self, groups: tuple[str, ...] | None = None, kinds: tuple[str, ...] | None = None) -> Counter:
        c: Counter = Counter()
        for ps, kind, errs in self.results:
            if kinds is not None and kind not in kinds:
                continue
            if groups is not None and self.group(ps, kind) not in groups:
                continue
            c.update(code_of(e) for e in errs)
        return c

    def errors(self, groups: tuple[str, ...] | None = None) -> list[str]:
        return [e for ps, kind, errs in self.results if groups is None or self.group(ps, kind) in groups for e in errs]

    def summary(self) -> str:
        n_fail = sum(bool(errs) for _, _, errs in self.results)
        n_err = sum(len(errs) for _, _, errs in self.results)
        return f"검사 {len(self.results):,}개, 실패 {n_fail:,}개, 오류 {n_err:,}개, {self.seconds:.0f}초"

    def by_kind(self) -> str:
        k: dict[str, list[int]] = {}
        for _, kind, errs in self.results:
            x = k.setdefault(kind, [0, 0])
            x[0] += 1
            x[1] += bool(errs)
        return ", ".join(f"{kind} {n:,}개 중 실패 {f:,}" for kind, (n, f) in sorted(k.items()))


def run_validate(workers: int | None = None, ov: dict[str, str] | None = None) -> VRes:
    """`sinkprep validate`(인자 없음)와 같은 파일 목록·같은 검사. 오류를 상한 없이 모은다(파일당 validate.MAX_ERRORS 까지)."""
    t0 = time.time()
    files, cover = validate._collect(argparse.Namespace(paths=None, docs=None, limit=None))
    jobs = sorted((str(p) for p in files), key=lambda s: -(Path(s).stat().st_size if Path(s).exists() else 0))
    workers = workers or min(os.cpu_count() or 1, 12)
    if workers > 1 and len(jobs) >= 16:
        kw = {"initializer": apply_paths, "initargs": (ov,)} if ov else {}
        with ProcessPoolExecutor(max_workers=workers, **kw) as ex:
            results = list(ex.map(validate._check_file_worker, jobs, chunksize=max(1, len(jobs) // (workers * 8))))
    else:
        results = [validate._check_file_worker(j) for j in jobs]
    out = []
    for ps, kind, errs in results:
        if not Path(ps).exists():
            errs = errs or ["[io] 파일이 없음"]
        out.append((ps, kind, errs))
    out += [(name, "coverage", errs) for name, errs in cover]
    return VRes(out, time.time() - t0)


def verdict(errs: list[str] | None) -> str:
    if errs is None:
        return "검사 안 함"
    if not errs:
        return "검사 통과"
    return f"검사 오류 {len(errs):,}개({counts(Counter(code_of(e) for e in errs))})"


# ------------------------------------------------------------------ 영상 쪽
def video_films(rows: list[dict]) -> list[tuple[str, dict]]:
    """catalog 에서 영상(video_path)이 있는 영화마다 (film_id, 대표 행). 대표(is_primary) 행이 없으면 첫 행."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("status") != "excluded" and r.get("video_path") and r.get("film_id"):
            by[r["film_id"]].append(r)
    return [(fid, next((r for r in rs if r.get("is_primary") == "true"), rs[0])) for fid, rs in sorted(by.items())]


def _code_versions() -> dict[str, str]:
    """지금 코드의 단계 판(결과물이 낡았는지 견주려고). 모듈을 못 읽으면 빈 값."""
    out = {}
    for name, attr in (("media", "MADE"), ("asr", "VERSION"), ("align", "VERSION"), ("final", "VERSION")):
        try:
            mod = __import__(f"sinkprep.{name}", fromlist=[attr])
            out[name] = str(getattr(mod, attr))
        except Exception:  # noqa: BLE001 - 다른 작업자가 고치는 중이어도 보고는 만든다
            out[name] = ""
    return out


def model_summary(model: str | None) -> str:
    """pass 의 model 문자열 요약: 모델 저장소@스냅숏 · 장치·계산형 · 코드 판(+ rescue 규칙)."""
    if not model:
        return "-"
    parts = [x.strip() for x in model.split("|")]
    if len(parts) < 3:
        return model[:120]
    keep = [parts[1], parts[2]] + [x for x in parts if x.startswith("sinkprep.")]
    k = next((i for i, x in enumerate(parts) if x.startswith("rescue:")), None)
    s = " · ".join(keep)
    if k is not None:
        s += " · " + " | ".join(parts[k:])
    return s


def _ver_in(text: str | None, mod: str) -> str | None:
    m = re.search(rf"sinkprep\.{mod} (\S+)", text or "")
    return m.group(1) if m else None


def _stale(have: str | None, now: str, what: str) -> str:
    return f" (지금 코드는 {what} {now}: 다시 만들면 바뀜)" if have and now and have != now else ""


def _num(v, fmt: str = "") -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return format(v, fmt) if fmt else str(v)
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _qval(v) -> str:
    if isinstance(v, list):
        return f"{len(v)}개"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k} {_qval(x)}" for k, x in v.items()) + "}"
    return _num(v)


def _pct(a: int, b: int) -> str:
    return f"{a:,}/{b:,}({100 * a / b:.1f}%)" if b else f"{a}/0"


def film_block(fid: str, row: dict, vres: VRes, nr_rows: list[dict] | None, ver: dict[str, str]) -> tuple[list[str], dict]:
    """영화 하나의 영상 쪽 묶음(글 줄 목록, 요약 표 한 행에 쓸 값)."""
    L: list[str] = []
    w = L.append
    sm = {"film": fid, "doc": row.get("doc_id") or "-", "dur": row.get("video_duration") or "-", "words": "없음",
          "match": "-", "items": "없음", "final": "없음"}
    sha = row.get("video_sha1") or ""
    w(f"### {fid} (대표 문서 {row.get('doc_id')}, catalog status {row.get('status') or '-'}, match {row.get('match') or '-'})")
    w("")
    # 영상
    vid = (f"`{row.get('video_path')}`, 길이 {row.get('video_duration') or '-'}초, 언어 {row.get('video_lang') or '-'}, "
           f"sha1 {sha[:12] + '…' if sha else '-'}")
    tm_p = paths.AUDIO / f"{sha}.timemap.json"
    tm = read_json_or_none(tm_p) if sha else None
    if tm:
        gaps = tm.get("gaps")
        gaps = len(gaps) if isinstance(gaps, list) else gaps
        made = str(tm.get("made") or "")
        w(f"- 영상: {vid}. 시각 대응표(`{show(tm_p)}`): mode {tm.get('mode')}, 틈 {_num(gaps)}곳 합 "
          f"{_num(tm.get('gap_total'), '.3f')}초, 소리 {_num(tm.get('audio_duration'), '.3f')}초 → 영상 "
          f"{_num(tm.get('video_duration'), '.3f')}초, {made or 'made 없음'}"
          + _stale(made or None, ver.get("media", ""), "대응표"))
    else:
        w(f"- 영상: {vid}. 시각 대응표: 없음(`{show(tm_p) if sha else 'cache/audio/{video_sha1}.timemap.json'}`)")

    work = paths.FILMS / fid / "work"
    # 받아쓰기
    ap = work / "asr.json"
    a = read_json_or_none(ap)
    if a is None:
        w(f"- 받아쓰기: 없음(`{show(ap)}`)")
    else:
        vad = a.get("vad") or []
        speech = sum(float(x[1]) - float(x[0]) for x in vad if isinstance(x, (list, tuple)) and len(x) >= 2)
        passes = a.get("passes") or []
        w(f"- 받아쓰기(`{show(ap)}`, 언어 {a.get('lang')}): 말 구간 {len(vad):,}개 합 {speech:,.1f}초, 묶음 "
          f"{len(a.get('chunks') or []):,}개, pass {len(passes)}개, {verdict(vres.of(ap))}")
        words = []
        for ps in passes:
            ws = ps.get("words") or []
            pp = [float(x["p"]) for x in ws if x.get("p") is not None]
            extra = {k: v for k, v in ps.items() if k not in ("id", "model", "context", "words")}
            w(f"  - `{ps.get('id')}`: 낱말 {len(ws):,}개, 평균 확률 {f'{statistics.mean(pp):.3f}' if pp else '-'}, "
              f"context {ps.get('context') or '-'}, 모델 {model_summary(ps.get('model'))}"
              + _stale(_ver_in(ps.get("model"), "asr"), ver.get("asr", ""), "asr")
              + (f", 그 밖의 칸 {json.dumps(extra, ensure_ascii=False)[:200]}" if extra else ""))
            words.append(f"{ps.get('id')} {len(ws):,}")
        rescue = any(ps.get("id") == "rescue" for ps in passes)
        w("  - rescue 통계: " + ("asr.json 에는 rescue pass 의 낱말만 있고 조각 수·버린 수 같은 통계 칸은 없음(asr 단계 "
                                "출력에만)" if rescue else "rescue pass 없음"))
        sm["words"] = ", ".join(words) or "0"

    # 맞추기
    alp = work / "align.json"
    al = read_json_or_none(alp)
    if al is None:
        w(f"- 맞추기: 없음(`{show(alp)}`)")
    else:
        m = al.get("method") or {}
        st = m.get("stats") or {}
        aligner = m.get("aligner")
        w(f"- 맞추기(`{show(alp)}`, {aligner}{_stale(_ver_in(aligner, 'align'), ver.get('align', ''), 'align')}, "
          f"기준 pass {m.get('base_pass') or '-'}): 대사 {_num(st.get('dialogue'))} = 말 있는 대본 대사 "
          f"{_num(st.get('spoken'))} + 말 없는 대사 {_num(st.get('silent_turns'))} + added {_num(st.get('added'))}; "
          f"same {_num(st.get('same'))}, changed {_num(st.get('changed'))}, absent {_num(st.get('absent'))}"
          f"(말 있는 대사 가운데 {_num(st.get('absent_spoken'))}), 일치율 {_num(st.get('match'))}, "
          f"낱말 평균 확률 {_num(st.get('mean_word_p'))}, {verdict(vres.of(alp))}")
        spoken, same, changed = st.get("spoken"), st.get("same") or 0, st.get("changed") or 0
        if isinstance(spoken, int) and spoken:
            w(f"  - 말 있는 대본 대사 가운데 받아쓰기와 짝지은 줄(same+changed) {_pct(same + changed, spoken)}")
        ps_ = m.get("passes") or {}
        if ps_:
            w("  - pass 별(method.passes): " + "; ".join(
                f"`{pid}` " + ", ".join(f"{k} {_num(v)}" for k, v in p.items() if k != "words") for pid, p in ps_.items()
                if isinstance(p, dict)))
        ops = al.get("ops") or []
        used = Counter(str(o["asr"][0]).split(":")[0] for o in ops if isinstance(o.get("asr"), list) and o["asr"])
        w(f"  - 고른 줄의 받아쓰기 pass(ops 의 asr 범위): {counts(used)}; decided_by: "
          f"{counts(Counter(str(o.get('decided_by')) for o in ops))}")
        dec = {k: v for k, v in (m.get("decide") or {}).items() if v not in (0, None, [], {})}
        if dec:
            w("  - pass 고르기(method.decide, 0 아닌 값): " + ", ".join(f"{k} {_qval(v)}" for k, v in dec.items()))
        rep = {f"{where}.{k}": v for where in ("stats", "decide") for k, v in (m.get(where) or {}).items()
               if "repair" in k or "slot" in k}
        w("  - 다시 짝지은 줄: " + (", ".join(f"{k} {_qval(v)}" for k, v in rep.items()) if rep
                                   else "align.json 에 세는 칸 없음"))
        tf = m.get("time_fix")
        if isinstance(tf, dict):
            w(f"  - 시각 보정(method.time_fix): 줄 {_num(tf.get('lines'))}, 낱말 {_num(tf.get('words'))}, "
              f"믿기 어려운 줄 {_qval(tf.get('suspect'))}")
        w(f"  - 씬 구간(align scenes how): {counts(Counter(str(s.get('how')) for s in al.get('scenes') or []))}")
        sm["match"] = _num(st.get("match"))

    # 최종본
    fp = paths.FILMS / fid / "final.json"
    fi = read_json_or_none(fp)
    if fi is None:
        w(f"- 최종본: 없음(`{show(fp)}`)")
    else:
        items = fi.get("items") or []
        made = (fi.get("made") or {}).get("final")
        qd = fi.get("quality") or {}
        w(f"- 최종본(`{show(fp)}`, {made}{_stale(_ver_in(made, 'final'), ver.get('final', ''), 'final')}): 항목 "
          f"{len(items):,}개({counts(Counter(str(i.get('type')) for i in items))}), {verdict(vres.of(fp))}")
        w("  - quality: " + (", ".join(f"{k} {_qval(v)}" for k, v in qd.items()) or "없음"))
        dl = [i for i in items if i.get("type") == "dialogue"]
        w(f"  - 대사 status: {counts(Counter(str(i.get('status')) for i in dl))}; 항목 timing: "
          f"{counts(Counter(str(i.get('timing')) for i in items))}")
        scenes = fi.get("scenes") or []
        w(f"  - 씬 {len(scenes):,}개, 구간(start)이 있는 씬 {sum(s.get('start') is not None for s in scenes):,}개")
        sd = [i for i in dl if i.get("status") != "added"]
        w(f"  - 화자 있는 대사(speakers 가 빈 목록이 아님): 대본 대사 {_pct(sum(bool(i.get('speakers')) for i in sd), len(sd))}, "
          f"added 포함 {_pct(sum(bool(i.get('speakers')) for i in dl), len(dl))}; speaker_from "
          f"{counts(Counter(str(i.get('speaker_from')) for i in dl))}")
        w(f"  - 시각 보정: time_fixed {_qval(qd.get('time_fixed'))}줄(낱말 {_qval(qd.get('time_fixed_words'))}), "
          f"time_suspect {_qval(qd.get('time_suspect') or [])}, pass_overlaps {_qval(qd.get('pass_overlaps') or [])}")
        fm = ((fi.get("film") or {}).get("script") or {}).get("match")
        w(f"  - film.script.match {_num(fm)}, catalog match {row.get('match') or '-'}")
        sm["items"] = f"{len(items):,}"
        sm["final"] = verdict(vres.of(fp))
    # 확인 필요
    if nr_rows is None:
        w(f"- 확인 필요(`{show(paths.NEEDS_REVIEW)}`): 없음(파일 없음)")
    else:
        mine = [x for x in nr_rows if x.get("film_id") == fid and x.get("level") in ("film", "item")]
        lv = {k: Counter(x["problem"] for x in mine if x["level"] == k) for k in ("film", "item")}
        w(f"- 확인 필요(`{show(paths.NEEDS_REVIEW)}`): film 수준 {sum(lv['film'].values())}건({counts(lv['film'])}), "
          f"item 수준 {sum(lv['item'].values())}건({counts(lv['item'])})")
    stages = []
    for stg in PROBLEM_STAGES:
        rows_ = read_json_or_none(paths.PROBLEMS / stg / f"{fid}.json")
        if rows_ is None:
            stages.append(f"{stg} 없음")
        else:
            c = Counter(str(x.get("level")) for x in rows_)
            stages.append(f"{stg} " + "·".join(f"{k} {v}" for k, v in sorted(c.items())))
    w(f"  - 단계 문제 기록(`{show(paths.PROBLEMS)}/{{asr,align,final}}/{fid}.json`, review 가 위 목록으로 모음): "
      + ", ".join(stages))
    return L, sm


def video_section(rows: list[dict], vres: VRes) -> list[str]:
    """6b절: catalog 에서 영상이 있는 영화마다 한 묶음."""
    L: list[str] = []
    w = L.append
    w("## 6b. 영상 쪽(영화별: 영상 등록·받아쓰기·맞추기·최종본)")
    w("")
    films = video_films(rows)
    if not films:
        w("- catalog 에 영상(video_path)이 있는 영화가 없음")
        w("")
        return L
    nr_rows = None
    if paths.NEEDS_REVIEW.exists():
        with open(paths.NEEDS_REVIEW, encoding="utf-8-sig", newline="") as f:
            nr_rows = list(csv.DictReader(f))
    ver = _code_versions()
    blocks, sums = [], []
    for fid, row in films:
        b, s = film_block(fid, row, vres, nr_rows, ver)
        blocks.append(b)
        sums.append(s)
    w(f"catalog 에서 영상이 있는 영화 {len(films)}편. 결과물은 `{show(paths.FILMS)}/{{film_id}}/` (work/asr.json, "
      "work/align.json, final.json). 파일이 없으면 '없음'.")
    w("")
    w("| 영화 | 대표 문서 | 영상 길이(초) | 받아쓰기 낱말(pass) | 일치율 | 최종본 항목 | final.json 검사 |")
    w("|---|---|---|---|---|---|---|")
    for s in sums:
        w(f"| {s['film']} | {s['doc']} | {s['dur']} | {s['words']} | {s['match']} | {s['items']} | {s['final']} |")
    w("")
    for b in blocks:
        L.extend(b)
        w("")
    return L


# ------------------------------------------------------------------ 수락 기준 C·D
C_HOW = {
    "C1": "validate_asr: pass 마다 낱말 번호 i 가 0..n-1 로 빠짐·겹침 없음, s<=e. validate_align: op 의 asr 참조(pass:i)가 "
          "asr.json 안에 있고 script_el 이 script.json 에 있음, asr_sha1 = asr.json 파일 sha1",
    "C2": "validate_align: op 의 item 과 final.json 대사 항목이 1:1, 같은 pass 의 같은 낱말을 두 op 이 갖지 않음, added op 에 "
          "asr 범위가 있음, 대사 항목 words = 그 op 의 asr 범위 낱말(timeline.op_words). final 의 film.id = 폴더 이름",
    "C3": "asr.json 스키마(sink.asr/1)에 정렬 칸이 없고 align.json 의 asr_sha1 이 지금 asr.json 과 같음([C1] 로 냄). "
          "같은 asr.json 에 받아쓰기 캐시(cache/asr 의 rescue 기록: 버릇 글귀)와 LLM 캐시(cache/llm 짝 판정)도 같을 때 다시 "
          "맞추면 바이트가 같은지는 tests/test_align.py·test_final.py(align 은 asr.json 말고 두 캐시도 입력으로 읽는다)",
}
D_HOW = {
    "D1": "인자 없는 validate 가 catalog 에서 영상이 있는 대표 문서마다 films/{film_id}/final.json 이 있는지, 그 film 칸(대표 "
          "문서·영상 sha1·길이·match)이 catalog 와 같은지 봄([D1]), 스키마는 [schema]",
    "D2": "validate_final: same·changed·added 는 start<end, words 있음, 낱말이 줄 안, heard = words, 시각이 영상 길이 "
          "+0.5초 안, 줄 안 낱말 시작 시각이 줄지 않음",
    "D3": "validate_final: absent 는 start=end=null·script_ref 있음, added 는 script_ref=null",
    "D4": "validate_final: changed 는 diffs 있음·same 은 없음, match=text 줄은 textnorm.key(heard)와 key(script_ref.text)가 "
          "same 이면 같고 changed 면 다름",
    "D5": "validate_final: 시각 있는 대사에 ko 가 있고, lang=ko 면 ko_src=same·ko = heard, 시각 있는 대사의 들린 글이 "
          "영상 언어의 글자면 lang 이 영상 언어(textnorm.spoken_lang)",
    "D6": "validate_final: match=translation 이면 diffs 없음",
    "D7": "validate_final: items 가 시각·id 차례, 시각 없는 항목은 대본 순서로 앞뒤 항목 사이",
    "D8": "validate_final: scene·speakers·script_ref.el·src_el 참조가 있음, script.json 을 찾으면 그 scene·action·dialogue·"
          "onscreen·transition 요소가 한 번씩 나옴",
    "D9": "validate_final: script.json 을 찾았을 때 대본에서 복사한 칸(대사의 speakers·how·lang·paren·sung·overlap·scene·"
          "script_ref.text, 대사 아닌 항목의 text·scene·kind, 씬·인물 칸)이 대본과 같고(시각 있는 대사의 lang 은 "
          "textnorm.spoken_lang 으로 정한 값) 씬 구간이 그 씬 항목 시각과 맞음. "
          "곧 CBoE 입력 칸이 final.json 만으로 채워짐",
}


def criteria_cd(rows: list[dict], vres: VRes) -> list[str]:
    """수락 기준 C1~C3, D1~D9 줄: 검사 결과(오류 코드별 수)와 검사 방법."""
    L: list[str] = []
    w = L.append
    films_c = vres.codes(groups=("films",))
    ex_c = vres.codes(groups=("examples",))
    cov_c = vres.codes(groups=("coverage",))
    kinds = Counter(kind for ps, kind, _ in vres.results if vres.group(ps, kind) == "films")
    w(f"- 영상 쪽 검사 대상: films 의 {counts(kinds) if kinds else '파일 없음'}, examples 의 asr·align·final 도 같은 검사")
    asr_schema = vres.codes(groups=("films", "examples"), kinds=("asr",))["[schema]"]
    sha_bad = sum("asr_sha1" in e for e in vres.errors(groups=("films", "examples")) if e.startswith("[C1]"))
    for k in ("C1", "C2"):
        w(f"- {k} 오류 films {films_c[f'[{k}]']:,} · examples {ex_c[f'[{k}]']:,}. 검사: {C_HOW[k]}")
    w(f"- C3 asr_sha1 어긋남 {sha_bad:,}개, asr.json 스키마 오류 {asr_schema:,}개(films·examples). 검사: {C_HOW['C3']}")
    vf = video_films(rows)
    finals = [(fid, paths.FILMS / fid / "final.json") for fid, _ in vf]
    have = [(fid, p) for fid, p in finals if p.is_file()]
    d1_cover = [e for e in vres.errors(groups=("coverage",)) if e.startswith("[D1]")]
    fin_schema = vres.codes(groups=("films",), kinds=("final",))["[schema]"]
    w(f"- D1 영상이 있는 영화 {len(vf)}편 가운데 final.json 이 있는 영화 {len(have)}편; [D1] 오류 {cov_c['[D1]']:,}개"
      f"(없는 final.json·catalog 와 어긋남), films 의 final.json 스키마 오류 {fin_schema:,}개. 검사: {D_HOW['D1']}")
    for k in ("D2", "D3", "D4", "D5", "D6", "D7", "D8"):
        w(f"- {k} 오류 films {films_c[f'[{k}]']:,} · examples {ex_c[f'[{k}]']:,}. 검사: {D_HOW[k]}")
    with_script = 0                      # validate_final 이 대본과 견준(script.json 을 찾은) final.json 수
    for _, p in have:
        d = read_json_or_none(p) or {}
        doc = ((d.get("film") or {}).get("script") or {}).get("doc_id") if isinstance(d, dict) else None
        with_script += validate._find_script(doc, p.parent) is not None
    ok = []
    for fid, p in have:
        errs = vres.of(p)
        if errs == [] and not any(e.startswith(f"[D1] {fid}:") for e in d1_cover):
            ok.append(fid)
    w(f"- D9 오류 films {films_c['[D9]']:,} · examples {ex_c['[D9]']:,}, 대본(script.json)과 견준 final.json "
      f"{with_script}/{len(have)}개. **영상이 있는 영화 {len(vf)}편 중 final.json 이 검사를 통과한 영화 {len(ok)}편**"
      + (f"({', '.join(ok)})" if ok else "") + f". 검사: {D_HOW['D9']}")
    other = Counter({k: v for k, v in films_c.items() if not re.fullmatch(r"\[[CD]\d\]", k)})
    w(f"- films 의 그 밖의 오류 코드(스키마·NFC·시각 자리수 등): {counts(other)}")
    return L


# ------------------------------------------------------------------ 대본 쪽 보강(공백 복원·파서 판·새 규칙)
def new_rule_tags() -> list[tuple[str, re.Pattern]]:
    """parse 0.3.x 가 quality.notes 에 남기는 새 꼬리표(tools/make_types.py 분기 표의 값과 같은 이름)."""
    import make_types

    rx = dict(make_types._BRANCH_RE)
    return [(k, rx[k]) for k in make_types.PARSE_03_BRANCHES]


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="REPORT.md 생성")
    ap.add_argument("--out", default=None, help="보고를 쓸 파일(기본 REPORT.md)")
    ap.add_argument("--root", default=None, help="이 폴더를 저장소로 보고 센다")
    ap.add_argument("--films", default=None, help="films/ 만 이 폴더로 바꿔 센다")
    ap.add_argument("--no-eval", action="store_true", help="5절 파서 정확도(eval_parse)를 건너뛴다")
    ap.add_argument("--eval-dir", default=None, help="eval_parse 결과와 그 캐시를 이 폴더에(저장소 cache/ 를 건드리지 않음)")
    ap.add_argument("--workers", type=int, default=None, help="검사 병렬 수")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else ROOT / "REPORT.md"
    ov = path_overrides(a.root, a.films)
    old = apply_paths(ov)
    try:
        text = build(a, ov)
    finally:
        apply_paths({k: str(v) for k, v in old.items()})
    util.write_text(out, text)
    print(text)
    return 0


def build(a: argparse.Namespace, ov: dict[str, str]) -> str:
    t0 = time.time()
    rows = catalog.load()
    docs = [r for r in rows if r["status"] != "excluded"]
    excl = [r for r in rows if r["status"] == "excluded"]
    L = []
    w = L.append

    w("# sink 결과 보고: 대본 형식 통일 전처리와 영상 맞추기")
    w("")
    w(f"만든 시각: {time.strftime('%Y-%m-%d %H:%M')} · 생성기 `tools/make_report.py`(모든 숫자는 디스크 결과물에서 다시 셈)")
    if ov:
        w("")
        w("경로를 바꿔 셈: " + ", ".join(f"{k}={Path(v).as_posix()}" for k, v in sorted(ov.items()) if k in ("ROOT", "FILMS")))
    w("")
    # ---------- 원본과 카탈로그
    raw_files = [p for p in paths.RAW.rglob("*") if p.is_file() and p.name != "manifest.csv"] if paths.RAW.exists() else []
    ext_files = [p for p in paths.EXTRACTED.rglob("*") if p.is_file()] if paths.EXTRACTED.exists() else []
    films = {r["film_id"] for r in docs if r["film_id"]}
    w("## 1. 원본과 카탈로그")
    w(f"- 원본 파일 {len(raw_files)}개(manifest 제외) + 압축에서 나온 파일 {len(ext_files)}개 = {len(raw_files) + len(ext_files)}개")
    w(f"- catalog.csv {len(rows)}행: 문서 {len(docs)}개, 제외 {len(excl)}개, 영화 {len(films)}편")
    fmt_files = Counter(f for r in rows for f in r["file_formats_list"])
    w("- 파일별 실제 형식: " + ", ".join(f"{k} {v}" for k, v in fmt_files.most_common()))
    fmt_docs = Counter(r["real_format"] for r in docs)
    w("- 문서별 형식: " + ", ".join(f"{k} {v}" for k, v in fmt_docs.most_common()))
    exn = Counter((r["note"] or "").split(":")[0] for r in excl)
    w("- 제외 사유: " + ", ".join(f"{k} {v}" for k, v in exn.most_common()))
    lang = Counter(r["script_lang"] for r in docs)
    w("- 대본 언어(review 판정): " + ", ".join(f"{k} {v}" for k, v in lang.most_common()))
    w(f"- 번역 대본: {sum(1 for r in docs if r['is_translation'] == 'true')}개")
    # A3: 확장자와 실제 형식이 다른 파일
    ext_of = {".hwp": {"hwp3", "hwp5"}, ".doc": {"doc"}, ".docx": {"docx"}, ".pdf": {"pdf_text", "pdf_scan", "pdf_broken"},
              ".txt": {"txt"}, ".rtf": {"rtf"}, ".xls": {"xls"}}
    disguised = []
    for r in rows:
        for f, fmt in zip(r["files_list"], r["file_formats_list"]):
            e = Path(f).suffix.lower()
            if e in ext_of and fmt not in ext_of[e] and fmt != "archive":
                disguised.append((f, fmt))
    w(f"- 확장자와 실제 형식이 다른 파일 {len(disguised)}개(앞 바이트로 판별): "
      + "; ".join(f"{Path(f).name} -> {fmt}" for f, fmt in disguised[:10]))
    w("")
    # ---------- 변환
    log = defaultdict(dict)
    if (paths.CACHE / "convert_log.jsonl").exists():
        for x in util.read_jsonl(paths.CACHE / "convert_log.jsonl"):
            log[(x["doc_id"], x["k"])] = x
    w("## 2. 한컴 변환(HWP3/HWP5/doc/docx/rtf -> PDF)")
    if log:
        ok = Counter((x["real_format"], x["ok"], x["method"]) for x in log.values())
        w(f"- 변환 대상 파일 {len(log)}개(최근 기록 기준)")
        for (fmt, okv, m), n in sorted(ok.items()):
            w(f"  - {fmt}: {'성공' if okv else '실패'} {m} {n}")
        secs = [x["seconds"] for x in log.values() if x.get("seconds")]
        if secs:
            w(f"- 파일당 시간 중앙값 {statistics.median(secs):.1f}초, 최대 {max(secs):.1f}초")
        losses = [x["loss"] for x in log.values() if x.get("loss") is not None]
        switched = [k for k, x in log.items() if "hancom_pdf_incomplete" in (x.get("note") or "")]
        warned = [k for k, x in log.items() if x.get("method") == "hancom" and (x.get("loss") or 0) > 0.01]
        if losses:
            w(f"- 교차 검사(대체 경로로 읽은 원본 글자 중 PDF 에 없는 비율 loss): {len(losses)}개 파일 검사, "
              f"중앙값 {statistics.median(losses):.4f}, 0.002 미만 {sum(v < 0.002 for v in losses)}개; "
              f"PDF 대신 원본 글 사용(loss>0.03) {len(switched)}개 {[d for d, _ in switched]}, "
              f"PDF 쓰고 확인 필요(0.01~0.03) {len(warned)}개 {[d for d, _ in warned]}")
    chk = paths.CACHE / "convert_check.json"
    if chk.exists():
        rows_chk = [x for x in util.read_json(chk)["rows"] if "error" not in x]
        by = defaultdict(list)
        for x in rows_chk:
            by[x["fmt"]].append(x)
        w("- 형식별 내용 대조(`tools/check_convert.py`, 한컴 변환 성공 파일 전부, 공백 뺀 글자 다중집합 겹침 / 20자 구간 일치):")
        for fmt in sorted(by):
            ov_ = [x["overlap"] for x in by[fmt]]
            sg = [x["seg_pct"] for x in by[fmt]]
            w(f"  - {fmt}: {len(ov_)}개, 겹침 중앙값 {statistics.median(ov_):.3f}, 구간 일치 중앙값 {statistics.median(sg):.0f}%")
    w("")
    # ---------- OCR
    ocr_files = sorted(paths.OCR_CACHE.glob("*/*.json")) if paths.OCR_CACHE.exists() else []
    w("## 3. OCR (RapidOCR, PP-OCRv5 korean / PP-OCRv6 en)")
    if ocr_files:
        why = Counter()
        pages = 0
        confs = []
        failed = 0
        for p in ocr_files:
            d = util.read_json(p)
            why[d.get("why", "?")] += 1
            pages += len({pg["page"] for pg in d.get("pages", [])})
            failed += len(d.get("failed_pages") or [])
            if d.get("conf_mean") is not None:
                confs.append(d["conf_mean"])
        w(f"- OCR 파일 {len(ocr_files)}개, {pages}쪽, 그림 실패 쪽 {failed}개; 사유: " + ", ".join(f"{k} {v}" for k, v in why.most_common()))
        sup = sorted(paths.OCR_SUPERSEDED.glob("*/*.json")) if paths.OCR_SUPERSEDED.exists() else []
        if sup:
            w(f"- 글자층 복원(lines)으로 충분해 쓰지 않는 예전 OCR 결과 {len(sup)}개(cache/ocr_superseded): "
              + ", ".join(f"{p.parent.name}/{p.stem}" for p in sup))
        if confs:
            w(f"- 파일별 평균 신뢰도: 중앙값 {statistics.median(confs):.3f}, 최저 {min(confs):.3f}, 0.85 미만 {sum(c < 0.85 for c in confs)}개")
    w("")
    # ---------- lines / script
    w("## 4. 형식 통일 줄(lines.jsonl)과 구조 파싱(script.json)")
    n_lines, src = [], Counter()
    n_script = 0
    types = Counter()
    parsers = Counter()
    parser_ver = Counter()
    lines_ver = Counter()
    respaced, skipped = [], []
    tags = new_rule_tags()
    tag_docs: dict[str, list[str]] = {k: [] for k, _ in tags}
    qs = []
    q_items = defaultdict(list)
    empty = []
    for r in docs:
        lp = paths.DOCS / r["doc_id"] / "lines.jsonl"
        if lp.exists():
            allr = util.read_jsonl(lp)
            head, lr = (allr[0], allr[1:]) if allr and "schema" in allr[0] else ({}, allr)
            n_lines.append(len(lr))
            if not lr:
                empty.append(r["doc_id"])
            for x in lr:
                src[x["src"]] += 1
            made = head.get("made") or {}
            lines_ver[str(made.get("version"))] += 1
            for rec in made.get("spacing") or []:
                if rec.get("from") == "source":
                    respaced.append((r["doc_id"], rec.get("file"), rec.get("loss")))
                else:
                    skipped.append((r["doc_id"], rec.get("file"), rec.get("skipped")))
        sp = paths.DOCS / r["doc_id"] / "script.json"
        if sp.exists():
            n_script += 1
            s = util.read_json(sp)
            pz = s.get("parser") or {}
            parsers[pz.get("name")] += 1
            parser_ver[f"{pz.get('name')} {pz.get('version')}"] += 1
            for el in s.get("elements", []):
                types[el["type"]] += len(el.get("lines", []))
            qq = s.get("quality") or {}
            if qq.get("score") is not None:
                qs.append(qq["score"])
            for k in ("typed_ratio", "dialogue_ratio", "unknown_ratio"):
                if qq.get(k) is not None:
                    q_items[k].append(qq[k])
            notes = qq.get("notes") or []
            for k, rx in tags:
                if any(rx.search(nt) for nt in notes):
                    tag_docs[k].append(r["doc_id"])
    w(f"- lines.jsonl 있는 문서 {len(n_lines)}/{len(docs)}, 줄 0개 문서 {len(empty)}개, 총 줄 {sum(n_lines):,}")
    w("- 줄 출처: " + ", ".join(f"{k} {v:,}" for k, v in src.most_common()))
    w("- lines 판(머리 행 made.version): " + counts(lines_ver))
    w(f"- 공백 복원(머리 행 `made.spacing` 에 from=source, 한컴 PDF 줄의 공백·탭을 원본 문서 글로 맞춤): {len({d for d, _, _ in respaced})}문서"
      + (": " + ", ".join(f"{d}({'' if k in (0, None) else f'파일 {k} '}손실 {_num(lo)})" for d, k, lo in respaced)
         if respaced else ""))
    w(f"- 공백 손실을 못 견준 파일(from=null, skipped): {len(skipped)}개"
      + (": " + ", ".join(f"{d}/{k} {why}" for d, k, why in skipped) if skipped else ""))
    w(f"- script.json 있는 문서 {n_script}/{len(docs)}")
    w("- 파서 판(parser.name·version): " + counts(parser_ver))
    w("- parse 0.3.x 새 규칙이 걸린 문서(docs 의 script.json quality.notes 꼬리표, 값 이름은 TYPES.md 5.2절): "
      + ", ".join(f"`{k}` {len(v)}" + (f"({', '.join(v[:12])}{' …' if len(v) > 12 else ''})" if v else "")
                  for k, v in tag_docs.items()))
    tot = sum(types.values()) or 1
    w("- 요소 종류별 줄 비율: " + ", ".join(f"{k} {100 * v / tot:.1f}%" for k, v in types.most_common()))
    if qs:
        w(f"- 문서 품질 점수 분포: 10% {q(qs, .1):.2f} · 중앙값 {q(qs, .5):.2f} · 90% {q(qs, .9):.2f}")
    for k, v in q_items.items():
        w(f"  - {k}: 10% {q(v, .1):.2f} · 중앙값 {q(v, .5):.2f} · 90% {q(v, .9):.2f}")
    w("")
    # ---------- 정답 표본
    t_eval = time.time()
    w("## 5. 파서 정확도(따로 만든 정답 표본)")
    w("규칙 판 = 규칙 파서(sink-parse)만, 지금 판 = docs/*/script.json(llm 단계가 받아들인 문서는 LLM 판). 값은 요소 종류% / 화자%. "
      "hard-test 는 처음 열었을 때(LLM 단계 고치기 전) LLM 창 채점 100/100(120줄 조각 98.9/100)이었고, 그 뒤 hard-test 문서 사례를 보고 "
      "조립을 고쳤으므로 지금 값은 개발용으로 본다.")
    w("")
    if a.no_eval:
        w("- 건너뜀(`--no-eval`)")
    else:
        edir = Path(a.eval_dir).resolve() if a.eval_dir else paths.CACHE
        eov = dict(ov, CACHE=str(edir / "cache")) if a.eval_dir else ov
        env = dict(os.environ, SINK_REPORT_PATHS=json.dumps(eov, ensure_ascii=False)) if eov else None
        for name, sset in (("개발용(dev)", "dev"), ("확인용(test, 파서 개발 중 안 봄)", "test"), ("dev+test+hard-dev", "all"),
                           ("어려운 꼴 개발용(hard-dev)", "hard-dev"), ("어려운 꼴 확인용(hard-test)", "hard-test")):
            got = {}
            for which in ("rules", "current"):
                p = edir / f"eval_{sset}_{which}.json"
                if p.exists():
                    p.unlink()                  # 이번 실행이 실패했는데 예전 값을 적지 않게
                args = [str(ROOT / "tools" / "eval_parse.py"), "--set", sset, "--which", which, "--json", str(p)]
                cmd = [PY, "-I", "-c", _EVAL_RUNNER, str(ROOT)] + args if eov else [PY, "-I"] + args
                r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
                if r.returncode == 0 and p.exists():
                    got[which] = json.loads(p.read_text(encoding="utf-8"))["total"]
            if got:
                t = next(iter(got.values()))
                w(f"- {name}: 줄 {t.get('lines', 0)}개(대사 줄 {t.get('spk_total', 0)}개), " + ", ".join(
                    f"{'규칙 판' if k == 'rules' else '지금 판'} {100 * v.get('type_ok', 0) / max(1, v.get('lines', 0)):.1f} / "
                    f"{100 * v.get('spk_ok', 0) / max(1, v.get('spk_total', 0)):.1f}" for k, v in got.items()))
            else:
                w(f"- {name}: eval_parse 실패(결과 없음)")
    t_eval = time.time() - t_eval
    gold = [p for p in (ROOT / "tests" / "gold").glob("*.jsonl")]
    gdocs = {util.read_jsonl(p)[0]["doc_id"] for p in gold if util.read_jsonl(p)}
    w(f"- 정답 표본: 구간 {len(gold)}개, 문서 {len(gdocs)}개. Claude 모델(서브에이전트) 둘이 따로 표시하고 갈린 줄은 셋째 "
      "서브에이전트가 판정한 정답이다(claude-opus-5-5, 사람이 붙인 정답이 아님). 위 정확도는 이 정답에 대한 값이고, LLM 판"
      "(5b 절의 모델)은 같은 모델이 붙인 정답과 견준 값이다")
    w("")
    # ---------- llm 단계
    from sinkprep import llm, llm_parse
    w("## 5b. LLM 파싱 단계(llm)")
    dec = Counter(util.read_json(f).get("decision") for f in (paths.STATE / "llm").glob("*.json"))
    w(f"- 모델 {llm.MODEL}(effort {llm.EFFORT}), 프롬프트 {llm_parse.PROMPT_VERSION}, 조립 {llm_parse.VERSION}. "
      f"다룬 문서 {sum(dec.values())}개: LLM 판 사용 {dec['llm']}, 규칙 판 유지 {dec['rules']}")
    w("- docs 의 파서: " + ", ".join(f"{k} {v}" for k, v in parsers.most_common()))
    w(f"- claude 응답 캐시 {len(list((paths.CACHE / 'llm').glob('*.json')))}개(cache/llm). 같은 입력은 다시 부르지 않아 "
      "다시 돌려도 바이트 단위로 같다. `sinkprep all` 은 캐시만 쓴다.")
    w("")
    # ---------- needs_review
    w("## 6. 확인 필요 목록(qa/needs_review.csv)")
    if paths.NEEDS_REVIEW.exists():
        with open(paths.NEEDS_REVIEW, encoding="utf-8-sig") as f:
            nr = list(csv.DictReader(f))
        c = Counter(x["problem"] for x in nr)
        w(f"- {len(nr)}건(문서 {len({x['doc_id'] for x in nr if x['doc_id']})}개): " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
        lv = Counter(x.get("level") for x in nr)
        w("- 수준별: " + counts(lv))
    else:
        w(f"- 없음(`{show(paths.NEEDS_REVIEW)}` 파일 없음)")
    w("")
    # ---------- 검사(영상 쪽 절과 수락 기준이 같이 쓴다)
    vres = run_validate(a.workers, ov)
    # ---------- 영상 쪽
    L.extend(video_section(rows, vres))
    # ---------- 수락 기준
    w("## 7. 수락 기준 확인(A1~A3, B1~B5, C1~C3, D1~D9, E1~E2)")
    w("")
    w(f"검사는 `sinkprep validate`(인자 없음)와 같은 파일·같은 함수(catalog, needs_review, docs 전체, films, examples, 덮기 "
      f"B1·D1)를 이 도구 안에서 돌려 오류를 모두 셌다: {vres.summary()}. 종류별 {vres.by_kind()}.")
    w("")
    cat_c = vres.codes(kinds=("catalog",))
    w(f"- A1 모든 원본·압축 해제 파일이 정확히 한 행에: 오류 {cat_c['[A1]']}개")
    w(f"- A2 영화마다 is_primary 최대 1개: 오류 {cat_c['[A2]']}개")
    w(f"- A3 앞 바이트 판별: 위장 파일 {len(disguised)}개 판별(위 1절), tests/test_detect.py")
    w(f"- B1·B2·B3·E1·E2 검사 결과 오류 코드별(전체, 아래 C·D 포함): {counts(vres.codes())}")
    for g in ("docs", "coverage", "catalog", "needs_review", "films", "examples"):
        w(f"  - {g}: {counts(vres.codes(groups=(g,)))}")
    w("- B4 needs_review 기준값: typed_ratio<0.85, dialogue_ratio<0.15, 화자 없는 대사>5%, 말 없는 대사(이름 줄만)>=10%, "
      "추정 화자(speaker_inferred)>10%, OCR conf<0.85, 추출 실패·쪽 그림 실패, 한컴 PDF 글자 손실>1%")
    w("- B5 까다로운 경우 11가지: tests/fixtures/parse/, tests/test_parse.py")
    L.extend(criteria_cd(rows, vres))
    w("")
    w(f"(보고 생성 {time.time() - t0:.0f}초: 검사 {vres.seconds:.0f}초, 파서 정확도 {t_eval:.0f}초)")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
