"""기존 판과 논문 그대로 판 결과를 줄 id 로 맞춰 비교한다(명세 3.13).

  .venv\\Scripts\\python.exe -X utf8 scripts\\compare_versions.py
  .venv\\Scripts\\python.exe -X utf8 scripts\\compare_versions.py --main out\\demo --paper out\\paper_demo ^
      --out out\\compare\\compare.md

-X utf8 은 UTF-8 모드다(run.cmd, run_paper.cmd 가 켜는 PYTHONUTF8=1 과 같음). 여기서 쓰는 G2P 와 근사 프런트엔드는 UTF-8
모드가 아니어도 돌지만, 기존 판과 같은 조건으로 돌리려고 켠다. 실행 중 오류는 한국어 한 줄로 알리고 끝 코드 2 로 끝난다.

읽는 것: 두 폴더의 results.jsonl 과 run.json(읽기만). 쓰는 것: --out 의 compare.md 하나.
줄마다 영어, 기존 판이 고른 대사(음절), 논문 판 ISO, PS, PS-Comet 대사(음절), K' 와 K, DTW 를 적는다.
음절은 두 판 모두 기존 판의 셈(pstts.textnorm.ko_syllable_count: 한글 글자 수, 숫자는 한자어로 읽음)이다.

기존 판 DTW 는 반경 26 이고 프레임을 만드는 법도 달라 논문 판 DTW 와 견줄 수 없다. 그래서 논문 판이 approx 로 돌았으면
기존 판이 고른 대사를 논문 판과 같은 자(approx 프런트엔드, 논문 판 run.json 의 설정과 반경)로 다시 재서
K 와 DTW 를 함께 적는다.
모델은 올리지 않는다(G2P 와 tslearn 만 씀).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pstts.textnorm import ko_syllable_count  # noqa: E402


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """results.jsonl 을 id -> 줄 사전으로(파일 순서 유지)."""
    if not path.is_file():
        raise FileNotFoundError(f"결과 파일이 없습니다: {path}")
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                row = json.loads(ln)
                out[str(row["id"])] = row
    return out


def read_run(folder: Path) -> dict[str, Any]:
    path = folder / "run.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def paper_config(run: dict[str, Any]) -> Any:
    """논문 판 run.json 의 config 로 PaperConfig 를 다시 만든다(모르는 칸은 버림)."""
    from pstts.paper.config import PaperConfig
    names = {f.name for f in fields(PaperConfig)}
    return PaperConfig(**{k: v for k, v in (run.get("config") or {}).items() if k in names})


class PaperRuler:
    """기존 판 대사를 논문 판과 같은 자(approx)로 잰다: K(T) = R x dur_T / f(식 4), 모음 DTW(식 10)."""

    def __init__(self, run: dict[str, Any]) -> None:
        from pstts.paper.frontend import ApproxFrontend
        self.cfg = paper_config(run)
        self.front = ApproxFrontend(self.cfg)

    def measure(self, source_en: str, text: str, r: float) -> tuple[float, float | None]:
        from pstts.paper.iso import k_frames, tts_seconds
        from pstts.paper.ps import measure_line
        (dur,) = tts_seconds(self.front, [text], "ko")
        (cand,), _ = measure_line(self.front, source_en, [text], self.cfg.sakoe_chiba_radius)
        return k_frames(r, dur, self.cfg.frame_sec_sa), cand.dtw

    def close(self) -> None:
        self.front.close()


def _syl(text: str) -> str:
    return f"{text} ({ko_syllable_count(text)})" if text else "-"


def _md(text: Any) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _num(x: Any, digits: int = 1) -> str:
    return "-" if x is None else f"{float(x):.{digits}f}"


def build_rows(main: dict[str, dict], paper: dict[str, dict], ruler: PaperRuler | None) -> list[dict[str, Any]]:
    """줄마다 비교 값. 한쪽에만 있는 id 도 적는다(논문 판 순서, 그다음 기존 판에만 있는 것)."""
    ids = list(paper) + [i for i in main if i not in paper]
    rows = []
    for i in ids:
        m, p = main.get(i, {}), paper.get(i, {})
        main_ko = m.get("final_ko", "")
        row: dict[str, Any] = {
            "id": i, "en": p.get("source_en") or m.get("source_en", ""),
            "main_ko": main_ko, "main_target": (m.get("target") or {}).get("n"),
            "main_tol": (m.get("target") or {}).get("tol"), "main_status": m.get("status", ""),
            "iso_ko": p.get("iso_ko", ""), "iso_status": p.get("iso_status", ""),
            "iso_iterations": p.get("iso_iterations"), "k_prime": p.get("k_prime"), "iso_k": p.get("iso_k"),
            "ps_ko": p.get("ps_ko", ""), "ps_dtw": p.get("ps_dtw"),
            "final_ko": p.get("final_ko", ""), "final_dtw": p.get("ps_comet_dtw"),
            "final_score": p.get("ps_comet_score"), "final_comet": p.get("ps_comet_comet"),
            "n_ps": p.get("n_ps_candidates"), "main_k": None, "main_dtw": None,
        }
        if ruler is not None and main_ko and p.get("r") is not None:
            row["main_k"], row["main_dtw"] = ruler.measure(row["en"], main_ko, float(p["r"]))
        row["same_final"] = bool(main_ko) and main_ko == row["final_ko"]
        row["main_in_ps"] = bool(main_ko) and main_ko in [c["text"] for c in (p.get("ps") or {}).get("candidates", [])]
        rows.append(row)
    return rows


def markdown(rows: list[dict[str, Any]], main_dir: Path, paper_dir: Path, main_run: dict, paper_run: dict,
             measured: bool) -> str:
    pc = paper_run.get("config") or {}
    radius = pc.get("sakoe_chiba_radius")
    out = ["# 기존 판과 논문 그대로 판 비교", ""]
    out.append(f"- 기존 판: {_md(main_dir)} (시작 {main_run.get('started_at', '-')}, 모델 2 "
               f"{(main_run.get('llm') or {}).get('backend', '-')})")
    out.append(f"- 논문 판: {_md(paper_dir)} (시작 {paper_run.get('started_at', '-')}, TTS "
               f"{(paper_run.get('frontend') or {}).get('name', '-')}, 모델 2 "
               f"{(paper_run.get('llm') or {}).get('backend', '-')})")
    out.append("- 음절: 두 판 모두 기존 판의 셈(한글 글자 수, 숫자는 한자어로 읽음). 괄호 안 수.")
    out.append(f"- 논문 판 DTW: 근사 프런트엔드, Sakoe-Chiba 반경 {'tslearn 기본(1)' if radius is None else radius}. "
               "기존 판 DTW(반경 26, 음절 시간 나누기)와는 자가 달라 견줄 수 없다.")
    if measured:
        out.append("- '기존 판 K, DTW' 칸은 기존 판이 고른 대사를 논문 판과 같은 자(approx, 같은 R 과 반경)로 "
                   "다시 잰 값이다.")
    out += ["", "| id | 영어 | 기존 판 고른 대사(음절) | 논문 판 ISO(음절) | 논문 판 PS(음절) | "
                "논문 판 PS-Comet(음절) | K' / K(ISO) | DTW PS / PS-Comet | 기존 판 K, DTW |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        target = f" [목표 {r['main_target']}±{r['main_tol']}]" if r["main_target"] is not None else ""
        iso = _syl(r["iso_ko"]) + (f" [{r['iso_status']}, {r['iso_iterations']}번]" if r["iso_status"] else "")
        out.append(f"| {_md(r['id'])} | {_md(r['en'])} | {_md(_syl(r['main_ko']))}{target} | {_md(iso)} | "
                   f"{_md(_syl(r['ps_ko']))} | {_md(_syl(r['final_ko']))} | "
                   f"{_num(r['k_prime'])} / {_num(r['iso_k'])} | "
                   f"{_num(r['ps_dtw'])} / {_num(r['final_dtw'])} | {_num(r['main_k'])}, {_num(r['main_dtw'])} |")
    both = [r for r in rows if r["main_ko"] and r["final_ko"]]
    out += ["", "## 요약", ""]
    out.append(f"- 두 판이 같은 대사를 고른 줄: {sum(r['same_final'] for r in both)} / {len(both)}")
    out.append(f"- 기존 판 대사가 논문 판 PS 후보 안에 있던 줄: {sum(r['main_in_ps'] for r in both)} / {len(both)}")
    if both:
        dm = [ko_syllable_count(r["main_ko"]) for r in both]
        dp = [ko_syllable_count(r["final_ko"]) for r in both]
        out.append(f"- 음절 합: 기존 판 {sum(dm)}, 논문 판 PS-Comet {sum(dp)} (줄마다 차이 "
                   f"{', '.join(f'{b - a:+d}' for a, b in zip(dm, dp))})")
    if measured:
        cmp = [r for r in both if r["main_dtw"] is not None and r["final_dtw"] is not None]
        if cmp:
            lower = sum(1 for r in cmp if r["final_dtw"] < r["main_dtw"] - 1e-9)
            out.append(f"- 논문 판 자로 잰 DTW 가 논문 판 PS-Comet 대사에서 더 작은 줄: {lower} / {len(cmp)} "
                       f"(평균 기존 판 {sum(r['main_dtw'] for r in cmp) / len(cmp):.1f}, 논문 판 "
                       f"{sum(r['final_dtw'] for r in cmp) / len(cmp):.1f})")
        tol = pc.get("iso_tol_frames", 26)
        ks = [r for r in both if r["main_k"] is not None and r["k_prime"] is not None]
        if ks:
            ok = sum(1 for r in ks if abs(r["main_k"] - r["k_prime"]) <= tol + 1e-9)
            out.append(f"- 기존 판 대사가 논문 판 식 6(K' ±{tol}프레임)을 채운 줄: {ok} / {len(ks)}")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="기존 판(out\\demo)과 논문 판(out\\paper_demo) 결과를 줄마다 비교한다.")
    ap.add_argument("--main", type=Path, default=ROOT / "out" / "demo", help="기존 판 결과 폴더")
    ap.add_argument("--paper", type=Path, default=ROOT / "out" / "paper_demo", help="논문 판 결과 폴더")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "compare" / "compare.md", help="쓸 compare.md")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    try:
        main_rows = read_jsonl(args.main / "results.jsonl")
        paper_rows = read_jsonl(args.paper / "results.jsonl")
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"오류: {e}", file=sys.stderr)
        return 2
    try:
        main_run, paper_run = read_run(args.main), read_run(args.paper)
        measured = (paper_run.get("frontend") or {}).get("name") == "approx"
        ruler = PaperRuler(paper_run) if measured else None
        try:
            rows = build_rows(main_rows, paper_rows, ruler)
        finally:
            if ruler is not None:
                ruler.close()
    except (RuntimeError, ValueError, KeyError, OSError) as e:
        print(f"오류: 비교하다 멈췄습니다. {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    text = markdown(rows, args.main, args.paper, main_run, paper_run, measured)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"저장: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
