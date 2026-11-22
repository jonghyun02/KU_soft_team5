"""말 빠르기(음절/초)와 환산 계수 coef(= 한국어 음절 속도 / 영어 음절 속도)를 잰다.

1) 매니페스트(csv/tsv, 머리글 lang, text, seconds): lang 은 en 또는 ko, seconds 는 쉼을 뺀 말 길이(초).
   언어별 음절 속도를 '전체 합 비율'(음절 합 / 초 합)과 '문장별 중앙값' 두 가지로 내고 coef = ko / en.
   AI Hub 71524 를 받으면 그 대본·음성 길이로 매니페스트를 만들어 이 방법으로 다시 잰다.
2) --from-dub-rhythm C:\\projects\\dub-rhythm-demo\\work\\lines.json: usable 줄의 n_target
   (OmniVoice 길이 규칙으로 정한 '칸을 채우는 한국어 음절 수') / 영어 음절 수(eSpeak) 로 임시 coef 를 잰다.

실행 예:
  .venv\\Scripts\\python.exe scripts\\measure_rate.py manifest.csv
  .venv\\Scripts\\python.exe scripts\\measure_rate.py --from-dub-rhythm C:\\projects\\dub-rhythm-demo\\work\\lines.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pstts.phonetics import english_syllable_count  # noqa: E402
from pstts.textnorm import ko_syllable_count  # noqa: E402

LANGS = ("en", "ko")


def count_syllables(text: str, lang: str) -> int:
    if lang == "en":
        return english_syllable_count(text)
    if lang == "ko":
        return ko_syllable_count(text)
    raise ValueError(f"lang 은 en 또는 ko 여야 합니다: {lang!r}")


def read_manifest(path: Path) -> list[dict[str, str]]:
    """csv(쉼표) 또는 tsv(탭) 매니페스트를 읽는다. lang, text, seconds 칸이 있어야 한다."""
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        missing = {"lang", "text", "seconds"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"매니페스트에 칸이 없습니다: {', '.join(sorted(missing))} ({path})")
        return [dict(row) for row in reader]


def _rates(pairs: list[tuple[int, float]]) -> dict[str, float | int]:
    syl = sum(n for n, _ in pairs)
    sec = sum(s for _, s in pairs)
    return {
        "n": len(pairs),
        "syllables": syl,
        "seconds": round(sec, 3),
        "rate_total": syl / sec,
        "rate_median": statistics.median(n / s for n, s in pairs),
    }


def measure_manifest(rows: Iterable[dict[str, str]]) -> dict:
    """언어별 음절 속도와 coef. 길이가 0 이하이거나 음절이 0 인 줄은 빼고 센다."""
    pairs: dict[str, list[tuple[int, float]]] = {lang: [] for lang in LANGS}
    skipped: list[str] = []
    for i, row in enumerate(rows, start=2):           # 머리글이 1행
        lang = (row.get("lang") or "").strip().lower()
        text = row.get("text") or ""
        try:
            sec = float(row.get("seconds") or "nan")
        except ValueError:
            sec = float("nan")
        if lang not in LANGS:
            raise ValueError(f"{i}행: lang 은 en 또는 ko 여야 합니다: {lang!r}")
        n = count_syllables(text, lang)
        if not sec > 0 or n == 0:
            skipped.append(f"{i}행")
            continue
        pairs[lang].append((n, sec))
    out: dict = {"skipped": skipped}
    for lang in LANGS:
        if not pairs[lang]:
            raise ValueError(f"{lang} 줄이 하나도 없어서 음절 속도를 잴 수 없습니다.")
        out[lang] = _rates(pairs[lang])
    out["coef_total"] = out["ko"]["rate_total"] / out["en"]["rate_total"]
    out["coef_median"] = out["ko"]["rate_median"] / out["en"]["rate_median"]
    return out


def dub_rhythm_coef(lines: list[dict], usable_only: bool = True) -> dict:
    """dub-rhythm-demo lines.json 에서 n_target / 영어 음절 수 로 임시 coef 를 잰다."""
    ratios: list[float] = []
    ko_sum = en_sum = 0
    dur_sum = 0.0
    skipped: list[str] = []
    for line in lines:
        if usable_only and not line.get("usable"):
            continue
        en = english_syllable_count(line["text_en"])
        if en == 0:
            skipped.append(str(line.get("id")))
            continue
        ko = int(line["n_target"])
        ratios.append(ko / en)
        ko_sum += ko
        en_sum += en
        dur_sum += float(line.get("dur") or 0.0)
    if not ratios:
        raise ValueError("쓸 수 있는 줄이 없습니다.")
    return {
        "n_lines": len(ratios),
        "skipped": skipped,
        "ko_syllables": ko_sum,
        "en_syllables": en_sum,
        "seconds": round(dur_sum, 3),
        "coef_total": ko_sum / en_sum,
        "coef_median": statistics.median(ratios),
        # 칸 길이(dur)는 0.35초 이하 쉼을 품고 있어서 이 값은 말 빠르기의 아래쪽 어림이다.
        "en_rate_total": en_sum / dur_sum if dur_sum > 0 else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="음절 속도와 환산 계수 coef 를 잰다.")
    ap.add_argument("manifest", nargs="?", type=Path, help="lang,text,seconds 칸이 있는 csv/tsv")
    ap.add_argument("--from-dub-rhythm", type=Path, help="dub-rhythm-demo 의 work/lines.json")
    ap.add_argument("--all-lines", action="store_true", help="--from-dub-rhythm 에서 usable 이 아닌 줄도 쓴다")
    ap.add_argument("--json", type=Path, help="결과를 JSON 으로도 저장")
    args = ap.parse_args(argv)
    if (args.manifest is None) == (args.from_dub_rhythm is None):
        ap.error("매니페스트 경로나 --from-dub-rhythm 중 하나만 주세요.")

    if args.manifest is not None:
        res = measure_manifest(read_manifest(args.manifest))
        for lang, name in (("en", "영어"), ("ko", "한국어")):
            r = res[lang]
            print(f"{name}: {r['n']}줄, 음절 {r['syllables']}, {r['seconds']}초, "
                  f"전체 합 비율 {r['rate_total']:.3f} 음절/초, 문장별 중앙값 {r['rate_median']:.3f} 음절/초")
        if res["skipped"]:
            print(f"뺀 줄(길이 0 이하 또는 음절 0): {', '.join(res['skipped'])}")
        print(f"coef(전체 합 비율) = {res['coef_total']:.3f}")
        print(f"coef(문장별 중앙값) = {res['coef_median']:.3f}")
        print(f"추천: --coef {res['coef_total']:.2f} (전체 합 비율). "
              f"영어 음절 속도 en_rate = {res['en']['rate_total']:.2f} 음절/초 는 config.py 에서 바꾼다.")
    else:
        lines = json.loads(args.from_dub_rhythm.read_text(encoding="utf-8"))
        res = dub_rhythm_coef(lines, usable_only=not args.all_lines)
        print(f"쓴 줄 {res['n_lines']}개(usable{' 아닌 줄 포함' if args.all_lines else ' 만'})")
        print(f"한국어 목표 음절 합 {res['ko_syllables']}, 영어 음절 합(eSpeak en-us) {res['en_syllables']}")
        print(f"coef(전체 합 비율) = {res['coef_total']:.3f}")
        print(f"coef(줄별 중앙값) = {res['coef_median']:.3f}")
        if res["en_rate_total"] is not None:
            print(f"참고: 칸 길이 합 {res['seconds']}초 기준 영어 음절 속도 {res['en_rate_total']:.2f} 음절/초(짧은 쉼 포함)")
        if res["skipped"]:
            print(f"영어 음절이 0 이라 뺀 줄: {', '.join(res['skipped'])}")
        print(f"추천(임시): --coef {res['coef_total']:.2f}. n_target 은 OmniVoice 글자 가중치로 정한 값이라 "
              "실제 한국어 낭독 속도로 잰 값이 아니다. AI Hub 71524 로 다시 잴 것.")
    if args.json:
        args.json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"저장: {args.json}")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):     # IPA 기호·한글이 cp949 로 깨지지 않게
        stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
