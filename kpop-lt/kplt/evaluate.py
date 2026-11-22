"""4.3~4.4절 평가: 표 5 (SCD, error rate, Semsec, NSP), 표 6 (Phodeg, Phovar).

- 정답 번역 없이 원문 한국어와 모델 번역만 비교한다 (4.4절).
- 곡마다 SCD, Semsec, NSP, Phodeg, Phovar 를 구하고 곡 평균과 표준편차를 낸다 (표 5 "average value and
  associated standard deviation"). error rate 는 모든 줄 중 음절 수가 틀린 줄의 비율.
- Dataset 줄: 논문 데이터셋 K-pop 곡의 실제 번역 (원문 한국어 대 영어 번역). Source 줄: 평가 원문 한국어.

실행: python -m kplt.evaluate  (out/translations/*.json 이 있는 모델만 계산)
"""
from __future__ import annotations

import argparse
import json

from .config import DATA, OUT, RUNS, VENDOR
from .gtranslate import GoogleTranslator
from .metrics import NSPScorer, STS, empty_lines, error_lines, mean_std, song_scd
from .model.infer import all_model_names
from .phoneme import song_pho
from .text import load_meta

ROW_NAMES = {
    "baseline": "Baseline", "line_semi": "Line / Semi-supervised", "line_syl_semi": "Line / Semi-supervised (+<SYL>)",
    "line_ft": "Line / Fine-tuned", "line_syl_ft": "Line / Fine-tuned (+<SYL>)",
    "sect_semi": "Section / Semi-supervised", "sect_syl_semi": "Section / Semi-supervised (+<SYL>)",
    "sect_ft": "Section / Fine-tuned", "sect_syl_ft": "Section / Fine-tuned (+<SYL>)",
}
ORDER = ["baseline", "line_semi", "line_syl_semi", "line_ft", "line_syl_ft",
         "sect_semi", "sect_syl_semi", "sect_ft", "sect_syl_ft"]


def score_pairs(pairs, sts: STS, nsp: NSPScorer | None):
    """pairs: [(id, src_sections, tgt_sections)]"""
    scd, sem, nsps, deg, var = [], [], [], [], []
    wrong = total = empty = 0
    for _, src, tgt in pairs:
        empty += empty_lines(tgt)
        scd.append(song_scd(src, tgt))
        w, n = error_lines(src, tgt)
        wrong, total = wrong + w, total + n
        sem.append(sts.song(src, tgt)["semsec"])
        if nsp is not None:
            nsps.append(nsp.song(tgt))
        p = song_pho(tgt)
        if p:
            deg.append(p[0])
            var.append(p[1])
    r = {"n_songs": len(pairs), "n_lines": total, "empty_output_lines": empty,
         "SCD": mean_std(scd), "ErrorRate": wrong / total, "Semsec": mean_std(sem),
         "Phodeg": sum(deg) / len(deg), "Phovar": sum(var) / len(var)}
    if nsps:
        r["NSP"] = mean_std(nsps)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--suffix", default="", help="infer --limit 로 만든 파일이면 _limitN")
    a = ap.parse_args()
    songs = json.loads((DATA / "eval_kpop.json").read_text(encoding="utf-8"))
    sts = STS(translator=GoogleTranslator(), device=a.device)
    nsp_dir = RUNS / "nsp" / "best"
    nsp = NSPScorer(nsp_dir, device=a.device) if (nsp_dir / "config.json").exists() else None
    results = {}
    for name in all_model_names():
        f = OUT / "translations" / f"{name}{a.suffix}.json"
        if not f.exists():
            continue
        tr = json.loads(f.read_text(encoding="utf-8"))
        pairs = [(s["id"], s["sections"], tr[s["id"]]) for s in songs if s["id"] in tr]
        results[name] = score_pairs(pairs, sts, nsp)
        print(name, results[name], flush=True)

    # Dataset 줄 (표 5) 과 Source 줄 (표 6)
    meta = load_meta(VENDOR / "meta.csv")
    lyrics = json.loads((DATA / "lyrics.json").read_text(encoding="utf-8"))
    ds_pairs = [(lid, [[k for k, _ in sec] for sec in song], [[e for _, e in sec] for sec in song])
                for lid, song in lyrics.items() if meta[lid]["GENRE"] == "k-pop"]
    results["dataset"] = score_pairs(ds_pairs, sts, nsp)
    src_pho = [p for p in (song_pho(s["sections"]) for s in songs) if p]
    results["source"] = {"Phodeg": sum(p[0] for p in src_pho) / len(src_pho),
                         "Phovar": sum(p[1] for p in src_pho) / len(src_pho)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"eval{a.suffix}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    write_tables(results, OUT / f"tables{a.suffix}.md")


def _ms(x):
    return f"{x[0]:.2f} ({x[1]:.2f})"


def write_tables(res: dict, path):
    lines = ["## 표 5", "", "| 모델 | SCD | Error Rate | Semsec | NSP |", "|---|---|---|---|---|"]
    for k in ORDER + ["dataset"]:
        if k not in res:
            continue
        r = res[k]
        name = ROW_NAMES.get(k, "Dataset")
        er = "-" if k == "dataset" else f"{r['ErrorRate']:.2f}"
        nsp = _ms(r["NSP"]) if "NSP" in r else "-"
        lines.append(f"| {name} | {_ms(r['SCD'])} | {er} | {_ms(r['Semsec'])} | {nsp} |")
    lines += ["", "## 표 6", "", "| 모델 | Phodeg | Phovar |", "|---|---|---|"]
    for k in ORDER + ["source"]:
        if k in res:
            lines.append(f"| {ROW_NAMES.get(k, 'Source')} | {res[k]['Phodeg']:.2f} | {res[k]['Phovar']:.2f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
