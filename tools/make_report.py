"""REPORT.md 생성: 전체 실행 결과와 수락 기준(A1~A3, B1~B5, E1~E2) 증거.

사용: .venv/Scripts/python -I tools/make_report.py
숫자는 모두 지금 디스크의 결과물에서 다시 센다(손으로 쓴 숫자 없음).
"""
from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sinkprep import catalog, paths, util, validate  # noqa: E402

PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    rows = catalog.load()
    docs = [r for r in rows if r["status"] != "excluded"]
    excl = [r for r in rows if r["status"] == "excluded"]
    L = []
    w = L.append

    w("# sink 대본 형식 통일 전처리 결과 보고")
    w("")
    w(f"만든 시각: {time.strftime('%Y-%m-%d %H:%M')} · 생성기 `tools/make_report.py`(모든 숫자는 디스크 결과물에서 다시 셈)")
    w("")
    # ---------- 원본과 카탈로그
    raw_files = [p for p in paths.RAW.rglob("*") if p.is_file() and p.name != "manifest.csv"]
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
            ov = [x["overlap"] for x in by[fmt]]
            sg = [x["seg_pct"] for x in by[fmt]]
            w(f"  - {fmt}: {len(ov)}개, 겹침 중앙값 {statistics.median(ov):.3f}, 구간 일치 중앙값 {statistics.median(sg):.0f}%")
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
    qs = []
    q_items = defaultdict(list)
    empty = []
    for r in docs:
        lp = paths.DOCS / r["doc_id"] / "lines.jsonl"
        if lp.exists():
            lr = util.read_jsonl(lp)[1:]
            n_lines.append(len(lr))
            if not lr:
                empty.append(r["doc_id"])
            for x in lr:
                src[x["src"]] += 1
        sp = paths.DOCS / r["doc_id"] / "script.json"
        if sp.exists():
            n_script += 1
            s = util.read_json(sp)
            for el in s.get("elements", []):
                types[el["type"]] += len(el.get("lines", []))
            qq = s.get("quality") or {}
            if qq.get("score") is not None:
                qs.append(qq["score"])
            for k in ("typed_ratio", "dialogue_ratio", "unknown_ratio"):
                if qq.get(k) is not None:
                    q_items[k].append(qq[k])
    w(f"- lines.jsonl 있는 문서 {len(n_lines)}/{len(docs)}, 줄 0개 문서 {len(empty)}개, 총 줄 {sum(n_lines):,}")
    w("- 줄 출처: " + ", ".join(f"{k} {v:,}" for k, v in src.most_common()))
    w(f"- script.json 있는 문서 {n_script}/{len(docs)}")
    tot = sum(types.values()) or 1
    w("- 요소 종류별 줄 비율: " + ", ".join(f"{k} {100 * v / tot:.1f}%" for k, v in types.most_common()))
    if qs:
        w(f"- 문서 품질 점수 분포: 10% {q(qs, .1):.2f} · 중앙값 {q(qs, .5):.2f} · 90% {q(qs, .9):.2f}")
    for k, v in q_items.items():
        w(f"  - {k}: 10% {q(v, .1):.2f} · 중앙값 {q(v, .5):.2f} · 90% {q(v, .9):.2f}")
    w("")
    # ---------- 정답 표본
    w("## 5. 파서 정확도(독립 정답 표본)")
    for name, sset in (("개발용(dev)", "dev"), ("확인용(test, 파서 개발 중 안 봄)", "test"), ("전체(dev+test)", "all")):
        p = paths.CACHE / f"eval_{sset}.json"
        subprocess.run([PY, "-I", str(ROOT / "tools" / "eval_parse.py"), "--set", sset, "--json", str(p)],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            t = d["total"]
            w(f"- {name}: 줄 {t.get('lines', 0)}개, 요소 종류 {100 * t.get('type_ok', 0) / max(1, t.get('lines', 0)):.1f}%, "
              f"화자 {100 * t.get('spk_ok', 0) / max(1, t.get('spk_total', 0)):.1f}% (대사 줄 {t.get('spk_total', 0)}개)")
    gold = [p for p in (ROOT / "tests" / "gold").glob("*.jsonl")]
    gdocs = {util.read_jsonl(p)[0]["doc_id"] for p in gold if util.read_jsonl(p)}
    w(f"- 정답 표본: 구간 {len(gold)}개, 문서 {len(gdocs)}개, 두 사람 독립 표시 후 갈린 줄만 판정")
    w("")
    # ---------- needs_review
    w("## 6. 확인 필요 목록(qa/needs_review.csv)")
    if paths.NEEDS_REVIEW.exists():
        with open(paths.NEEDS_REVIEW, encoding="utf-8-sig") as f:
            nr = list(csv.DictReader(f))
        c = Counter(x["problem"] for x in nr)
        w(f"- {len(nr)}건(문서 {len({x['doc_id'] for x in nr if x['doc_id']})}개): " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    w("")
    # ---------- 수락 기준
    w("## 7. 수락 기준 확인")
    cat_errs = validate.validate_catalog(check_files=True)
    a1 = [e for e in cat_errs if e.startswith("[A1]")]
    a2 = [e for e in cat_errs if e.startswith("[A2]")]
    w(f"- A1 모든 원본·압축 해제 파일이 정확히 한 행에: 오류 {len(a1)}개")
    w(f"- A2 영화마다 is_primary 최대 1개: 오류 {len(a2)}개")
    w(f"- A3 앞 바이트 판별: 위장 파일 {len(disguised)}개 판별(위 1절), tests/test_detect.py")
    vr = subprocess.run([PY, "-I", "-m", "sinkprep", "validate"], cwd=ROOT, capture_output=True, text=True,
                        encoding="utf-8", errors="replace")
    tail = vr.stdout.strip().splitlines()[-1] if vr.stdout.strip() else vr.stderr.strip()[-200:]
    codes = Counter()
    for ln in vr.stdout.splitlines():
        ln = ln.strip()
        if ln.startswith("["):
            codes[ln.split("]")[0] + "]"] += 1
    w(f"- B1·B2·B3·E1·E2 `sinkprep validate`(catalog, needs_review, docs 전체, examples): {tail}"
      + (f" / 오류 종류 {dict(codes)}" if codes else ""))
    w("- B4 needs_review 기준값: typed_ratio<0.85, dialogue_ratio<0.15, 화자 없는 대사>5%, 말 없는 대사(이름 줄만)>=10%, "
      "추정 화자(speaker_inferred)>10%, OCR conf<0.85, 추출 실패·쪽 그림 실패, 한컴 PDF 글자 손실>1%")
    w("- B5 까다로운 경우 11가지: tests/fixtures/parse/, tests/test_parse.py")
    w("")
    w(f"(보고 생성 {time.time() - t0:.0f}초)")
    util.write_text(ROOT / "REPORT.md", "\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
