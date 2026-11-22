"""scripts/compare_versions.py: 두 판의 results.jsonl 을 id 로 맞춰 compare.md 를 쓴다(작은 가짜 결과 폴더로)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pstts.config import ROOT
from pstts.paper.config import PaperConfig

sys.path.insert(0, str(ROOT / "scripts"))
import compare_versions  # noqa: E402

EN = "We have to leave before the storm hits."


def _write(folder: Path, rows: list[dict], run: dict) -> None:
    folder.mkdir(parents=True)
    (folder / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                          encoding="utf-8")
    (folder / "run.json").write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")


def test_compare_writes_table_and_summary(tmp_path: Path) -> None:
    main_dir, paper_dir, out = tmp_path / "main", tmp_path / "paper", tmp_path / "cmp" / "compare.md"
    _write(main_dir, [{"id": "a", "source_en": EN, "final_ko": "폭풍 전에 가자.", "status": "ok",
                       "target": {"n": 12, "tol": 2}},
                      {"id": "only_main", "source_en": "Hi.", "final_ko": "안녕.", "status": "ok",
                       "target": {"n": 2, "tol": 2}}],
           {"started_at": "t0", "llm": {"backend": "claude-cli"}})
    cfg = PaperConfig(device="cpu")
    paper_row = {"id": "a", "source_en": EN, "iso_ko": "폭풍이 오기 전에 떠나야 해.", "iso_status": "draft",
                 "iso_iterations": 0, "k_prime": 125.23, "iso_k": 119.58, "r": 1.0,
                 "ps_ko": "폭풍이 치기 전에 떠나자.", "ps_dtw": 60.5, "final_ko": "폭풍 전에 가자.", "ps_comet_dtw": 61.0,
                 "ps_comet_score": 1.7, "ps_comet_comet": 0.3, "n_ps_candidates": 2,
                 "ps": {"candidates": [{"text": "폭풍이 치기 전에 떠나자."}, {"text": "폭풍 전에 가자."}]}}
    _write(paper_dir, [paper_row], {"started_at": "t1", "frontend": {"name": "approx"},
                                    "llm": {"backend": "file"}, "config": {**vars(cfg), "unknown_field": 1}})
    assert compare_versions.main(["--main", str(main_dir), "--paper", str(paper_dir), "--out", str(out)]) == 0
    md = out.read_text(encoding="utf-8")
    row_a = next(line for line in md.splitlines() if line.startswith("| a |"))
    assert "폭풍 전에 가자. (6) [목표 12±2]" in row_a and "[draft, 0번]" in row_a and "125.2 / 119.6" in row_a
    assert "60.5 / 61.0" in row_a
    main_k, main_dtw = row_a.rstrip(" |").split("| ")[-1].split(", ")
    assert abs(float(main_k) - 65.2) < 0.1 and float(main_dtw) > 0     # 6음절을 논문 판 자(approx)로 다시 잼
    assert any(line.startswith("| only_main |") for line in md.splitlines())
    assert "두 판이 같은 대사를 고른 줄: 1 / 1" in md and "PS 후보 안에 있던 줄: 1 / 1" in md
    assert "식 6(K' ±26프레임)을 채운 줄: 0 / 1" in md


def test_compare_missing_folder_exits_2(tmp_path: Path) -> None:
    assert compare_versions.main(["--main", str(tmp_path / "no"), "--paper", str(tmp_path / "no2"),
                                  "--out", str(tmp_path / "c.md")]) == 2


def test_compare_runtime_error_is_one_korean_line(tmp_path: Path, monkeypatch, capsys) -> None:
    """자를 만들다 난 오류(예: 라이브러리 RuntimeError)도 추적 없이 한국어 한 줄과 끝 코드 2 로 끝난다."""
    main_dir, paper_dir, out = tmp_path / "main", tmp_path / "paper", tmp_path / "c.md"
    _write(main_dir, [{"id": "a", "source_en": EN, "final_ko": "가자.", "status": "ok"}], {})
    _write(paper_dir, [{"id": "a", "source_en": EN, "final_ko": "가자.", "r": 1.0}], {"frontend": {"name": "approx"}})

    def broken(run: dict) -> None:
        raise RuntimeError("PanPhon 자료 파일을 읽다 실패")

    monkeypatch.setattr(compare_versions.PaperRuler, "__init__", lambda self, run: broken(run))
    assert compare_versions.main(["--main", str(main_dir), "--paper", str(paper_dir), "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("오류: 비교하다 멈췄습니다. RuntimeError: PanPhon") and "Traceback" not in err
    assert not out.exists()


def test_compare_runs_without_utf8_mode(tmp_path: Path) -> None:
    """한국어 Windows 기본(cp949, UTF-8 모드 아님)에서도 자를 만들고 비교한다(근사 프런트엔드가 PanPhon 을 부르지 않음)."""
    import os
    import subprocess

    main_dir, paper_dir, out = tmp_path / "main", tmp_path / "paper", tmp_path / "c.md"
    _write(main_dir, [{"id": "a", "source_en": EN, "final_ko": "폭풍 전에 가자.", "status": "ok"}], {})
    _write(paper_dir, [{"id": "a", "source_en": EN, "final_ko": "폭풍 전에 가자.", "r": 1.0, "k_prime": 125.2}],
           {"frontend": {"name": "approx"}, "config": {"device": "cpu"}})
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUTF8"}
    r = subprocess.run([sys.executable, "-X", "utf8=0", str(ROOT / "scripts" / "compare_versions.py"), "--main",
                        str(main_dir), "--paper", str(paper_dir), "--out", str(out)], capture_output=True,
                       cwd=ROOT, env=env, timeout=300)
    assert r.returncode == 0, r.stderr.decode("utf-8", errors="replace")[-2000:]
    row = next(line for line in out.read_text(encoding="utf-8").splitlines() if line.startswith("| a |"))
    main_k, main_dtw = row.rstrip(" |").split("| ")[-1].split(", ")
    assert float(main_k) > 0 and float(main_dtw) > 0
