"""Typer CLI entrypoints for MoS-SDH Studio (5-Stage Pipeline + FastAPI Backend)."""

from __future__ import annotations

from pathlib import Path
import shutil
import pysubs2
from rich.console import Console
from rich.table import Table
import typer

from mos_sdh import __version__
from mos_sdh.config import AnalysisConfig, get_default_cache_dir
from mos_sdh.core.vram import bind_windows_cuda_dlls
from mos_sdh.exporters.report_exporter import export_reports
from mos_sdh.exporters.subtitle_exporter import export_subtitles
from mos_sdh.pipeline import run_full_pipeline
from mos_sdh.stage4_mos.korean_honorifics import get_kiwi
from mos_sdh.stage4_mos.swer_evaluator import evaluate_cer_wer_swer

app = typer.Typer(
    name="mos-sdh",
    help="5-Stage 시청각 화법(MoS) 정량 분석 및 배리어프리(SDH/CC) 자막 프로토타입 엔진",
    add_completion=False,
)
console = Console()


@app.command("doctor")
def doctor_cmd() -> None:
    """로컬 GPU(CUDA), FFmpeg, 시각(OpenCV/EasyOCR), VLM/LLM 양자화, 형태소(Kiwi), 음향 분석기 준비 상태를 진단합니다."""
    dll_dirs = bind_windows_cuda_dlls()
    import cv2
    import easyocr
    import fastapi
    import parselmouth
    import scenedetect
    import sentence_transformers
    import torch
    import transformers

    table = Table(title=f"MoS-SDH Studio v{__version__} 로컬 환경 진단 결과")
    table.add_column("구성 요소", style="cyan", no_wrap=True)
    table.add_column("상태", style="green")
    table.add_column("세부 정보")

    ffmpeg_path = shutil.which("ffmpeg")
    table.add_row("FFmpeg", "정상" if ffmpeg_path else "누락", ffmpeg_path or "PATH에서 찾을 수 없음")

    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_detail = f"{gpu_name} ({vram_gb:.2f} GB VRAM, PyTorch {torch.__version__})"
    else:
        gpu_detail = f"CPU 폴백 모드 (PyTorch {torch.__version__})"
    table.add_row("GPU / CUDA", "가속 활성" if cuda_ok else "CPU 전용", gpu_detail)
    table.add_row("Windows CUDA DLL", "바인딩 완료", f"{len(dll_dirs)}개 경로 등록")

    kiwi = get_kiwi()
    sample_toks = kiwi.tokenize("아버지께서 방에 들어가신다.")
    table.add_row("Stage 1/4 음향·형태소", "정상", f"Parselmouth v{parselmouth.__version__} | Kiwi ({len(sample_toks)} tok)")
    table.add_row(
        "Stage 2 고밀도 시각/OCR",
        "정상",
        f"OpenCV v{cv2.__version__} | PySceneDetect v{scenedetect.__version__} | EasyOCR v{easyocr.__version__}",
    )
    table.add_row(
        "Stage 3/5 VLM·LLM·임베딩",
        "정상",
        f"Transformers v{transformers.__version__} | SentenceTransformers v{sentence_transformers.__version__}",
    )
    import PySide6

    table.add_row("순수 데스크톱 앱 (Qt6)", "정상", f"PySide6 v{PySide6.__version__} (웹 브라우저 불필요)")
    table.add_row("FastAPI 백엔드 서버", "정상", f"FastAPI v{fastapi.__version__}")
    table.add_row("비-OneDrive 캐시 경로", "활성", str(get_default_cache_dir()))

    console.print(table)


@app.command("gui")
def gui_cmd(
    report: Path | None = typer.Option(None, "--report", "-r", help="시작 시 바로 열 mos_report.json 경로"),
) -> None:
    """웹 브라우저 없이 동작하는 Windows 순수 네이티브 데스크톱 스튜디오 앱(Qt6)을 실행합니다."""
    from mos_sdh.desktop import launch_desktop_app

    extra = ["--report", str(report)] if report is not None else []
    raise typer.Exit(code=launch_desktop_app(extra))


@app.command("analyze")
def analyze_cmd(
    input_media: Path = typer.Argument(..., exists=True, help="분석할 영상(.mp4/.mkv) 또는 오디오(.wav) 경로"),
    subtitle: Path | None = typer.Option(None, "--subtitle", "-s", exists=True, help="비교·진단할 기존 자막(.srt/.vtt)"),
    out_dir: Path = typer.Option(Path("./output"), "--out-dir", "-o", help="리포트 및 프로토타입 자막 출력 폴더"),
    whisper_model: str = typer.Option("large-v3-turbo", "--whisper-model", help="Faster-Whisper 모델 이름"),
    vlm_model: str = typer.Option("Qwen/Qwen2.5-VL-3B-Instruct", "--vlm-model", help="Stage 3 로컬 4-bit VLM 모델"),
    llm_model: str = typer.Option("Qwen/Qwen2.5-3B-Instruct", "--llm-model", help="Stage 5 로컬 4-bit LLM 모델"),
    fast: bool = typer.Option(False, "--fast", help="Demucs 배경음 분리를 생략하여 고속 실행"),
    skip_visual: bool = typer.Option(False, "--skip-visual", help="Stage 2 시각(ASD/FER/OCR) 분석 생략"),
    skip_vlm: bool = typer.Option(False, "--skip-vlm", help="Stage 3 로컬 VLM 씬/관계 추론 생략 (휴리스틱 관계만 사용)"),
    skip_llm: bool = typer.Option(False, "--skip-llm", help="Stage 5 로컬 LLM 재작성 생략 (위치 태그 다중 후보만 생성)"),
) -> None:
    """5-Stage 시청각 화법(MoS) 파이프라인을 실행하고 배리어프리(.srt/.vtt/.ass) 프로토타입을 생성합니다."""
    cfg = AnalysisConfig(
        whisper_model=whisper_model,
        vlm_model=vlm_model,
        llm_model=llm_model,
        enable_source_separation=not fast,
        enable_visual_stage=not skip_visual,
        enable_vlm_stage=not skip_vlm,
        enable_llm_stage=not skip_llm,
    )
    console.print(f"[bold cyan]MoS-SDH 5-Stage 파이프라인 시작[/bold cyan]: {input_media.name}")
    doc = run_full_pipeline(input_media=input_media, config=cfg, existing_subtitle=subtitle)

    sub_paths = export_subtitles(doc, out_dir=out_dir)
    rep_paths = export_reports(doc, out_dir=out_dir)

    m = doc.metrics
    summary = Table(title="5-Stage 분석 완료 요약")
    summary.add_column("지표 / 산출물", style="cyan")
    summary.add_column("결과")
    summary.add_row("총 씬 / 세그먼트 / 화자 수", f"{m.scene_count}씬 / {m.total_segments}개 / {m.speaker_count}명")
    summary.add_row("6화계 문맥 일치율", f"{m.honorific_consistency_rate:.2f}%")
    summary.add_row(
        "화계 오인식 / 의도적 전환 / 반어법 탐지",
        f"{m.mismatch_error_count}건 / {m.dramatic_shift_count}건 / {m.irony_segment_count}건",
    )
    summary.add_row(
        "화면 밖 발화(ASD) / 하단 OCR 충돌 회피",
        f"{m.off_screen_segment_count}건 / {m.ocr_collision_avoided_count}건",
    )
    summary.add_row(
        "음향 강조 단어 (<toe_1>~<toe_4>)",
        f"{m.emphasized_word_count}개 (강한 강조 {m.strong_emphasis_word_count}개)",
    )
    summary.add_row(
        "후보 의미 보존도(C_i) / 조화 점수(O_i)",
        f"C_i={m.mean_semantic_ci:.4f} | O_i={m.mean_harmonic_oi:.4f}",
    )
    if m.swer_comparison is not None:
        sw = m.swer_comparison
        summary.add_row(
            "기존 자막 CER / WER / sWER",
            f"{sw.cer_percent:.2f}% / {sw.wer_percent:.2f}% / {sw.swer_percent:.2f}%",
        )
    if doc.vram_telemetry:
        max_peak = max(t.peak_allocated_mb for t in doc.vram_telemetry)
        summary.add_row("최대 스테이지 피크 VRAM", f"{max_peak:.1f} MB ({len(doc.vram_telemetry)}개 GPU 스테이지 순차 회수)")
    summary.add_row("표준 지문 자막 (.srt / .vtt)", f"{sub_paths['srt']} | {sub_paths['vtt']}")
    summary.add_row("시각 스타일 자막 (.ass)", str(sub_paths["ass"]))
    summary.add_row("진단 리포트 (.md / .json)", f"{rep_paths['markdown']} | {rep_paths['json']}")
    console.print(summary)


@app.command("eval-swer")
def eval_swer_cmd(
    ref_file: Path = typer.Argument(..., exists=True, help="정답 참조 자막(.srt/.vtt) 또는 줄단위 텍스트 파일"),
    hyp_file: Path = typer.Argument(..., exists=True, help="비교 가설 자막(.srt/.vtt) 또는 줄단위 텍스트 파일"),
) -> None:
    """Bang et al. (2020) Algorithm A1 & A2 기반 CER, WER, 공백 정규화 sWER을 산출합니다."""
    ref_lines = _load_lines(ref_file)
    hyp_lines = _load_lines(hyp_file)
    if len(ref_lines) != len(hyp_lines):
        ref_lines = [" ".join(ref_lines)]
        hyp_lines = [" ".join(hyp_lines)]

    res = evaluate_cer_wer_swer(ref_lines, hyp_lines)
    table = Table(title="한국어 전사 오류율 비교 (Bang et al., 2020)")
    table.add_column("지표", style="cyan")
    table.add_column("수치 (%)", justify="right", style="bold")
    table.add_row("음절 오류율 (CER)", f"{res.cer_percent:.2f}%")
    table.add_row("일반 어절 오류율 (WER)", f"{res.wer_percent:.2f}%")
    table.add_row("공백 정규화 어절 오류율 (sWER)", f"{res.swer_percent:.2f}%")
    console.print(table)


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="바인딩 호스트 주소"),
    port: int = typer.Option(8000, "--port", "-p", help="바인딩 포트 번호"),
    reload: bool = typer.Option(False, "--reload", help="코드 변경 시 자동 리로드 활성화"),
) -> None:
    """MoS-SDH Studio 인터랙티브 웹 스튜디오 및 로컬 FastAPI 백엔드 서버를 실행합니다."""
    import uvicorn

    bind_windows_cuda_dlls()
    console.print(
        f"[bold green]MoS-SDH Studio v{__version__} 구동[/bold green]: "
        f"스튜디오 웹 UI [underline]http://{host}:{port}/[/underline] | API 문서 [underline]http://{host}:{port}/docs[/underline]"
    )
    uvicorn.run("mos_sdh.api.server:app", host=host, port=port, reload=reload)


def _load_lines(path: Path) -> list[str]:
    if path.suffix.lower() in {".srt", ".vtt", ".ass"}:
        subs = pysubs2.load(str(path), encoding="utf-8")
        return [
            ev.plaintext.replace("\n", " ").strip()
            for ev in subs.events
            if not ev.is_comment and ev.plaintext.strip()
        ]
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


if __name__ == "__main__":
    app()
