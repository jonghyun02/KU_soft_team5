"""FastAPI backend & Interactive Studio server for MoS-SDH Studio v1.0."""

from __future__ import annotations

from dataclasses import dataclass
import mimetypes
from pathlib import Path
import re
import shutil
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mos_sdh import __version__
from mos_sdh.config import AnalysisConfig, get_default_cache_dir
from mos_sdh.core.vram import bind_windows_cuda_dlls
from mos_sdh.exporters.report_exporter import export_reports
from mos_sdh.exporters.subtitle_exporter import export_subtitles
from mos_sdh.pipeline import run_full_pipeline
from mos_sdh.schemas.document import MoSDocument, SegmentMoS, SWERComparison
from mos_sdh.stage4_mos.swer_evaluator import evaluate_cer_wer_swer
from mos_sdh.stage5_llm.overstylization_pruner import prune_overstylized_cues

STUDIO_STATIC_DIR = Path(__file__).resolve().parent.parent / "studio"
_AUDIO_ONLY_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


@dataclass
class StoredProject:
    project_id: str
    label: str
    document: MoSDocument
    out_dir: Path


class ProjectStore:
    """In-memory + disk-backed store of analyzed MoSDocument sessions for Studio inspection."""

    def __init__(self, root_dir: Path | None = None, auto_discover: bool = True) -> None:
        self.root_dir = root_dir or (get_default_cache_dir() / "projects")
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, StoredProject] = {}
        if auto_discover:
            self.discover_existing_outputs([self.root_dir, Path("./output")])

    def register(
        self,
        document: MoSDocument,
        out_dir: Path,
        project_id: str | None = None,
        label: str | None = None,
    ) -> StoredProject:
        pid = project_id or uuid.uuid4().hex[:12]
        out_dir.mkdir(parents=True, exist_ok=True)
        media_name = Path(document.input_media_path).name if document.input_media_path else pid
        mode_tag = "자막QA" if document.mode == "mode_b_subtitle_qa" else "영상직행"
        resolved_label = label or f"{media_name} ({mode_tag} · {out_dir.name})"
        entry = StoredProject(
            project_id=pid,
            label=resolved_label,
            document=document,
            out_dir=out_dir,
        )
        self._items[pid] = entry
        return entry

    def discover_existing_outputs(self, search_roots: list[Path]) -> int:
        """Scan output folders for previously generated `mos_report.json` files."""
        count = 0
        for root in search_roots:
            if not root.exists():
                continue
            for json_file in sorted(root.rglob("mos_report.json")):
                try:
                    doc = MoSDocument.model_validate_json(json_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                rel_parts = json_file.parent.parts[-2:] if len(json_file.parent.parts) >= 2 else (json_file.parent.name,)
                slug = re.sub(r"[^a-zA-Z0-9가-힣_-]+", "_", "_".join(rel_parts)).strip("_")
                pid = slug or uuid.uuid4().hex[:10]
                if pid not in self._items:
                    self.register(doc, json_file.parent, project_id=pid)
                    count += 1
        return count

    def get(self, project_id: str) -> StoredProject | None:
        if project_id in self._items:
            return self._items[project_id]
        disk_json = self.root_dir / project_id / "mos_report.json"
        if disk_json.exists():
            doc = MoSDocument.model_validate_json(disk_json.read_text(encoding="utf-8"))
            return self.register(doc, disk_json.parent, project_id=project_id)
        return None

    def list_all(self) -> list[StoredProject]:
        return list(self._items.values())


class AnalyzeRequest(BaseModel):
    input_media_path: str
    input_subtitle_path: str | None = None
    out_dir: str | None = None
    whisper_model: str = "large-v3-turbo"
    enable_source_separation: bool = True
    enable_visual_stage: bool = True
    enable_vlm_stage: bool = True
    enable_llm_stage: bool = True
    vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct"


class AnalyzeResponse(BaseModel):
    project_id: str
    label: str
    out_dir: str
    output_files: dict[str, str]
    document: MoSDocument


class SelectCandidateRequest(BaseModel):
    candidate_id: int
    custom_srt_text: str | None = None
    custom_ass_text: str | None = None


class UpdateSegmentRequest(BaseModel):
    sdh_srt_text: str | None = None
    sdh_ass_text: str | None = None
    speaker_id: str | None = None


class PruneProjectRequest(BaseModel):
    max_cue_density_ratio: float = Field(default=0.45, ge=0.0, le=1.0)
    recommended_max_cps: float = Field(default=14.0, ge=5.0, le=35.0)


class SelectCandidateResponse(BaseModel):
    project_id: str
    segment: SegmentMoS
    mean_semantic_ci: float
    mean_harmonic_oi: float
    mean_cps: float
    cps_compliance_rate: float
    cue_attachment_ratio: float
    output_files: dict[str, str]


class SWERRequest(BaseModel):
    ref_lines: list[str] = Field(..., min_length=1)
    hyp_lines: list[str] = Field(..., min_length=1)


def _refresh_readability_and_export(entry: StoredProject, recommended_max_cps: float = 14.0) -> dict[str, str]:
    """Recompute segment CPS, global C_i/O_i/CPS/cue_ratio metrics, and persist all dual subtitle & report files."""
    doc = entry.document
    if doc.segments:
        for seg in doc.segments:
            dur_s = max(0.5, (seg.end_ms - seg.start_ms) / 1000.0)
            vis_chars = len(re.sub(r"\s+", "", seg.sdh_srt_text or seg.raw_transcript))
            seg.cps = round(vis_chars / dur_s, 2)
            seg.cps_exceeded = seg.cps > recommended_max_cps

        selected_cands = [next((c for c in s.candidates if c.selected), None) for s in doc.segments]
        valid = [c for c in selected_cands if c is not None]
        if valid:
            doc.metrics.mean_semantic_ci = round(sum(c.semantic_ci for c in valid) / len(valid), 4)
            doc.metrics.mean_harmonic_oi = round(sum(c.harmonic_oi for c in valid) / len(valid), 4)

        doc.metrics.mean_cps = round(sum(s.cps for s in doc.segments) / len(doc.segments), 2)
        cps_ok = sum(1 for s in doc.segments if not s.cps_exceeded)
        doc.metrics.cps_compliance_rate = round((cps_ok / len(doc.segments)) * 100.0, 2)
        has_dir = sum(1 for s in doc.segments if "(" in s.sdh_srt_text)
        doc.metrics.cue_attachment_ratio = round(has_dir / len(doc.segments), 4)
        doc.metrics.speaker_count = len({s.speaker_id for s in doc.segments})

    sub_files = export_subtitles(doc, entry.out_dir)
    rep_files = export_reports(doc, entry.out_dir)
    return {k: str(v.resolve()) for k, v in {**sub_files, **rep_files}.items()}


def create_app(store: ProjectStore | None = None) -> FastAPI:
    """Create and configure the MoS-SDH Studio v1.0 FastAPI + Web UI application."""
    bind_windows_cuda_dlls()

    application = FastAPI(
        title="MoS-SDH Studio API",
        version=__version__,
        description="5-Stage Audio-Visual Manner of Speech Quantitative Analyzer & Interactive Barrier-Free Subtitle Studio",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.state.store = store if store is not None else ProjectStore()

    if STUDIO_STATIC_DIR.exists():
        application.mount(
            "/static",
            StaticFiles(directory=str(STUDIO_STATIC_DIR)),
            name="studio_static",
        )

    @application.get("/", include_in_schema=False)
    def serve_studio_index() -> FileResponse:
        index_html = STUDIO_STATIC_DIR / "index.html"
        if not index_html.exists():
            raise HTTPException(status_code=404, detail="Studio UI bundle not found.")
        return FileResponse(index_html, media_type="text/html; charset=utf-8")

    @application.get("/api/v1/health")
    def health_check() -> dict[str, object]:
        import torch

        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "cpu"
        total_vram_gb = (
            round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            if cuda_ok
            else 0.0
        )
        return {
            "status": "ok",
            "version": __version__,
            "cuda_available": cuda_ok,
            "gpu_name": gpu_name,
            "total_vram_gb": total_vram_gb,
            "ffmpeg_available": shutil.which("ffmpeg") is not None,
            "cache_dir": str(get_default_cache_dir()),
            "studio_ui_ready": (STUDIO_STATIC_DIR / "index.html").exists(),
        }

    @application.post("/api/v1/eval-swer", response_model=SWERComparison)
    def eval_swer_endpoint(req: SWERRequest) -> SWERComparison:
        ref = [ln.strip() for ln in req.ref_lines if ln.strip()]
        hyp = [ln.strip() for ln in req.hyp_lines if ln.strip()]
        if not ref or not hyp:
            raise HTTPException(status_code=400, detail="ref_lines and hyp_lines must contain non-empty text.")
        if len(ref) != len(hyp):
            ref = [" ".join(ref)]
            hyp = [" ".join(hyp)]
        return evaluate_cer_wer_swer(ref, hyp)

    @application.post("/api/v1/analyze", response_model=AnalyzeResponse)
    def analyze_media_endpoint(req: AnalyzeRequest) -> AnalyzeResponse:
        media_path = Path(req.input_media_path)
        if not media_path.exists():
            raise HTTPException(status_code=404, detail=f"Input media not found: {media_path}")

        sub_path: Path | None = None
        if req.input_subtitle_path:
            sub_path = Path(req.input_subtitle_path)
            if not sub_path.exists():
                raise HTTPException(status_code=404, detail=f"Input subtitle not found: {sub_path}")

        cfg = AnalysisConfig(
            whisper_model=req.whisper_model,
            enable_source_separation=req.enable_source_separation,
            enable_visual_stage=req.enable_visual_stage,
            enable_vlm_stage=req.enable_vlm_stage,
            enable_llm_stage=req.enable_llm_stage,
            vlm_model=req.vlm_model,
            llm_model=req.llm_model,
        )

        pid = uuid.uuid4().hex[:12]
        target_out = Path(req.out_dir) if req.out_dir else (application.state.store.root_dir / pid)

        doc = run_full_pipeline(input_media=media_path, config=cfg, existing_subtitle=sub_path)
        sub_files = export_subtitles(doc, target_out)
        rep_files = export_reports(doc, target_out)

        entry = application.state.store.register(doc, target_out, project_id=pid)
        all_files = {k: str(v.resolve()) for k, v in {**sub_files, **rep_files}.items()}
        return AnalyzeResponse(
            project_id=entry.project_id,
            label=entry.label,
            out_dir=str(target_out.resolve()),
            output_files=all_files,
            document=doc,
        )

    @application.post("/api/v1/analyze-upload", response_model=AnalyzeResponse)
    def analyze_upload_endpoint(
        media_file: UploadFile = File(...),
        subtitle_file: UploadFile | None = File(None),
        enable_source_separation: bool = Form(True),
        enable_visual_stage: bool = Form(True),
        enable_vlm_stage: bool = Form(False),
        enable_llm_stage: bool = Form(False),
    ) -> AnalyzeResponse:
        upload_id = uuid.uuid4().hex[:12]
        upload_dir = get_default_cache_dir() / "uploads" / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        orig_stem = Path(media_file.filename or "uploaded_clip.mp4").stem
        media_suffix = Path(media_file.filename or "input.mp4").suffix or ".mp4"
        saved_media = upload_dir / f"{orig_stem}{media_suffix}"
        with saved_media.open("wb") as f_out:
            shutil.copyfileobj(media_file.file, f_out)

        saved_sub: Path | None = None
        if subtitle_file is not None and subtitle_file.filename:
            sub_suffix = Path(subtitle_file.filename).suffix or ".srt"
            saved_sub = upload_dir / f"input_sub{sub_suffix}"
            with saved_sub.open("wb") as f_sub:
                shutil.copyfileobj(subtitle_file.file, f_sub)

        req = AnalyzeRequest(
            input_media_path=str(saved_media),
            input_subtitle_path=str(saved_sub) if saved_sub else None,
            enable_source_separation=enable_source_separation,
            enable_visual_stage=enable_visual_stage,
            enable_vlm_stage=enable_vlm_stage,
            enable_llm_stage=enable_llm_stage,
        )
        return analyze_media_endpoint(req)

    @application.get("/api/v1/projects")
    def list_projects_endpoint() -> list[dict[str, object]]:
        items = application.state.store.list_all()
        result: list[dict[str, object]] = []
        for item in items:
            media_p = Path(item.document.input_media_path)
            has_media = media_p.exists()
            has_video = has_media and media_p.suffix.lower() not in _AUDIO_ONLY_EXTS
            result.append(
                {
                    "project_id": item.project_id,
                    "label": item.label,
                    "mode": item.document.mode,
                    "input_media_path": item.document.input_media_path,
                    "has_media": has_media,
                    "has_video": has_video,
                    "duration_ms": item.document.duration_ms,
                    "scene_count": item.document.metrics.scene_count,
                    "total_segments": item.document.metrics.total_segments,
                    "irony_segment_count": item.document.metrics.irony_segment_count,
                    "mismatch_error_count": item.document.metrics.mismatch_error_count,
                    "out_dir": str(item.out_dir),
                }
            )
        return result

    @application.get("/api/v1/projects/{project_id}", response_model=MoSDocument)
    def get_project_endpoint(project_id: str) -> MoSDocument:
        entry = application.state.store.get(project_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
        return entry.document

    @application.get("/api/v1/projects/{project_id}/media")
    def stream_project_media_endpoint(project_id: str) -> FileResponse:
        entry = application.state.store.get(project_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
        media_path = Path(entry.document.input_media_path)
        if not media_path.exists():
            raise HTTPException(status_code=404, detail=f"Media file missing on disk: {media_path}")
        mime, _ = mimetypes.guess_type(str(media_path))
        return FileResponse(media_path, media_type=mime or "video/mp4")

    @application.get("/api/v1/projects/{project_id}/export/{fmt}")
    def download_exported_file_endpoint(project_id: str, fmt: str) -> FileResponse:
        entry = application.state.store.get(project_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

        fmt_key = fmt.lower().strip(".")
        all_files = _refresh_readability_and_export(entry)
        key_map = {
            "srt": ("srt", "prototype_sdh.srt", "text/plain; charset=utf-8"),
            "vtt": ("vtt", "prototype_sdh.vtt", "text/vtt; charset=utf-8"),
            "ass": ("ass", "prototype_sdh.ass", "text/plain; charset=utf-8"),
            "md": ("markdown", "mos_report.md", "text/markdown; charset=utf-8"),
            "markdown": ("markdown", "mos_report.md", "text/markdown; charset=utf-8"),
            "json": ("json", "mos_report.json", "application/json; charset=utf-8"),
        }
        if fmt_key not in key_map:
            raise HTTPException(status_code=400, detail=f"Unsupported export format: {fmt}")

        dict_key, dl_name, media_type = key_map[fmt_key]
        file_path = Path(all_files[dict_key])
        return FileResponse(file_path, media_type=media_type, filename=f"{project_id}_{dl_name}")

    @application.post(
        "/api/v1/projects/{project_id}/segments/{segment_id}/select-candidate",
        response_model=SelectCandidateResponse,
    )
    def select_segment_candidate_endpoint(
        project_id: str,
        segment_id: int,
        req: SelectCandidateRequest,
    ) -> SelectCandidateResponse:
        entry = application.state.store.get(project_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

        doc = entry.document
        seg = next((s for s in doc.segments if s.segment_id == segment_id), None)
        if seg is None:
            raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found.")

        target_cand = next((c for c in seg.candidates if c.candidate_id == req.candidate_id), None)
        if target_cand is None:
            raise HTTPException(
                status_code=404,
                detail=f"Candidate {req.candidate_id} not found on segment {segment_id}.",
            )

        for cand in seg.candidates:
            cand.selected = cand.candidate_id == req.candidate_id

        if req.custom_srt_text is not None:
            target_cand.srt_text = req.custom_srt_text
        if req.custom_ass_text is not None:
            target_cand.ass_text = req.custom_ass_text

        seg.sdh_srt_text = target_cand.srt_text
        seg.sdh_ass_text = target_cand.ass_text
        seg.stage_direction_pruned = False

        all_files = _refresh_readability_and_export(entry)
        return SelectCandidateResponse(
            project_id=project_id,
            segment=seg,
            mean_semantic_ci=doc.metrics.mean_semantic_ci,
            mean_harmonic_oi=doc.metrics.mean_harmonic_oi,
            mean_cps=doc.metrics.mean_cps,
            cps_compliance_rate=doc.metrics.cps_compliance_rate,
            cue_attachment_ratio=doc.metrics.cue_attachment_ratio,
            output_files=all_files,
        )

    @application.patch(
        "/api/v1/projects/{project_id}/segments/{segment_id}",
        response_model=SelectCandidateResponse,
    )
    def manual_edit_segment_endpoint(
        project_id: str,
        segment_id: int,
        req: UpdateSegmentRequest,
    ) -> SelectCandidateResponse:
        entry = application.state.store.get(project_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

        doc = entry.document
        seg = next((s for s in doc.segments if s.segment_id == segment_id), None)
        if seg is None:
            raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found.")

        if req.speaker_id is not None and req.speaker_id.strip():
            old_spk = seg.speaker_id
            new_spk = req.speaker_id.strip()
            seg.speaker_id = new_spk
            if req.sdh_srt_text is None:
                seg.sdh_srt_text = seg.sdh_srt_text.replace(f"[{old_spk}", f"[{new_spk}", 1)
            if req.sdh_ass_text is None:
                seg.sdh_ass_text = seg.sdh_ass_text.replace(f"[{old_spk}", f"[{new_spk}", 1)

        if req.sdh_srt_text is not None:
            seg.sdh_srt_text = req.sdh_srt_text.strip()
        if req.sdh_ass_text is not None:
            seg.sdh_ass_text = req.sdh_ass_text.strip()

        all_files = _refresh_readability_and_export(entry)
        return SelectCandidateResponse(
            project_id=project_id,
            segment=seg,
            mean_semantic_ci=doc.metrics.mean_semantic_ci,
            mean_harmonic_oi=doc.metrics.mean_harmonic_oi,
            mean_cps=doc.metrics.mean_cps,
            cps_compliance_rate=doc.metrics.cps_compliance_rate,
            cue_attachment_ratio=doc.metrics.cue_attachment_ratio,
            output_files=all_files,
        )

    @application.post("/api/v1/projects/{project_id}/prune", response_model=MoSDocument)
    def reprune_project_cues_endpoint(
        project_id: str,
        req: PruneProjectRequest,
    ) -> MoSDocument:
        entry = application.state.store.get(project_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

        doc = entry.document
        for seg in doc.segments:
            sel = next((c for c in seg.candidates if c.selected), None)
            if sel is not None:
                seg.sdh_srt_text = sel.srt_text
                seg.sdh_ass_text = sel.ass_text
            seg.stage_direction_pruned = False

        cfg = AnalysisConfig(
            max_cue_density_ratio=req.max_cue_density_ratio,
            recommended_max_cps=req.recommended_max_cps,
        )
        prune_overstylized_cues(doc.segments, cfg)
        _refresh_readability_and_export(entry, recommended_max_cps=req.recommended_max_cps)
        return doc

    return application


app = create_app()
