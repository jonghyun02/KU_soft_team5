"""Scene & burned-in subtitle OCR with automatic SDH bottom-collision avoidance."""

from __future__ import annotations

from pathlib import Path
import cv2

from mos_sdh.config import AnalysisConfig
from mos_sdh.core.vram import StageVRAMReport, gpu_stage
from mos_sdh.schemas.document import OCRBox, SceneContext, SegmentMoS


def run_scene_ocr_and_collision_detection(
    safe_video_path: Path,
    segments: list[SegmentMoS],
    scenes: list[SceneContext],
    config: AnalysisConfig,
    telemetry_sink: list[StageVRAMReport] | None = None,
) -> None:
    """Detect Korean/English on-screen text per utterance and flag bottom subtitle collisions."""
    if not segments:
        return

    cap = cv2.VideoCapture(str(safe_video_path))
    if not cap.isOpened():
        return

    width = max(1, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = max(1, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    with gpu_stage("Stage 2-B: EasyOCR Scene Text & Collision Scan", telemetry_sink=telemetry_sink) as device:
        import easyocr

        reader = easyocr.Reader(["ko", "en"], gpu=(device == "cuda"), verbose=False)

        for seg in segments:
            mid_ms = (seg.start_ms + seg.end_ms) // 2
            cap.set(cv2.CAP_PROP_POS_MSEC, float(mid_ms))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue

            detections = reader.readtext(frame_bgr, detail=1)
            boxes: list[OCRBox] = []
            seen_texts: list[str] = []
            bottom_hit = False

            for coords, text, conf in detections:
                clean_txt = str(text).strip()
                if not clean_txt or float(conf) < 0.35:
                    continue

                xs = [float(pt[0]) / width for pt in coords]
                ys = [float(pt[1]) / height for pt in coords]
                x_min = round(max(0.0, min(xs)), 4)
                y_min = round(max(0.0, min(ys)), 4)
                x_max = round(min(1.0, max(xs)), 4)
                y_max = round(min(1.0, max(ys)), 4)

                boxes.append(
                    OCRBox(
                        text=clean_txt,
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                        confidence=round(float(conf), 4),
                    )
                )
                if clean_txt not in seen_texts:
                    seen_texts.append(clean_txt)
                if y_max >= config.ocr_bottom_zone_ymin:
                    bottom_hit = True

            seg.visual.ocr_boxes = boxes
            seg.visual.ocr_texts = seen_texts
            seg.visual.avoid_bottom_region = bottom_hit

        del reader

    cap.release()

    # Aggregate unique OCR texts per SceneContext for Stage 3 VLM context enrichment
    scene_map = {sc.scene_id: sc for sc in scenes}
    for seg in segments:
        sc = scene_map.get(seg.visual.scene_id)
        if sc is not None:
            for txt in seg.visual.ocr_texts:
                if txt not in sc.ocr_summary:
                    sc.ocr_summary.append(txt)
