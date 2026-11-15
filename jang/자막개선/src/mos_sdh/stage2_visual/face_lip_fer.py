"""Active Speaker Lip-Motion Detection (ASD via OpenCV YuNet 5-point landmarks) & Continuous ViT FER."""

from __future__ import annotations

from pathlib import Path
import urllib.request
import cv2
import numpy as np
from PIL import Image

from mos_sdh.config import AnalysisConfig, get_default_cache_dir
from mos_sdh.core.vram import StageVRAMReport, gpu_stage
from mos_sdh.schemas.document import SegmentMoS, SpeakerZone

_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"

_EXPRESSION_MAP_KO = {
    "angry": "분노",
    "disgust": "냉소",
    "fear": "공포",
    "happy": "기쁨",
    "sad": "슬픔",
    "surprise": "놀람",
    "neutral": "중립",
}


def _ensure_yunet_model_path(cache_dir: Path) -> Path | None:
    """Locate or download the 232KB OpenCV YuNet ONNX face+landmark detector into persistent local cache."""
    global_model_dir = get_default_cache_dir() / "models"
    global_model_path = global_model_dir / _YUNET_FILENAME
    if global_model_path.exists() and global_model_path.stat().st_size > 100_000:
        return global_model_path

    local_model_dir = cache_dir / "models"
    local_model_path = local_model_dir / _YUNET_FILENAME
    if local_model_path.exists() and local_model_path.stat().st_size > 100_000:
        return local_model_path

    try:
        global_model_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_YUNET_URL, global_model_path)
        return global_model_path
    except Exception:
        return None


def analyze_faces_lips_and_expressions(
    safe_video_path: Path,
    segments: list[SegmentMoS],
    config: AnalysisConfig,
    telemetry_sink: list[StageVRAMReport] | None = None,
) -> dict[int, np.ndarray]:
    """Track faces & 5-point mouth/eye landmarks per segment for on/off-screen ASD and run GPU ViT FER.

    Returns:
        Mapping `segment_id -> frame_fer_matrix` of shape (T_frames, 5) ordered
        `[angry, sad, happy, fear, disgust_sneer]` for Stage 4 Trimodal CmGI.
    """
    temporal_fer_sequences: dict[int, np.ndarray] = {}
    if not segments:
        return temporal_fer_sequences

    cap = cv2.VideoCapture(str(safe_video_path))
    if not cap.isOpened():
        return temporal_fer_sequences

    width = max(1, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = max(1, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    yunet_path = _ensure_yunet_model_path(config.cache_dir)
    face_detector = None
    if yunet_path is not None and hasattr(cv2, "FaceDetectorYN"):
        face_detector = cv2.FaceDetectorYN.create(
            str(yunet_path),
            "",
            (width, height),
            0.55,
            0.3,
            5000,
        )

    seg_face_crops: dict[int, list[Image.Image]] = {}
    step_ms = max(120, int(round(1000.0 / max(1.0, config.visual_sample_fps))))

    for seg in segments:
        sample_times = list(range(seg.start_ms, max(seg.start_ms + 1, seg.end_ms), step_ms))[:16]
        if not sample_times:
            sample_times = [(seg.start_ms + seg.end_ms) // 2]

        mouth_patches: list[np.ndarray] = []
        upper_patches: list[np.ndarray] = []
        x_centers: list[float] = []
        crops: list[Image.Image] = []

        for t_ms in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t_ms))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue

            if face_detector is None:
                continue

            fh_img, fw_img = frame_bgr.shape[:2]
            face_detector.setInputSize((fw_img, fh_img))
            _, detections = face_detector.detect(frame_bgr)
            if detections is None or len(detections) == 0:
                continue

            # Pick largest high-confidence face: row = [x, y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rmc_x, rmc_y, lmc_x, lmc_y, score]
            best = max(detections, key=lambda r: float(r[2]) * float(r[3]))
            fx = max(0, int(round(float(best[0]))))
            fy = max(0, int(round(float(best[1]))))
            fw = max(12, min(fw_img - fx, int(round(float(best[2])))))
            fh = max(12, min(fh_img - fy, int(round(float(best[3])))))
            if fw < 20 or fh < 20:
                continue

            x_centers.append((fx + 0.5 * fw) / fw_img)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # Exact mouth ROI centered on YuNet mouth corners (best[10:14])
            mouth_cx = 0.5 * (float(best[10]) + float(best[12]))
            mouth_cy = 0.5 * (float(best[11]) + float(best[13]))
            mouth_half_w = max(8, int(round(0.30 * fw)))
            mouth_half_h = max(5, int(round(0.14 * fh)))

            mx1 = max(0, int(round(mouth_cx)) - mouth_half_w)
            mx2 = min(fw_img, int(round(mouth_cx)) + mouth_half_w)
            my1 = max(0, int(round(mouth_cy)) - mouth_half_h)
            my2 = min(fh_img, int(round(mouth_cy)) + mouth_half_h)

            # Rigid upper-face baseline ROI centered on eye bridge (best[4:8])
            eye_cx = 0.5 * (float(best[4]) + float(best[6]))
            eye_cy = 0.5 * (float(best[5]) + float(best[7]))
            ux1 = max(0, int(round(eye_cx)) - mouth_half_w)
            ux2 = min(fw_img, int(round(eye_cx)) + mouth_half_w)
            uy1 = max(0, int(round(eye_cy)) - mouth_half_h)
            uy2 = min(fh_img, int(round(eye_cy)) + mouth_half_h)

            if my2 - my1 >= 6 and mx2 - mx1 >= 8:
                m_patch = cv2.resize(gray[my1:my2, mx1:mx2], (32, 16)).astype(np.float32) / 255.0
                mouth_patches.append(m_patch)
            if uy2 - uy1 >= 6 and ux2 - ux1 >= 8:
                u_patch = cv2.resize(gray[uy1:uy2, ux1:ux2], (32, 16)).astype(np.float32) / 255.0
                upper_patches.append(u_patch)

            rgb_crop = cv2.cvtColor(frame_bgr[fy : fy + fh, fx : fx + fw], cv2.COLOR_BGR2RGB)
            crops.append(Image.fromarray(rgb_crop))

        if crops:
            seg.visual.face_detected = True
            seg_face_crops[seg.segment_id] = crops
            mean_xc = float(np.mean(x_centers))
            zone: SpeakerZone = "left" if mean_xc < 0.36 else ("right" if mean_xc > 0.64 else "center")
            seg.visual.speaker_zone = zone

            if len(mouth_patches) >= 2:
                m_diffs = [
                    float(np.mean(np.abs(mouth_patches[i + 1] - mouth_patches[i])))
                    for i in range(len(mouth_patches) - 1)
                ]
                u_diffs = (
                    [
                        float(np.mean(np.abs(upper_patches[i + 1] - upper_patches[i])))
                        for i in range(len(upper_patches) - 1)
                    ]
                    if len(upper_patches) == len(mouth_patches)
                    else [0.0] * len(m_diffs)
                )
                net_lip = float(np.mean([max(0.0, md - 0.65 * ud) for md, ud in zip(m_diffs, u_diffs)]))
                seg.visual.lip_activity_score = round(net_lip, 4)
                seg.visual.on_screen_speaker = net_lip >= config.lip_active_threshold
            else:
                seg.visual.lip_activity_score = 0.02
                seg.visual.on_screen_speaker = True
        else:
            seg.visual.face_detected = False
            seg.visual.lip_activity_score = 0.0
            seg.visual.on_screen_speaker = False

    cap.release()

    if not seg_face_crops:
        return temporal_fer_sequences

    with gpu_stage("Stage 2-A: ViT Facial Expression Recognition", telemetry_sink=telemetry_sink) as device:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        try:
            processor = AutoImageProcessor.from_pretrained(config.fer_model)
            model = AutoModelForImageClassification.from_pretrained(config.fer_model).to(device)
            model.eval()
            id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}

            for seg in segments:
                crops = seg_face_crops.get(seg.segment_id)
                if not crops:
                    continue

                inputs = processor(images=crops, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()

                mean_probs = np.mean(probs, axis=0)
                prob_dict: dict[str, float] = {}
                for idx_lbl, p_val in enumerate(mean_probs):
                    raw_lbl = id2label.get(idx_lbl, f"cls_{idx_lbl}")
                    ko_lbl = _EXPRESSION_MAP_KO.get(raw_lbl, raw_lbl)
                    prob_dict[ko_lbl] = round(prob_dict.get(ko_lbl, 0.0) + float(p_val), 4)

                seg.visual.expression_probs = prob_dict
                if prob_dict:
                    seg.visual.facial_expression = max(prob_dict.items(), key=lambda kv: kv[1])[0]

                seq_5 = np.zeros((len(crops), 5), dtype=np.float32)
                col_order = ["분노", "슬픔", "기쁨", "공포", "냉소"]
                for f_i in range(len(crops)):
                    for idx_lbl, p_val in enumerate(probs[f_i]):
                        ko_lbl = _EXPRESSION_MAP_KO.get(id2label.get(idx_lbl, ""), "중립")
                        if ko_lbl in col_order:
                            seq_5[f_i, col_order.index(ko_lbl)] += float(p_val)
                temporal_fer_sequences[seg.segment_id] = seq_5

            del model, processor
        except Exception:
            pass

    return temporal_fer_sequences
