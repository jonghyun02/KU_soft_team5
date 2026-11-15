"""Scene boundary detection (PySceneDetect) and ASCII-safe video stream wrapper for Windows."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
import cv2
from PIL import Image

from mos_sdh.schemas.document import SceneContext, SegmentMoS

_AUDIO_ONLY_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


def ensure_ascii_safe_video_path(input_media: Path, cache_dir: Path) -> Path | None:
    """Return an OpenCV-readable video path (creating hardlink/copy in ASCII cache if Korean path fails)."""
    if input_media.suffix.lower() in _AUDIO_ONLY_EXTS or not input_media.exists():
        return None

    cap = cv2.VideoCapture(str(input_media))
    if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) > 0:
        cap.release()
        return input_media
    cap.release()

    stat = input_media.stat()
    fp = hashlib.sha256(f"{input_media.resolve()}|{stat.st_size}".encode("utf-8")).hexdigest()[:16]
    safe_dir = cache_dir / f"media_{fp}"
    safe_dir.mkdir(parents=True, exist_ok=True)
    safe_video = safe_dir / f"video_safe{input_media.suffix.lower()}"
    if not safe_video.exists():
        try:
            os.link(input_media, safe_video)
        except OSError:
            shutil.copy2(input_media, safe_video)

    cap2 = cv2.VideoCapture(str(safe_video))
    ok = cap2.isOpened() and int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)) > 0
    cap2.release()
    return safe_video if ok else None


def detect_scenes_and_keyframes(
    safe_video_path: Path,
    duration_ms: int,
    segments: list[SegmentMoS],
    cache_dir: Path,
) -> tuple[list[SceneContext], dict[int, Path]]:
    """Detect visual scene cuts via PySceneDetect ContentDetector and save one RGB keyframe per scene."""
    from scenedetect import ContentDetector, detect

    keyframe_dir = cache_dir / "keyframes" / safe_video_path.stem
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    raw_scenes = []
    try:
        raw_scenes = detect(str(safe_video_path), ContentDetector(threshold=27.0, min_scene_len=15))
    except Exception:
        raw_scenes = []

    scenes: list[SceneContext] = []
    if not raw_scenes:
        mid_ms = max(0, duration_ms // 2)
        scenes.append(
            SceneContext(
                scene_id=1,
                start_ms=0,
                end_ms=max(1000, duration_ms),
                keyframe_ms=mid_ms,
            )
        )
    else:
        for idx, (s_tc, e_tc) in enumerate(raw_scenes, start=1):
            s_ms = int(round(s_tc.get_seconds() * 1000.0))
            e_ms = int(round(e_tc.get_seconds() * 1000.0))
            mid_ms = (s_ms + e_ms) // 2
            scenes.append(
                SceneContext(
                    scene_id=idx,
                    start_ms=s_ms,
                    end_ms=max(s_ms + 100, e_ms),
                    keyframe_ms=mid_ms,
                )
            )

    # Assign each segment its enclosing scene_id
    for seg in segments:
        seg_mid = (seg.start_ms + seg.end_ms) // 2
        matched = next(
            (sc.scene_id for sc in scenes if sc.start_ms <= seg_mid <= sc.end_ms),
            scenes[0].scene_id,
        )
        seg.visual.scene_id = matched

    # Extract keyframe JPEGs for Stage 3 VLM
    keyframe_paths: dict[int, Path] = {}
    cap = cv2.VideoCapture(str(safe_video_path))
    if cap.isOpened():
        for sc in scenes:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(sc.keyframe_ms))
            ok, frame_bgr = cap.read()
            if ok and frame_bgr is not None:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img_path = keyframe_dir / f"scene_{sc.scene_id:03d}.jpg"
                Image.fromarray(rgb).save(img_path, quality=90)
                keyframe_paths[sc.scene_id] = img_path
    cap.release()

    return scenes, keyframe_paths
