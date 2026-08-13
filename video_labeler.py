#!/usr/bin/env python3
"""Interactive video labeling utility.

Labels object bounding boxes and gesture frame ranges for gesture-classification
datasets. Annotation sidecars are saved beside each video as
<video_stem>.labels.json.
"""

from __future__ import annotations

import json
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
    ".webm",
    ".mpg",
    ".mpeg",
}

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2 inclusive
SAM2_REPOSITORY = "git+https://github.com/facebookresearch/sam2.git"
SAM2_DEFAULT_MODEL = "facebook/sam2-hiera-tiny"


def ensure_sam2_installed() -> Tuple[bool, str]:
    """Install Meta SAM 2 into the active Python environment when needed."""
    if importlib.util.find_spec("sam2") is not None:
        return True, ""
    print("SAM2 is not installed; installing it into the active Python environment...")
    env = os.environ.copy()
    env["SAM2_BUILD_CUDA"] = "0"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", SAM2_REPOSITORY],
        env=env,
        check=False,
    )
    importlib.invalidate_caches()
    if result.returncode != 0 or importlib.util.find_spec("sam2") is None:
        return False, "Automatic SAM2 installation failed; see the terminal output."
    return True, ""


@dataclass
class Category:
    id: int
    name: str


@dataclass
class LabelConfig:
    object_categories: List[Category]
    gesture_categories: List[Category]
    sam2_model_cfg: str = ""
    sam2_checkpoint: str = ""
    sam2_device: str = "auto"


@dataclass
class VideoItem:
    input_path: Path
    label_path: Path
    status: str


@dataclass
class ObjectTrack:
    id: int
    category_id: int
    boxes: Dict[int, BBox] = field(default_factory=dict)


@dataclass
class GestureRange:
    id: int
    category_id: int
    start_frame: int
    end_frame: int


@dataclass
class AnnotationState:
    video_path: Path
    label_path: Path
    width: int
    height: int
    frame_count: int
    fps: float
    config: LabelConfig
    tracks: List[ObjectTrack] = field(default_factory=list)
    gestures: List[GestureRange] = field(default_factory=list)
    next_track_id: int = 1
    next_gesture_id: int = 1
    status: str = "in_progress"

    def track_by_id(self, track_id: int) -> Optional[ObjectTrack]:
        for track in self.tracks:
            if track.id == track_id:
                return track
        return None

    def ensure_track(self, track_id: Optional[int], category_id: int) -> ObjectTrack:
        if track_id is not None:
            existing = self.track_by_id(track_id)
            if existing is not None:
                existing.category_id = category_id
                return existing

        track = ObjectTrack(id=self.next_track_id, category_id=category_id)
        self.next_track_id += 1
        self.tracks.append(track)
        return track


class Sam2Helper:
    """Optional SAM2 wrapper.

    The app stays usable without SAM2. When SAM2 is configured and importable,
    a selected box can be used as a prompt and propagated to future frames.
    """

    def __init__(self, config: LabelConfig, video_path: Path, frame_count: int) -> None:
        self.config = config
        self.video_path = video_path
        self.frame_count = frame_count
        self.available = False
        self.error = ""
        self.predictor = None
        self.state = None
        self.device = config.sam2_device
        self._try_init()

    def _try_init(self) -> None:
        installed, install_error = ensure_sam2_installed()
        if not installed:
            self.error = install_error
            return
        try:
            import torch
        except Exception as exc:
            self.error = f"SAM2 not importable: {exc}"
            return

        try:
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            sam2_input = prepare_sam2_frame_dir(self.video_path, self.frame_count)
            if self.config.sam2_model_cfg and self.config.sam2_checkpoint:
                from sam2.build_sam import build_sam2_video_predictor
                self.predictor = build_sam2_video_predictor(
                    self.config.sam2_model_cfg,
                    self.config.sam2_checkpoint,
                    device=self.device,
                )
            else:
                from sam2.sam2_video_predictor import SAM2VideoPredictor
                print(f"Downloading/loading default SAM2 model {SAM2_DEFAULT_MODEL}...")
                self.predictor = SAM2VideoPredictor.from_pretrained(
                    SAM2_DEFAULT_MODEL,
                    device=self.device,
                )
            self.state = self.predictor.init_state(str(sam2_input))
            self.available = True
        except Exception as exc:
            self.error = f"SAM2 init failed: {exc}"
            self.predictor = None
            self.state = None
            self.available = False

    def propagate_box(
        self,
        frame_idx: int,
        object_id: int,
        box: BBox,
        max_frame: int,
    ) -> Dict[int, BBox]:
        if not self.available or self.predictor is None or self.state is None:
            raise RuntimeError(self.error or "SAM2 unavailable.")

        import torch

        x1, y1, x2, y2 = box
        prompt_box = np.array([x1, y1, x2, y2], dtype=np.float32)
        results: Dict[int, BBox] = {}

        with torch.inference_mode():
            self.predictor.add_new_points_or_box(
                inference_state=self.state,
                frame_idx=frame_idx,
                obj_id=object_id,
                box=prompt_box,
            )
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.state,
                start_frame_idx=frame_idx,
            ):
                if out_frame_idx > max_frame:
                    break
                if object_id not in out_obj_ids:
                    continue
                obj_pos = list(out_obj_ids).index(object_id)
                mask = (out_mask_logits[obj_pos] > 0.0).detach().cpu().numpy()
                mask = np.squeeze(mask)
                bbox = bbox_from_mask(mask)
                if bbox is not None:
                    results[int(out_frame_idx)] = bbox
        return results


def pick_directory(title: str, initial: Optional[Path] = None) -> Optional[Path]:
    if sys.platform == "darwin":
        script = f'set chosenFolder to choose folder with prompt "{title}"\nPOSIX path of chosenFolder'
        args = ["osascript", "-e", script]
        if initial and initial.is_dir():
            script = (
                f'set startFolder to POSIX file "{str(initial)}"\n'
                f'set chosenFolder to choose folder with prompt "{title}" default location startFolder\n'
                "POSIX path of chosenFolder"
            )
            args = ["osascript", "-e", script]
        try:
            result = subprocess.run(args, capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                return Path(result.stdout.strip()).resolve()
            if result.returncode not in (0, 1) and result.stderr.strip():
                print(f"Folder picker failed: {result.stderr.strip()}")
        except Exception:
            pass

        # Some macOS/Tk builds abort the interpreter instead of raising a
        # Python exception. If AppleScript is unavailable, use stdin.
        print(f"Enter path for {title}:")
        raw = input("> ").strip().strip('"')
        if not raw:
            return None
        path = Path(raw).expanduser().resolve()
        return path if path.is_dir() else None

    tk_pair = import_tkinter()
    if tk_pair is None:
        print(f"Enter path for {title}:")
        raw = input("> ").strip().strip('"')
        if not raw:
            return None
        path = Path(raw).expanduser().resolve()
        return path if path.is_dir() else None
    tk, filedialog = tk_pair

    root = tk.Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askdirectory(
        title=title,
        initialdir=str(initial) if initial else str(Path.cwd()),
        mustexist=True,
    )
    root.destroy()
    return Path(selected).resolve() if selected else None


def pick_file(title: str, initial: Optional[Path] = None) -> Optional[Path]:
    if sys.platform == "darwin":
        script = f'set chosenFile to choose file with prompt "{title}"\nPOSIX path of chosenFile'
        args = ["osascript", "-e", script]
        if initial and initial.is_dir():
            script = (
                f'set startFolder to POSIX file "{str(initial)}"\n'
                f'set chosenFile to choose file with prompt "{title}" default location startFolder\n'
                "POSIX path of chosenFile"
            )
            args = ["osascript", "-e", script]
        try:
            result = subprocess.run(args, capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                return Path(result.stdout.strip()).resolve()
            if result.returncode not in (0, 1) and result.stderr.strip():
                print(f"File picker failed: {result.stderr.strip()}")
        except Exception:
            pass

        print(f"Enter path for {title} (empty for labels.example.json):")
        raw = input("> ").strip().strip('"')
        if not raw:
            return None
        path = Path(raw).expanduser().resolve()
        return path if path.is_file() else None

    tk_pair = import_tkinter()
    if tk_pair is None:
        print(f"Enter path for {title} (empty for labels.example.json):")
        raw = input("> ").strip().strip('"')
        if not raw:
            return None
        path = Path(raw).expanduser().resolve()
        return path if path.is_file() else None
    tk, filedialog = tk_pair

    root = tk.Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askopenfilename(
        title=title,
        initialdir=str(initial) if initial else str(Path.cwd()),
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(selected).resolve() if selected else None


def import_tkinter():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    return tk, filedialog


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS and path.is_file()


def resolve_input_video_dir(selected_dir: Path) -> Path:
    mkv_subdir = selected_dir / "mkv"
    if mkv_subdir.is_dir():
        return mkv_subdir
    return selected_dir


def label_path_for(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}.labels.json")


def collect_videos(input_dir: Path) -> List[VideoItem]:
    videos = sorted([p for p in input_dir.iterdir() if is_video_file(p)], key=lambda p: p.name.lower())
    items = []
    for video_path in videos:
        label_path = label_path_for(video_path)
        status = "pending"
        if label_path.is_file():
            status = "completed"
            try:
                with label_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                if payload.get("info", {}).get("labeling_status") == "in_progress":
                    status = "in_progress"
            except (OSError, json.JSONDecodeError, AttributeError):
                pass
        items.append(VideoItem(video_path, label_path, status))
    return items


def clamp(val: int, low: int, high: int) -> int:
    return max(low, min(high, val))


def normalize_rect(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> BBox:
    xa = clamp(min(x1, x2), 0, w - 1)
    xb = clamp(max(x1, x2), 0, w - 1)
    ya = clamp(min(y1, y2), 0, h - 1)
    yb = clamp(max(y1, y2), 0, h - 1)
    return xa, ya, xb, yb


def point_in_rect(x: int, y: int, rect: BBox) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def coco_bbox(rect: BBox) -> List[int]:
    x1, y1, x2, y2 = rect
    return [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)]


def rect_from_coco(bbox: Iterable[float], width: int, height: int) -> Optional[BBox]:
    vals = list(bbox)
    if len(vals) != 4:
        return None
    x, y, w, h = vals
    if w <= 0 or h <= 0:
        return None
    return normalize_rect(int(round(x)), int(round(y)), int(round(x + w - 1)), int(round(y + h - 1)), width, height)


def bbox_from_mask(mask: np.ndarray) -> Optional[BBox]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def prepare_sam2_frame_dir(video_path: Path, frame_count: int) -> Path:
    """Create/reuse a JPEG frame directory for SAM2 video prediction."""
    try:
        st = video_path.stat()
        sig = f"{int(st.st_mtime_ns)}_{st.st_size}"
    except OSError:
        sig = "unknown"
    base = "".join(ch if ch.isalnum() else "_" for ch in video_path.stem).strip("_") or "video"
    frame_dir = video_path.parent / ".video_labeler_sam2_frames" / f"{base}_{sig}"
    done_file = frame_dir / ".complete"
    if done_file.is_file():
        return frame_dir

    frame_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for SAM2 frame extraction: {video_path}")

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out_path = frame_dir / f"{idx:06d}.jpg"
            if not cv2.imwrite(str(out_path), frame):
                raise RuntimeError(f"Could not write SAM2 frame: {out_path}")
            idx += 1
    finally:
        cap.release()

    if idx == 0:
        raise RuntimeError("No frames extracted for SAM2.")
    if frame_count > 0 and idx < min(frame_count, 2):
        raise RuntimeError(f"Only extracted {idx} frame(s) for SAM2.")
    done_file.write_text(f"{idx}\n", encoding="utf-8")
    return frame_dir


def find_ffmpeg_binary() -> Optional[str]:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        return ffmpeg_bin
    try:
        import imageio_ffmpeg

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_bin and Path(ffmpeg_bin).exists():
            return ffmpeg_bin
    except Exception:
        pass
    return None


def ensure_readable_input(input_path: Path, cache_root: Path) -> Path:
    cap = cv2.VideoCapture(str(input_path))
    if cap.isOpened():
        ok, _frame = cap.read()
        cap.release()
        if ok:
            return input_path
    else:
        cap.release()

    if input_path.suffix.lower() != ".mkv":
        return input_path

    ffmpeg_bin = find_ffmpeg_binary()
    if not ffmpeg_bin:
        return input_path

    try:
        st = input_path.stat()
    except OSError:
        return input_path

    cache_root.mkdir(parents=True, exist_ok=True)
    base = "".join(ch if ch.isalnum() else "_" for ch in input_path.stem).strip("_") or "video"
    sig = f"{int(st.st_mtime_ns)}_{st.st_size}"
    converted = cache_root / f"{base}_{sig}.mp4"
    if converted.is_file():
        return converted

    print(f"Direct MKV ingest unavailable for {input_path.name}; creating MP4 import cache...")
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryslow",
        "-crf",
        "10",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(converted),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        if result.stderr.strip():
            print(f"MKV ingest conversion failed for {input_path.name}: {result.stderr.strip()}")
        return input_path
    return converted


def parse_categories(payload: Dict, key: str) -> List[Category]:
    raw = payload.get(key, [])
    categories: List[Category] = []
    if not isinstance(raw, list):
        raise ValueError(f"{key} must be a list.")
    for idx, item in enumerate(raw, start=1):
        if isinstance(item, str):
            categories.append(Category(id=idx, name=item))
        elif isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            cat_id = int(item.get("id", idx))
            if not name:
                raise ValueError(f"{key} item {idx} is missing a name.")
            categories.append(Category(id=cat_id, name=name))
        else:
            raise ValueError(f"{key} item {idx} must be an object or string.")
    if not categories:
        raise ValueError(f"{key} cannot be empty.")
    ids = [c.id for c in categories]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{key} has duplicate ids.")
    return categories


def load_label_config(path: Path) -> LabelConfig:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Label config must be a JSON object.")
    sam2 = payload.get("sam2", {})
    if not isinstance(sam2, dict):
        sam2 = {}
    return LabelConfig(
        object_categories=parse_categories(payload, "object_categories"),
        gesture_categories=parse_categories(payload, "gesture_categories"),
        sam2_model_cfg=str(sam2.get("model_cfg", "") or ""),
        sam2_checkpoint=str(sam2.get("checkpoint", "") or ""),
        sam2_device=str(sam2.get("device", "auto") or "auto"),
    )


def category_name(categories: List[Category], cat_id: int) -> str:
    for cat in categories:
        if cat.id == cat_id:
            return cat.name
    return f"id:{cat_id}"


def serialize_annotations(state: AnnotationState) -> Dict:
    images = [
        {
            "id": idx + 1,
            "frame_index": idx,
            "file_name": f"{state.video_path.name}#frame={idx}",
            "width": state.width,
            "height": state.height,
        }
        for idx in range(state.frame_count)
    ]
    annotations = []
    ann_id = 1
    for track in state.tracks:
        for frame_idx, rect in sorted(track.boxes.items()):
            bbox = coco_bbox(rect)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": frame_idx + 1,
                    "frame_index": frame_idx,
                    "category_id": track.category_id,
                    "track_id": track.id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return {
        "info": {
            "description": "Video object and gesture labels",
            "version": "1.0",
            "created_by": "video_labeler.py",
            "labeling_status": state.status,
        },
        "video": {
            "file_name": state.video_path.name,
            "path": str(state.video_path),
            "width": state.width,
            "height": state.height,
            "frame_count": state.frame_count,
            "fps": state.fps,
        },
        "images": images,
        "categories": [
            {"id": cat.id, "name": cat.name, "supercategory": "object"}
            for cat in state.config.object_categories
        ],
        "annotations": annotations,
        "tracks": [
            {
                "id": track.id,
                "category_id": track.category_id,
                "frames": [
                    {"frame_index": frame_idx, "bbox": coco_bbox(rect)}
                    for frame_idx, rect in sorted(track.boxes.items())
                ],
            }
            for track in state.tracks
        ],
        "gesture_categories": [
            {"id": cat.id, "name": cat.name}
            for cat in state.config.gesture_categories
        ],
        "gestures": [
            {
                "id": gesture.id,
                "category_id": gesture.category_id,
                "start_frame": min(gesture.start_frame, gesture.end_frame),
                "end_frame": max(gesture.start_frame, gesture.end_frame),
            }
            for gesture in state.gestures
        ],
    }


def save_annotations(state: AnnotationState) -> None:
    payload = serialize_annotations(state)
    with state.label_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_annotations(
    video_path: Path,
    label_path: Path,
    width: int,
    height: int,
    frame_count: int,
    fps: float,
    config: LabelConfig,
) -> AnnotationState:
    state = AnnotationState(video_path, label_path, width, height, frame_count, fps, config)
    if not label_path.is_file():
        return state

    with label_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    info = payload.get("info", {})
    state.status = "in_progress" if isinstance(info, dict) and info.get("labeling_status") == "in_progress" else "completed"

    tracks_payload = payload.get("tracks", [])
    if isinstance(tracks_payload, list):
        for item in tracks_payload:
            if not isinstance(item, dict):
                continue
            track_id = int(item.get("id", state.next_track_id))
            category_id = int(item.get("category_id", config.object_categories[0].id))
            track = ObjectTrack(id=track_id, category_id=category_id)
            for frame_item in item.get("frames", []):
                if not isinstance(frame_item, dict):
                    continue
                frame_idx = int(frame_item.get("frame_index", -1))
                rect = rect_from_coco(frame_item.get("bbox", []), width, height)
                if rect is not None and 0 <= frame_idx < frame_count:
                    track.boxes[frame_idx] = rect
            state.tracks.append(track)

    if not state.tracks:
        grouped: Dict[int, ObjectTrack] = {}
        for item in payload.get("annotations", []):
            if not isinstance(item, dict):
                continue
            track_id = int(item.get("track_id", item.get("id", state.next_track_id)))
            category_id = int(item.get("category_id", config.object_categories[0].id))
            frame_idx = int(item.get("frame_index", int(item.get("image_id", 1)) - 1))
            rect = rect_from_coco(item.get("bbox", []), width, height)
            if rect is None or not (0 <= frame_idx < frame_count):
                continue
            track = grouped.setdefault(track_id, ObjectTrack(id=track_id, category_id=category_id))
            track.boxes[frame_idx] = rect
        state.tracks.extend(sorted(grouped.values(), key=lambda t: t.id))

    for item in payload.get("gestures", []):
        if not isinstance(item, dict):
            continue
        gesture = GestureRange(
            id=int(item.get("id", state.next_gesture_id)),
            category_id=int(item.get("category_id", config.gesture_categories[0].id)),
            start_frame=clamp(int(item.get("start_frame", 0)), 0, max(frame_count - 1, 0)),
            end_frame=clamp(int(item.get("end_frame", 0)), 0, max(frame_count - 1, 0)),
        )
        state.gestures.append(gesture)

    if state.tracks:
        state.next_track_id = max(track.id for track in state.tracks) + 1
    if state.gestures:
        state.next_gesture_id = max(gesture.id for gesture in state.gestures) + 1
    return state


class VideoLabelEditor:
    def __init__(self, input_path: Path, config: LabelConfig) -> None:
        self.input_path = input_path
        self.config = config
        self.label_path = label_path_for(input_path)
        self.ingest_cache_dir = input_path.parent / ".video_labeler_import_cache"
        self.read_path = ensure_readable_input(input_path, self.ingest_cache_dir)

        self.cap = cv2.VideoCapture(str(self.read_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.state = load_annotations(
            self.input_path,
            self.label_path,
            self.width,
            self.height,
            self.frame_count,
            self.fps,
            self.config,
        )

        self.frame_idx = 0
        self.last_frame_idx = -1
        self.last_frame = None
        self.playing = False
        self.speed_levels = [round(1 + i * (49.0 / 8.0)) for i in range(9)]
        self.speed_idx = 0
        self.last_tick = time.time()
        self.window = f"Video Labeler - {self.input_path.name}"

        self.object_category_idx = 0
        self.gesture_category_idx = 0
        self.active_track_id: Optional[int] = self.state.tracks[0].id if self.state.tracks else None
        self.active_box_frame: Optional[int] = None

        self.drawing = False
        self.draw_start = (0, 0)
        self.draw_current = (0, 0)
        self.moving = False
        self.move_offset = (0, 0)
        self.move_size = (0, 0)
        self.move_preview: Optional[BBox] = None
        self.wheel_accum = 0
        self.gesture_start: Optional[int] = None
        self.selected_gesture_id: Optional[int] = None
        self.open_menu: Optional[str] = None
        self.menu_scroll = {
            "object": 0,
            "gesture_category": 0,
            "track": 0,
            "gesture": 0,
        }
        self.message = ""
        self.message_until = 0.0
        self.dirty = False

        self.sam2: Optional[Sam2Helper] = None

    def close(self) -> None:
        self.cap.release()
        try:
            cv2.destroyWindow(self.window)
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            cv2.waitKey(1)
        except Exception:
            pass

    def set_message(self, text: str, seconds: float = 2.0) -> None:
        self.message = text
        self.message_until = time.time() + seconds
        print(text)

    def seek(self, idx: int) -> None:
        self.frame_idx = clamp(idx, 0, max(self.frame_count - 1, 0))

    def _read_frame(self, idx: int):
        if idx == self.last_frame_idx and self.last_frame is not None:
            return self.last_frame.copy()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.last_frame_idx = idx
        self.last_frame = frame
        return frame.copy()

    def _wheel_delta(self, flags: int) -> int:
        raw = (flags >> 16) & 0xFFFF
        if raw & 0x8000:
            raw -= 0x10000
        return raw

    def _current_object_category_id(self) -> int:
        return self.config.object_categories[self.object_category_idx].id

    def _current_gesture_category_id(self) -> int:
        return self.config.gesture_categories[self.gesture_category_idx].id

    def _tracks_on_frame(self, frame_idx: int) -> List[Tuple[ObjectTrack, BBox]]:
        found = []
        for track in self.state.tracks:
            if frame_idx in track.boxes:
                found.append((track, track.boxes[frame_idx]))
        return found

    def _find_track_at(self, x: int, y: int) -> Optional[ObjectTrack]:
        for track, rect in reversed(self._tracks_on_frame(self.frame_idx)):
            if point_in_rect(x, y, rect):
                return track
        return None

    def _on_mouse(self, event, x, y, flags, _userdata):
        if event in (cv2.EVENT_MOUSEWHEEL, cv2.EVENT_MOUSEHWHEEL):
            delta = self._wheel_delta(flags)
            if delta == 0:
                return
            if self.open_menu is not None:
                self._scroll_open_menu(-1 if delta > 0 else 1)
                return
            self.wheel_accum += delta
            step_count = 0
            while self.wheel_accum >= 120:
                step_count += 1
                self.wheel_accum -= 120
            while self.wheel_accum <= -120:
                step_count -= 1
                self.wheel_accum += 120
            if step_count:
                self.playing = False
                self.seek(self.frame_idx + step_count)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self._handle_menu_click(x, y):
                return
            track = self._find_track_at(x, y)
            if track is not None:
                rect = track.boxes[self.frame_idx]
                self.active_track_id = track.id
                self.object_category_idx = self._category_index(self.config.object_categories, track.category_id)
                self.moving = True
                self.move_offset = (x - rect[0], y - rect[1])
                self.move_size = (rect[2] - rect[0], rect[3] - rect[1])
                self.move_preview = rect
            else:
                self.drawing = True
                self.draw_start = (x, y)
                self.draw_current = (x, y)
                self.active_box_frame = self.frame_idx

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.draw_current = (x, y)
            if self.moving:
                w, h = self.move_size
                ox, oy = self.move_offset
                nx1 = clamp(x - ox, 0, self.width - 1)
                ny1 = clamp(y - oy, 0, self.height - 1)
                nx2 = clamp(nx1 + w, 0, self.width - 1)
                ny2 = clamp(ny1 + h, 0, self.height - 1)
                self.move_preview = normalize_rect(nx1, ny1, nx2, ny2, self.width, self.height)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                x1, y1 = self.draw_start
                x2, y2 = self.draw_current
                rect = normalize_rect(x1, y1, x2, y2, self.width, self.height)
                if (rect[2] - rect[0]) >= 2 and (rect[3] - rect[1]) >= 2:
                    track = self.state.ensure_track(self.active_track_id, self._current_object_category_id())
                    track.boxes[self.frame_idx] = rect
                    self.active_track_id = track.id
                    self.active_box_frame = self.frame_idx
                    self.dirty = True
                self.drawing = False

            if self.moving and self.move_preview is not None and self.active_track_id is not None:
                track = self.state.track_by_id(self.active_track_id)
                if track is not None:
                    track.boxes[self.frame_idx] = self.move_preview
                    track.category_id = self._current_object_category_id()
                    self.dirty = True
                self.moving = False
                self.move_preview = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            track = self._find_track_at(x, y)
            if track is not None and self.frame_idx in track.boxes:
                del track.boxes[self.frame_idx]
                if not track.boxes:
                    self.state.tracks = [t for t in self.state.tracks if t.id != track.id]
                    if self.active_track_id == track.id:
                        self.active_track_id = self.state.tracks[0].id if self.state.tracks else None
                self.dirty = True

    def _selected_gesture(self) -> Optional[GestureRange]:
        for gesture in self.state.gestures:
            if gesture.id == self.selected_gesture_id:
                return gesture
        return None

    def _menu_definitions(self):
        tracks = [("New track", None)] + [
            (f"T{track.id}: {category_name(self.config.object_categories, track.category_id)}", track.id)
            for track in self.state.tracks
        ]
        gestures = [("New gesture", None)] + [
            (f"G{g.id}: {category_name(self.config.gesture_categories, g.category_id)} "
             f"[{min(g.start_frame, g.end_frame) + 1}-{max(g.start_frame, g.end_frame) + 1}]", g.id)
            for g in self.state.gestures
        ]
        return {
            "object": (10, 34, 190, [(cat.name, i) for i, cat in enumerate(self.config.object_categories)]),
            "gesture_category": (205, 34, 190, [(cat.name, i) for i, cat in enumerate(self.config.gesture_categories)]),
            "track": (400, 34, 190, tracks),
            "gesture": (595, 34, 300, gestures),
        }

    def _handle_menu_click(self, x: int, y: int) -> bool:
        menus = self._menu_definitions()
        if self.open_menu is not None:
            mx, my, mw, options = menus[self.open_menu]
            offset, visible = self._menu_window(self.open_menu, options, my)
            if mx <= x < mx + mw and my + 24 <= y < my + 24 * (visible + 1):
                index = offset + (y - my - 24) // 24
                self._select_menu_option(self.open_menu, options[index][1])
                self.open_menu = None
                return True
        for name, (mx, my, mw, _options) in menus.items():
            if mx <= x < mx + mw and my <= y < my + 24:
                self.open_menu = None if self.open_menu == name else name
                return True
        if self.open_menu is not None:
            self.open_menu = None
            return True
        return False

    def _menu_window(self, name: str, options, menu_y: int) -> Tuple[int, int]:
        # Keep lists compact and above the bottom help text, even on tall videos.
        visible = min(len(options), max(1, min(20, (self.height - menu_y - 72) // 24)))
        max_offset = max(0, len(options) - visible)
        offset = clamp(self.menu_scroll.get(name, 0), 0, max_offset)
        self.menu_scroll[name] = offset
        return offset, visible

    def _scroll_open_menu(self, direction: int) -> None:
        if self.open_menu is None:
            return
        _x, menu_y, _width, options = self._menu_definitions()[self.open_menu]
        offset, visible = self._menu_window(self.open_menu, options, menu_y)
        max_offset = max(0, len(options) - visible)
        self.menu_scroll[self.open_menu] = clamp(offset + direction * 3, 0, max_offset)

    def _select_menu_option(self, menu: str, value) -> None:
        if menu == "object":
            self.object_category_idx = value
            track = self.state.track_by_id(self.active_track_id) if self.active_track_id is not None else None
            if track is not None:
                track.category_id = self._current_object_category_id()
                self.dirty = True
        elif menu == "gesture_category":
            self.gesture_category_idx = value
            gesture = self._selected_gesture()
            if gesture is not None:
                gesture.category_id = self._current_gesture_category_id()
                self.dirty = True
        elif menu == "track":
            self.active_track_id = value
            track = self.state.track_by_id(value) if value is not None else None
            if track is not None:
                self.object_category_idx = self._category_index(self.config.object_categories, track.category_id)
        elif menu == "gesture":
            self.selected_gesture_id = value
            gesture = self._selected_gesture()
            if gesture is not None:
                self.gesture_category_idx = self._category_index(self.config.gesture_categories, gesture.category_id)
                self.playing = False
                self.seek(min(gesture.start_frame, gesture.end_frame))

    def _category_index(self, categories: List[Category], cat_id: int) -> int:
        for idx, cat in enumerate(categories):
            if cat.id == cat_id:
                return idx
        return 0

    def _advance_playback(self) -> None:
        if not self.playing:
            return
        now = time.time()
        elapsed = now - self.last_tick
        step = int(elapsed * self.fps * self.speed_levels[self.speed_idx])
        if step > 0:
            self.seek(self.frame_idx + step)
            self.last_tick = now
            if self.frame_idx >= self.frame_count - 1:
                self.playing = False

    def _raise_window(self) -> None:
        try:
            cv2.resizeWindow(self.window, min(self.width, 1400), min(self.height, 900))
        except Exception:
            pass
        try:
            cv2.moveWindow(self.window, 80, 80)
        except Exception:
            pass

    def _draw_overlay(self, frame) -> None:
        for track, rect in self._tracks_on_frame(self.frame_idx):
            x1, y1, x2, y2 = rect
            active = track.id == self.active_track_id
            color = (0, 255, 255) if active else (0, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"T{track.id}:{category_name(self.config.object_categories, track.category_id)}"
            cv2.putText(frame, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

        if self.drawing:
            x1, y1 = self.draw_start
            x2, y2 = self.draw_current
            rect = normalize_rect(x1, y1, x2, y2, self.width, self.height)
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

        if self.moving and self.move_preview is not None:
            x1, y1, x2, y2 = self.move_preview
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        for gesture in self.state.gestures:
            start = min(gesture.start_frame, gesture.end_frame)
            end = max(gesture.start_frame, gesture.end_frame)
            if start <= self.frame_idx <= end:
                text = f"Gesture: {category_name(self.config.gesture_categories, gesture.category_id)} [{start + 1}-{end + 1}]"
                if gesture.id == self.selected_gesture_id:
                    text += " (selected)"
                cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 255, 80), 2)
                break

        status = "PLAY" if self.playing else "PAUSE"
        dirty = "*" if self.dirty else ""
        txt1 = (
            f"Frame {self.frame_idx + 1}/{max(self.frame_count, 1)} | {status} | "
            f"speed {self.speed_levels[self.speed_idx]}x | {dirty}{self.label_path.name}"
        )
        cv2.putText(frame, txt1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        self._draw_menus(frame)
        txt3 = "Drag=Draw/Move  RightClick=Delete  B/E=Set gesture bounds  X=Delete gesture"
        txt4 = "Scroll/Arrows/J/L=Scrub  P=Init/Propagate SAM2  S=Save  I=InProgress  Q=Complete"
        cv2.putText(frame, txt3, (10, self.height - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(frame, txt4, (10, self.height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        if self.message and time.time() < self.message_until:
            cv2.putText(frame, self.message[:90], (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

    def _draw_menus(self, frame) -> None:
        selected = self._selected_gesture()
        labels = {
            "object": f"Object: {category_name(self.config.object_categories, self._current_object_category_id())}",
            "gesture_category": f"Type: {category_name(self.config.gesture_categories, self._current_gesture_category_id())}",
            "track": f"Track: {self.active_track_id if self.active_track_id is not None else 'new'}",
            "gesture": f"Gesture: {selected.id if selected is not None else 'new'}",
        }
        for name, (x, y, width, options) in self._menu_definitions().items():
            header = labels[name]
            if self.open_menu == name:
                offset, visible = self._menu_window(name, options, y)
                if len(options) > visible:
                    header += f" [{offset + 1}-{offset + visible}/{len(options)}]"
            cv2.rectangle(frame, (x, y), (x + width, y + 23), (55, 55, 55), -1)
            cv2.rectangle(frame, (x, y), (x + width, y + 23), (210, 210, 210), 1)
            cv2.putText(frame, header[:36], (x + 5, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
            cv2.putText(frame, "v", (x + width - 15, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            if self.open_menu == name:
                for i, (label, _value) in enumerate(options[offset:offset + visible]):
                    option_y = y + 24 * (i + 1)
                    cv2.rectangle(frame, (x, option_y), (x + width, option_y + 23), (35, 35, 35), -1)
                    cv2.rectangle(frame, (x, option_y), (x + width, option_y + 23), (160, 160, 160), 1)
                    cv2.putText(frame, label[:42], (x + 5, option_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _cycle_object_category(self) -> None:
        self.object_category_idx = (self.object_category_idx + 1) % len(self.config.object_categories)
        if self.active_track_id is not None:
            track = self.state.track_by_id(self.active_track_id)
            if track is not None:
                track.category_id = self._current_object_category_id()
                self.dirty = True

    def _cycle_gesture_category(self) -> None:
        self.gesture_category_idx = (self.gesture_category_idx + 1) % len(self.config.gesture_categories)

    def _new_track(self) -> None:
        self.active_track_id = None
        self.set_message("Drawing will create a new track.")

    def _next_track(self) -> None:
        if not self.state.tracks:
            self.active_track_id = None
            return
        ids = sorted(track.id for track in self.state.tracks)
        if self.active_track_id not in ids:
            self.active_track_id = ids[0]
        else:
            self.active_track_id = ids[(ids.index(self.active_track_id) + 1) % len(ids)]
        track = self.state.track_by_id(self.active_track_id)
        if track is not None:
            self.object_category_idx = self._category_index(self.config.object_categories, track.category_id)

    def _begin_gesture(self) -> None:
        selected = self._selected_gesture()
        if selected is not None:
            selected.start_frame = self.frame_idx
            self.dirty = True
            self.set_message(f"Gesture {selected.id} start moved to frame {self.frame_idx + 1}.")
            return
        self.gesture_start = self.frame_idx
        self.set_message(f"Gesture start set at frame {self.frame_idx + 1}.")

    def _end_gesture(self) -> None:
        selected = self._selected_gesture()
        if selected is not None:
            selected.end_frame = self.frame_idx
            self.dirty = True
            self.set_message(f"Gesture {selected.id} end moved to frame {self.frame_idx + 1}.")
            return
        if self.gesture_start is None:
            self.set_message("Set gesture start with B first.")
            return
        gesture = GestureRange(
            id=self.state.next_gesture_id,
            category_id=self._current_gesture_category_id(),
            start_frame=self.gesture_start,
            end_frame=self.frame_idx,
        )
        self.state.next_gesture_id += 1
        self.state.gestures.append(gesture)
        self.selected_gesture_id = gesture.id
        self.gesture_start = None
        self.dirty = True
        self.set_message("Gesture range added.")

    def _delete_current_gesture(self) -> None:
        selected = self._selected_gesture()
        if selected is not None:
            self.state.gestures.remove(selected)
            self.selected_gesture_id = None
            self.dirty = True
            self.set_message("Selected gesture deleted.")
            return
        for idx in range(len(self.state.gestures) - 1, -1, -1):
            gesture = self.state.gestures[idx]
            start = min(gesture.start_frame, gesture.end_frame)
            end = max(gesture.start_frame, gesture.end_frame)
            if start <= self.frame_idx <= end:
                del self.state.gestures[idx]
                self.dirty = True
                self.set_message("Gesture range deleted.")
                return
        self.set_message("No gesture range on this frame.")

    def _save(self, status: Optional[str] = None) -> None:
        if status is not None:
            self.state.status = status
        save_annotations(self.state)
        self.dirty = False
        self.set_message(f"Saved {self.label_path.name}.")

    def _sam2_propagate(self) -> None:
        if self.active_track_id is None:
            self.set_message("Select or draw a track before SAM2 propagation.")
            return
        track = self.state.track_by_id(self.active_track_id)
        if track is None or self.frame_idx not in track.boxes:
            self.set_message("Active track needs a box on this frame.")
            return
        if self.sam2 is None:
            self.set_message("Initializing SAM2 (first use may install/download it)...", seconds=5.0)
            self.sam2 = Sam2Helper(self.config, self.read_path, self.frame_count)
        if not self.sam2.available:
            self.set_message(self.sam2.error or "SAM2 unavailable.", seconds=4.0)
            return
        self.set_message("SAM2 propagation running; UI may pause.")
        try:
            boxes = self.sam2.propagate_box(
                self.frame_idx,
                track.id,
                track.boxes[self.frame_idx],
                self.frame_count - 1,
            )
        except Exception as exc:
            self.set_message(f"SAM2 propagation failed: {exc}", seconds=4.0)
            return
        track.boxes.update(boxes)
        self.dirty = True
        self.set_message(f"SAM2 propagated {len(boxes)} boxes. Drag any box to correct it.", seconds=4.0)

    def run(self) -> str:
        if self.frame_count <= 0:
            print(f"No frames found in video: {self.input_path}")
            return ""

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        self._raise_window()
        cv2.setMouseCallback(self.window, self._on_mouse)

        try:
            while True:
                self._advance_playback()
                frame = self._read_frame(self.frame_idx)
                if frame is None:
                    break

                self._draw_overlay(frame)
                cv2.imshow(self.window, frame)

                key = cv2.waitKeyEx(10)
                key_low = key & 0xFF
                if key == -1:
                    continue
                if key_low in (ord("q"), 27):
                    self._save("completed")
                    break
                if key_low == ord("i"):
                    self._save("in_progress")
                    break
                if key_low == ord(" "):
                    self.playing = not self.playing
                    self.last_tick = time.time()
                elif key in (2424832, 65361) or key_low == ord("j"):
                    self.playing = False
                    self.seek(self.frame_idx - 1)
                elif key in (2555904, 65363) or key_low == ord("l"):
                    self.playing = False
                    self.seek(self.frame_idx + 1)
                elif key_low == ord("a"):
                    self.playing = False
                    self.seek(self.frame_idx - 10)
                elif key_low == ord("d"):
                    self.playing = False
                    self.seek(self.frame_idx + 10)
                elif key_low == ord("["):
                    self.playing = False
                    self.seek(self.frame_idx - 100)
                elif key_low == ord("]"):
                    self.playing = False
                    self.seek(self.frame_idx + 100)
                elif ord("1") <= key_low <= ord("9"):
                    self.speed_idx = key_low - ord("1")
                elif key_low == ord("s"):
                    self._save()
                elif key_low == ord("n"):
                    self._new_track()
                elif key_low == ord("t"):
                    self._next_track()
                elif key_low == ord("o"):
                    self._cycle_object_category()
                elif key_low == ord("g"):
                    self._cycle_gesture_category()
                elif key_low == ord("b"):
                    self._begin_gesture()
                elif key_low == ord("e"):
                    self._end_gesture()
                elif key_low == ord("x"):
                    self._delete_current_gesture()
                elif key_low == ord("p"):
                    self._sam2_propagate()
        finally:
            self.close()
        return "saved" if self.label_path.exists() else ""


def print_menu(items: List[VideoItem], input_dir: Path, config_path: Path) -> None:
    clear_screen()
    print("Video Labeler")
    print(f"Input : {input_dir}")
    print(f"Config: {config_path}")
    print()
    if not items:
        print("No video files found in input directory.")
    else:
        print("Idx  Status   Video")
        print("---  -------  ------------------------------")
        for i, item in enumerate(items, start=1):
            status = {"completed": "LABELED", "in_progress": "IN PROG"}.get(item.status, "PENDING")
            print(f"{i:>3}  {status:<7}  {item.input_path.name}")
    print()
    print("Commands:")
    print("  <number> : open video editor")
    print("  n        : open next unlabeled")
    print("  i        : choose input directory")
    print("  c        : choose label config")
    print("  r        : refresh menu")
    print("  q        : quit")


def select_config(initial_dir: Path) -> Path:
    default = Path.cwd() / "labels.example.json"
    selected = pick_file("Select Label Config JSON", initial=initial_dir)
    if selected is None:
        selected = default
    return selected.resolve()


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] in {"--help", "-h"}:
        print("Usage: python video_labeler.py [label_config.json] [video_directory]")
        return 0

    if len(sys.argv) >= 3:
        input_dir = resolve_input_video_dir(Path(sys.argv[2]).expanduser().resolve())
    else:
        print("Select input video directory.")
        selected = pick_directory("Select Input Video Directory")
        if selected is None:
            print("No input directory selected.")
            return 1
        input_dir = resolve_input_video_dir(selected)

    if len(sys.argv) >= 2:
        config_path = Path(sys.argv[1]).expanduser().resolve()
    else:
        config_path = select_config(input_dir)

    try:
        config = load_label_config(config_path)
    except Exception as exc:
        print(f"Failed to load label config {config_path}: {exc}")
        return 1

    last_opened_idx = -1
    while True:
        items = collect_videos(input_dir)
        print_menu(items, input_dir, config_path)
        cmd = input("\nEnter command: ").strip().lower()
        if not cmd:
            continue
        if cmd == "q":
            return 0
        if cmd == "r":
            continue
        if cmd == "i":
            chosen = pick_directory("Select Input Video Directory", initial=input_dir)
            if chosen:
                input_dir = resolve_input_video_dir(chosen)
            continue
        if cmd == "c":
            chosen_config = select_config(input_dir)
            try:
                config = load_label_config(chosen_config)
                config_path = chosen_config
                print(f"Config set to: {config_path}")
            except Exception as exc:
                print(f"Failed to load config: {exc}")
                time.sleep(1.5)
            continue
        if cmd == "n":
            pending = [idx for idx, item in enumerate(items) if item.status != "completed"]
            if not pending:
                print("All videos are labeled.")
                time.sleep(1)
                continue
            start = (last_opened_idx + 1) % len(items) if items else 0
            chosen_idx = pending[0]
            for offset in range(len(items)):
                idx = (start + offset) % len(items)
                if idx in pending:
                    chosen_idx = idx
                    break
            last_opened_idx = chosen_idx
            VideoLabelEditor(items[chosen_idx].input_path, config).run()
            continue
        if cmd.isdigit():
            idx = int(cmd)
            if idx < 1 or idx > len(items):
                print("Invalid index.")
                time.sleep(1)
                continue
            last_opened_idx = idx - 1
            VideoLabelEditor(items[idx - 1].input_path, config).run()
            continue
        print("Unknown command.")
        time.sleep(1)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nExited.")
        raise SystemExit(130)
