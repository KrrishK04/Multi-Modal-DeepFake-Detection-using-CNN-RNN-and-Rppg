#!/usr/bin/env python3
"""
Preprocess FaceForensics++ videos into a reusable face-frame cache.

The output is designed for the ViT-S + temporal transformer training path:
each video becomes a directory of uniformly sampled face crops. Training then
loads cached images instead of decoding and face-detecting the original mp4 on
every epoch.

Default layout:
  FaceForensics++_C23/preprocessed/faces_v1_224/
    manifest.csv
    train_manifest.csv
    val_manifest.csv
    test_manifest.csv
    frames/<category>/<video_stem>/000.png
    frames/<category>/<video_stem>/001.png
    ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent

_CFG: Dict[str, Any] = {}
_MTCNN: Any = None


@dataclass(frozen=True)
class Sample:
    rel_path: str
    label: str
    category: str
    split: str


@dataclass
class ProcessResult:
    rel_video_path: str
    label: str
    category: str
    split: str
    cache_dir: str
    cached_frames: int
    source_frames: int
    face_failures: int
    center_fallbacks: int
    reused_previous_box: int
    success: int
    reason: str
    seconds: float


def _clean_rel_path(path: str) -> str:
    return path.strip().replace("\\", "/")


def _find_columns(header: Sequence[str]) -> Tuple[int, int]:
    h = [c.strip() for c in header]
    try:
        return h.index("File Path"), h.index("Label")
    except ValueError:
        if len(h) >= 3:
            return 1, 2
        raise ValueError(f"Could not find File Path / Label columns in {header!r}")


def _category_from_rel_path(rel_path: str) -> str:
    return _clean_rel_path(rel_path).split("/", 1)[0]


def load_metadata_samples(metadata_csv: Path) -> List[Sample]:
    samples: List[Sample] = []
    with metadata_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        path_idx, label_idx = _find_columns(header)
        for row in reader:
            if len(row) <= max(path_idx, label_idx):
                continue
            rel_path = _clean_rel_path(row[path_idx])
            label = row[label_idx].strip().upper()
            if not rel_path or label not in {"FAKE", "REAL"}:
                continue
            samples.append(
                Sample(
                    rel_path=rel_path,
                    label=label,
                    category=_category_from_rel_path(rel_path),
                    split="all",
                )
            )
    return samples


def load_split_samples(split_csv: Path, split: str) -> List[Sample]:
    samples: List[Sample] = []
    with split_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = _clean_rel_path(row["File Path"])
            label = row["Label"].strip().upper()
            if label not in {"FAKE", "REAL"}:
                raise ValueError(f"Invalid label {label!r} in {split_csv}")
            samples.append(
                Sample(
                    rel_path=rel_path,
                    label=label,
                    category=_category_from_rel_path(rel_path),
                    split=split,
                )
            )
    return samples


def collect_samples(
    metadata_csv: Path,
    split_dir: Optional[Path],
    splits: Sequence[str],
) -> List[Sample]:
    if split_dir is not None:
        out: List[Sample] = []
        for split in splits:
            split_csv = split_dir / f"{split}_split.csv"
            if not split_csv.is_file():
                raise FileNotFoundError(f"Split CSV not found: {split_csv}")
            out.extend(load_split_samples(split_csv, split))
        return out
    return load_metadata_samples(metadata_csv)


def filter_samples(
    samples: List[Sample],
    categories: Optional[set[str]],
    limit: int,
    limit_per_split: int,
    seed: int,
) -> List[Sample]:
    if categories is not None:
        samples = [s for s in samples if s.category in categories]

    if limit_per_split > 0:
        rng = random.Random(seed)
        grouped: Dict[str, List[Sample]] = {}
        for sample in samples:
            grouped.setdefault(sample.split, []).append(sample)
        limited: List[Sample] = []
        for split in sorted(grouped):
            rows = grouped[split]
            rng.shuffle(rows)
            limited.extend(rows[:limit_per_split])
        samples = limited

    if limit > 0 and len(samples) > limit:
        rng = random.Random(seed)
        samples = samples[:]
        rng.shuffle(samples)
        samples = samples[:limit]

    return samples


def cache_dir_for_sample(output_root: Path, rel_path: str) -> Path:
    rel = Path(_clean_rel_path(rel_path))
    return output_root / "frames" / rel.with_suffix("")


def image_extension(image_format: str) -> str:
    return ".jpg" if image_format == "jpg" else ".png"


def expected_image_paths(cache_dir: Path, count: int, image_format: str) -> List[Path]:
    ext = image_extension(image_format)
    return [cache_dir / f"{idx:03d}{ext}" for idx in range(count)]


def list_cached_images(cache_dir: Path, image_format: str) -> List[Path]:
    ext = image_extension(image_format)
    return sorted(cache_dir.glob(f"*{ext}"))


def select_frame_indices(total_frames: int, frames_per_video: int) -> List[int]:
    if total_frames <= 0:
        return []
    if frames_per_video <= 1:
        return [0]
    if total_frames >= frames_per_video:
        return np.linspace(0, total_frames - 1, frames_per_video, dtype=int).tolist()
    base = list(range(total_frames))
    base.extend([total_frames - 1] * (frames_per_video - total_frames))
    return base


def iter_selected_frames(
    cap: cv2.VideoCapture,
    indices: Sequence[int],
) -> Iterable[Tuple[int, int, Optional[np.ndarray], bool]]:
    """Yield selected frames by walking the video forward once.

    Each item is (output_idx, source_idx, frame_bgr_or_none, reuse_previous).
    When reuse_previous is True, the frame matches the last decoded frame;
    callers should reuse the previous crop/box without decoding or MTCNN again.
    When frame is None and reuse_previous is False, decoding failed.
    """
    current_frame = 0
    last_source_idx = -1
    last_frame: Optional[np.ndarray] = None

    for output_idx, source_idx in enumerate(indices):
        source_idx = int(source_idx)

        if last_frame is not None and source_idx == last_source_idx:
            yield output_idx, source_idx, None, True
            continue

        while current_frame < source_idx:
            ok = cap.grab()
            if not ok:
                yield output_idx, source_idx, None, False
                return
            current_frame += 1

        ok, frame = cap.read()
        if not ok or frame is None:
            yield output_idx, source_idx, None, False
            return

        current_frame += 1
        last_source_idx = source_idx
        last_frame = frame
        yield output_idx, source_idx, frame, False


def init_worker(config: Dict[str, Any]) -> None:
    global _CFG, _MTCNN
    _CFG = config
    cv2.setNumThreads(0)
    torch.set_num_threads(1)

    from facenet_pytorch import MTCNN

    _MTCNN = MTCNN(keep_all=True, device=torch.device("cpu"))


def _box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def choose_face_box(
    boxes: Optional[np.ndarray],
    probs: Optional[np.ndarray],
    last_box: Optional[np.ndarray],
    frame_shape: Tuple[int, int, int],
    min_face_prob: float,
) -> Optional[np.ndarray]:
    if boxes is None or len(boxes) == 0:
        return None

    valid: List[np.ndarray] = []
    for idx, box in enumerate(boxes):
        prob = 1.0 if probs is None else float(probs[idx])
        if prob >= min_face_prob:
            valid.append(np.asarray(box, dtype=np.float32))
    if not valid:
        return None

    if last_box is None:
        return max(valid, key=_box_area)

    h, w = frame_shape[:2]
    diag = float((w * w + h * h) ** 0.5)
    last_center = np.array(
        [(last_box[0] + last_box[2]) / 2.0, (last_box[1] + last_box[3]) / 2.0],
        dtype=np.float32,
    )

    def score(box: np.ndarray) -> float:
        center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)
        distance = float(np.linalg.norm(center - last_center)) / max(diag, 1.0)
        area_bonus = _box_area(box) / max(float(w * h), 1.0)
        return distance - 0.15 * area_bonus

    return min(valid, key=score)


def center_box(frame_shape: Tuple[int, int, int]) -> np.ndarray:
    h, w = frame_shape[:2]
    side = min(h, w)
    left = (w - side) // 2
    top = (h - side) // 2
    return np.array([left, top, left + side, top + side], dtype=np.float32)


def expand_square_box(
    box: np.ndarray,
    frame_shape: Tuple[int, int, int],
    margin: float,
) -> Tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + margin)
    side = max(side, 2.0)

    left = int(round(cx - side / 2.0))
    right = int(round(cx + side / 2.0))
    top = int(round(cy - side / 2.0))
    bottom = int(round(cy + side / 2.0))

    left = max(0, min(left, w - 1))
    right = max(left + 1, min(right, w))
    top = max(0, min(top, h - 1))
    bottom = max(top + 1, min(bottom, h))
    return left, top, right, bottom


def crop_resize_rgb(
    frame_bgr: np.ndarray,
    box: np.ndarray,
    output_size: int,
    margin: float,
) -> np.ndarray:
    left, top, right, bottom = expand_square_box(box, frame_bgr.shape, margin)
    crop_bgr = frame_bgr[top:bottom, left:right]
    if crop_bgr.size == 0:
        left, top, right, bottom = expand_square_box(center_box(frame_bgr.shape), frame_bgr.shape, 0.0)
        crop_bgr = frame_bgr[top:bottom, left:right]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    interpolation = cv2.INTER_AREA if crop_rgb.shape[0] >= output_size else cv2.INTER_CUBIC
    return cv2.resize(crop_rgb, (output_size, output_size), interpolation=interpolation)


def detect_boxes_on_scaled_frame(
    frame_rgb: np.ndarray,
    detect_max_side: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run MTCNN on a smaller copy of high-res frames, then scale boxes back.

    FaceForensics++ C23 videos are often 1920x1080. Running CPU MTCNN on the
    full frame is much slower than detecting on a 640px-long-side image while
    still cropping from the original frame.
    """
    h, w = frame_rgb.shape[:2]
    max_side = max(h, w)
    with torch.inference_mode():
        if detect_max_side > 0 and max_side > detect_max_side:
            scale = float(detect_max_side) / float(max_side)
            det_w = max(1, int(round(w * scale)))
            det_h = max(1, int(round(h * scale)))
            det_rgb = cv2.resize(frame_rgb, (det_w, det_h), interpolation=cv2.INTER_AREA)
            boxes, probs = _MTCNN.detect(Image.fromarray(det_rgb))
            if boxes is not None:
                boxes = np.asarray(boxes, dtype=np.float32) / scale
            return boxes, probs

        return _MTCNN.detect(Image.fromarray(frame_rgb))


def write_image(path: Path, rgb: np.ndarray, image_format: str, jpeg_quality: int, png_compression: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if image_format == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]
    ok = cv2.imwrite(str(path), bgr, params)
    if not ok:
        raise OSError(f"Could not write image: {path}")


def _success_result_from_cache(sample: Sample, cache_dir: Path, output_root: Path, seconds: float) -> ProcessResult:
    success_path = cache_dir / "_SUCCESS.json"
    payload: Dict[str, Any] = {}
    if success_path.is_file():
        try:
            payload = json.loads(success_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    cache_rel = cache_dir.relative_to(output_root).as_posix()
    return ProcessResult(
        rel_video_path=sample.rel_path,
        label=sample.label,
        category=sample.category,
        split=sample.split,
        cache_dir=cache_rel,
        cached_frames=int(payload.get("cached_frames", len(list_cached_images(cache_dir, _CFG["image_format"])))),
        source_frames=int(payload.get("source_frames", 0)),
        face_failures=int(payload.get("face_failures", 0)),
        center_fallbacks=int(payload.get("center_fallbacks", 0)),
        reused_previous_box=int(payload.get("reused_previous_box", 0)),
        success=1,
        reason="cached",
        seconds=seconds,
    )


def process_one(sample_payload: Dict[str, str]) -> ProcessResult:
    sample = Sample(**sample_payload)
    t0 = time.time()

    ffpp_root = Path(_CFG["ffpp_root"])
    output_root = Path(_CFG["output_root"])
    video_path = ffpp_root / sample.rel_path
    cache_dir = cache_dir_for_sample(output_root, sample.rel_path)
    cache_rel = cache_dir.relative_to(output_root).as_posix()
    success_path = cache_dir / "_SUCCESS.json"
    image_format = _CFG["image_format"]
    frames_per_video = int(_CFG["frames_per_video"])

    if not video_path.is_file():
        return ProcessResult(
            sample.rel_path,
            sample.label,
            sample.category,
            sample.split,
            cache_rel,
            0,
            0,
            0,
            0,
            0,
            0,
            "missing_video",
            time.time() - t0,
        )

    existing = list_cached_images(cache_dir, image_format)
    if not _CFG["overwrite"] and success_path.is_file() and len(existing) >= frames_per_video:
        return _success_result_from_cache(sample, cache_dir, output_root, time.time() - t0)

    if _CFG["overwrite"] and cache_dir.is_dir():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = select_frame_indices(source_frames, frames_per_video)
    if not indices:
        cap.release()
        return ProcessResult(
            sample.rel_path,
            sample.label,
            sample.category,
            sample.split,
            cache_rel,
            0,
            source_frames,
            0,
            0,
            0,
            0,
            "no_source_frames",
            time.time() - t0,
        )

    face_failures = 0
    center_fallbacks = 0
    reused_previous_box = 0
    written = [False] * frames_per_video
    last_box: Optional[np.ndarray] = None
    last_crop: Optional[np.ndarray] = None
    image_paths = expected_image_paths(cache_dir, frames_per_video, image_format)

    try:
        for output_idx, _source_idx, frame_bgr, reuse_previous in iter_selected_frames(cap, indices):
            if reuse_previous:
                if last_crop is not None:
                    write_image(
                        image_paths[output_idx],
                        last_crop,
                        image_format,
                        int(_CFG["jpeg_quality"]),
                        int(_CFG["png_compression"]),
                    )
                    written[output_idx] = True
                continue

            if frame_bgr is None:
                if last_crop is not None:
                    write_image(
                        image_paths[output_idx],
                        last_crop,
                        image_format,
                        int(_CFG["jpeg_quality"]),
                        int(_CFG["png_compression"]),
                    )
                    written[output_idx] = True
                continue

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            boxes, probs = detect_boxes_on_scaled_frame(
                rgb,
                int(_CFG["detect_max_side"]),
            )
            box = choose_face_box(
                boxes,
                probs,
                last_box,
                frame_bgr.shape,
                float(_CFG["min_face_prob"]),
            )

            if box is None:
                face_failures += 1
                if last_box is not None:
                    box = last_box
                    reused_previous_box += 1
                else:
                    box = center_box(frame_bgr.shape)
                    center_fallbacks += 1
            else:
                last_box = box

            crop_rgb = crop_resize_rgb(
                frame_bgr,
                box,
                int(_CFG["image_size"]),
                float(_CFG["margin"]),
            )
            write_image(
                image_paths[output_idx],
                crop_rgb,
                image_format,
                int(_CFG["jpeg_quality"]),
                int(_CFG["png_compression"]),
            )
            last_crop = crop_rgb
            written[output_idx] = True
    finally:
        cap.release()

    saved = sum(1 for item in written if item)
    if saved == 0:
        return ProcessResult(
            sample.rel_path,
            sample.label,
            sample.category,
            sample.split,
            cache_rel,
            0,
            source_frames,
            face_failures,
            center_fallbacks,
            reused_previous_box,
            0,
            "no_decodable_selected_frames",
            time.time() - t0,
        )

    if last_crop is not None:
        for output_idx, is_written in enumerate(written):
            if is_written:
                continue
            write_image(
                image_paths[output_idx],
                last_crop,
                image_format,
                int(_CFG["jpeg_quality"]),
                int(_CFG["png_compression"]),
            )
            written[output_idx] = True
    saved = sum(1 for item in written if item)

    result = ProcessResult(
        sample.rel_path,
        sample.label,
        sample.category,
        sample.split,
        cache_rel,
        saved,
        source_frames,
        face_failures,
        center_fallbacks,
        reused_previous_box,
        1 if saved >= frames_per_video else 0,
        "ok" if saved >= frames_per_video else "partial",
        time.time() - t0,
    )

    if result.success:
        metadata = asdict(result)
        metadata.update(
            {
                "frames_per_video": frames_per_video,
                "image_size": int(_CFG["image_size"]),
                "image_format": image_format,
                "margin": float(_CFG["margin"]),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        success_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return result


def write_manifest(path: Path, rows: Sequence[ProcessResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ProcessResult.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_manifests(output_root: Path, rows: Sequence[ProcessResult]) -> None:
    all_rows = list(rows)
    write_manifest(output_root / "manifest.csv", all_rows)

    split_names = sorted({r.split for r in all_rows})
    for split in split_names:
        if split == "all":
            continue
        split_rows = [r for r in all_rows if r.split == split and r.success]
        write_manifest(output_root / f"{split}_manifest.csv", split_rows)

    success_rows = [r for r in all_rows if r.success]
    write_manifest(output_root / "successful_manifest.csv", success_rows)


def print_summary(rows: Sequence[ProcessResult]) -> None:
    total = len(rows)
    ok = sum(1 for r in rows if r.success)
    failed = total - ok
    face_failures = sum(r.face_failures for r in rows)
    center_fallbacks = sum(r.center_fallbacks for r in rows)
    seconds = sum(r.seconds for r in rows)

    print("\nPreprocessing summary")
    print("---------------------")
    print(f"Videos processed : {total}")
    print(f"Successful       : {ok}")
    print(f"Failed/partial   : {failed}")
    print(f"Face misses      : {face_failures}")
    print(f"Center fallbacks : {center_fallbacks}")
    print(f"Total time       : {seconds / 60.0:.1f} min")

    by_split: Dict[str, int] = {}
    for row in rows:
        if row.success:
            by_split[row.split] = by_split.get(row.split, 0) + 1
    if by_split:
        print("Successful by split:")
        for split in sorted(by_split):
            print(f"  {split:5s}: {by_split[split]}")


def parse_args() -> argparse.Namespace:
    default_root = PROJECT_ROOT / "FaceForensics++_C23"
    default_output = default_root / "preprocessed" / "faces_v1_224"
    default_split_dir = PROJECT_ROOT / "checkpoints" / "FFpp_C23" / "splits"

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ffpp-root", type=Path, default=default_root)
    p.add_argument("--metadata", type=Path, default=None, help="Default: <ffpp-root>/csv/FF++_Metadata.csv")
    p.add_argument("--output-root", type=Path, default=default_output)
    p.add_argument(
        "--split-dir",
        type=str,
        default=str(default_split_dir) if default_split_dir.is_dir() else "",
        help="Directory containing train_split.csv, val_split.csv, test_split.csv. Use '' to ignore.",
    )
    p.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated split names to preprocess.")
    p.add_argument("--categories", type=str, default="", help="Comma-separated FF++ folders to include.")
    p.add_argument("--frames-per-video", type=int, default=64)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--image-format", choices=["png", "jpg"], default="png")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--png-compression", type=int, default=3)
    p.add_argument("--margin", type=float, default=0.30, help="Extra face bbox margin as a fraction of side length.")
    p.add_argument("--min-face-prob", type=float, default=0.90)
    p.add_argument(
        "--detect-max-side",
        type=int,
        default=640,
        help="Resize frames so their longest side is at most this before MTCNN detection. 0 = full resolution.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel processes (each loads MTCNN on CPU). Match physical cores if RAM allows; use 1 to debug.",
    )
    p.add_argument(
        "--imap-chunksize",
        type=int,
        default=1,
        help="Tasks per imap round-trip when workers>1. Keep 1 for per-video progress updates.",
    )
    p.add_argument("--limit", type=int, default=0, help="Debug cap across all selected samples.")
    p.add_argument("--limit-per-split", type=int, default=0, help="Debug cap per split.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="Rebuild existing cache directories.")
    p.add_argument("--dry-run", action="store_true", help="List how many samples would be processed and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ffpp_root = args.ffpp_root.resolve()
    metadata_csv = (args.metadata or (ffpp_root / "csv" / "FF++_Metadata.csv")).resolve()
    output_root = args.output_root.resolve()

    split_dir: Optional[Path]
    if args.split_dir is None or str(args.split_dir).strip() == "":
        split_dir = None
    else:
        split_dir = Path(args.split_dir).resolve()

    if not ffpp_root.is_dir():
        sys.exit(f"FF++ root not found: {ffpp_root}")
    if not metadata_csv.is_file():
        sys.exit(f"Metadata CSV not found: {metadata_csv}")
    if args.frames_per_video <= 0:
        sys.exit("--frames-per-video must be > 0")
    if args.image_size <= 0:
        sys.exit("--image-size must be > 0")
    if args.detect_max_side < 0:
        sys.exit("--detect-max-side must be >= 0")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        sys.exit("--splits must include at least one split name")

    categories = {c.strip() for c in args.categories.split(",") if c.strip()} or None
    samples = collect_samples(metadata_csv, split_dir, splits)
    samples = filter_samples(samples, categories, args.limit, args.limit_per_split, args.seed)

    print("FaceForensics++ face preprocessing")
    print("----------------------------------")
    print(f"FF++ root      : {ffpp_root}")
    print(f"Metadata       : {metadata_csv}")
    print(f"Split dir      : {split_dir if split_dir is not None else 'none'}")
    print(f"Output root    : {output_root}")
    print(f"Samples        : {len(samples)}")
    print(f"Frames/video   : {args.frames_per_video}")
    print(f"Image          : {args.image_size}x{args.image_size} {args.image_format}")
    print(f"Detect max side: {args.detect_max_side if args.detect_max_side > 0 else 'full resolution'}")
    print(f"Workers        : {args.workers}")
    if args.workers > 1:
        print(f"imap chunksize : {max(1, int(args.imap_chunksize))}")

    if args.dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "ffpp_root": str(ffpp_root),
        "output_root": str(output_root),
        "frames_per_video": int(args.frames_per_video),
        "image_size": int(args.image_size),
        "image_format": args.image_format,
        "jpeg_quality": int(args.jpeg_quality),
        "png_compression": int(args.png_compression),
        "margin": float(args.margin),
        "min_face_prob": float(args.min_face_prob),
        "detect_max_side": int(args.detect_max_side),
        "overwrite": bool(args.overwrite),
    }
    (output_root / "preprocess_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    payloads = [asdict(sample) for sample in samples]
    results: List[ProcessResult] = []

    if args.workers <= 1:
        init_worker(config)
        iterator: Iterable[Dict[str, str]] = payloads
        for payload in tqdm(iterator, total=len(payloads), desc="Videos"):
            results.append(process_one(payload))
    else:
        import multiprocessing as mp

        n = len(payloads)
        chunksize = max(1, int(args.imap_chunksize))
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers, initializer=init_worker, initargs=(config,)) as pool:
            for result in tqdm(
                pool.imap_unordered(process_one, payloads, chunksize=chunksize),
                total=n,
                desc="Videos",
            ):
                results.append(result)

    results.sort(key=lambda r: (r.split, r.category, r.rel_video_path))
    write_manifests(output_root, results)
    print_summary(results)
    print(f"\nManifest written to: {output_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
