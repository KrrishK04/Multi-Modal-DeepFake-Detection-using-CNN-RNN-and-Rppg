#!/usr/bin/env python3
"""
Evaluate ViTS_TemporalTransformer on FaceForensics++ (C23) videos.

Expected layout under the dataset root (same as the official download scripts):
  DeepFakeDetection/, Deepfakes/, Face2Face/, FaceShifter/, FaceSwap/,
  NeuralTextures/, original/, csv/

Metadata: by default uses csv/FF++_Metadata.csv (one row per video, columns
include "File Path" and "Label"). Videos not yet downloaded are skipped when
--skip-missing is set (default).

Preprocessing (default --face-crop): MTCNN face detection + 112×112 crop, same
idea as rPPG/preprocessing.ipynb; then ToPILImage → Resize(224) → normalize
like training on CroppedFaces clips. Use --no-face-crop for full-frame RGB
→ 224 (train.ipynb video_dataset style).

Use --max-per-category / --videos-per-folder N to evaluate only N videos per
top-level folder (random subsample, see --seed).

Label convention matches training: tensor class 0 = FAKE, 1 = REAL.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from tqdm import tqdm

# Project imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import Config
from models.vit_temporal import ViTS_TemporalTransformer


# ---------------------------------------------------------------------------
# CSV parsing (FaceForensics++ script export: leading index column in rows)
# ---------------------------------------------------------------------------

def _find_columns(header: Sequence[str]) -> Tuple[int, int]:
    h = [c.strip() for c in header]
    try:
        ip = h.index("File Path")
        il = h.index("Label")
        return ip, il
    except ValueError:
        pass
    # Fallback: second and third columns as in ",File Path,Label,..."
    if len(h) >= 3:
        return 1, 2
    raise ValueError(f"Could not find 'File Path' / 'Label' in CSV header: {header!r}")


def load_metadata_rows(metadata_csv: str) -> List[Dict[str, str]]:
    """Return list of dicts: rel_path, label (FAKE|REAL), category (top folder)."""
    rows: List[Dict[str, str]] = []
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        ip, il = _find_columns(header)
        for parts in reader:
            if len(parts) <= max(ip, il):
                continue
            rel = parts[ip].strip()
            lab = parts[il].strip().upper()
            if not rel or lab not in ("FAKE", "REAL"):
                continue
            cat = rel.split("/")[0] if "/" in rel else ""
            rows.append({"rel_path": rel, "label": lab, "category": cat})
    return rows


def subsample_rows(
    rows: List[Dict[str, str]],
    max_per_category: Optional[int],
    seed: int,
) -> List[Dict[str, str]]:
    if max_per_category is None or max_per_category <= 0:
        return rows
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        buckets[r["category"]].append(r)
    out: List[Dict[str, str]] = []
    for cat, items in sorted(buckets.items()):
        rng.shuffle(items)
        out.extend(items[:max_per_category])
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Video → clip tensor (same logic as train.ipynb video_dataset)
# ---------------------------------------------------------------------------

def read_frames_rgb(path: str, max_read_frames: int = 0) -> List[np.ndarray]:
    """Full-frame RGB frames. If max_read_frames > 0, stop after that many reads."""
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(path)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        n += 1
        if max_read_frames and n >= max_read_frames:
            break
    cap.release()
    return frames


def _mtcnn_detect_device(inference_device: torch.device) -> torch.device:
    """facenet_pytorch MTCNN is reliable on CUDA/CPU; fall back to CPU on MPS."""
    if inference_device.type == "cuda":
        return inference_device
    return torch.device("cpu")


def read_frames_face_crop_mtcnn(
    path: str,
    mtcnn: Any,
    max_read_frames: int,
    face_size: Tuple[int, int] = (112, 112),
) -> List[np.ndarray]:
    """
    Match rPPG/preprocessing.ipynb: MTCNN detect → first face crop → resize to 112×112.
    Returns RGB uint8 arrays (same as read_frames_rgb) for downstream Resize(224).
    """
    width, height = face_size
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(path)
    frame_idx = 0
    while True:
        if max_read_frames and frame_idx >= max_read_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            boxes, _ = mtcnn.detect(img)
            if boxes is None or len(boxes) == 0:
                frame_idx += 1
                continue
            left, top, right, bottom = [int(b) for b in boxes[0]]
            top = max(0, top)
            left = max(0, left)
            bottom = min(frame.shape[0], bottom)
            right = min(frame.shape[1], right)
            face_bgr = frame[top:bottom, left:right]
            if face_bgr.size == 0:
                frame_idx += 1
                continue
            face_resized = cv2.resize(face_bgr, (width, height))
            frames.append(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
        except Exception:
            pass
        frame_idx += 1
    cap.release()
    return frames


def load_mtcnn(device: torch.device) -> Any:
    from facenet_pytorch import MTCNN

    return MTCNN(keep_all=True, device=device)


def frames_to_clip(
    frames: List[np.ndarray],
    clip_len: int,
    tfm: transforms.Compose,
) -> torch.Tensor:
    n = len(frames)
    if n == 0:
        raise ValueError("No frames")
    if n >= clip_len:
        indices = np.linspace(0, n - 1, clip_len, dtype=int)
    else:
        indices = list(range(n)) + [n - 1] * (clip_len - n)
    stacked = torch.stack([tfm(frames[i]) for i in indices])
    return stacked


def build_eval_transform(im_size: int, mean: Sequence[float], std: Sequence[float]):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(list(mean), list(std)),
        ]
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> ViTS_TemporalTransformer:
    model = ViTS_TemporalTransformer(
        num_classes=Config.NUM_CLASSES,
        pretrained=False,
        temporal_layers=Config.TEMPORAL_LAYERS,
        temporal_heads=Config.TEMPORAL_HEADS,
        temporal_ff=Config.TEMPORAL_FF,
        temporal_dropout=Config.TEMPORAL_DROPOUT,
        t_max=Config.T_MAX,
        use_cls_token=Config.USE_CLS_TOKEN,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


@torch.inference_mode()
def predict_logits(model: ViTS_TemporalTransformer, clip: torch.Tensor, device: torch.device) -> torch.Tensor:
    # clip: [T, C, H, W] → [1, T, C, H, W]
    x = clip.unsqueeze(0).to(device)
    return model(x)[0]


def logits_to_fake_prob(logits: torch.Tensor) -> float:
    """Class 0 = FAKE, 1 = REAL → probability of fake."""
    p = torch.softmax(logits, dim=-1)
    return float(p[0].item())


def true_class_from_label(label: str) -> int:
    """Training convention: 0 = FAKE, 1 = REAL."""
    return 0 if label.upper() == "FAKE" else 1


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    y_true: List[int]
    y_pred: List[int]
    p_fake: List[float]
    categories: List[str]
    skipped_missing: int
    skipped_empty: int
    skipped_no_face: int
    skipped_error: int
    detail_rows: List[Dict[str, str]]


def run_evaluation(
    ffpp_root: str,
    metadata_csv: str,
    checkpoint_path: str,
    categories_filter: Optional[set],
    max_per_category: Optional[int],
    seed: int,
    skip_missing: bool,
    device: torch.device,
    collect_details: bool,
    use_face_crop: bool,
    max_read_frames: int,
) -> EvalResult:
    rows = load_metadata_rows(metadata_csv)
    if categories_filter is not None:
        rows = [r for r in rows if r["category"] in categories_filter]
    rows = subsample_rows(rows, max_per_category, seed)

    tfm = build_eval_transform(Config.IM_SIZE, Config.MEAN, Config.STD)
    model = load_model(checkpoint_path, device)

    mtcnn = None
    mtcnn_dev: Optional[torch.device] = None
    if use_face_crop:
        mtcnn_dev = _mtcnn_detect_device(device)
        mtcnn = load_mtcnn(mtcnn_dev)

    y_true: List[int] = []
    y_pred: List[int] = []
    p_fake: List[float] = []
    cats_out: List[str] = []
    detail_rows: List[Dict[str, str]] = []
    skipped_missing = skipped_empty = skipped_no_face = skipped_error = 0
    preproc = "mtcnn_112" if use_face_crop else "full_frame"

    for r in tqdm(rows, desc="Videos"):
        full = os.path.join(ffpp_root, r["rel_path"])
        if not os.path.isfile(full):
            if skip_missing:
                skipped_missing += 1
                continue
            raise FileNotFoundError(f"Video missing: {full}")

        try:
            if use_face_crop:
                assert mtcnn is not None
                frames = read_frames_face_crop_mtcnn(
                    full, mtcnn, max_read_frames=max_read_frames, face_size=(112, 112)
                )
                if not frames:
                    skipped_no_face += 1
                    continue
            else:
                frames = read_frames_rgb(full, max_read_frames=max_read_frames or 0)
                if not frames:
                    skipped_empty += 1
                    continue
            clip = frames_to_clip(frames, Config.CLIP_LEN, tfm)
            logits = predict_logits(model, clip, device)
            pred = int(logits.argmax().item())
            tc = true_class_from_label(r["label"])
            pf = logits_to_fake_prob(logits)
            y_true.append(tc)
            y_pred.append(pred)
            p_fake.append(pf)
            cats_out.append(r["category"])
            if collect_details:
                detail_rows.append(
                    {
                        "rel_path": r["rel_path"],
                        "category": r["category"],
                        "label": r["label"],
                        "preprocessing": preproc,
                        "pred_class": str(pred),
                        "p_fake": f"{pf:.6f}",
                        "correct": str(int(pred == tc)),
                    }
                )
        except Exception:
            skipped_error += 1
            continue

    return EvalResult(
        y_true=y_true,
        y_pred=y_pred,
        p_fake=p_fake,
        categories=cats_out,
        skipped_missing=skipped_missing,
        skipped_empty=skipped_empty,
        skipped_no_face=skipped_no_face,
        skipped_error=skipped_error,
        detail_rows=detail_rows,
    )


def print_metrics(res: EvalResult) -> None:
    n = len(res.y_true)
    print("\n--- Skips ---")
    print(f"  Missing files: {res.skipped_missing}")
    print(f"  Empty videos: {res.skipped_empty}")
    print(f"  No face detected (MTCNN): {res.skipped_no_face}")
    print(f"  Other errors: {res.skipped_error}")
    print(f"  Evaluated: {n}")

    if n == 0:
        print("No videos evaluated. Add videos under the dataset root or disable --skip-missing.")
        return

    acc = accuracy_score(res.y_true, res.y_pred)
    # AUC: positive class = FAKE (1), score = P(fake)
    y_bin = [1 if t == 0 else 0 for t in res.y_true]  # FAKE → 1
    try:
        auc = roc_auc_score(y_bin, res.p_fake)
    except ValueError:
        auc = float("nan")

    print("\n--- Overall ---")
    print(f"  Accuracy: {acc * 100:.2f}%")
    print(f"  ROC-AUC (FAKE positive): {auc:.4f}")
    # Names for report: 0=Fake, 1=Real to match train notebook confusion labels
    print("\n" + classification_report(
        res.y_true,
        res.y_pred,
        labels=[0, 1],
        target_names=["Fake (0)", "Real (1)"],
        digits=4,
    ))

    by_cat: Dict[str, List[Tuple[int, int, float]]] = defaultdict(list)
    for t, p, pf, c in zip(res.y_true, res.y_pred, res.p_fake, res.categories):
        by_cat[c].append((t, p, pf))

    print("--- Per category (accuracy) ---")
    for cat in sorted(by_cat.keys()):
        items = by_cat[cat]
        yt = [x[0] for x in items]
        yp = [x[1] for x in items]
        a = accuracy_score(yt, yp) if items else 0.0
        print(f"  {cat:20s}  n={len(items):5d}  acc={a * 100:.2f}%")


def save_detail_csv(path: str, detail_rows: List[Dict[str, str]]) -> None:
    if not detail_rows:
        return
    fieldnames = list(detail_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(detail_rows)
    print(f"\nPer-video results written to {path}")


def main() -> None:
    default_root = os.path.join(PROJECT_ROOT, "FaceForensics++_C23")
    default_meta = os.path.join(default_root, "csv", "FF++_Metadata.csv")
    default_ckpt = os.path.join(Config.CHECKPOINT_DIR, "checkpoint.pt")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ffpp-root", type=str, default=os.environ.get("FFD_FFPP_ROOT", default_root),
                   help="FaceForensics++ root (contains manipulation folders + csv/).")
    p.add_argument("--metadata", type=str, default=default_meta,
                   help="CSV with File Path + Label (default: .../csv/FF++_Metadata.csv).")
    p.add_argument("--checkpoint", type=str, default=default_ckpt,
                   help="checkpoint.pt from training (or set Config.MODEL_NAME).")
    p.add_argument("--categories", type=str, default="",
                   help="Comma-separated top folders to include, e.g. 'original,Deepfakes'. Empty = all.")
    p.add_argument(
        "--max-per-category",
        "--videos-per-folder",
        type=int,
        default=0,
        dest="max_per_category",
        metavar="N",
        help="Test at most N videos per top-level folder (random, --seed). 0 = all in metadata.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for per-folder subsampling.")
    p.set_defaults(face_crop=True)
    p.add_argument(
        "--face-crop",
        dest="face_crop",
        action="store_true",
        help="MTCNN face crop 112×112 like rPPG/preprocessing.ipynb (default).",
    )
    p.add_argument(
        "--no-face-crop",
        dest="face_crop",
        action="store_false",
        help="Full frames only (no MTCNN).",
    )
    p.add_argument(
        "--max-read-frames",
        type=int,
        default=150,
        help="Max source frames to scan per video (0 = entire file). Preprocessing notebook default 150.",
    )
    p.add_argument("--skip-missing", action="store_true", default=True,
                   help="Skip rows whose video file is not on disk (default: on).")
    p.add_argument("--no-skip-missing", action="store_false", dest="skip_missing",
                   help="Fail if a file from metadata is missing.")
    p.add_argument("--list-metadata", action="store_true",
                   help="Print metadata row counts per category and exit (no model load).")
    p.add_argument("--output-csv", type=str, default="",
                   help="Optional path to save per-video predictions.")
    p.add_argument("--device", type=str, default="", help="cuda | cpu | mps | empty=Config.DEVICE")
    args = p.parse_args()

    meta_path = os.path.abspath(args.metadata)
    if not os.path.isfile(meta_path):
        sys.exit(f"Metadata CSV not found: {meta_path}")

    rows = load_metadata_rows(meta_path)
    if args.list_metadata:
        counts: Dict[str, int] = defaultdict(int)
        for r in rows:
            counts[r["category"]] += 1
        print(f"Metadata: {meta_path} ({len(rows)} rows)\n")
        for k in sorted(counts.keys()):
            print(f"  {k:20s}  {counts[k]}")
        return

    cats_filter: Optional[set] = None
    if args.categories.strip():
        cats_filter = {c.strip() for c in args.categories.split(",") if c.strip()}

    max_pc = args.max_per_category if args.max_per_category > 0 else None

    if args.device:
        device = torch.device(args.device)
    else:
        device = Config.DEVICE

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt):
        sys.exit(f"Checkpoint not found: {ckpt}\nTrain or set --checkpoint / Config.MODEL_NAME.")

    Config.print_config()
    print(f"\nFF++ root     : {os.path.abspath(args.ffpp_root)}")
    print(f"Metadata      : {meta_path}")
    print(f"Checkpoint    : {ckpt}")
    print(f"Device        : {device}")
    print(f"Face crop     : {args.face_crop} (MTCNN on {_mtcnn_detect_device(device)})")
    mr = args.max_read_frames
    print(f"Max read frames / video: {mr if mr > 0 else 'unlimited'}")

    out_csv = args.output_csv.strip()
    res = run_evaluation(
        ffpp_root=os.path.abspath(args.ffpp_root),
        metadata_csv=meta_path,
        checkpoint_path=ckpt,
        categories_filter=cats_filter,
        max_per_category=max_pc,
        seed=args.seed,
        skip_missing=args.skip_missing,
        device=device,
        collect_details=bool(out_csv),
        use_face_crop=args.face_crop,
        max_read_frames=mr if mr > 0 else 0,
    )
    print_metrics(res)
    if out_csv:
        save_detail_csv(os.path.abspath(out_csv), res.detail_rows)


if __name__ == "__main__":
    main()
