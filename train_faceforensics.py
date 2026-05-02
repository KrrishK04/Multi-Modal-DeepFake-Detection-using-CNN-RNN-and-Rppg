#!/usr/bin/env python3
"""
Train ViTS_TemporalTransformer on FaceForensics++ using the same recipe as train.ipynb:

  • Metadata CSV (File Path, Label) under the FF++ root
  • OpenCV RGB frames, uniform temporal sampling, resize 224, ImageNet normalize
  • Labels: 0 = FAKE, 1 = REAL (same as notebook / test_faceforensics.py)
  • AdamW with separate LRs for ViT backbone vs temporal+head; cosine LR after warmup
  • Checkpoints under checkpoints/<MODEL_NAME>/ (override with --model-name)
  • Stratified train / validation / test split (--val-fraction, --test-fraction);
    after training, reports metrics on the held-out test set using best_model.pt.

Example:
  python train_faceforensics.py --ffpp-root ./FaceForensics++_C23 --epochs 15
  python train_faceforensics.py --preprocessed-root ./FaceForensics++_C23/preprocessed/faces_v1_224 --epochs 15
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
import time
from datetime import datetime
from math import cos, pi
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import Config
from models.vit_temporal import ViTS_TemporalTransformer
from utils.device import empty_cache

# Reuse CSV + frame helpers from the eval script (same preprocessing contract).
import test_faceforensics as ffpp  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FaceForensicsClipDataset(Dataset):
    """One clip per video; paths must exist and be readable."""

    def __init__(
        self,
        video_paths: Sequence[str],
        labels: Sequence[int],
        clip_len: int,
        transform: transforms.Compose,
    ):
        if len(video_paths) != len(labels):
            raise ValueError("video_paths and labels length mismatch")
        self.paths = list(video_paths)
        self.labels = list(labels)
        self.clip_len = clip_len
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.paths[idx]
        frames = ffpp.read_frames_rgb(path)
        if not frames:
            raise ValueError(f"No frames in video: {path}")
        clip = ffpp.frames_to_clip(frames, self.clip_len, self.transform)
        return clip, self.labels[idx]


class FaceForensicsCachedClipDataset(Dataset):
    """One clip per cached face-frame directory."""

    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

    def __init__(
        self,
        cache_dirs: Sequence[str],
        labels: Sequence[int],
        clip_len: int,
        transform: transforms.Compose,
        random_temporal_sampling: bool = False,
        seed: int = 42,
    ):
        if len(cache_dirs) != len(labels):
            raise ValueError("cache_dirs and labels length mismatch")
        self.cache_dirs = [os.path.normpath(p) for p in cache_dirs]
        self.labels = list(labels)
        self.clip_len = clip_len
        self.transform = transform
        self.random_temporal_sampling = random_temporal_sampling
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.cache_dirs)

    def _list_images(self, cache_dir: str) -> List[str]:
        base = Path(cache_dir)
        paths: List[Path] = []
        for ext in self.IMAGE_EXTENSIONS:
            paths.extend(base.glob(f"*{ext}"))
        return [str(p) for p in sorted(paths)]

    def _sample_indices(self, n: int) -> List[int]:
        if n <= 0:
            return []
        if n >= self.clip_len:
            if self.random_temporal_sampling:
                edges = np.linspace(0, n, self.clip_len + 1, dtype=int)
                indices: List[int] = []
                for start, end in zip(edges[:-1], edges[1:]):
                    end = max(start + 1, end)
                    indices.append(self.rng.randrange(start, min(end, n)))
                return indices
            return np.linspace(0, n - 1, self.clip_len, dtype=int).tolist()
        return list(range(n)) + [n - 1] * (self.clip_len - n)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        cache_dir = self.cache_dirs[idx]
        images = self._list_images(cache_dir)
        if not images:
            raise ValueError(f"No cached frames in directory: {cache_dir}")

        indices = self._sample_indices(len(images))
        frames = []
        for frame_idx in indices:
            bgr = cv2.imread(images[frame_idx], cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Could not read cached frame: {images[frame_idx]}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(rgb))
        return torch.stack(frames), self.labels[idx]


def load_cached_split_manifest(
    preprocessed_root: str,
    split: str,
    categories_filter: Optional[set],
    max_items: Optional[int],
    seed: int,
) -> Tuple[List[str], List[int]]:
    manifest_path = os.path.join(preprocessed_root, f"{split}_manifest.csv")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Cached split manifest not found: {manifest_path}\n"
            "Run preprocess_ffpp_faces.py first, or omit --preprocessed-root."
        )

    rows: List[Tuple[str, int, str]] = []
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("success", "1")).strip() not in {"1", "true", "True"}:
                continue
            category = row["category"].strip()
            if categories_filter is not None and category not in categories_filter:
                continue
            label = row["label"].strip().upper()
            if label not in {"FAKE", "REAL"}:
                raise ValueError(f"Invalid label {label!r} in {manifest_path}")
            cache_dir = row["cache_dir"].strip()
            if not os.path.isabs(cache_dir):
                cache_dir = os.path.join(preprocessed_root, cache_dir)
            if not os.path.isdir(cache_dir):
                raise FileNotFoundError(f"Cached frame directory missing: {cache_dir}")
            rows.append((os.path.normpath(cache_dir), 0 if label == "FAKE" else 1, category))

    if max_items is not None and max_items > 0 and len(rows) > max_items:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_items]

    if not rows:
        raise RuntimeError(f"No usable cached samples found for split '{split}' in {preprocessed_root}")

    return [r[0] for r in rows], [r[1] for r in rows]


# ---------------------------------------------------------------------------
# Training utilities (aligned with train.ipynb)
# ---------------------------------------------------------------------------


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    return 100 * correct.float().sum().item() / batch_size


def train_epoch(
    epoch: int,
    num_epochs: int,
    data_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    pbar = tqdm(
        data_loader,
        desc=f"  Train {epoch}/{num_epochs}",
        bar_format="{l_bar}{bar:30}{r_bar}",
        leave=True,
    )
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(calculate_accuracy(outputs, targets), inputs.size(0))
        pbar.set_postfix_str(f"loss={losses.avg:.4f}  acc={accuracies.avg:.1f}%")
    return losses.avg, accuracies.avg


def validate(
    epoch: int,
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pbar_desc: Optional[str] = None,
) -> Tuple[list, list, np.ndarray, float, float]:
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    true_labels: List[int] = []
    predictions: List[int] = []
    all_probs: List[np.ndarray] = []
    desc = pbar_desc if pbar_desc is not None else f"  Val   {epoch}     "
    pbar = tqdm(
        data_loader,
        desc=desc,
        bar_format="{l_bar}{bar:30}{r_bar}",
        leave=True,
    )
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, dtype=torch.long, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            probs = torch.softmax(outputs, dim=1)
            true_labels.extend(targets.cpu().numpy().tolist())
            predictions.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(calculate_accuracy(outputs, targets), inputs.size(0))
            pbar.set_postfix_str(f"loss={losses.avg:.4f}  acc={accuracies.avg:.1f}%")
    return true_labels, predictions, np.array(all_probs), losses.avg, accuracies.avg


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    train_loss_avg: List[float],
    train_accuracy: List[float],
    test_loss_avg: List[float],
    test_accuracy: List[float],
    val_true: Optional[list] = None,
    val_preds: Optional[list] = None,
    val_probs: Optional[np.ndarray] = None,
    best_val_acc: float = 0.0,
    filename: str = "checkpoint.pt",
) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss_avg": train_loss_avg,
            "train_accuracy": train_accuracy,
            "test_loss_avg": test_loss_avg,
            "test_accuracy": test_accuracy,
            "val_true_labels": val_true,
            "val_predictions": val_preds,
            "val_probabilities": val_probs,
            "best_val_acc": best_val_acc,
        },
        path,
    )
    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
) -> Tuple:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return (
        ckpt["epoch"],
        ckpt["train_loss_avg"],
        ckpt["train_accuracy"],
        ckpt["test_loss_avg"],
        ckpt["test_accuracy"],
        ckpt.get("val_true_labels"),
        ckpt.get("val_predictions"),
        ckpt.get("val_probabilities"),
        ckpt.get("best_val_acc", 0.0),
    )


def cooldown(seconds: int) -> None:
    if seconds <= 0:
        return
    print(f"\n  Cooldown: waiting {seconds // 60}m {seconds % 60}s ...", end="", flush=True)
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        print(f"\r  Cooldown: {mins:02d}:{secs:02d} remaining ...  ", end="", flush=True)
        time.sleep(1)
    print("\r  Cooldown: done!                          ")


def collect_samples(
    ffpp_root: str,
    metadata_csv: str,
    categories_filter: Optional[set],
    skip_missing: bool,
    max_total: Optional[int],
    seed: int,
) -> Tuple[List[str], List[int]]:
    rows = ffpp.load_metadata_rows(metadata_csv)
    if categories_filter is not None:
        rows = [r for r in rows if r["category"] in categories_filter]

    paths: List[str] = []
    labels: List[int] = []
    skipped_missing = 0
    for r in rows:
        full = os.path.join(ffpp_root, r["rel_path"])
        if not os.path.isfile(full):
            if skip_missing:
                skipped_missing += 1
                continue
            raise FileNotFoundError(f"Video missing: {full}")
        paths.append(os.path.normpath(full))
        labels.append(0 if r["label"].upper() == "FAKE" else 1)

    if skipped_missing:
        print(f"  [Data] Skipped {skipped_missing} missing file(s) from metadata.")

    if len(set(labels)) < 2:
        raise SystemExit("Need both FAKE and REAL samples after filtering. Check metadata and --categories.")

    if max_total is not None and max_total > 0 and len(paths) > max_total:
        rng = random.Random(seed)
        idx = list(range(len(paths)))
        rng.shuffle(idx)
        idx = idx[:max_total]
        paths = [paths[i] for i in idx]
        labels = [labels[i] for i in idx]
        print(f"  [Data] Subsampled to {max_total} videos (seed={seed}).")

    return paths, labels


def apply_config_model_name(model_name: str) -> str:
    """Point checkpoints at checkpoints/<model_name>/."""
    Config.MODEL_NAME = model_name
    Config.CHECKPOINT_DIR = os.path.join(Config.PROJECT_ROOT, "checkpoints", model_name)
    return Config.CHECKPOINT_DIR


def stratified_train_val_test_split(
    paths: List[str],
    labels: List[int],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
    """
    Split so that (approximately) val_fraction and test_fraction of all samples
    go to validation and test; the rest is training. Stratified by label.
    """
    if not (0 < test_fraction < 1 and 0 < val_fraction < 1):
        raise ValueError("val_fraction and test_fraction must be in (0, 1).")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1 (need room for training).")

    train_val_p, test_p, train_val_y, test_y = train_test_split(
        paths,
        labels,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels,
    )
    # Val should be val_fraction of the original pool → fraction of (train+val) pool:
    val_ratio_of_remainder = val_fraction / (1.0 - test_fraction)
    train_p, val_p, train_y, val_y = train_test_split(
        train_val_p,
        train_val_y,
        test_size=val_ratio_of_remainder,
        random_state=seed,
        stratify=train_val_y,
    )
    return train_p, val_p, test_p, train_y, val_y, test_y


def save_split_manifest(
    checkpoint_dir: str,
    ffpp_root: str,
    train_p: Sequence[str],
    val_p: Sequence[str],
    test_p: Sequence[str],
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> str:
    """Write train/val/test paths for reproducibility."""

    def rel_or_abs(p: str) -> str:
        p = os.path.normpath(p)
        root = os.path.normpath(ffpp_root)
        if p.startswith(root + os.sep):
            return os.path.relpath(p, root)
        return p

    manifest = {
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "train_size": len(train_p),
        "val_size": len(val_p),
        "test_size": len(test_p),
        "train_paths": [rel_or_abs(x) for x in train_p],
        "val_paths": [rel_or_abs(x) for x in val_p],
        "test_paths": [rel_or_abs(x) for x in test_p],
    }
    path = os.path.join(checkpoint_dir, "split_manifest.json")
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def main() -> None:
    default_root = os.path.join(PROJECT_ROOT, "FaceForensics++_C23")
    default_meta = os.path.join(default_root, "csv", "FF++_Metadata.csv")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ffpp-root", type=str, default=os.environ.get("FFD_FFPP_ROOT", default_root))
    p.add_argument("--metadata", type=str, default=default_meta)
    p.add_argument(
        "--preprocessed-root",
        type=str,
        default=os.environ.get("FFD_FFPP_PREPROCESSED_ROOT", ""),
        help="Use cached face-frame manifests from preprocess_ffpp_faces.py instead of reading mp4s.",
    )
    p.add_argument("--categories", type=str, default="", help="Comma-separated folders, e.g. original,Deepfakes. Empty=all.")
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of all videos for validation (stratified). Default 0.15.",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Fraction of all videos for held-out test (stratified). Default 0.15.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-videos", type=int, default=0, help="Cap dataset size for debugging (0 = all).")
    p.add_argument("--skip-missing", action="store_true", default=True)
    p.add_argument("--no-skip-missing", action="store_false", dest="skip_missing")
    p.add_argument(
        "--model-name",
        type=str,
        default=os.environ.get("FFD_FFPP_MODEL_NAME", "FFpp_C23"),
        help="Subfolder under checkpoints/ for saves (default: FFpp_C23).",
    )
    p.add_argument("--epochs", type=int, default=None, help="Override Config.NUM_EPOCHS when set.")
    p.add_argument("--batch-size", type=int, default=0, help="0 = use Config.BATCH_SIZE.")
    p.add_argument("--no-pretrained", action="store_true", help="Train ViT from scratch (not recommended).")
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoints/<model-name>/checkpoint.pt (default: load it if present, like train.ipynb).",
    )
    p.add_argument("--cooldown-seconds", type=int, default=300, help="Pause between epochs (0 = off).")
    p.add_argument("--no-plots", action="store_true", help="Skip matplotlib figures at the end.")
    p.add_argument(
        "--skip-final-test",
        action="store_true",
        help="Do not evaluate on the held-out test set after training.",
    )
    p.add_argument("--device", type=str, default="", help="cuda | cpu | mps | empty = Config.DEVICE")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_dir = apply_config_model_name(args.model_name)
    device = torch.device(args.device) if args.device else Config.DEVICE

    meta_path = os.path.abspath(args.metadata)
    if not os.path.isfile(meta_path):
        sys.exit(f"Metadata CSV not found: {meta_path}")

    ffpp_root = os.path.abspath(args.ffpp_root)
    cats_filter: Optional[set] = None
    if args.categories.strip():
        cats_filter = {c.strip() for c in args.categories.split(",") if c.strip()}

    tfm = ffpp.build_eval_transform(Config.IM_SIZE, Config.MEAN, Config.STD)
    preprocessed_root = os.path.abspath(args.preprocessed_root) if args.preprocessed_root.strip() else ""

    if preprocessed_root:
        if not os.path.isdir(preprocessed_root):
            sys.exit(f"Preprocessed root not found: {preprocessed_root}")
        max_items = args.max_videos if args.max_videos > 0 else None
        train_p, train_y = load_cached_split_manifest(preprocessed_root, "train", cats_filter, max_items, args.seed)
        val_p, val_y = load_cached_split_manifest(preprocessed_root, "val", cats_filter, max_items, args.seed)
        test_p, test_y = load_cached_split_manifest(preprocessed_root, "test", cats_filter, max_items, args.seed)

        train_ds = FaceForensicsCachedClipDataset(
            train_p,
            train_y,
            Config.CLIP_LEN,
            tfm,
            random_temporal_sampling=True,
            seed=args.seed,
        )
        val_ds = FaceForensicsCachedClipDataset(val_p, val_y, Config.CLIP_LEN, tfm)
        test_ds = FaceForensicsCachedClipDataset(test_p, test_y, Config.CLIP_LEN, tfm)
        split_manifest_path = os.path.join(preprocessed_root, "manifest.csv")
    else:
        max_v = args.max_videos if args.max_videos > 0 else None
        paths, labels = collect_samples(
            ffpp_root, meta_path, cats_filter, args.skip_missing, max_v, args.seed,
        )

        train_p, val_p, test_p, train_y, val_y, test_y = stratified_train_val_test_split(
            paths,
            labels,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )

        train_ds = FaceForensicsClipDataset(train_p, train_y, Config.CLIP_LEN, tfm)
        val_ds = FaceForensicsClipDataset(val_p, val_y, Config.CLIP_LEN, tfm)
        test_ds = FaceForensicsClipDataset(test_p, test_y, Config.CLIP_LEN, tfm)

        split_manifest_path = save_split_manifest(
            checkpoint_dir, ffpp_root, train_p, val_p, test_p, args.seed, args.val_fraction, args.test_fraction
        )

    batch_size = args.batch_size if args.batch_size > 0 else Config.BATCH_SIZE
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )
    valid_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    num_epochs = args.epochs if args.epochs is not None else Config.NUM_EPOCHS

    Config.print_config()
    print(f"\n  FF++ root        : {ffpp_root}")
    print(f"  Metadata         : {meta_path}")
    if preprocessed_root:
        print(f"  Preprocessed    : {preprocessed_root}")
    print(f"  Checkpoint dir   : {checkpoint_dir}")
    def fake_real_counts(ys: Sequence[int]) -> Tuple[int, int]:
        c = Counter(ys)
        return c[0], c[1]

    tfk, trk = fake_real_counts(train_y)
    vfk, vrk = fake_real_counts(val_y)
    xfk, xrk = fake_real_counts(test_y)
    print(f"  Train / Val / Test : {len(train_ds)} / {len(val_ds)} / {len(test_ds)} clips")
    print(f"    (FAKE/REAL) train {tfk}/{trk}  val {vfk}/{vrk}  test {xfk}/{xrk}")
    print(f"  Split manifest     : {split_manifest_path}")
    print(f"  Batch size       : {batch_size}")
    print(f"  Epochs           : {num_epochs}")

    model = ViTS_TemporalTransformer(
        num_classes=Config.NUM_CLASSES,
        pretrained=not args.no_pretrained,
        temporal_layers=Config.TEMPORAL_LAYERS,
        temporal_heads=Config.TEMPORAL_HEADS,
        temporal_ff=Config.TEMPORAL_FF,
        temporal_dropout=Config.TEMPORAL_DROPOUT,
        t_max=Config.T_MAX,
        use_cls_token=Config.USE_CLS_TOKEN,
    ).to(device)

    vit_params = model.frame_encoder.parameters()
    temp_head_params = list(model.temporal.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": vit_params, "lr": 3e-5},
            {"params": temp_head_params, "lr": 1e-4},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    total_steps = max(1, num_epochs * len(train_loader))
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + cos(pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss().to(device)

    train_loss_avg: List[float] = []
    train_accuracy: List[float] = []
    test_loss_avg: List[float] = []
    test_accuracy: List[float] = []
    start_epoch = 1
    best_val_acc = 0.0

    resume_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

    if not args.no_resume and os.path.isfile(resume_path):
        result = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        start_epoch = int(result[0]) + 1
        train_loss_avg = list(result[1])
        train_accuracy = list(result[2])
        test_loss_avg = list(result[3])
        test_accuracy = list(result[4])
        best_val_acc = float(result[8]) if len(result) > 8 else 0.0
        if os.path.isfile(best_model_path):
            bck = torch.load(best_model_path, map_location=device, weights_only=False)
            best_val_acc = max(best_val_acc, float(bck.get("best_val_acc", 0.0)))
        print(f"Resumed from epoch {start_epoch - 1}; best val acc tracked: {best_val_acc:.2f}%")
    elif not args.no_resume:
        print(f"No checkpoint at {resume_path}; training from scratch.")

    print("=" * 60)
    print(f"  Training: epochs {start_epoch}..{num_epochs}  |  device={device}")
    print("=" * 60)

    true: List[int] = []
    pred: List[int] = []
    probs = np.zeros((0, 2))

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        lr_vit = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]
        print(f"\n{'─' * 60}")
        print(
            f"  Epoch {epoch}/{num_epochs}  |  lr_vit={lr_vit:.2e}  lr_head={lr_head:.2e}  |  "
            f"{datetime.now().strftime('%H:%M:%S')}"
        )
        print(f"{'─' * 60}")

        tl, ta = train_epoch(epoch, num_epochs, train_loader, model, criterion, optimizer, scheduler, device)
        train_loss_avg.append(tl)
        train_accuracy.append(ta)

        true, pred, probs, vl, va = validate(epoch, model, valid_loader, criterion, device)
        test_loss_avg.append(vl)
        test_accuracy.append(va)

        elapsed = time.time() - t0
        print(f"\n  Train Loss : {tl:.4f}   Acc : {ta:6.2f}%")
        print(f"  Val   Loss : {vl:.4f}   Acc : {va:6.2f}%")
        print(f"  Time       : {elapsed / 60:.1f} min")

        improved = va >= best_val_acc
        if improved:
            best_val_acc = va
            best_path = save_checkpoint(
                checkpoint_dir,
                epoch,
                model,
                optimizer,
                scheduler,
                train_loss_avg,
                train_accuracy,
                test_loss_avg,
                test_accuracy,
                val_true=true,
                val_preds=pred,
                val_probs=probs,
                best_val_acc=best_val_acc,
                filename="best_model.pt",
            )
            print(f"  Best model saved → {best_path} (Val Acc = {va:.2f}%)")
        else:
            print(f"  Val acc {va:.2f}% did not improve best ({best_val_acc:.2f}%)")

        ckpt_path = save_checkpoint(
            checkpoint_dir,
            epoch,
            model,
            optimizer,
            scheduler,
            train_loss_avg,
            train_accuracy,
            test_loss_avg,
            test_accuracy,
            val_true=true,
            val_preds=pred,
            val_probs=probs,
            best_val_acc=best_val_acc,
            filename="checkpoint.pt",
        )
        print(f"  Checkpoint saved → {ckpt_path}")

        empty_cache(device)
        if epoch < num_epochs:
            cooldown(args.cooldown_seconds)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best Val Acc: {best_val_acc:.2f}%")
    print("=" * 60)

    print("\nClassification report (validation — last epoch):")
    print(
        classification_report(
            true,
            pred,
            labels=[0, 1],
            target_names=["Fake (0)", "Real (1)"],
            digits=4,
        )
    )

    if not args.skip_final_test:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        if os.path.isfile(best_path):
            b = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(b["model_state_dict"])
            print("\n--- Held-out test set (weights from best_model.pt) ---")
        else:
            print("\n--- Held-out test set (current epoch weights; best_model.pt not found) ---")
        true_te, pred_te, probs_te, loss_te, acc_te = validate(
            0,
            model,
            test_loader,
            criterion,
            device,
            pbar_desc="  Test        ",
        )
        print(f"  Test loss: {loss_te:.4f}   Test acc: {acc_te:.2f}%")
        p_fake = probs_te[:, 0]
        y_bin = [1 if t == 0 else 0 for t in true_te]
        try:
            auc = roc_auc_score(y_bin, p_fake)
            print(f"  ROC-AUC (FAKE positive, P(fake)): {auc:.4f}")
        except ValueError:
            print("  ROC-AUC: n/a (single class in test set)")
        print("\nClassification report (held-out test):")
        print(
            classification_report(
                true_te,
                pred_te,
                labels=[0, 1],
                target_names=["Fake (0)", "Real (1)"],
                digits=4,
            )
        )

    if not args.no_plots:
        import matplotlib.pyplot as plt

        def _style_axes(ax):
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#333333")
                spine.set_linewidth(1.5)
            ax.tick_params(axis="both", colors="#333333", labelsize=12, length=5, width=1.2)
            ax.grid(True, alpha=0.3, linestyle="--", color="gray")
            ax.set_facecolor("white")

        ne = len(train_loss_avg)
        ep = range(1, ne + 1)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ep, train_loss_avg, color="#2ca02c", label="Training loss", marker="o", markersize=4)
        ax.plot(ep, test_loss_avg, color="#1f77b4", label="Validation loss", marker="s", markersize=4)
        ax.set_title("Training and Validation Loss (FaceForensics++)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        _style_axes(ax)
        ax.legend()
        os.makedirs(checkpoint_dir, exist_ok=True)
        fig.savefig(os.path.join(checkpoint_dir, "loss_plot_ffpp.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ep, train_accuracy, color="#2ca02c", label="Training acc", marker="o", markersize=4)
        ax.plot(ep, test_accuracy, color="#1f77b4", label="Validation acc", marker="s", markersize=4)
        ax.set_title("Training and Validation Accuracy (FaceForensics++)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        _style_axes(ax)
        ax.legend()
        fig.savefig(os.path.join(checkpoint_dir, "accuracy_plot_ffpp.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved loss/accuracy plots under {checkpoint_dir}")


if __name__ == "__main__":
    main()
