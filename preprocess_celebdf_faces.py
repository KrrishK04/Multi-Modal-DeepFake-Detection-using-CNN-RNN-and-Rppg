#!/usr/bin/env python3
"""
Preprocess Celeb-DF videos into the same reusable face-frame cache format
used by preprocess_ffpp_faces.py.

Expected Celeb-DF layout:
  ../celebdf/
    Celeb-real/*.mp4        -> REAL
    YouTube-real/*.mp4      -> REAL
    Celeb-synthesis/*.mp4   -> FAKE

Default output:
  ../celebdf/preprocessed/faces_v1_224/
    splits/train_split.csv
    splits/val_split.csv
    splits/test_split.csv
    manifest.csv
    train_manifest.csv
    val_manifest.csv
    test_manifest.csv
    frames/<category>/<video_stem>/000.jpg
    ...
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from preprocess_ffpp_faces import (
    PROJECT_ROOT,
    ProcessResult,
    Sample,
    init_worker,
    process_one,
    print_summary,
    write_manifests,
)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
REAL_FOLDERS = {"Celeb-real", "YouTube-real"}
FAKE_FOLDERS = {"Celeb-synthesis"}


def label_for_category(category: str) -> str:
    if category in REAL_FOLDERS:
        return "REAL"
    if category in FAKE_FOLDERS:
        return "FAKE"
    raise ValueError(f"Unsupported Celeb-DF category: {category}")


def collect_celebdf_samples(celebdf_root: Path) -> List[Sample]:
    samples: List[Sample] = []
    categories = sorted(REAL_FOLDERS | FAKE_FOLDERS)
    for category in categories:
        folder = celebdf_root / category
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected Celeb-DF folder not found: {folder}")
        for video_path in sorted(folder.iterdir()):
            if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            rel_path = video_path.relative_to(celebdf_root).as_posix()
            samples.append(
                Sample(
                    rel_path=rel_path,
                    label=label_for_category(category),
                    category=category,
                    split="all",
                )
            )
    if not samples:
        raise RuntimeError(f"No Celeb-DF videos found under {celebdf_root}")
    return samples


def stratified_train_val_test_split(
    samples: Sequence[Sample],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1):
        raise ValueError("val_fraction and test_fraction must be in (0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1")

    labels = [sample.label for sample in samples]
    train_val, test = train_test_split(
        list(samples),
        test_size=test_fraction,
        random_state=seed,
        stratify=labels,
    )
    train_val_labels = [sample.label for sample in train_val]
    val_ratio = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_val_labels,
    )

    def with_split(rows: Sequence[Sample], split: str) -> List[Sample]:
        return [
            Sample(
                rel_path=row.rel_path,
                label=row.label,
                category=row.category,
                split=split,
            )
            for row in rows
        ]

    return with_split(train, "train"), with_split(val, "val"), with_split(test, "test")


def write_split_csv(path: Path, samples: Sequence[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["File Path", "Label"])
        writer.writeheader()
        for sample in samples:
            writer.writerow({"File Path": sample.rel_path, "Label": sample.label})


def read_split_csv(path: Path, split: str) -> List[Sample]:
    rows: List[Sample] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = row["File Path"].strip().replace("\\", "/")
            label = row["Label"].strip().upper()
            category = rel_path.split("/", 1)[0]
            if label not in {"FAKE", "REAL"}:
                raise ValueError(f"Invalid label {label!r} in {path}")
            rows.append(Sample(rel_path=rel_path, label=label, category=category, split=split))
    return rows


def split_paths(split_dir: Path) -> Dict[str, Path]:
    return {
        "train": split_dir / "train_split.csv",
        "val": split_dir / "val_split.csv",
        "test": split_dir / "test_split.csv",
    }


def prepare_or_load_splits(
    celebdf_root: Path,
    split_dir: Path,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    overwrite_splits: bool,
    dry_run: bool,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    paths = split_paths(split_dir)
    split_files_exist = all(path.is_file() for path in paths.values())

    if split_files_exist and not overwrite_splits:
        return (
            read_split_csv(paths["train"], "train"),
            read_split_csv(paths["val"], "val"),
            read_split_csv(paths["test"], "test"),
        )

    all_samples = collect_celebdf_samples(celebdf_root)
    train, val, test = stratified_train_val_test_split(all_samples, val_fraction, test_fraction, seed)

    if not dry_run:
        write_split_csv(paths["train"], train)
        write_split_csv(paths["val"], val)
        write_split_csv(paths["test"], test)
        meta = {
            "seed": seed,
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "source_root": str(celebdf_root),
            "split_strategy": "stratified_by_label",
        }
        (split_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return train, val, test


def maybe_limit_samples(samples: Sequence[Sample], limit: int, limit_per_split: int, seed: int) -> List[Sample]:
    rows = list(samples)
    rng = random.Random(seed)

    if limit_per_split > 0:
        out: List[Sample] = []
        by_split: Dict[str, List[Sample]] = {}
        for sample in rows:
            by_split.setdefault(sample.split, []).append(sample)
        for split in sorted(by_split):
            group = by_split[split]
            rng.shuffle(group)
            out.extend(group[:limit_per_split])
        rows = out

    if limit > 0 and len(rows) > limit:
        rng.shuffle(rows)
        rows = rows[:limit]
    return rows


def counts_by_label(samples: Sequence[Sample]) -> str:
    c = Counter(sample.label for sample in samples)
    return f"FAKE={c['FAKE']} REAL={c['REAL']}"


def parse_args() -> argparse.Namespace:
    default_root = (PROJECT_ROOT / ".." / "celebdf").resolve()
    default_output = default_root / "preprocessed" / "faces_v1_224"

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--celebdf-root", type=Path, default=default_root)
    p.add_argument("--output-root", type=Path, default=default_output)
    p.add_argument("--split-dir", type=Path, default=None, help="Default: <output-root>/splits")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--frames-per-video", type=int, default=64)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--image-format", choices=["png", "jpg"], default="jpg")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--png-compression", type=int, default=3)
    p.add_argument("--margin", type=float, default=0.30)
    p.add_argument("--min-face-prob", type=float, default=0.90)
    p.add_argument("--detect-max-side", type=int, default=640)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--imap-chunksize", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="Debug cap after full split creation/loading.")
    p.add_argument("--limit-per-split", type=int, default=0, help="Debug cap per split after full split creation/loading.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="Rebuild existing cached frame directories.")
    p.add_argument("--overwrite-splits", action="store_true", help="Recreate split CSVs even if they already exist.")
    p.add_argument("--dry-run", action="store_true", help="Show discovered/split counts and exit without writing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    celebdf_root = args.celebdf_root.resolve()
    output_root = args.output_root.resolve()
    split_dir = args.split_dir.resolve() if args.split_dir is not None else output_root / "splits"

    if not celebdf_root.is_dir():
        sys.exit(f"Celeb-DF root not found: {celebdf_root}")
    if args.frames_per_video <= 0:
        sys.exit("--frames-per-video must be > 0")
    if args.image_size <= 0:
        sys.exit("--image-size must be > 0")
    if args.detect_max_side < 0:
        sys.exit("--detect-max-side must be >= 0")

    all_samples = collect_celebdf_samples(celebdf_root)
    train, val, test = prepare_or_load_splits(
        celebdf_root=celebdf_root,
        split_dir=split_dir,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        overwrite_splits=args.overwrite_splits,
        dry_run=args.dry_run,
    )
    selected = maybe_limit_samples(train + val + test, args.limit, args.limit_per_split, args.seed)

    print("Celeb-DF face preprocessing")
    print("---------------------------")
    print(f"Celeb-DF root  : {celebdf_root}")
    print(f"Output root    : {output_root}")
    print(f"Split dir      : {split_dir}")
    print(f"Discovered     : {len(all_samples)} ({counts_by_label(all_samples)})")
    print(f"Train          : {len(train)} ({counts_by_label(train)})")
    print(f"Val            : {len(val)} ({counts_by_label(val)})")
    print(f"Test           : {len(test)} ({counts_by_label(test)})")
    print(f"Selected       : {len(selected)}")
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
        # process_one is shared with FF++ preprocessing and expects this key.
        "ffpp_root": str(celebdf_root),
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

    payloads = [asdict(sample) for sample in selected]
    results: List[ProcessResult] = []

    if args.workers <= 1:
        init_worker(config)
        for payload in tqdm(payloads, total=len(payloads), desc="Videos"):
            results.append(process_one(payload))
    else:
        import multiprocessing as mp

        chunksize = max(1, int(args.imap_chunksize))
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers, initializer=init_worker, initargs=(config,)) as pool:
            iterator: Iterable[ProcessResult] = pool.imap_unordered(process_one, payloads, chunksize=chunksize)
            for result in tqdm(iterator, total=len(payloads), desc="Videos"):
                results.append(result)

    results.sort(key=lambda r: (r.split, r.category, r.rel_video_path))
    write_manifests(output_root, results)
    print_summary(results)
    print(f"\nManifest written to: {output_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
