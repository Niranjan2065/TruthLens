"""
TruthLens - Data Preparation: Stratified Train/Val Split
-----------------------------------------------------------
Builds the train_targeted / val_holdout directories used by
screen_capture_finetune.py, from your existing folders:

    D:\\TruthLens\\RealWorld_Training
    D:\\TruthLens\\Mobile_Adapted_Training

IMPORTANT: D:\\TruthLens\\RealWorld_Test is NEVER read or referenced here.
It is your final held-out test set and must stay untouched by anything
in this script or in screen_capture_finetune.py.

What it does:
  1. Pools RealWorld_Training + Mobile_Adapted_Training per class.
  2. Stratified split (default 85% train / 15% val) per class, so val
     has a proportional, class-balanced sample.
  3. Creates data/train_targeted/<class>/ and data/val_holdout/<class>/
     using symlinks by default (fast, no duplicate disk usage) with a
     --copy flag to physically copy files instead (e.g. if symlinks
     aren't supported on your filesystem/permissions).
  4. Prints a per-class count summary so you can sanity-check balance
     before training.

Usage:
    python prepare_data_split.py \
        --sources "D:\\TruthLens\\RealWorld_Training" "D:\\TruthLens\\Mobile_Adapted_Training" \
        --output_root data \
        --val_fraction 0.15 \
        --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

CLASS_NAMES = ["AI_Generated", "Deepfake", "Manipulated", "Real"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Explicit safety guard - never allow this path to be used as a source
FORBIDDEN_SUBSTRING = "realworld_test"


def collect_class_files(source_dirs, class_name):
    files = []
    for src in source_dirs:
        class_dir = Path(src) / class_name
        if not class_dir.exists():
            print(f"  [warn] {class_dir} does not exist, skipping")
            continue
        for f in class_dir.iterdir():
            if f.suffix.lower() in IMG_EXTENSIONS:
                files.append(f)
    return files


def guard_against_test_set(source_dirs):
    for src in source_dirs:
        if FORBIDDEN_SUBSTRING in str(src).lower():
            raise ValueError(
                f"Refusing to run: '{src}' looks like the RealWorld_Test set. "
                f"That set must never be used for training or validation."
            )


def link_or_copy(src_path, dst_path, copy=False):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return
    if copy:
        shutil.copy2(src_path, dst_path)
    else:
        try:
            dst_path.symlink_to(src_path.resolve())
        except OSError:
            # Symlinks may require admin privileges on Windows -> fall back to copy
            shutil.copy2(src_path, dst_path)


def build_split(sources, output_root, val_fraction, seed, copy):
    guard_against_test_set(sources)
    random.seed(seed)

    output_root = Path(output_root)
    train_root = output_root / "train_targeted"
    val_root = output_root / "val_holdout"

    print(f"Building split from: {[str(s) for s in sources]}")
    print(f"Output: {train_root}  and  {val_root}")
    print(f"RealWorld_Test is NOT referenced anywhere in this script.\n")

    summary = {}

    for class_name in CLASS_NAMES:
        files = collect_class_files(sources, class_name)
        random.shuffle(files)

        n_val = max(1, int(len(files) * val_fraction))
        val_files = files[:n_val]
        train_files = files[n_val:]

        for f in train_files:
            link_or_copy(f, train_root / class_name / f.name, copy=copy)
        for f in val_files:
            link_or_copy(f, val_root / class_name / f.name, copy=copy)

        summary[class_name] = (len(train_files), len(val_files))

    print(f"{'Class':<15} {'Train':>8} {'Val':>8}")
    print("-" * 33)
    total_train, total_val = 0, 0
    for class_name, (n_train, n_val) in summary.items():
        print(f"{class_name:<15} {n_train:>8} {n_val:>8}")
        total_train += n_train
        total_val += n_val
    print("-" * 33)
    print(f"{'TOTAL':<15} {total_train:>8} {total_val:>8}")

    if total_train == 0:
        raise RuntimeError("No training files found - check your --sources paths.")

    return train_root, val_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", required=True,
                         help="e.g. D:\\TruthLens\\RealWorld_Training D:\\TruthLens\\Mobile_Adapted_Training")
    parser.add_argument("--output_root", default="data")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true",
                         help="Physically copy files instead of symlinking")
    args = parser.parse_args()

    build_split(args.sources, args.output_root, args.val_fraction, args.seed, args.copy)
