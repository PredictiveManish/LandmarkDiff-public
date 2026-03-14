"""Augmentation preview and analysis tool.

Visualizes the augmentation pipeline on sample training images,
producing before/after grids for geometric, photometric, and
clinical augmentations. Useful for verifying augmentation quality
before launching training.

Usage:
    # Preview augmentations on training samples
    python scripts/augmentation_preview.py \
        --data_dir data/training_combined \
        --output augmentation_preview

    # Show specific augmentation types
    python scripts/augmentation_preview.py \
        --data_dir data/training_combined \
        --augmentations geometric photometric clinical \
        --num_samples 10

    # Analyze Fitzpatrick distribution
    python scripts/augmentation_preview.py \
        --data_dir data/training_combined \
        --analyze-distribution
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from landmarkdiff.augmentation import (
    AugmentationConfig,
    FitzpatrickBalancer,
    augment_skin_tone,
    augment_training_sample,
)
from landmarkdiff.evaluation import classify_fitzpatrick_ita


def load_training_sample(data_dir: Path, idx: int = 0) -> dict | None:
    """Load a training sample from the dataset directory."""
    inputs = sorted(data_dir.glob("*_input.png"))
    if idx >= len(inputs):
        return None

    input_path = inputs[idx]
    prefix = input_path.stem.replace("_input", "")

    input_img = cv2.imread(str(input_path))
    target_path = data_dir / f"{prefix}_target.png"
    cond_path = data_dir / f"{prefix}_conditioning.png"
    mask_path = data_dir / f"{prefix}_mask.png"

    target = cv2.imread(str(target_path)) if target_path.exists() else input_img.copy()
    conditioning = cv2.imread(str(cond_path)) if cond_path.exists() else np.zeros_like(input_img)
    mask = (
        cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_path.exists()
        else np.ones(input_img.shape[:2], dtype=np.uint8) * 255
    )
    mask = mask.astype(np.float32) / 255.0

    return {
        "input_image": input_img,
        "target_image": target,
        "conditioning": conditioning,
        "mask": mask,
        "prefix": prefix,
    }


def preview_geometric(
    sample: dict,
    n_variants: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Generate grid showing geometric augmentation variants."""
    size = 256
    rng = np.random.default_rng(seed)

    config = AugmentationConfig(
        random_flip=True,
        random_rotation_deg=10.0,
        random_scale=(0.9, 1.1),
        random_translate=0.05,
        brightness_range=(1.0, 1.0),  # disable photometric
        contrast_range=(1.0, 1.0),
        saturation_range=(1.0, 1.0),
        hue_shift_range=0.0,
        conditioning_dropout_prob=0.0,
        conditioning_noise_std=0.0,
    )

    panels = [cv2.resize(sample["input_image"], (size, size))]
    labels = ["Original"]

    for i in range(n_variants):
        augmented = augment_training_sample(
            sample["input_image"],
            sample["target_image"],
            sample["conditioning"],
            sample["mask"],
            config=config,
            rng=np.random.default_rng(rng.integers(2**32)),
        )
        panels.append(cv2.resize(augmented["input_image"], (size, size)))
        labels.append(f"Geo #{i + 1}")

    # Add labels
    labeled = []
    for panel, label in zip(panels, labels, strict=False):
        lbl = np.full((25, size, 3), (30, 30, 30), dtype=np.uint8)
        cv2.putText(
            lbl, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA
        )
        labeled.append(np.vstack([lbl, panel]))

    return np.hstack(labeled)


def preview_photometric(
    sample: dict,
    n_variants: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Generate grid showing photometric augmentation variants."""
    size = 256
    rng = np.random.default_rng(seed)

    config = AugmentationConfig(
        random_flip=False,
        random_rotation_deg=0.0,
        random_scale=(1.0, 1.0),
        random_translate=0.0,
        brightness_range=(0.7, 1.3),
        contrast_range=(0.7, 1.3),
        saturation_range=(0.7, 1.3),
        hue_shift_range=15.0,
        conditioning_dropout_prob=0.0,
        conditioning_noise_std=0.0,
    )

    panels = [cv2.resize(sample["input_image"], (size, size))]
    labels = ["Original"]

    for i in range(n_variants):
        augmented = augment_training_sample(
            sample["input_image"],
            sample["target_image"],
            sample["conditioning"],
            sample["mask"],
            config=config,
            rng=np.random.default_rng(rng.integers(2**32)),
        )
        panels.append(cv2.resize(augmented["input_image"], (size, size)))
        labels.append(f"Photo #{i + 1}")

    labeled = []
    for panel, label in zip(panels, labels, strict=False):
        lbl = np.full((25, size, 3), (30, 30, 30), dtype=np.uint8)
        cv2.putText(
            lbl, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA
        )
        labeled.append(np.vstack([lbl, panel]))

    return np.hstack(labeled)


def preview_conditioning_aug(
    sample: dict,
    n_variants: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Generate grid showing conditioning augmentation (dropout + noise)."""
    size = 256
    rng = np.random.default_rng(seed)

    panels = [cv2.resize(sample["conditioning"], (size, size))]
    labels = ["Original Cond."]

    for i in range(n_variants):
        config = AugmentationConfig(
            random_flip=False,
            random_rotation_deg=0.0,
            random_scale=(1.0, 1.0),
            random_translate=0.0,
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
            saturation_range=(1.0, 1.0),
            hue_shift_range=0.0,
            conditioning_dropout_prob=0.3 if i % 2 == 0 else 0.0,
            conditioning_noise_std=0.05 * (i + 1),
        )
        augmented = augment_training_sample(
            sample["input_image"],
            sample["target_image"],
            sample["conditioning"],
            sample["mask"],
            config=config,
            rng=np.random.default_rng(rng.integers(2**32)),
        )
        panels.append(cv2.resize(augmented["conditioning"], (size, size)))
        if i % 2 == 0:
            labels.append("Drop+Noise")
        else:
            labels.append(f"Noise {0.05 * (i + 1):.2f}")

    labeled = []
    for panel, label in zip(panels, labels, strict=False):
        lbl = np.full((25, size, 3), (30, 30, 30), dtype=np.uint8)
        cv2.putText(
            lbl, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA
        )
        labeled.append(np.vstack([lbl, panel]))

    return np.hstack(labeled)


def preview_skin_tone(
    sample: dict,
    ita_deltas: list[float] | None = None,
) -> np.ndarray:
    """Generate grid showing skin tone augmentation."""
    if ita_deltas is None:
        ita_deltas = [-15, -10, -5, 0, 5, 10, 15]

    size = 256
    panels = []
    labels = []

    for delta in ita_deltas:
        augmented = augment_skin_tone(sample["input_image"], ita_delta=delta)
        panels.append(cv2.resize(augmented, (size, size)))
        fitz = classify_fitzpatrick_ita(augmented)
        labels.append(f"ITA{delta:+.0f} (F{fitz})")

    labeled = []
    for panel, label in zip(panels, labels, strict=False):
        lbl = np.full((25, size, 3), (30, 30, 30), dtype=np.uint8)
        cv2.putText(
            lbl, label, (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA
        )
        labeled.append(np.vstack([lbl, panel]))

    return np.hstack(labeled)


def preview_combined(
    sample: dict,
    n_variants: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """Full augmentation pipeline preview (all transforms combined)."""
    size = 256
    rng = np.random.default_rng(seed)

    config = AugmentationConfig(
        random_flip=True,
        random_rotation_deg=5.0,
        random_scale=(0.95, 1.05),
        random_translate=0.02,
        brightness_range=(0.9, 1.1),
        contrast_range=(0.9, 1.1),
        saturation_range=(0.9, 1.1),
        hue_shift_range=5.0,
        conditioning_dropout_prob=0.1,
        conditioning_noise_std=0.02,
    )

    # Input row
    input_panels = [cv2.resize(sample["input_image"], (size, size))]
    # Conditioning row
    cond_panels = [cv2.resize(sample["conditioning"], (size, size))]

    input_labels = ["Original"]
    for i in range(n_variants):
        augmented = augment_training_sample(
            sample["input_image"],
            sample["target_image"],
            sample["conditioning"],
            sample["mask"],
            config=config,
            rng=np.random.default_rng(rng.integers(2**32)),
        )
        input_panels.append(cv2.resize(augmented["input_image"], (size, size)))
        cond_panels.append(cv2.resize(augmented["conditioning"], (size, size)))
        input_labels.append(f"Aug #{i + 1}")

    def add_labels(panels, labels, row_label):
        labeled = []
        for panel, label in zip(panels, labels, strict=False):
            lbl = np.full((25, size, 3), (30, 30, 30), dtype=np.uint8)
            cv2.putText(
                lbl,
                f"{row_label}: {label}",
                (3, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            labeled.append(np.vstack([lbl, panel]))
        return np.hstack(labeled)

    input_row = add_labels(input_panels, input_labels, "Input")
    cond_row = add_labels(cond_panels, input_labels, "Cond")
    return np.vstack([input_row, cond_row])


def analyze_distribution(data_dir: Path, max_samples: int = 500) -> dict:
    """Analyze Fitzpatrick distribution in a dataset."""
    inputs = sorted(data_dir.glob("*_input.png"))[:max_samples]
    counts: dict[str, int] = {}

    for i, img_path in enumerate(inputs):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        fitz = classify_fitzpatrick_ita(img)
        counts[fitz] = counts.get(fitz, 0) + 1
        if (i + 1) % 100 == 0:
            print(f"  Analyzed {i + 1}/{len(inputs)}...")

    total = sum(counts.values())
    distribution = {k: round(v / total, 3) for k, v in sorted(counts.items())}

    # Compute balancer weights
    balancer = FitzpatrickBalancer()
    for ftype, count in counts.items():
        for _ in range(count):
            balancer.register_sample(ftype)

    fitz_types = []
    for ftype, count in counts.items():
        fitz_types.extend([ftype] * count)
    weights = balancer.get_sampling_weights(fitz_types)
    weight_by_type: dict[str, float] = {}
    idx = 0
    for ftype, count in sorted(counts.items()):
        weight_by_type[ftype] = round(float(weights[idx]), 4)
        idx += count

    return {
        "total_samples": total,
        "counts": dict(sorted(counts.items())),
        "distribution": distribution,
        "balancer_weights": weight_by_type,
    }


def main():
    parser = argparse.ArgumentParser(description="Augmentation preview tool")
    parser.add_argument("--data_dir", required=True, help="Training data directory")
    parser.add_argument("--output", default="augmentation_preview", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to preview")
    parser.add_argument(
        "--augmentations",
        nargs="+",
        default=["geometric", "photometric", "conditioning", "skin_tone", "combined"],
        choices=["geometric", "photometric", "conditioning", "skin_tone", "combined"],
        help="Augmentation types to preview",
    )
    parser.add_argument(
        "--analyze-distribution",
        action="store_true",
        help="Analyze Fitzpatrick distribution in dataset",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # Analyze distribution
    if args.analyze_distribution:
        print("Analyzing Fitzpatrick distribution...")
        dist = analyze_distribution(data_dir)
        print(f"\nFitzpatrick Distribution (n={dist['total_samples']}):")
        for ftype, count in dist["counts"].items():
            pct = dist["distribution"][ftype] * 100
            bar = "#" * int(pct)
            print(f"  Type {ftype}: {count:5d} ({pct:5.1f}%) {bar}")
        print("\nBalancer weights:")
        for ftype, weight in dist["balancer_weights"].items():
            print(f"  Type {ftype}: {weight:.4f}")

        with open(out_dir / "distribution.json", "w") as f:
            json.dump(dist, f, indent=2)
        print(f"\nSaved: {out_dir / 'distribution.json'}")
        return

    # Preview augmentations
    preview_funcs = {
        "geometric": preview_geometric,
        "photometric": preview_photometric,
        "conditioning": preview_conditioning_aug,
        "skin_tone": preview_skin_tone,
        "combined": preview_combined,
    }

    for sample_idx in range(args.num_samples):
        sample = load_training_sample(data_dir, sample_idx)
        if sample is None:
            print(f"Cannot load sample {sample_idx}")
            continue

        print(f"\nSample {sample_idx}: {sample['prefix']}")

        for aug_type in args.augmentations:
            func = preview_funcs[aug_type]
            grid = func(sample, seed=args.seed + sample_idx)
            out_path = out_dir / f"sample{sample_idx}_{aug_type}.png"
            cv2.imwrite(str(out_path), grid)
            print(f"  {aug_type}: {out_path}")

    print(f"\nAll previews saved to {out_dir}")


if __name__ == "__main__":
    main()
