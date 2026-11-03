import argparse
import math
from pathlib import Path

import numpy as np
import torch


def compare_values(a, b, path="root", differences=None, stats=None):
    if differences is None:
        differences = []

    if stats is None:
        stats = {"tensors": 0, "equal_tensors": 0}

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        stats["tensors"] += 1

        if a.shape != b.shape:
            differences.append(f"{path}: tensor shape differs: {a.shape} vs {b.shape}")
        elif a.dtype != b.dtype:
            differences.append(f"{path}: tensor dtype differs: {a.dtype} vs {b.dtype}")
        elif torch.equal(a, b):
            stats["equal_tensors"] += 1
        else:
            unequal = int(torch.count_nonzero(a != b).item())

            if a.is_floating_point() or a.is_complex():
                try:
                    max_diff = (a - b).abs().max().item()
                except RuntimeError:
                    max_diff = float("nan")
            else:
                max_diff = "N/A"

            differences.append(
                f"{path}: tensor differs "
                f"(unequal elements={unequal}, max difference={max_diff})"
            )

        return differences, stats

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape or a.dtype != b.dtype:
            differences.append(f"{path}: NumPy array metadata differs")
        elif not np.array_equal(a, b):
            differences.append(f"{path}: NumPy array values differ")

        return differences, stats

    if isinstance(a, dict) and isinstance(b, dict):
        keys_a = set(a.keys())
        keys_b = set(b.keys())

        if keys_a != keys_b:
            differences.append(
                f"{path}: dictionary keys differ "
                f"(only A={keys_a - keys_b}, only B={keys_b - keys_a})"
            )

        for key in keys_a & keys_b:
            compare_values(
                a[key],
                b[key],
                f"{path}.{key}",
                differences,
                stats,
            )

        return differences, stats

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b):
            differences.append(f"{path}: sequence types differ")
            return differences, stats

        if len(a) != len(b):
            differences.append(
                f"{path}: sequence lengths differ ({len(a)} vs {len(b)})"
            )
            return differences, stats

        for index, (value_a, value_b) in enumerate(zip(a, b)):
            compare_values(
                value_a,
                value_b,
                f"{path}[{index}]",
                differences,
                stats,
            )

        return differences, stats

    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return differences, stats

    if type(a) != type(b) or a != b:
        differences.append(f"{path}: values differ (A={a!r}, B={b!r})")

    return differences, stats


def load_checkpoint(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare two training checkpoints for exact equality."
    )

    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)

    args = parser.parse_args()

    print("Loading checkpoints...")

    checkpoint_a = load_checkpoint(args.checkpoint_a)
    checkpoint_b = load_checkpoint(args.checkpoint_b)

    sections = [
        "model",
        "optimizer",
        "scaler",
        "rng",
        "model_config",
        "parameters",
        "tokenizer_type",
        "tokenizer_vocab_size",
        "tokenizer_dir",
        "dataset_manifest",
    ]

    all_equal = True

    print("\n========== CHECKPOINT COMPARISON ==========")

    for section in sections:
        print(f"\nComparing: {section}")

        if section not in checkpoint_a or section not in checkpoint_b:
            print("FAIL: Section missing from one or both checkpoints.")
            all_equal = False
            continue

        differences, stats = compare_values(
            checkpoint_a[section],
            checkpoint_b[section],
            path=section,
        )

        if not differences:
            print("PASS: Exact match.")
        else:
            print(f"FAIL: {len(differences)} difference(s).")

            for difference in differences[:20]:
                print(f"  - {difference}")

            if len(differences) > 20:
                print(f"  ... and {len(differences) - 20} more.")

            all_equal = False

        if stats["tensors"]:
            print(f"  Tensors equal: " f"{stats['equal_tensors']}/{stats['tensors']}")

    print("\n========== TRAINING CONFIGURATION ==========")

    config_a = dict(checkpoint_a.get("train_config") or {})
    config_b = dict(checkpoint_b.get("train_config") or {})

    # These fields are expected to differ between separate experiments.
    ignored_config_fields = {"run_name", "out_dir"}

    for field in ignored_config_fields:
        config_a.pop(field, None)
        config_b.pop(field, None)

    differences, _ = compare_values(
        config_a,
        config_b,
        path="train_config",
    )

    if not differences:
        print("PASS: Training configurations match.")
    else:
        print("FAIL: Training configuration differences:")
        for difference in differences:
            print(f"  - {difference}")
        all_equal = False

    print("\n========== FINAL CHECK ==========")

    step_a = checkpoint_a.get("step")
    step_b = checkpoint_b.get("step")

    print(f"Checkpoint A step: {step_a}")
    print(f"Checkpoint B step: {step_b}")

    if step_a != step_b:
        print("FAIL: Training steps differ.")
        all_equal = False

    if all_equal:
        print("\nSUCCESS: All compared checkpoint states match exactly.")
    else:
        print("\nRESULT: Checkpoint differences detected.")

    print("============================================")


if __name__ == "__main__":
    main()
