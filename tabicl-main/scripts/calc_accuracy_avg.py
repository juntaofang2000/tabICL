#!/usr/bin/env python3
"""Compute the average accuracy reported in a benchmark log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean

ACCURACY_RE = re.compile(r"accuracy\s*=\s*([0-9]*\.?[0-9]+)")
DATASET_ACCURACY_RE = re.compile(
    r"^(?P<name>[^:]+):\s*accuracy\s*=\s*(?P<value>[0-9]*\.?[0-9]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average accuracies in a log file")
    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to the log file that contains 'accuracy=' entries",
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        help="Optional file that lists dataset names to include (format: '<name>: value')",
    )
    parser.add_argument(
        "--expected",
        type=int,
        default=None,
        help="Expected number of datasets; defaults to all detected entries",
    )
    return parser.parse_args()


def collect_accuracies(log_path: Path) -> tuple[list[float], dict[str, float]]:
    accuracies: list[float] = []
    per_dataset: dict[str, float] = {}
    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            ds_match = DATASET_ACCURACY_RE.match(line)
            if ds_match:
                value = float(ds_match.group("value"))
                name = ds_match.group("name").strip()
                accuracies.append(value)
                per_dataset[name] = value
                continue

            match = ACCURACY_RE.search(line)
            if match:
                accuracies.append(float(match.group(1)))

    return accuracies, per_dataset


def read_dataset_names(dataset_file: Path) -> list[str]:
    names: list[str] = []
    with dataset_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if ":" in stripped:
                stripped = stripped.split(":", 1)[0]
            if stripped:
                names.append(stripped)
    return names


def main() -> None:
    args = parse_args()
    accuracies, per_dataset = collect_accuracies(args.log_file)

    if not accuracies:
        raise SystemExit("No accuracy entries found in the provided log file.")

    if args.datasets is not None:
        dataset_names = read_dataset_names(args.datasets)
        if not dataset_names:
            raise SystemExit(
                f"Dataset list {args.datasets} did not contain any valid dataset names."
            )

        selected: list[float] = []
        missing: list[str] = []
        for name in dataset_names:
            if name in per_dataset:
                selected.append(per_dataset[name])
            else:
                missing.append(name)

        if not selected:
            missing_str = ", ".join(dataset_names)
            raise SystemExit(
                "None of the datasets from the list were found in the log file. "
                f"Missing: {missing_str}"
            )

        avg_acc = mean(selected)
        expected = args.expected or len(dataset_names)
        if expected and expected != len(selected):
            print(
                f"Warning: expected {expected} datasets, "
                f"but matched {len(selected)} datasets present in the log."
            )
        if missing:
            print(
                "Datasets not found in log: " + ", ".join(sorted(set(missing)))
            )

        print(f"Datasets requested: {len(dataset_names)}")
        print(f"Datasets matched: {len(selected)}")
        print(f"Average accuracy: {avg_acc:.4f}")
        return

    avg_acc = mean(accuracies)
    expected = args.expected or len(accuracies)
    if expected and expected != len(accuracies):
        print(
            f"Warning: expected {expected} entries, "
            f"but found {len(accuracies)} entries in {args.log_file}"
        )

    print(f"Datasets counted: {len(accuracies)}")
    print(f"Average accuracy: {avg_acc:.4f}")


if __name__ == "__main__":
    main()
