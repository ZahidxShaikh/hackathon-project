"""Validate and canonicalize an authoritative Phase 5B training dataset.

This does not generate labels or sample pixels. It only validates an already
prepared authoritative feature table and writes a normalized CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.app.services.ml.dataset import TrainingDataError, load_training_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Authoritative CSV/GeoJSON/GPKG/SHP input.")
    parser.add_argument("--output", type=Path, default=Path("backend/data/training/validated_training_dataset.csv"))
    args = parser.parse_args()
    try:
        dataset = load_training_dataset(args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(args.output, index=False)
    except (TrainingDataError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Validated authoritative training rows: {len(dataset)}")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

