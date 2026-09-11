"""Train only when an authoritative labeled Phase 5B table is supplied."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.app.services.ml.dataset import TrainingDataError, load_training_dataset
from backend.app.services.ml.model import train_random_forest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-description", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    try:
        dataset = load_training_dataset(args.input)
        result = train_random_forest(
            dataset, source_dataset_description=args.source_description,
            random_seed=args.seed, threshold=args.threshold,
        )
    except (TrainingDataError, ValueError) as error:
        print(f"TRAINING STOPPED: {error}", file=sys.stderr)
        return 1
    print(f"Model: {result.model_path.resolve()}")
    print(f"Metadata: {result.metadata_path.resolve()}")
    print(f"Metrics: {result.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

