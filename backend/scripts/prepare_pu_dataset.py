"""Prepare reproducible spatially buffered positive-unlabeled training data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from backend.app.services.ml.pu_learning import prepare_pu_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIRECTORY = PROJECT_ROOT / "backend" / "data" / "training"
PROVENANCE_PATH = PROJECT_ROOT / "backend" / "data" / "processed" / "gsi_49367" / "provenance.json"


def main() -> int:
    try:
        provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
        result = prepare_pu_dataset(
            TRAINING_DIRECTORY / "training_samples.csv",
            PROVENANCE_PATH.parent / provenance["feature_path"],
            TRAINING_DIRECTORY / "background_samples.csv",
            TRAINING_DIRECTORY / "pu_training_dataset.csv",
            TRAINING_DIRECTORY / "pu_dataset_metadata.json",
        )
    except Exception as error:
        print(f"PU DATASET PREPARATION FAILED: {error}", file=sys.stderr)
        return 1
    print("PU DATASET PREPARATION COMPLETE")
    print(f"Authoritative positives: {result.positive_count}")
    print(f"Unlabeled/background samples: {result.background_count}")
    print(f"Valid candidates after 300 m exclusion: {result.candidate_background_count}")
    print(f"Background: {result.background_path}")
    print(f"PU dataset: {result.pu_dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
