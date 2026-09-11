"""Create a positive-only authoritative GSI Phase 5 training-sample table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from backend.app.services.ml.gsi_training_dataset import prepare_gsi_positive_training_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH_DIRECTORY = PROJECT_ROOT / "backend" / "data" / "ground_truth"
TRAINING_DIRECTORY = PROJECT_ROOT / "backend" / "data" / "training"
PROVENANCE_PATH = PROJECT_ROOT / "backend" / "data" / "processed" / "gsi_49367" / "provenance.json"


def main() -> int:
    try:
        provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
        feature_raster = PROVENANCE_PATH.parent / provenance["feature_path"]
        result = prepare_gsi_positive_training_dataset(
            GROUND_TRUTH_DIRECTORY / "manganese_positive.geojson",
            GROUND_TRUTH_DIRECTORY / "metadata.json",
            feature_raster,
            TRAINING_DIRECTORY / "training_samples.csv",
            TRAINING_DIRECTORY / "dataset_metadata.json",
        )
    except Exception as error:
        print(f"GSI TRAINING-DATA PREPARATION FAILED: {error}", file=sys.stderr)
        return 1
    print("GSI TRAINING-DATA PREPARATION COMPLETE")
    print(f"Total GSI records: {result.total_ground_truth_points}")
    print(f"Valid GSI records sampled: {result.sampled_points}/{result.valid_ground_truth_points}")
    print(f"Dropped due to invalid/nodata features: {result.dropped_points}")
    print(f"MnO available/missing: {result.mno_available}/{result.mno_missing}")
    print("Label distribution: authoritative positive (1) only; no background/negative labels created")
    print(f"Dataset: {result.output_path}")
    print(f"Metadata: {result.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
