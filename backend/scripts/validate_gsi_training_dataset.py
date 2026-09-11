"""Validate the authoritative positive-only GSI Phase 5 training dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.services.ml.gsi_training_dataset import FEATURE_COLUMNS, OUTPUT_COLUMNS, WGS84


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIRECTORY = PROJECT_ROOT / "backend" / "data" / "training"


def main() -> int:
    try:
        csv_path = TRAINING_DIRECTORY / "training_samples.csv"
        metadata_path = TRAINING_DIRECTORY / "dataset_metadata.json"
        dataset = pd.read_csv(csv_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        missing = [column for column in OUTPUT_COLUMNS if column not in dataset.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")
        if metadata.get("dataset_type") != "authoritative_positive_only":
            raise ValueError("Dataset is not declared positive-only authoritative GSI data.")
        if metadata.get("crs", {}).get("point_coordinates") != WGS84 or metadata.get("crs", {}).get("sampling_raster") != "EPSG:32644":
            raise ValueError("Dataset CRS metadata is inconsistent with GSI WGS84 points and the sampled raster.")
        counts = metadata["counts"]
        if counts["total_ground_truth_points"] != 172 or counts["valid_ground_truth_points"] != 170:
            raise ValueError("Unexpected GSI ground-truth counts in metadata.")
        if len(dataset) != counts["sampled_training_points"]:
            raise ValueError("CSV row count does not match metadata.")
        if dataset["point_id"].isna().any() or dataset["point_id"].duplicated().any():
            raise ValueError("Point IDs must be present and unique.")
        if not dataset["longitude"].between(-180, 180).all() or not dataset["latitude"].between(-90, 90).all():
            raise ValueError("WGS84 longitude/latitude values are invalid.")
        values = dataset.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        nan_inf_count = int((~np.isfinite(values)).sum())
        if nan_inf_count:
            raise ValueError(f"Feature columns contain {nan_inf_count} NaN/Inf values.")
        if not dataset["label"].eq(1).all() or counts["label_counts"] != {"1": len(dataset), "0": 0}:
            raise ValueError("Only authoritative positive label=1 samples may be present.")
        if not dataset["sample_class"].eq("authoritative_manganese_positive").all():
            raise ValueError("Sample class must identify authoritative manganese positives.")
        if dataset["source"].isna().any() or not dataset["source"].str.contains("Geological Survey of India", regex=False).all():
            raise ValueError("Authoritative GSI source metadata is missing from one or more rows.")
        mno_available = int(dataset["MnO"].notna().sum())
        if mno_available != counts["mno_available"]:
            raise ValueError("MnO availability count does not match metadata.")
        ranges = {
            column: {"min": float(dataset[column].min()), "max": float(dataset[column].max())}
            for column in FEATURE_COLUMNS
        }
    except Exception as error:
        print(f"GSI TRAINING-DATA VALIDATION FAILED: {error}", file=sys.stderr)
        return 1

    print("GSI TRAINING-DATA VALIDATION PASSED")
    print(f"Total GSI points: {counts['total_ground_truth_points']}")
    print(f"Valid GSI points: {counts['valid_ground_truth_points']}")
    print(f"Successfully sampled training points: {len(dataset)}")
    print(f"Dropped points: {counts['dropped_points']}")
    print(f"Label counts: {counts['label_counts']}")
    print(f"MnO available/missing: {mno_available}/{len(dataset) - mno_available}")
    print(f"NaN/Inf in feature columns: {nan_inf_count}")
    print(f"Coincident-coordinate sample records retained: {counts['coincident_coordinate_records_retained']}")
    print("Feature ranges:")
    for column, value_range in ranges.items():
        print(f"  {column}: {value_range['min']:.6g} to {value_range['max']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
