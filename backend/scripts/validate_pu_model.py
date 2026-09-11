"""Validate PU data integrity, spatial exclusion, and the saved PU model artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from backend.app.services.ml.pu_learning import FEATURE_COLUMNS, POSITIVE_EXCLUSION_BUFFER_METERS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIRECTORY = PROJECT_ROOT / "backend" / "data" / "training"
MODELS_DIRECTORY = PROJECT_ROOT / "backend" / "models"


def main() -> int:
    try:
        positives = pd.read_csv(TRAINING_DIRECTORY / "training_samples.csv")
        background = pd.read_csv(TRAINING_DIRECTORY / "background_samples.csv")
        pu = pd.read_csv(TRAINING_DIRECTORY / "pu_training_dataset.csv")
        dataset_metadata = json.loads((TRAINING_DIRECTORY / "pu_dataset_metadata.json").read_text(encoding="utf-8"))
        model_metadata = json.loads((MODELS_DIRECTORY / "model_metadata.json").read_text(encoding="utf-8"))
        artifact = joblib.load(MODELS_DIRECTORY / "manganese_pu_model.joblib")
        if dataset_metadata.get("dataset_type") != "positive_unlabeled":
            raise ValueError("PU dataset metadata type is invalid.")
        if dataset_metadata["positive_samples"]["count"] != 170 or len(positives) != 170:
            raise ValueError("Expected 170 authoritative GSI positive samples.")
        if not positives["label"].eq(1).all():
            raise ValueError("Positive source data must retain only authoritative label=1 samples.")
        if len(background) != dataset_metadata["unlabeled_background"]["count"]:
            raise ValueError("Background sample count differs from metadata.")
        if not background["sample_class"].eq("unlabeled_background").all():
            raise ValueError("Background samples are not explicitly marked unlabeled.")
        if not background["source"].str.contains("not confirmed non-manganese", case=False, regex=False).all():
            raise ValueError("Background source metadata must explicitly state that it is not confirmed non-manganese.")
        feature_values = pu.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(feature_values).all():
            raise ValueError("PU dataset has NaN/Inf feature values.")
        positive_pu = pu.loc[pu["sample_class"] == "authoritative_manganese_positive"]
        unlabeled_pu = pu.loc[pu["sample_class"] == "unlabeled_background"]
        if len(positive_pu) != 170 or len(unlabeled_pu) != len(background):
            raise ValueError("PU class counts are inconsistent.")
        if not positive_pu["pu_training_target"].eq(1).all() or not unlabeled_pu["pu_training_target"].eq(0).all():
            raise ValueError("PU surrogate targets are inconsistent.")
        if unlabeled_pu["authoritative_label"].notna().any():
            raise ValueError("Unlabeled pixels must not carry an authoritative label.")
        tree = KDTree(positive_pu[["x_utm", "y_utm"]].to_numpy(dtype=float))
        minimum_distance = float(tree.query(unlabeled_pu[["x_utm", "y_utm"]].to_numpy(dtype=float), k=1)[0].min())
        buffer_meters = float(dataset_metadata["unlabeled_background"]["exclusion_buffer_meters"])
        if minimum_distance < buffer_meters:
            raise ValueError("At least one background sample is inside the positive exclusion buffer.")
        if buffer_meters != POSITIVE_EXCLUSION_BUFFER_METERS:
            raise ValueError("Unexpected positive exclusion buffer.")
        if artifact.get("feature_columns") != list(FEATURE_COLUMNS) or artifact.get("model_kind") != "positive_vs_unlabeled_random_forest":
            raise ValueError("Saved model artifact has an unexpected PU schema.")
        scores = artifact["model"].predict_proba(pu.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float))[:, 1]
        if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
            raise ValueError("PU model produced invalid scores.")
        metrics = model_metadata.get("metrics", {})
        if metrics.get("validation_kind") != "spatial GroupKFold positive-vs-unlabeled surrogate validation":
            raise ValueError("Model metadata does not document spatial PU validation.")
    except Exception as error:
        print(f"PU MODEL VALIDATION FAILED: {error}", file=sys.stderr)
        return 1

    print("PU MODEL VALIDATION PASSED")
    print(f"Authoritative positives: {len(positive_pu)}")
    print(f"Unlabeled/background: {len(unlabeled_pu)}")
    print(f"Positive exclusion buffer: {buffer_meters:.0f} m")
    print(f"Nearest background-to-positive distance: {minimum_distance:.2f} m")
    print(f"Feature count: {len(FEATURE_COLUMNS)}")
    print(f"Model type: {model_metadata['model_type']}")
    print(f"Validation strategy: {metrics['validation_kind']}")
    print(f"ROC-AUC (positive vs unlabeled): {metrics['roc_auc_positive_vs_unlabeled']:.4f}")
    print(f"Average precision (positive vs unlabeled): {metrics['average_precision_positive_vs_unlabeled']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
