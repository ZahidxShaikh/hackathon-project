"""Train the initial spatially validated positive-unlabeled prospectivity ranker."""

from __future__ import annotations

import sys
from pathlib import Path

from backend.app.services.ml.pu_learning import train_pu_random_forest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    try:
        result = train_pu_random_forest(
            PROJECT_ROOT / "backend" / "data" / "training" / "pu_training_dataset.csv",
            PROJECT_ROOT / "backend" / "models" / "manganese_pu_model.joblib",
            PROJECT_ROOT / "backend" / "models" / "model_metadata.json",
        )
    except Exception as error:
        print(f"PU MODEL TRAINING FAILED: {error}", file=sys.stderr)
        return 1
    print("PU MODEL TRAINING COMPLETE")
    print(f"Model: {result.model_path}")
    print(f"Validation strategy: {result.metrics['validation_kind']}")
    print(f"ROC-AUC (positive vs unlabeled): {result.metrics['roc_auc_positive_vs_unlabeled']:.4f}")
    print(f"Average precision (positive vs unlabeled): {result.metrics['average_precision_positive_vs_unlabeled']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
