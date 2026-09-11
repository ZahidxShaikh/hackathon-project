"""Random Forest baseline and safe artifact persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from backend.app.services.ml.dataset import TRAINING_FEATURES, TrainingDataError, spatial_holdout_split
from backend.app.services.ml.evaluation import evaluate_predictions
from backend.app.services.ml.preprocessing import build_preprocessor, select_features

MODEL_PATH = Path("backend/models/manganese_random_forest.joblib")
METADATA_PATH = Path("backend/models/model_metadata.json")


@dataclass(frozen=True)
class TrainingResult:
    model_path: Path
    metadata_path: Path
    metrics: dict


def train_random_forest(
    frame, *, source_dataset_description: str, random_seed: int = 42, threshold: float = 0.5,
    model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH,
) -> TrainingResult:
    """Train only on sufficient validated real labels and save a pipeline artifact."""
    if not source_dataset_description.strip():
        raise TrainingDataError("A source dataset description is required for a model artifact.")
    counts = frame["label"].value_counts()
    if len(counts) != 2 or counts.min() < 2 or len(frame) < 10:
        raise TrainingDataError(
            "Insufficient real labeled data: need at least 10 records and at least 2 records per class."
        )
    train_frame, validation_frame = spatial_holdout_split(frame, test_size=0.25, random_seed=random_seed)
    pipeline = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=300, random_state=random_seed, class_weight="balanced", n_jobs=-1
        )),
    ])
    pipeline.fit(select_features(train_frame), train_frame["label"])
    probabilities = pipeline.predict_proba(select_features(validation_frame))[:, 1]
    metrics = evaluate_predictions(validation_frame["label"], probabilities, threshold)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metadata = {
        "model_type": "RandomForestClassifier",
        "feature_list": TRAINING_FEATURES,
        "training_sample_count": int(len(frame)),
        "positive_count": int((frame.label == 1).sum()),
        "negative_count": int((frame.label == 0).sum()),
        "metrics": metrics,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "classification_threshold": threshold,
        "source_dataset_description": source_dataset_description,
        "scientific_disclaimer": "Manganese prospectivity / spectral-geological proxy only; not a reserve estimate.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return TrainingResult(model_path, metadata_path, metrics)

