"""Leakage-safe sklearn preprocessing factory."""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from backend.app.services.ml.dataset import TRAINING_FEATURES


def build_preprocessor() -> Pipeline:
    """Impute inside the fitted pipeline so holdout data cannot leak statistics."""
    return Pipeline([("imputer", SimpleImputer(strategy="median"))])


def select_features(frame):
    return frame[TRAINING_FEATURES].copy()

