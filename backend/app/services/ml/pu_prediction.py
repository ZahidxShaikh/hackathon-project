"""Raster prediction for the validated positive-unlabeled Random Forest model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import rasterio
from rasterio.windows import Window

from backend.app.services.ml.pu_learning import FEATURE_COLUMNS, TARGET_CRS
from backend.app.services.raster import OUTPUT_NODATA


RASTER_BAND_BY_FEATURE = {
    "B02": "B02", "B03": "B03", "B04": "B04", "B08": "B08", "B11": "B11", "B12": "B12",
    "NDVI": "NDVI", "NDMI": "NDMI", "iron_oxide_proxy": "Iron/Oxide Proxy", "swir_ratio": "SWIR Ratio",
}
CLASS_NAMES = ("Very Low", "Low", "Moderate", "High", "Very High")
PREDICTION_WINDOW_ROWS = 100


class PuPredictionError(RuntimeError):
    """Raised when a PU model or feature raster is incompatible for prediction."""


@dataclass(frozen=True)
class PuPredictionResult:
    output_path: Path
    summary_path: Path
    valid_pixels: int
    nodata_pixels: int
    minimum: float
    maximum: float
    mean: float
    median: float
    quantile_summary: list[dict[str, Any]]


def _feature_band_indexes(source: rasterio.DatasetReader) -> list[int]:
    descriptions = {description: index for index, description in enumerate(source.descriptions, start=1) if description}
    missing = [name for name in RASTER_BAND_BY_FEATURE.values() if name not in descriptions]
    if missing:
        raise PuPredictionError(f"Feature raster is missing required PU bands: {missing}")
    return [descriptions[RASTER_BAND_BY_FEATURE[feature]] for feature in FEATURE_COLUMNS]


def _load_pu_model(model_path: str | Path):
    path = Path(model_path)
    if not path.is_file():
        raise PuPredictionError(f"PU model artifact does not exist: {path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("model_kind") != "positive_vs_unlabeled_random_forest":
        raise PuPredictionError("Model artifact is not the validated positive-unlabeled Random Forest model.")
    if artifact.get("feature_columns") != list(FEATURE_COLUMNS):
        raise PuPredictionError("PU model feature contract does not match the validated ten-band raster contract.")
    model = artifact.get("model")
    if not hasattr(model, "predict_proba") or 1 not in model.classes_:
        raise PuPredictionError("PU model cannot produce its positive-vs-unlabeled ranking score.")
    return model


def _quantile_summary(scores: np.ndarray) -> list[dict[str, Any]]:
    boundaries = np.quantile(scores, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    class_index = np.searchsorted(boundaries[1:-1], scores, side="right")
    return [
        {
            "class": name,
            "score_lower": float(boundaries[index]),
            "score_upper": float(boundaries[index + 1]),
            "pixel_count": int((class_index == index).sum()),
        }
        for index, name in enumerate(CLASS_NAMES)
    ]


def predict_gsi_pu_prospectivity(
    feature_raster_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
) -> PuPredictionResult:
    """Create a one-band relative PU prospectivity-ranking raster.

    The score is the model's positive-vs-unlabeled probability-like ranking
    output. It is deliberately not represented as a calibrated probability of
    manganese occurrence.
    """
    source_path = Path(feature_raster_path)
    destination = Path(output_path)
    summary_destination = Path(summary_path)
    if not source_path.is_file():
        raise PuPredictionError(f"Feature raster does not exist: {source_path}")
    model = _load_pu_model(model_path)
    all_scores: list[np.ndarray] = []
    valid_pixel_count = 0
    nodata_pixel_count = 0
    try:
        with rasterio.open(source_path) as source:
            if source.crs is None or source.crs.to_string() != TARGET_CRS:
                raise PuPredictionError(f"Feature raster must use {TARGET_CRS}, found {source.crs}.")
            if source.nodata != OUTPUT_NODATA:
                raise PuPredictionError(f"Feature raster must use nodata {OUTPUT_NODATA}.")
            indexes = _feature_band_indexes(source)
            profile = source.profile.copy()
            profile.update(driver="GTiff", count=1, dtype="float32", nodata=OUTPUT_NODATA, compress="deflate", predictor=3)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(destination, "w", **profile) as output:
                # The input is a compressed strip GeoTIFF with very short
                # native blocks.  Predicting one strip at a time spends most
                # runtime in sklearn call overhead, so use bounded 100-row
                # windows while still avoiding a full-raster feature matrix.
                for row_start in range(0, source.height, PREDICTION_WINDOW_ROWS):
                    window = Window(
                        col_off=0,
                        row_off=row_start,
                        width=source.width,
                        height=min(PREDICTION_WINDOW_ROWS, source.height - row_start),
                    )
                    features = source.read(indexes, window=window).astype(np.float32, copy=False)
                    valid = np.all(np.isfinite(features) & (features != OUTPUT_NODATA), axis=0)
                    score_block = np.full((int(window.height), int(window.width)), OUTPUT_NODATA, dtype=np.float32)
                    if valid.any():
                        matrix = features[:, valid].T
                        scores = model.predict_proba(matrix)[:, list(model.classes_).index(1)].astype(np.float32)
                        if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
                            raise PuPredictionError("PU model returned an invalid ranking score.")
                        score_block[valid] = scores
                        all_scores.append(scores)
                        valid_pixel_count += int(scores.size)
                    nodata_pixel_count += int((~valid).sum())
                    output.write(score_block, 1, window=window)
                output.set_band_description(1, "PU Manganese Prospectivity Ranking Score")
                output.update_tags(
                    score_kind="Relative positive-vs-unlabeled Random Forest ranking score; not a calibrated probability of manganese occurrence.",
                    scientific_interpretation="Manganese prospectivity based on Sentinel-2 spectral proxies and authoritative GSI manganese occurrences using a positive-unlabeled Random Forest ranking model.",
                    disclaimer="Not a confirmed manganese reserve, ore grade, tonnage, economic-viability result, or geological confirmation.",
                    classification_method="Valid-pixel score quintiles; data-driven display ranking only, not concentration thresholds.",
                )
    except rasterio.errors.RasterioError as error:
        raise PuPredictionError(f"Could not read or write prospectivity GeoTIFF: {error}") from error
    if not all_scores:
        raise PuPredictionError("Feature raster contains no valid pixels to score.")
    scores = np.concatenate(all_scores)
    quantiles = _quantile_summary(scores)
    summary = {
        "output_type": "Manganese prospectivity based on Sentinel-2 spectral proxies and authoritative GSI manganese occurrences using a positive-unlabeled Random Forest ranking model.",
        "score_interpretation": "Relative positive-vs-unlabeled ranking score. It is not a calibrated probability of manganese occurrence.",
        "feature_raster": source_path.name,
        "model": Path(model_path).name,
        "crs": TARGET_CRS,
        "valid_pixel_count": valid_pixel_count,
        "nodata_pixel_count": nodata_pixel_count,
        "score_statistics": {
            "minimum": float(scores.min()), "maximum": float(scores.max()),
            "mean": float(scores.mean()), "median": float(np.median(scores)),
        },
        "classification": {
            "method": "valid-pixel score quintiles (20th, 40th, 60th, 80th percentiles)",
            "limitation": "Display-ranking classes are data-driven and are not geological concentration thresholds.",
            "classes": quantiles,
        },
        "limitations": [
            "Unlabeled background can include unknown manganese occurrences.",
            "The ranking is not confirmation of a manganese deposit, reserve, ore grade, tonnage, economic viability, or geological interpretation.",
            "Spatial PU metrics are surrogate positive-vs-unlabeled measures and do not calibrate occurrence probability.",
        ],
    }
    summary_destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return PuPredictionResult(
        destination, summary_destination, valid_pixel_count, nodata_pixel_count,
        float(scores.min()), float(scores.max()), float(scores.mean()), float(np.median(scores)), quantiles,
    )
