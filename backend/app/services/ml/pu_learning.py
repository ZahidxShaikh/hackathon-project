"""Spatially aware positive-unlabeled (PU) data preparation and modeling.

The classifier distinguishes authoritative GSI manganese samples from an
unlabeled raster-background reference set.  A background target of zero is a
PU training surrogate only, never a claim that a location is non-manganese.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KDTree

from backend.app.services.ml.gsi_training_dataset import FEATURE_COLUMNS, WGS84
from backend.app.services.raster import OUTPUT_NODATA


TARGET_CRS = "EPSG:32644"
RANDOM_SEED = 42
POSITIVE_EXCLUSION_BUFFER_METERS = 300.0
BACKGROUND_TO_POSITIVE_RATIO = 10
SPATIAL_BLOCK_SIZE_METERS = 1000.0
BACKGROUND_COLUMNS = (
    "point_id", "longitude", "latitude", "x_utm", "y_utm", "pixel_row", "pixel_col",
    *FEATURE_COLUMNS, "sample_class", "source",
)
PU_COLUMNS = (
    "point_id", "longitude", "latitude", "x_utm", "y_utm", *FEATURE_COLUMNS, "MnO",
    "sample_class", "authoritative_label", "pu_training_target", "source",
)


class PuLearningError(RuntimeError):
    """Raised when PU preparation, spatial validation, or model fitting is unsafe."""


@dataclass(frozen=True)
class PuDatasetResult:
    background_path: Path
    pu_dataset_path: Path
    metadata_path: Path
    positive_count: int
    background_count: int
    candidate_background_count: int


@dataclass(frozen=True)
class PuModelResult:
    model_path: Path
    metadata_path: Path
    metrics: dict[str, Any]


def _json(path: str | Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PuLearningError(f"Could not read JSON: {path}") from error


def _feature_indexes(dataset: rasterio.DatasetReader) -> list[int]:
    descriptions = {name: index for index, name in enumerate(dataset.descriptions, start=1) if name}
    inverse = {
        "B02": "B02", "B03": "B03", "B04": "B04", "B08": "B08", "B11": "B11", "B12": "B12",
        "NDVI": "NDVI", "NDMI": "NDMI", "iron_oxide_proxy": "Iron/Oxide Proxy", "swir_ratio": "SWIR Ratio",
    }
    missing = [raster_band for raster_band in inverse.values() if raster_band not in descriptions]
    if missing:
        raise PuLearningError(f"Feature raster is missing PU features: {missing}")
    return [descriptions[inverse[column]] for column in FEATURE_COLUMNS]


def _project_wgs84_frame(frame: pd.DataFrame) -> gpd.GeoDataFrame:
    points = gpd.GeoDataFrame(
        frame.copy(), geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]), crs=WGS84
    ).to_crs(TARGET_CRS)
    points["x_utm"] = points.geometry.x
    points["y_utm"] = points.geometry.y
    return points


def prepare_pu_dataset(
    positives_path: str | Path,
    feature_raster_path: str | Path,
    background_path: str | Path,
    pu_dataset_path: str | Path,
    metadata_path: str | Path,
    *,
    random_seed: int = RANDOM_SEED,
    exclusion_buffer_meters: float = POSITIVE_EXCLUSION_BUFFER_METERS,
    background_to_positive_ratio: int = BACKGROUND_TO_POSITIVE_RATIO,
) -> PuDatasetResult:
    """Create a positive-plus-unlabeled reference table without asserting negatives."""
    if exclusion_buffer_meters <= 0 or background_to_positive_ratio <= 0:
        raise PuLearningError("PU buffer and background-to-positive ratio must be positive.")
    positives = pd.read_csv(positives_path)
    required_positive = {"point_id", "longitude", "latitude", *FEATURE_COLUMNS, "MnO", "label", "source"}
    missing_positive = required_positive.difference(positives.columns)
    if missing_positive:
        raise PuLearningError(f"Positive training samples are missing columns: {sorted(missing_positive)}")
    if positives.empty or not positives["label"].eq(1).all():
        raise PuLearningError("PU positives must be non-empty authoritative label=1 GSI samples only.")
    if positives["point_id"].duplicated().any():
        raise PuLearningError("Authoritative GSI point IDs must be unique.")
    positive_values = positives.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(positive_values).all():
        raise PuLearningError("Positive training features must be finite.")
    projected_positives = _project_wgs84_frame(positives)
    feature_raster_path = Path(feature_raster_path)
    with rasterio.open(feature_raster_path) as raster:
        if raster.crs is None or raster.crs.to_string() != TARGET_CRS:
            raise PuLearningError(f"Feature raster must use {TARGET_CRS}, found {raster.crs}.")
        if raster.nodata != OUTPUT_NODATA:
            raise PuLearningError("Feature raster must retain nodata -9999.")
        indexes = _feature_indexes(raster)
        arrays = raster.read(indexes).astype(np.float32, copy=False)
        valid_pixels = np.all(np.isfinite(arrays) & (arrays != OUTPUT_NODATA), axis=0)
        exclusion = geometry_mask(
            [geometry.buffer(exclusion_buffer_meters) for geometry in projected_positives.geometry],
            out_shape=(raster.height, raster.width), transform=raster.transform, invert=True,
        )
        candidates = np.flatnonzero((valid_pixels & ~exclusion).ravel())
        # Rasterizing a curved vector buffer to a 20 m grid can include an
        # edge cell whose centre is marginally inside the requested radius.
        # Apply an exact UTM point-distance check before random selection.
        candidate_rows, candidate_cols = np.unravel_index(candidates, valid_pixels.shape)
        candidate_xs, candidate_ys = raster.xy(candidate_rows, candidate_cols, offset="center")
        positive_tree = KDTree(projected_positives[["x_utm", "y_utm"]].to_numpy(dtype=float))
        nearest_distances = positive_tree.query(np.column_stack((candidate_xs, candidate_ys)), k=1)[0][:, 0]
        candidates = candidates[nearest_distances >= exclusion_buffer_meters]
        desired = len(positives) * background_to_positive_ratio
        if len(candidates) < desired:
            raise PuLearningError(
                f"Only {len(candidates)} valid background candidates remain after the exclusion buffer; need {desired}."
            )
        selected = np.random.default_rng(random_seed).choice(candidates, size=desired, replace=False)
        rows, cols = np.unravel_index(selected, valid_pixels.shape)
        xs, ys = raster.xy(rows, cols, offset="center")
        background_geometry = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=raster.crs).to_crs(WGS84)
        sampled = arrays[:, rows, cols].T

    background_records = []
    for index, (row, col) in enumerate(zip(rows, cols), start=1):
        values = dict(zip(FEATURE_COLUMNS, (float(value) for value in sampled[index - 1])))
        background_records.append({
            "point_id": f"unlabeled_r{int(row)}_c{int(col)}",
            "longitude": float(background_geometry.iloc[index - 1].x),
            "latitude": float(background_geometry.iloc[index - 1].y),
            "x_utm": float(xs[index - 1]), "y_utm": float(ys[index - 1]),
            "pixel_row": int(row), "pixel_col": int(col), **values,
            "sample_class": "unlabeled_background",
            "source": "Valid Sentinel-2 Phase 4 raster pixel; unlabeled/background, not confirmed non-manganese.",
        })
    background = pd.DataFrame(background_records, columns=BACKGROUND_COLUMNS)
    if not np.isfinite(background.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise PuLearningError("Background sampling produced invalid feature values.")
    positive_pu = projected_positives.copy()
    positive_pu["sample_class"] = "authoritative_manganese_positive"
    positive_pu["authoritative_label"] = 1
    positive_pu["pu_training_target"] = 1
    positive_pu = positive_pu.loc[:, ["point_id", "longitude", "latitude", "x_utm", "y_utm", *FEATURE_COLUMNS, "MnO", "sample_class", "authoritative_label", "pu_training_target", "source"]]
    background_pu = background.copy()
    background_pu["MnO"] = np.nan
    background_pu["authoritative_label"] = np.nan
    background_pu["pu_training_target"] = 0
    background_pu = background_pu.loc[:, PU_COLUMNS]
    pu_dataset = pd.concat([positive_pu, background_pu], ignore_index=True)
    background_path = Path(background_path)
    pu_dataset_path = Path(pu_dataset_path)
    metadata_path = Path(metadata_path)
    background_path.parent.mkdir(parents=True, exist_ok=True)
    background.to_csv(background_path, index=False)
    pu_dataset.to_csv(pu_dataset_path, index=False)
    metadata = {
        "dataset_type": "positive_unlabeled",
        "random_seed": random_seed,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_raster_path": str(feature_raster_path).replace("\\", "/"),
        "crs": {"geographic": WGS84, "sampling": TARGET_CRS},
        "positive_samples": {
            "count": int(len(positives)),
            "status": "authoritative GSI manganese field/exploration samples",
            "authoritative_label": 1,
            "mno_available": int(positives["MnO"].notna().sum()),
            "mno_missing": int(positives["MnO"].isna().sum()),
        },
        "unlabeled_background": {
            "count": int(len(background)),
            "candidate_valid_pixels_after_buffer": int(len(candidates)),
            "status": "unlabeled/background only; not confirmed non-manganese and not authoritative label 0",
            "exclusion_buffer_meters": exclusion_buffer_meters,
            "sampling_ratio_to_positives": background_to_positive_ratio,
        },
        "training_target": {
            "positive": 1,
            "unlabeled": 0,
            "meaning": "surrogate target for distinguishing known positives from unlabeled background, not a confirmed negative label",
        },
        "limitations": [
            "Unlabeled background can contain unknown manganese occurrences.",
            "PU scores are relative prospectivity/ranking signals, not calibrated probability of a confirmed deposit.",
            "MnO is preserved as metadata and is not a model feature because it is unavailable for some authoritative samples.",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return PuDatasetResult(background_path, pu_dataset_path, metadata_path, len(positives), len(background), len(candidates))


def _spatial_groups(frame: pd.DataFrame, block_size_meters: float = SPATIAL_BLOCK_SIZE_METERS) -> np.ndarray:
    if block_size_meters <= 0:
        raise PuLearningError("Spatial block size must be positive.")
    east_block = np.floor(frame["x_utm"].to_numpy(dtype=float) / block_size_meters).astype(int)
    north_block = np.floor(frame["y_utm"].to_numpy(dtype=float) / block_size_meters).astype(int)
    return np.asarray([f"{east}_{north}" for east, north in zip(east_block, north_block)])


def train_pu_random_forest(
    pu_dataset_path: str | Path,
    model_path: str | Path,
    model_metadata_path: str | Path,
    *,
    random_seed: int = RANDOM_SEED,
    spatial_block_size_meters: float = SPATIAL_BLOCK_SIZE_METERS,
) -> PuModelResult:
    """Fit and spatially validate a positive-vs-unlabeled Random Forest ranker."""
    frame = pd.read_csv(pu_dataset_path)
    required = {"point_id", "x_utm", "y_utm", "sample_class", "pu_training_target", *FEATURE_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise PuLearningError(f"PU dataset is missing columns: {sorted(missing)}")
    if not frame.loc[frame["sample_class"] == "authoritative_manganese_positive", "pu_training_target"].eq(1).all():
        raise PuLearningError("Authoritative positive samples must have PU target 1.")
    if not frame.loc[frame["sample_class"] == "unlabeled_background", "pu_training_target"].eq(0).all():
        raise PuLearningError("Unlabeled background must have PU target 0 only as a surrogate target.")
    target = frame["pu_training_target"].to_numpy(dtype=int)
    features = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    if len(np.unique(target)) != 2 or not np.isfinite(features).all():
        raise PuLearningError("PU training requires finite features and both positive and unlabeled samples.")
    groups = _spatial_groups(frame, spatial_block_size_meters)
    group_count = len(np.unique(groups))
    splits = min(5, group_count)
    if splits < 2:
        raise PuLearningError("At least two spatial groups are required for PU validation.")
    splitter = GroupKFold(n_splits=splits)
    oof_scores = np.full(len(frame), np.nan, dtype=float)
    for train_index, test_index in splitter.split(features, target, groups):
        if len(np.unique(target[train_index])) < 2:
            raise PuLearningError("A spatial training fold does not contain both PU classes.")
        fold_model = RandomForestClassifier(
            n_estimators=400, random_state=random_seed, class_weight="balanced_subsample", n_jobs=-1,
            min_samples_leaf=2,
        )
        fold_model.fit(features[train_index], target[train_index])
        oof_scores[test_index] = fold_model.predict_proba(features[test_index])[:, 1]
    if not np.isfinite(oof_scores).all():
        raise PuLearningError("Spatial PU validation did not generate all out-of-fold scores.")
    predictions = (oof_scores >= 0.5).astype(int)
    metrics = {
        "validation_kind": "spatial GroupKFold positive-vs-unlabeled surrogate validation",
        "spatial_block_size_meters": spatial_block_size_meters,
        "spatial_group_count": group_count,
        "fold_count": splits,
        "roc_auc_positive_vs_unlabeled": float(roc_auc_score(target, oof_scores)),
        "average_precision_positive_vs_unlabeled": float(average_precision_score(target, oof_scores)),
        "precision_at_0_5_positive_vs_unlabeled": float(precision_score(target, predictions, zero_division=0)),
        "recall_at_0_5_known_positives": float(recall_score(target, predictions, zero_division=0)),
        "f1_at_0_5_positive_vs_unlabeled": float(f1_score(target, predictions, zero_division=0)),
        "surrogate_confusion_matrix": confusion_matrix(target, predictions, labels=[0, 1]).tolist(),
    }
    final_model = RandomForestClassifier(
        n_estimators=400, random_state=random_seed, class_weight="balanced_subsample", n_jobs=-1,
        min_samples_leaf=2,
    )
    final_model.fit(features, target)
    model_path = Path(model_path)
    model_metadata_path = Path(model_metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "feature_columns": list(FEATURE_COLUMNS), "model_kind": "positive_vs_unlabeled_random_forest"}, model_path)
    metadata = {
        "model_type": "RandomForestClassifier",
        "model_kind": "positive-unlabeled positive-vs-background ranker",
        "feature_list": list(FEATURE_COLUMNS),
        "positive_sample_count": int((target == 1).sum()),
        "unlabeled_background_count": int((target == 0).sum()),
        "random_seed": random_seed,
        "metrics": metrics,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "scientific_interpretation": "Manganese prospectivity based on Sentinel-2 spectral proxies and GSI ground truth; not confirmation of a manganese deposit, reserve, ore grade, or economic viability.",
        "limitations": [
            "The background class is unlabeled and can contain undiscovered manganese; it is not a confirmed non-manganese class.",
            "All validation metrics are positive-vs-unlabeled surrogate metrics, not deposit-detection accuracy.",
            "Scores are not calibrated probabilities of manganese occurrence without a defensible PU class-prior estimate and independent validation.",
            "Spatial blocks reduce nearby-pixel leakage but do not replace independent geographic validation.",
        ],
    }
    model_metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return PuModelResult(model_path, model_metadata_path, metrics)
