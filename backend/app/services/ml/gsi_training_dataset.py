"""Prepare authoritative positive-only samples from the corrected GSI feature raster.

This is a data-preparation service, not a model-training service.  It preserves
each distinct GSI sample ID even where separate field samples share coordinates.
Unlabeled raster pixels are deliberately excluded and are never represented as
confirmed non-manganese labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from backend.app.services.raster import OUTPUT_NODATA


WGS84 = "EPSG:4326"
TARGET_CRS_EPSG = 32644
FEATURE_BAND_TO_COLUMN = {
    "B02": "B02",
    "B03": "B03",
    "B04": "B04",
    "B08": "B08",
    "B11": "B11",
    "B12": "B12",
    "NDVI": "NDVI",
    "NDMI": "NDMI",
    "Iron/Oxide Proxy": "iron_oxide_proxy",
    "SWIR Ratio": "swir_ratio",
}
FEATURE_COLUMNS = tuple(FEATURE_BAND_TO_COLUMN.values())
OUTPUT_COLUMNS = (
    "point_id", "longitude", "latitude", *FEATURE_COLUMNS, "MnO", "label",
    "sample_class", "sample_type", "location", "source", "source_report", "NUID", "page",
)


class GsiTrainingDatasetError(RuntimeError):
    """Raised when an authoritative sample cannot be prepared safely."""


@dataclass(frozen=True)
class DatasetPreparationResult:
    output_path: Path
    metadata_path: Path
    total_ground_truth_points: int
    valid_ground_truth_points: int
    sampled_points: int
    dropped_points: int
    mno_available: int
    mno_missing: int


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GsiTrainingDatasetError(f"Could not read JSON metadata: {path}") from error


def _feature_indexes(dataset: rasterio.DatasetReader) -> dict[str, int]:
    descriptions = {name: index for index, name in enumerate(dataset.descriptions, start=1) if name}
    missing = [name for name in FEATURE_BAND_TO_COLUMN if name not in descriptions]
    if missing:
        raise GsiTrainingDatasetError(f"Feature raster is missing required bands: {', '.join(missing)}")
    return {name: descriptions[name] for name in FEATURE_BAND_TO_COLUMN}


def prepare_gsi_positive_training_dataset(
    ground_truth_path: str | Path,
    ground_truth_metadata_path: str | Path,
    feature_raster_path: str | Path,
    output_csv_path: str | Path,
    output_metadata_path: str | Path,
) -> DatasetPreparationResult:
    """Sample all Phase 4 features for valid authoritative GSI point records.

    Only a sample with a non-finite or nodata feature value is dropped.  The
    method does not create background samples or any label zero records.
    """
    ground_truth_path = Path(ground_truth_path)
    feature_raster_path = Path(feature_raster_path)
    output_csv_path = Path(output_csv_path)
    output_metadata_path = Path(output_metadata_path)
    report_metadata = _read_json(ground_truth_metadata_path)
    if not feature_raster_path.is_file():
        raise GsiTrainingDatasetError(f"Phase 4 feature raster does not exist: {feature_raster_path}")
    try:
        ground_truth = gpd.read_file(ground_truth_path)
    except Exception as error:
        raise GsiTrainingDatasetError(f"Could not read GSI ground-truth GeoJSON: {ground_truth_path}") from error
    if ground_truth.crs is None or ground_truth.crs.to_epsg() != 4326:
        raise GsiTrainingDatasetError("GSI ground truth must have WGS84 / EPSG:4326 CRS.")
    required_ground_truth_columns = {"sample_id", "validation_status", "MnO", "source_report", "NUID"}
    missing_columns = required_ground_truth_columns.difference(ground_truth.columns)
    if missing_columns:
        raise GsiTrainingDatasetError(f"GSI ground truth is missing columns: {sorted(missing_columns)}")
    if ground_truth["sample_id"].duplicated().any():
        raise GsiTrainingDatasetError("GSI ground truth contains duplicate sample IDs.")

    valid = ground_truth.loc[ground_truth["validation_status"] == "valid"].copy()
    if valid.empty:
        raise GsiTrainingDatasetError("No valid GSI ground-truth points are available.")
    projected = valid.to_crs(f"EPSG:{TARGET_CRS_EPSG}")
    records: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    with rasterio.open(feature_raster_path) as raster:
        if raster.crs is None or raster.crs.to_epsg() != TARGET_CRS_EPSG:
            raise GsiTrainingDatasetError(f"Feature raster must use EPSG:{TARGET_CRS_EPSG}, found {raster.crs}.")
        if raster.nodata != OUTPUT_NODATA:
            raise GsiTrainingDatasetError(f"Feature raster nodata must be {OUTPUT_NODATA}.")
        feature_indexes = _feature_indexes(raster)
        for source_index, point in projected.iterrows():
            sample_id = str(point["sample_id"])
            sampled = next(raster.sample([(point.geometry.x, point.geometry.y)], indexes=list(feature_indexes.values())))
            sampled = np.asarray(sampled, dtype=np.float32)
            invalid = ~np.isfinite(sampled) | (sampled == OUTPUT_NODATA)
            if invalid.any():
                dropped.append({"point_id": sample_id, "reason": "invalid_or_nodata_feature_value"})
                continue
            original = valid.loc[source_index]
            values = dict(zip(FEATURE_COLUMNS, (float(value) for value in sampled)))
            mno = pd.to_numeric(pd.Series([original["MnO"]]), errors="coerce").iloc[0]
            source = f"{report_metadata['source_organization']} / {report_metadata['source_portal']}; NUID {original['NUID']}"
            records.append({
                "point_id": sample_id,
                "longitude": float(original.geometry.x),
                "latitude": float(original.geometry.y),
                **values,
                "MnO": None if pd.isna(mno) else float(mno),
                "label": 1,
                "sample_class": "authoritative_manganese_positive",
                "sample_type": original.get("sample_type"),
                "location": original.get("location"),
                "source": source,
                "source_report": original["source_report"],
                "NUID": str(original["NUID"]),
                "page": original.get("page"),
            })

    dataset = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    if dataset.empty:
        raise GsiTrainingDatasetError("All valid GSI samples had invalid or nodata raster features.")
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_csv_path, index=False)
    coincident_coordinate_records = int(dataset.duplicated(["longitude", "latitude"], keep=False).sum())
    mno_available = int(dataset["MnO"].notna().sum())
    metadata = {
        "dataset_type": "authoritative_positive_only",
        "scientific_status": "GSI field/exploration evidence sampled from Sentinel-2 spectral/geological proxies; not a reserve, ore-grade, or economic assessment.",
        "background_samples": {
            "included": False,
            "status": "No unlabeled/background raster pixels were relabeled as confirmed non-manganese.",
        },
        "source": {
            "organization": report_metadata["source_organization"],
            "portal": report_metadata["source_portal"],
            "report_title": report_metadata["report_title"],
            "NUID": report_metadata["NUID"],
            "ground_truth_path": str(ground_truth_path).replace("\\", "/"),
        },
        "crs": {"point_coordinates": WGS84, "sampling_raster": f"EPSG:{TARGET_CRS_EPSG}"},
        "feature_raster_path": str(feature_raster_path).replace("\\", "/"),
        "feature_columns": list(FEATURE_COLUMNS),
        "label_definition": {"1": "authoritative GSI manganese field/exploration sample", "0": "not present in this positive-only dataset"},
        "counts": {
            "total_ground_truth_points": int(len(ground_truth)),
            "valid_ground_truth_points": int(len(valid)),
            "sampled_training_points": int(len(dataset)),
            "dropped_points": int(len(dropped)),
            "mno_available": mno_available,
            "mno_missing": int(len(dataset) - mno_available),
            "coincident_coordinate_records_retained": coincident_coordinate_records,
            "label_counts": {"1": int((dataset["label"] == 1).sum()), "0": 0},
        },
        "dropped_samples": dropped,
        "drop_rule": "Only invalid, non-finite, or nodata Phase 4 feature values cause a valid GSI sample to be dropped.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output_metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return DatasetPreparationResult(
        output_path=output_csv_path,
        metadata_path=output_metadata_path,
        total_ground_truth_points=len(ground_truth),
        valid_ground_truth_points=len(valid),
        sampled_points=len(dataset),
        dropped_points=len(dropped),
        mno_available=mno_available,
        mno_missing=len(dataset) - mno_available,
    )
