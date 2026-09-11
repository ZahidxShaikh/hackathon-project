"""Training-data contract and validation for authoritative labeled points."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


TRAINING_FEATURES = [
    "ndvi", "ndmi", "iron_oxide_proxy", "swir_ratio", "B02", "B03", "B04", "B08", "B11", "B12"
]
REQUIRED_COLUMNS = ["longitude", "latitude", "label", *TRAINING_FEATURES]
OPTIONAL_GEOLOGY_PREFIX = "geology_"
WGS84 = "EPSG:4326"


class TrainingDataError(ValueError):
    """Raised when a proposed training dataset violates the data contract."""


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "NDVI": "ndvi", "NDMI": "ndmi", "Iron/Oxide proxy": "iron_oxide_proxy",
        "iron_oxide": "iron_oxide_proxy", "SWIR ratio": "swir_ratio",
    }
    return frame.rename(columns={column: aliases.get(column, column) for column in frame.columns}).copy()


def load_training_dataset(path: str | Path, *, allow_synthetic_test_fixture: bool = False) -> pd.DataFrame:
    """Read CSV, GeoJSON, GeoPackage, or Shapefile point data into the contract.

    Real commands reject `synthetic_test_fixture=true`; this marker is accepted
    only by the test suite and prevents test data entering project training.
    """
    source = Path(path)
    if not source.is_file():
        raise TrainingDataError(f"Training data file does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix in {".geojson", ".json", ".gpkg", ".shp"}:
        geodata = gpd.read_file(source)
        if geodata.crs is None:
            raise TrainingDataError("GIS training data must declare a CRS.")
        geodata = geodata.to_crs(WGS84)
        if not geodata.geometry.geom_type.isin(["Point"]).all():
            raise TrainingDataError("GIS training data must contain Point geometries only.")
        frame = pd.DataFrame(geodata.drop(columns="geometry"))
        frame["longitude"] = geodata.geometry.x
        frame["latitude"] = geodata.geometry.y
    else:
        raise TrainingDataError("Supported training inputs are CSV, GeoJSON, GeoPackage, and Shapefile.")
    return validate_training_dataframe(frame, allow_synthetic_test_fixture=allow_synthetic_test_fixture)


def validate_training_dataframe(
    frame: pd.DataFrame, *, allow_synthetic_test_fixture: bool = False
) -> pd.DataFrame:
    """Validate the public training schema without manufacturing any labels."""
    frame = _normalise_columns(frame)
    if frame.empty:
        raise TrainingDataError("Training dataset is empty.")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise TrainingDataError(f"Training dataset is missing required columns: {', '.join(missing)}")
    if "synthetic_test_fixture" in frame and frame["synthetic_test_fixture"].astype(bool).any():
        if not allow_synthetic_test_fixture:
            raise TrainingDataError("Synthetic test fixtures cannot be used as project ground truth.")

    output = frame.copy()
    numeric_columns = ["longitude", "latitude", "label", *TRAINING_FEATURES]
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    if output[numeric_columns].isna().any().any() or not np.isfinite(output[numeric_columns].to_numpy()).all():
        raise TrainingDataError("Coordinates, labels, and required features must be finite non-missing numbers.")
    if not output["longitude"].between(-180, 180).all() or not output["latitude"].between(-90, 90).all():
        raise TrainingDataError("Coordinates fall outside valid WGS84 longitude/latitude bounds.")
    if not output["label"].isin([0, 1]).all():
        raise TrainingDataError("label must be 1 (authoritative manganese) or 0 (confirmed background).")
    if output.duplicated(["longitude", "latitude"]).any():
        raise TrainingDataError("Duplicate longitude/latitude training points are not allowed.")
    return output


def spatial_holdout_split(frame: pd.DataFrame, test_size: float, random_seed: int):
    """Deterministic holdout split; replace with spatial blocks when coverage grows.

    It is stratified by label and intended for a small baseline. Nearby points
    should be grouped into spatial blocks before operational evaluation.
    """
    from sklearn.model_selection import train_test_split

    return train_test_split(
        frame, test_size=test_size, random_state=random_seed, stratify=frame["label"]
    )

