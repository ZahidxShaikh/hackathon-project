"""Self-contained mechanics test; synthetic data never enters project training."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from backend.app.services.ml.dataset import TRAINING_FEATURES, TrainingDataError, validate_training_dataframe
from backend.app.services.ml.model import train_random_forest
from backend.app.services.ml.prediction import predict_prospectivity
from backend.app.services.raster import OUTPUT_NODATA


def synthetic_fixture() -> pd.DataFrame:
    rows = []
    for index in range(12):
        label = index % 2
        row = {"longitude": 80.11 + index * 0.001, "latitude": 21.71 + index * 0.001,
               "label": label, "synthetic_test_fixture": True}
        row.update({feature: float(index + 1 + label) for feature in TRAINING_FEATURES})
        rows.append(row)
    return pd.DataFrame(rows)


def write_feature_fixture(path: Path) -> None:
    names = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "NDWI", "Iron/Oxide Proxy", "SWIR Ratio"]
    profile = {"driver": "GTiff", "width": 3, "height": 2, "count": len(names), "dtype": "float32",
               "crs": "EPSG:32644", "transform": from_origin(500000, 2400000, 20, 20),
               "nodata": OUTPUT_NODATA, "compress": "deflate"}
    with rasterio.open(path, "w", **profile) as dataset:
        for band, name in enumerate(names, 1):
            dataset.write(np.full((2, 3), band + 1, dtype=np.float32), band)
            dataset.set_band_description(band, name)


def main() -> int:
    try:
        fixture = synthetic_fixture()
        try:
            validate_training_dataframe(fixture)
            raise AssertionError("Synthetic fixture was accepted as real ground truth.")
        except TrainingDataError:
            pass
        valid = validate_training_dataframe(fixture, allow_synthetic_test_fixture=True)
        invalid = valid.drop(columns=["B12"])
        try:
            validate_training_dataframe(invalid, allow_synthetic_test_fixture=True)
            raise AssertionError("Missing feature was not rejected.")
        except TrainingDataError:
            pass
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path, metadata_path = root / "model.joblib", root / "metadata.json"
            result = train_random_forest(
                valid, source_dataset_description="synthetic_test_fixture=true; test only",
                model_path=model_path, metadata_path=metadata_path,
            )
            assert result.model_path.is_file() and result.metadata_path.is_file()
            assert json.loads(metadata_path.read_text())["training_sample_count"] == 12
            feature_input, prediction_output = root / "features.tif", root / "prediction.tif"
            write_feature_fixture(feature_input)
            predict_prospectivity(feature_input, model_path, prediction_output)
            with rasterio.open(feature_input) as source, rasterio.open(prediction_output) as predicted:
                assert predicted.crs == source.crs and predicted.transform == source.transform
                assert predicted.width == source.width and predicted.height == source.height
                assert predicted.nodata == OUTPUT_NODATA and predicted.dtypes[0] == "float32"
                values = predicted.read(1)
                valid_values = values[values != OUTPUT_NODATA]
                assert np.isfinite(valid_values).all()
                assert ((valid_values >= 0) & (valid_values <= 1)).all()
    except (AssertionError, TrainingDataError, OSError, ValueError) as error:
        print(f"PHASE 5B TEST FAILED: {error}", file=sys.stderr)
        return 1
    print("PHASE 5B TEST PASSED")
    print("Synthetic fixture was isolated and explicitly rejected for real project ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

