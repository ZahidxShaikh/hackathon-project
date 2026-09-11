"""Pixel-wise prospectivity output from a compatible trained pipeline."""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import rasterio

from backend.app.services.ml.dataset import TRAINING_FEATURES
from backend.app.services.raster import OUTPUT_NODATA


class PredictionError(RuntimeError):
    pass


RASTER_NAME_TO_FEATURE = {
    "B02": "B02", "B03": "B03", "B04": "B04", "B08": "B08", "B11": "B11", "B12": "B12",
    "NDVI": "ndvi", "Iron/Oxide Proxy": "iron_oxide_proxy", "SWIR Ratio": "swir_ratio",
}


def _safe_ndmi(nir, swir1, valid):
    values = np.full(nir.shape, OUTPUT_NODATA, dtype=np.float32)
    denominator = nir + swir1
    np.divide(nir - swir1, denominator, out=values, where=valid & (denominator != 0))
    values[~np.isfinite(values)] = OUTPUT_NODATA
    return values


def predict_prospectivity(feature_raster: str | Path, model_path: str | Path, output_path: str | Path) -> Path:
    """Write a 0–1 float32 prospectivity/proxy raster; preserves input georeferencing."""
    model_file, input_file, destination = Path(model_path), Path(feature_raster), Path(output_path)
    if not model_file.is_file():
        raise PredictionError(f"Trained model artifact does not exist: {model_file}")
    if not input_file.is_file():
        raise PredictionError(f"Feature GeoTIFF does not exist: {input_file}")
    pipeline = joblib.load(model_file)
    expected = list(getattr(pipeline, "feature_names_in_", TRAINING_FEATURES))
    if expected != TRAINING_FEATURES:
        raise PredictionError("Model feature contract does not match the Phase 5B feature contract.")
    with rasterio.open(input_file) as source:
        descriptions = {name: i for i, name in enumerate(source.descriptions, 1) if name}
        missing = [name for name in RASTER_NAME_TO_FEATURE if name not in descriptions]
        if missing:
            raise PredictionError(f"Feature GeoTIFF is missing bands: {missing}")
        arrays = {feature: source.read(descriptions[name]).astype(np.float32) for name, feature in RASTER_NAME_TO_FEATURE.items()}
        valid = np.ones((source.height, source.width), dtype=bool)
        for values in arrays.values():
            valid &= np.isfinite(values) & (values != OUTPUT_NODATA)
        arrays["ndmi"] = _safe_ndmi(arrays["B08"], arrays["B11"], valid)
        valid &= arrays["ndmi"] != OUTPUT_NODATA
        output = np.full((source.height, source.width), OUTPUT_NODATA, dtype=np.float32)
        if valid.any():
            matrix = pd.DataFrame(
                {column: arrays[column][valid] for column in TRAINING_FEATURES},
                columns=TRAINING_FEATURES,
            )
            output[valid] = pipeline.predict_proba(matrix)[:, 1].astype(np.float32)
        profile = source.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="float32", nodata=OUTPUT_NODATA, compress="deflate", predictor=3)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(destination, "w", **profile) as target:
            target.write(output, 1)
            target.set_band_description(1, "Manganese Prospectivity / Spectral-Geological Proxy")
            target.update_tags(disclaimer="Not a confirmed deposit, reserve estimate, ore grade, or economic-viability result.")
    return destination
