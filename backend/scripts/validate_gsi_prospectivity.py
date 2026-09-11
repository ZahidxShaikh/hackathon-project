"""Validate geospatial preservation and finite PU prospectivity ranking scores."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio

from backend.app.services.raster import OUTPUT_NODATA


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GSI_OUTPUT_DIRECTORY = PROJECT_ROOT / "backend" / "data" / "processed" / "gsi_49367"
PREDICTION_DIRECTORY = PROJECT_ROOT / "backend" / "data" / "predictions"


def main() -> int:
    try:
        provenance = json.loads((GSI_OUTPUT_DIRECTORY / "provenance.json").read_text(encoding="utf-8"))
        summary = json.loads((PREDICTION_DIRECTORY / "balaghat_manganese_prospectivity_summary.json").read_text(encoding="utf-8"))
        source_path = GSI_OUTPUT_DIRECTORY / provenance["feature_path"]
        prediction_path = PREDICTION_DIRECTORY / "balaghat_manganese_prospectivity.tif"
        with rasterio.open(source_path) as source, rasterio.open(prediction_path) as prediction:
            if prediction.count != 1 or prediction.dtypes[0] != "float32":
                raise ValueError("Prediction must be a single-band float32 GeoTIFF.")
            if prediction.crs is None or prediction.crs.to_epsg() != 32644:
                raise ValueError(f"Prediction CRS must be EPSG:32644, found {prediction.crs}.")
            if (prediction.width, prediction.height) != (source.width, source.height):
                raise ValueError("Prediction dimensions do not match the feature raster.")
            if prediction.transform != source.transform or prediction.res != (20.0, 20.0):
                raise ValueError("Prediction transform/resolution does not match the 20 m feature raster.")
            if prediction.nodata != OUTPUT_NODATA:
                raise ValueError("Prediction nodata must be -9999.")
            values = prediction.read(1)
            valid = values != OUTPUT_NODATA
            invalid = valid & (~np.isfinite(values) | (values < 0) | (values > 1))
            if invalid.any():
                raise ValueError(f"Prediction has {int(invalid.sum())} invalid finite/range values.")
            valid_count = int(valid.sum())
            nodata_count = int((~valid).sum())
            if valid_count != summary["valid_pixel_count"] or nodata_count != summary["nodata_pixel_count"]:
                raise ValueError("Prediction pixel counts differ from the generated summary.")
            if not np.isclose(float(values[valid].min()), summary["score_statistics"]["minimum"]):
                raise ValueError("Prediction minimum differs from the generated summary.")
            if prediction.tags().get("score_kind", "").find("not a calibrated probability") == -1:
                raise ValueError("Prediction metadata lacks the PU score interpretation disclaimer.")
    except Exception as error:
        print(f"GSI PU PROSPECTIVITY VALIDATION FAILED: {error}", file=sys.stderr)
        return 1
    print("GSI PU PROSPECTIVITY VALIDATION PASSED")
    print(f"Output: {prediction_path}")
    print(f"CRS/dimensions/resolution: {prediction.crs}; {prediction.width} x {prediction.height}; {prediction.res}")
    print(f"Valid/NoData pixels: {valid_count}/{nodata_count}")
    print(f"Score min/max/mean/median: {values[valid].min():.6f} / {values[valid].max():.6f} / {values[valid].mean():.6f} / {np.median(values[valid]):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
