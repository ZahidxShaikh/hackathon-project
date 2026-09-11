"""Validate corrected GSI-study-area Phase 4 features and point overlap."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

from backend.app.services.gsi_study_aoi import GSI_FEATURE_DESCRIPTIONS
from backend.app.services.raster import OUTPUT_NODATA


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH = PROJECT_ROOT / "backend" / "data" / "ground_truth" / "manganese_positive.geojson"
OUTPUT_DIRECTORY = PROJECT_ROOT / "backend" / "data" / "processed" / "gsi_49367"
PROVENANCE_PATH = OUTPUT_DIRECTORY / "provenance.json"


def _expected_names() -> tuple[str, ...]:
    return tuple(description.split(":", maxsplit=1)[0] for description in GSI_FEATURE_DESCRIPTIONS)


def main() -> int:
    try:
        provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
        stack_path = OUTPUT_DIRECTORY / provenance["stack_path"]
        features_path = OUTPUT_DIRECTORY / provenance["feature_path"]
        if not stack_path.is_file() or not features_path.is_file():
            raise FileNotFoundError("Corrected stack or Phase 4 feature raster is missing.")
        ground_truth = gpd.read_file(GROUND_TRUTH)
        if ground_truth.crs is None or ground_truth.crs.to_epsg() != 4326:
            raise ValueError("Ground truth must use EPSG:4326.")
        valid_points = ground_truth.loc[ground_truth["validation_status"] == "valid"].copy()
        with rasterio.open(stack_path) as stack, rasterio.open(features_path) as features:
            expected_names = _expected_names()
            if features.count != len(expected_names):
                raise ValueError(f"Expected {len(expected_names)} bands, found {features.count}.")
            if tuple(features.descriptions) != expected_names:
                raise ValueError(f"Unexpected feature band descriptions: {features.descriptions}")
            if features.crs is None or features.crs.to_epsg() != 32644:
                raise ValueError(f"Expected EPSG:32644, found {features.crs}.")
            if features.width != stack.width or features.height != stack.height:
                raise ValueError("Feature dimensions do not match the corrected input stack.")
            if features.transform != stack.transform or features.res != (20.0, 20.0):
                raise ValueError("Feature transform/resolution does not preserve the 20 m corrected stack grid.")
            if features.nodata != OUTPUT_NODATA or features.dtypes != tuple(["float32"] * features.count):
                raise ValueError("Feature raster must be float32 with nodata -9999.")
            values = features.read()
            nodata = values == OUTPUT_NODATA
            nonfinite = ~np.isfinite(values)
            unexpected_nonfinite_count = int((nonfinite & ~nodata).sum())
            if unexpected_nonfinite_count:
                raise ValueError(f"Found {unexpected_nonfinite_count} unexpected NaN/Inf values.")
            valid_pixel_count = int(np.all(~nodata & ~nonfinite, axis=0).sum())
            points_in_raster_crs = valid_points.to_crs(features.crs)
            overlap_count = int(points_in_raster_crs.geometry.intersects(box(*features.bounds)).sum())
            if overlap_count == 0:
                raise ValueError("No valid GSI ground-truth point overlaps the feature raster.")
            if overlap_count != len(valid_points):
                raise ValueError(f"Only {overlap_count}/{len(valid_points)} valid GSI points overlap the feature raster.")
    except Exception as error:
        print(f"GSI PHASE 4 FEATURE VALIDATION FAILED: {error}", file=sys.stderr)
        return 1

    print("GSI PHASE 4 FEATURE VALIDATION PASSED")
    print(f"Feature raster: {features_path}")
    print(f"Band count: {features.count}")
    print(f"Band names: {', '.join(expected_names)}")
    print(f"Dimensions: {features.width} x {features.height}")
    print(f"CRS: {features.crs}")
    print(f"Resolution: {features.res}")
    print(f"NoData: {features.nodata}")
    print(f"Valid pixels (all bands): {valid_pixel_count}")
    print(f"Unexpected NaN/Inf count: {unexpected_nonfinite_count}")
    print(f"Valid ground-truth points overlapping raster: {overlap_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
