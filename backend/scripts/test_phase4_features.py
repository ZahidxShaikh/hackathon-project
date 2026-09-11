"""Validation script for the Phase 4 local feature GeoTIFF."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS

from backend.app.services.features import (
    FEATURE_BAND_DESCRIPTIONS,
    FeatureGenerationError,
    feature_output_path,
)
from backend.app.services.raster import OUTPUT_NODATA
from backend.scripts.generate_spectral_features import latest_phase3_stack


def main() -> int:
    try:
        input_path = latest_phase3_stack(Path("backend/data/processed"))
        output_path = feature_output_path(input_path)
        if not input_path.is_file():
            raise AssertionError(f"Input GeoTIFF does not exist: {input_path}")
        if not output_path.is_file():
            raise AssertionError(f"Output GeoTIFF does not exist: {output_path}")

        with rasterio.open(input_path) as input_dataset, rasterio.open(output_path) as output_dataset:
            assert output_dataset.count == len(FEATURE_BAND_DESCRIPTIONS), (
                f"Expected {len(FEATURE_BAND_DESCRIPTIONS)} bands, found {output_dataset.count}."
            )
            assert output_dataset.crs == CRS.from_epsg(32644), f"Unexpected CRS: {output_dataset.crs}"
            assert output_dataset.width == input_dataset.width
            assert output_dataset.height == input_dataset.height
            assert output_dataset.transform == input_dataset.transform
            assert abs(output_dataset.res[0] - 20.0) < 1e-6
            assert abs(output_dataset.res[1] - 20.0) < 1e-6
            assert output_dataset.nodata == OUTPUT_NODATA
            assert output_dataset.descriptions == tuple(
                description.split(":", maxsplit=1)[0] for description in FEATURE_BAND_DESCRIPTIONS
            )

            for band_index in range(1, output_dataset.count + 1):
                values = output_dataset.read(band_index)
                valid = values != OUTPUT_NODATA
                assert np.isfinite(values[valid]).all(), (
                    f"Band {band_index} contains NaN or infinity in valid pixels."
                )
            output_count = output_dataset.count
            output_crs = output_dataset.crs
            output_width = output_dataset.width
            output_height = output_dataset.height
            output_resolution = output_dataset.res[0]
            output_nodata = output_dataset.nodata
    except (AssertionError, FeatureGenerationError, OSError, rasterio.errors.RasterioError) as error:
        print(f"PHASE 4 TEST FAILED: {error}", file=sys.stderr)
        return 1

    print("PHASE 4 TEST PASSED")
    print(f"Input: {input_path.resolve()}")
    print(f"Output: {output_path.resolve()}")
    print(f"Bands: {output_count}; CRS: {output_crs}; dimensions: {output_width} x {output_height}")
    print(f"Resolution: {output_resolution} m; nodata: {output_nodata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
