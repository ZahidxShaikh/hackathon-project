"""Validate spatial overlap between authoritative GSI points and corrected rasters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH = PROJECT_ROOT / "backend" / "data" / "ground_truth" / "manganese_positive.geojson"
OUTPUT_DIRECTORY = PROJECT_ROOT / "backend" / "data" / "processed" / "gsi_49367"


def _default_stack() -> Path:
    provenance = json.loads((OUTPUT_DIRECTORY / "provenance.json").read_text(encoding="utf-8"))
    return OUTPUT_DIRECTORY / provenance["stack_path"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", type=Path, default=None)
    args = parser.parse_args()
    try:
        stack_path = args.stack or _default_stack()
        if not stack_path.is_file():
            raise FileNotFoundError(f"Corrected stack does not exist: {stack_path}")
        ground_truth = gpd.read_file(GROUND_TRUTH)
        if ground_truth.crs is None or ground_truth.crs.to_epsg() != 4326:
            raise ValueError("Ground truth must use EPSG:4326.")
        valid_points = ground_truth.loc[ground_truth["validation_status"] == "valid"].copy()
        with rasterio.open(stack_path) as raster:
            if raster.crs is None:
                raise ValueError("Corrected raster has no CRS.")
            projected_points = valid_points.to_crs(raster.crs)
            raster_extent = box(*raster.bounds)
            inside = projected_points.geometry.intersects(raster_extent)
            inside_count = int(inside.sum())
            if inside_count == 0:
                raise ValueError("No valid GSI ground-truth points intersect the corrected raster.")
            if inside_count != len(valid_points):
                raise ValueError(f"Only {inside_count}/{len(valid_points)} valid GSI points intersect the corrected raster.")
            print("GSI RASTER OVERLAP VALIDATION PASSED")
            print(f"Ground-truth CRS: {ground_truth.crs}")
            print(f"Raster CRS: {raster.crs}")
            print(f"Raster dimensions: {raster.width} x {raster.height}")
            print(f"Raster resolution: {raster.res}")
            print(f"Valid ground-truth points: {len(valid_points)}")
            print(f"Ground-truth points intersecting raster: {inside_count}")
            print("Non-zero spatial overlap: True")
    except Exception as error:
        print(f"GSI RASTER OVERLAP VALIDATION FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
