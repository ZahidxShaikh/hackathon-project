"""GSI-study-area AOI configuration and local spectral feature generation.

This module intentionally leaves the archived Phase 3/4 Balaghat outputs and
their feature-generation service untouched.  It derives a separate AOI from
the authoritative GSI report boundary recorded with the validated samples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.errors import RasterioError
from shapely.geometry import box

from backend.app.services.features import (
    FeatureGenerationError,
    _input_band_indexes,
    _read_valid_float32,
    _safe_ratio,
)
from backend.app.services.raster import OUTPUT_NODATA, REQUIRED_SPECTRAL_BANDS


GSI_AOI_MARGIN_DEGREES = 0.005
GSI_AOI_CONFIG_NAME = "gsi_study_aoi.json"
GSI_FEATURE_DESCRIPTIONS = (
    "B02: Sentinel-2 Blue reflectance (20 m aligned)",
    "B03: Sentinel-2 Green reflectance (20 m aligned)",
    "B04: Sentinel-2 Red reflectance (20 m aligned)",
    "B08: Sentinel-2 NIR reflectance (20 m aligned)",
    "B11: Sentinel-2 SWIR1 reflectance (20 m native target grid)",
    "B12: Sentinel-2 SWIR2 reflectance (20 m native target grid)",
    "NDVI: (B08 - B04) / (B08 + B04)",
    "NDMI: (B08 - B11) / (B08 + B11)",
    "Iron/Oxide Proxy: B04 / B02 red-to-blue spectral ratio",
    "SWIR Ratio: B11 / B12",
)


class GsiStudyAoiError(RuntimeError):
    """Raised when the GSI AOI or its derived raster cannot be verified."""


@dataclass(frozen=True)
class GsiStudyAoi:
    """WGS84 AOI derived from the GSI report's declared study boundary."""

    bbox: tuple[float, float, float, float]
    report_boundary: tuple[float, float, float, float]
    margin_degrees: float
    valid_ground_truth_count: int


def _load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    path = Path(metadata_path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GsiStudyAoiError(f"Could not read GSI ground-truth metadata: {path}") from error


def derive_gsi_study_aoi(
    ground_truth_path: str | Path,
    metadata_path: str | Path,
    margin_degrees: float = GSI_AOI_MARGIN_DEGREES,
) -> GsiStudyAoi:
    """Derive an expanded WGS84 AOI and verify it contains every valid point.

    The report boundary is preferred over only the point envelope because it
    covers the full exploration block, including unsampled portions.
    """
    if margin_degrees <= 0:
        raise GsiStudyAoiError("AOI margin must be positive.")
    metadata = _load_metadata(metadata_path)
    boundary = metadata.get("report_study_boundary_wgs84")
    if not isinstance(boundary, dict):
        raise GsiStudyAoiError("Ground-truth metadata has no report study boundary.")
    try:
        west = float(boundary["west"])
        south = float(boundary["south"])
        east = float(boundary["east"])
        north = float(boundary["north"])
    except (KeyError, TypeError, ValueError) as error:
        raise GsiStudyAoiError("Ground-truth metadata contains an invalid report boundary.") from error
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise GsiStudyAoiError("Report study boundary has invalid WGS84 coordinates.")

    try:
        points = gpd.read_file(ground_truth_path)
    except Exception as error:
        raise GsiStudyAoiError(f"Could not read validated ground truth: {ground_truth_path}") from error
    if points.crs is None or points.crs.to_epsg() != 4326:
        raise GsiStudyAoiError("Ground-truth GeoJSON must use EPSG:4326.")
    valid_points = points.loc[points["validation_status"] == "valid"].copy()
    if valid_points.empty:
        raise GsiStudyAoiError("No valid GSI points are available to verify the corrected AOI.")

    bbox = (west - margin_degrees, south - margin_degrees, east + margin_degrees, north + margin_degrees)
    aoi_geometry = box(*bbox)
    contained_count = int(valid_points.geometry.within(aoi_geometry).sum())
    if contained_count != len(valid_points):
        raise GsiStudyAoiError(
            f"Corrected AOI contains {contained_count}/{len(valid_points)} valid GSI points."
        )
    return GsiStudyAoi(
        bbox=bbox,
        report_boundary=(west, south, east, north),
        margin_degrees=margin_degrees,
        valid_ground_truth_count=contained_count,
    )


def write_gsi_aoi_config(aoi: GsiStudyAoi, output_path: str | Path) -> Path:
    """Write a reproducible, explicit configuration for the corrected AOI."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "GSI Balaghat manganese study area (NUID 49367)",
        "crs": "EPSG:4326",
        "bbox": list(aoi.bbox),
        "report_study_boundary_wgs84": {
            "west": aoi.report_boundary[0],
            "south": aoi.report_boundary[1],
            "east": aoi.report_boundary[2],
            "north": aoi.report_boundary[3],
        },
        "margin_degrees": aoi.margin_degrees,
        "derivation": "GSI report study boundary expanded by a 0.005 degree margin; verified against valid GSI samples.",
        "valid_ground_truth_points_inside": aoi.valid_ground_truth_count,
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def generate_gsi_study_features(
    input_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Create a separate ten-band local feature raster, including NDMI.

    All index bands are guarded against invalid input and zero denominators.
    They are spectral/geological proxies, not a direct manganese detector.
    """
    source_path = Path(input_path)
    destination = Path(output_path)
    if not source_path.is_file():
        raise GsiStudyAoiError(f"Corrected-AOI stack is missing: {source_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with rasterio.open(source_path) as source:
            if source.crs is None or source.nodata != OUTPUT_NODATA:
                raise GsiStudyAoiError("Corrected-AOI stack must have a CRS and nodata -9999.")
            band_indexes = _input_band_indexes(source)
            spectral: dict[str, np.ndarray] = {}
            valid: dict[str, np.ndarray] = {}
            for band_name in REQUIRED_SPECTRAL_BANDS:
                spectral[band_name], valid[band_name] = _read_valid_float32(source, band_indexes[band_name])

            ndvi = _safe_ratio(spectral["B08"] - spectral["B04"], spectral["B08"] + spectral["B04"], valid["B08"] & valid["B04"])
            ndmi = _safe_ratio(spectral["B08"] - spectral["B11"], spectral["B08"] + spectral["B11"], valid["B08"] & valid["B11"])
            iron_oxide = _safe_ratio(spectral["B04"], spectral["B02"], valid["B04"] & valid["B02"])
            swir_ratio = _safe_ratio(spectral["B11"], spectral["B12"], valid["B11"] & valid["B12"])
            arrays = [*(spectral[band] for band in REQUIRED_SPECTRAL_BANDS), ndvi, ndmi, iron_oxide, swir_ratio]
            profile = source.profile.copy()
            profile.update(driver="GTiff", count=len(arrays), dtype="float32", nodata=OUTPUT_NODATA, compress="deflate", predictor=3)
            with rasterio.open(destination, "w", **profile) as output:
                for index, (description, values) in enumerate(zip(GSI_FEATURE_DESCRIPTIONS, arrays), start=1):
                    output.write(values.astype(np.float32, copy=False), index)
                    output.set_band_description(index, description.split(":", maxsplit=1)[0])
                    output.update_tags(index, description=description)
                output.update_tags(
                    source_corrected_aoi_stack=source_path.name,
                    feature_disclaimer="Spectral/geological proxies only; not a direct manganese detector, reserve estimate, or ore-grade assessment.",
                    feature_processing="float32; nodata=-9999; guarded division; includes NDMI",
                )
    except FeatureGenerationError as error:
        raise GsiStudyAoiError(str(error)) from error
    except RasterioError as error:
        raise GsiStudyAoiError(f"Could not write corrected-AOI feature GeoTIFF: {error}") from error
    return destination
