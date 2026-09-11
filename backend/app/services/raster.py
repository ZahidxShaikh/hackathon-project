"""AOI-only raster preparation for signed Sentinel-2 COG assets.

Phase 3 deliberately produces an aligned spectral stack only. It does not
calculate indices, train a model, or make any mineral-prospectivity claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioError
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds
from pystac import Item


REQUIRED_SPECTRAL_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
TARGET_GRID_BAND = "B11"
OUTPUT_NODATA = -9999.0


class RasterPreparationError(RuntimeError):
    """Raised when a required asset cannot be read or aligned."""


@dataclass(frozen=True)
class RasterPreparationResult:
    """Details of an aligned AOI stack written to disk."""

    output_path: Path
    crs: str
    width: int
    height: int
    bands: tuple[str, ...]
    scale_offsets: dict[str, tuple[float, float]]


def _validate_aoi_bbox(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise RasterPreparationError("AOI bbox must contain [west, south, east, north].")
    try:
        west, south, east, north = (float(value) for value in bbox)
    except (TypeError, ValueError) as error:
        raise RasterPreparationError("AOI bbox values must be numeric.") from error
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise RasterPreparationError("AOI bbox has invalid longitude/latitude bounds.")
    return west, south, east, north


def _get_asset_scale_offset(item: Item, band_key: str) -> tuple[float, float]:
    """Read optional scale/offset from the selected asset's STAC metadata.

    Planetary Computer Sentinel-2 assets normally expose raster metadata. If a
    scale is absent, values remain in their published source units rather than
    applying an undocumented conversion.
    """
    asset = item.assets[band_key]
    raster_bands = asset.extra_fields.get("raster:bands", [])
    metadata = raster_bands[0] if raster_bands else {}
    try:
        scale = float(metadata.get("scale", 1.0))
        offset = float(metadata.get("offset", 0.0))
    except (TypeError, ValueError) as error:
        raise RasterPreparationError(
            f"Asset {band_key} has invalid raster scale or offset metadata."
        ) from error
    return scale, offset


def _target_grid(item: Item, bbox: tuple[float, float, float, float]) -> tuple[str, Affine, int, int]:
    """Create a 20 m target grid from B11, clipped to the AOI bounds."""
    if TARGET_GRID_BAND not in item.assets:
        raise RasterPreparationError(f"Required target-grid asset {TARGET_GRID_BAND} is unavailable.")

    try:
        with rasterio.open(item.assets[TARGET_GRID_BAND].href) as reference:
            if reference.crs is None:
                raise RasterPreparationError(f"Asset {TARGET_GRID_BAND} has no CRS metadata.")

            projected_bounds = transform_bounds(
                "EPSG:4326", reference.crs, *bbox, densify_pts=21
            )
            requested_window = from_bounds(
                *projected_bounds,
                transform=reference.transform,
            ).round_offsets().round_lengths()
            dataset_window = Window(0, 0, reference.width, reference.height)
            window = requested_window.intersection(dataset_window)
            if window.width <= 0 or window.height <= 0:
                raise RasterPreparationError("The requested AOI does not intersect the B11 raster.")

            return (
                reference.crs.to_string(),
                reference.window_transform(window),
                int(window.width),
                int(window.height),
            )
    except RasterioError as error:
        raise RasterPreparationError(
            f"Could not open the signed {TARGET_GRID_BAND} COG asset."
        ) from error


def _read_aligned_asset(
    item: Item,
    band_key: str,
    target_crs: str,
    target_transform: Affine,
    width: int,
    height: int,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Read just the target grid from one COG and return scaled float32 data."""
    if band_key not in item.assets:
        raise RasterPreparationError(f"Required Sentinel-2 asset {band_key} is unavailable.")

    scale, offset = _get_asset_scale_offset(item, band_key)
    try:
        with rasterio.open(item.assets[band_key].href) as source:
            # Average is appropriate when lowering 10 m reflectance bands to
            # the native 20 m SWIR target grid. B11/B12 are already 20 m.
            with WarpedVRT(
                source,
                crs=target_crs,
                transform=target_transform,
                width=width,
                height=height,
                resampling=Resampling.average,
            ) as aligned:
                masked = aligned.read(1, masked=True)
    except RasterioError as error:
        raise RasterPreparationError(f"Could not read AOI pixels for asset {band_key}.") from error

    values = masked.astype(np.float32)
    values = values * scale + offset
    return np.ma.filled(values, OUTPUT_NODATA).astype(np.float32), (scale, offset)


def prepare_sentinel2_aoi_stack(
    item: Item,
    bbox: Sequence[float],
    output_path: str | Path,
    band_keys: Sequence[str] = REQUIRED_SPECTRAL_BANDS,
) -> RasterPreparationResult:
    """Write an aligned 20 m float32 spectral stack for a small WGS84 AOI.

    Rasterio opens remote COGs and reads only the requested target window. No
    full Sentinel-2 tile is downloaded by this function.
    """
    aoi_bbox = _validate_aoi_bbox(bbox)
    selected_bands = tuple(band_keys)
    if not selected_bands:
        raise RasterPreparationError("At least one spectral band must be requested.")

    missing_bands = [band for band in selected_bands if band not in item.assets]
    if missing_bands:
        raise RasterPreparationError(f"Selected scene is missing required assets: {missing_bands}")

    target_crs, target_transform, width, height = _target_grid(item, aoi_bbox)
    arrays: list[np.ndarray] = []
    scale_offsets: dict[str, tuple[float, float]] = {}

    # These GDAL options make remote COG access more efficient and avoid
    # unnecessary directory listing requests. Each read remains AOI-windowed.
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
        GDAL_HTTP_MULTIRANGE="YES",
    ):
        for band_key in selected_bands:
            values, scale_offset = _read_aligned_asset(
                item,
                band_key,
                target_crs,
                target_transform,
                width,
                height,
            )
            arrays.append(values)
            scale_offsets[band_key] = scale_offset

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": len(selected_bands),
        "dtype": "float32",
        "crs": target_crs,
        "transform": target_transform,
        "nodata": OUTPUT_NODATA,
        "compress": "deflate",
        "predictor": 3,
    }
    try:
        with rasterio.open(destination, "w", **profile) as output:
            for index, (band_key, values) in enumerate(zip(selected_bands, arrays), start=1):
                output.write(values, index)
                output.set_band_description(index, band_key)
                scale, offset = scale_offsets[band_key]
                output.update_tags(index, source_scale=scale, source_offset=offset)
            output.update_tags(
                scene_id=item.id,
                source_collection=item.collection_id or "",
                processing_note="AOI-only COG reads; aligned to B11 native 20 m grid",
            )
    except RasterioError as error:
        raise RasterPreparationError(f"Could not write processed raster stack: {destination}") from error

    return RasterPreparationResult(
        output_path=destination,
        crs=target_crs,
        width=width,
        height=height,
        bands=selected_bands,
        scale_offsets=scale_offsets,
    )

