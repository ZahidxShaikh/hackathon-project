"""Spectral feature generation from an already processed Sentinel-2 stack.

These features are spectral proxies only. They can support later exploratory
analysis, but none is a direct manganese detector or geological confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioError

from backend.app.services.raster import OUTPUT_NODATA, REQUIRED_SPECTRAL_BANDS


FEATURE_BAND_DESCRIPTIONS = (
    "B02: Sentinel-2 Blue reflectance (20 m aligned)",
    "B03: Sentinel-2 Green reflectance (20 m aligned)",
    "B04: Sentinel-2 Red reflectance (20 m aligned)",
    "B08: Sentinel-2 NIR reflectance (20 m aligned)",
    "B11: Sentinel-2 SWIR1 reflectance (20 m native target grid)",
    "B12: Sentinel-2 SWIR2 reflectance (20 m native target grid)",
    "NDVI: (B08 - B04) / (B08 + B04)",
    "NDWI: (B03 - B08) / (B03 + B08)",
    "Iron/Oxide Proxy: B04 / B02 red-to-blue spectral ratio",
    "SWIR Ratio: B11 / B12",
)


class FeatureGenerationError(RuntimeError):
    """Raised when an input stack is missing or unsuitable for feature creation."""


@dataclass(frozen=True)
class FeatureGenerationResult:
    """Metadata for a written Phase 4 feature GeoTIFF."""

    input_path: Path
    output_path: Path
    width: int
    height: int
    crs: str
    feature_count: int


def feature_output_path(input_path: str | Path) -> Path:
    """Return the standard feature-stack name alongside a Phase 3 input stack."""
    source = Path(input_path)
    suffix = "_20m_stack"
    if not source.stem.endswith(suffix):
        raise FeatureGenerationError(
            "Input filename must end with '_20m_stack.tif' so its feature output can be named safely."
        )
    return source.with_name(f"{source.stem.removesuffix(suffix)}_20m_features.tif")


def _input_band_indexes(dataset: rasterio.DatasetReader) -> dict[str, int]:
    descriptions = dataset.descriptions
    indexes = {description: index for index, description in enumerate(descriptions, start=1) if description}
    missing = [band for band in REQUIRED_SPECTRAL_BANDS if band not in indexes]
    if missing:
        raise FeatureGenerationError(
            f"Input GeoTIFF is missing expected Phase 3 band descriptions: {', '.join(missing)}."
        )
    return indexes


def _read_valid_float32(dataset: rasterio.DatasetReader, band_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Read one source band as float32 and return data plus its valid-pixel mask."""
    data = dataset.read(band_index).astype(np.float32, copy=False)
    valid = np.isfinite(data) & (data != OUTPUT_NODATA)
    clean = np.where(valid, data, OUTPUT_NODATA).astype(np.float32)
    return clean, valid


def _safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Calculate a ratio without zero division, NaN, or infinity output."""
    result = np.full(numerator.shape, OUTPUT_NODATA, dtype=np.float32)
    calculation_mask = valid & np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    np.divide(numerator, denominator, out=result, where=calculation_mask)
    result[~np.isfinite(result)] = OUTPUT_NODATA
    return result


def generate_spectral_features(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> FeatureGenerationResult:
    """Create a 10-band float32 feature GeoTIFF from a Phase 3 Sentinel-2 stack.

    The input's CRS, transform, dimensions, resolution, extent, and nodata
    convention are retained. No satellite data is downloaded in this step.
    """
    source_path = Path(input_path)
    if not source_path.is_file():
        raise FeatureGenerationError(f"Input Phase 3 GeoTIFF does not exist: {source_path}")
    destination = Path(output_path) if output_path else feature_output_path(source_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(source_path) as source:
            if source.crs is None:
                raise FeatureGenerationError("Input GeoTIFF has no CRS.")
            if source.nodata != OUTPUT_NODATA:
                raise FeatureGenerationError(
                    f"Input GeoTIFF nodata must be {OUTPUT_NODATA}, found {source.nodata}."
                )
            band_indexes = _input_band_indexes(source)
            spectral_data: dict[str, np.ndarray] = {}
            valid_masks: dict[str, np.ndarray] = {}
            for band_name in REQUIRED_SPECTRAL_BANDS:
                spectral_data[band_name], valid_masks[band_name] = _read_valid_float32(
                    source, band_indexes[band_name]
                )

            # Each feature uses its own source-band validity mask. Invalid or
            # zero-denominator pixels remain OUTPUT_NODATA instead of becoming
            # NaN or infinity.
            ndvi = _safe_ratio(
                spectral_data["B08"] - spectral_data["B04"],
                spectral_data["B08"] + spectral_data["B04"],
                valid_masks["B08"] & valid_masks["B04"],
            )
            ndwi = _safe_ratio(
                spectral_data["B03"] - spectral_data["B08"],
                spectral_data["B03"] + spectral_data["B08"],
                valid_masks["B03"] & valid_masks["B08"],
            )
            iron_oxide_proxy = _safe_ratio(
                spectral_data["B04"],
                spectral_data["B02"],
                valid_masks["B04"] & valid_masks["B02"],
            )
            swir_ratio = _safe_ratio(
                spectral_data["B11"],
                spectral_data["B12"],
                valid_masks["B11"] & valid_masks["B12"],
            )

            output_arrays = [
                *(spectral_data[band] for band in REQUIRED_SPECTRAL_BANDS),
                ndvi,
                ndwi,
                iron_oxide_proxy,
                swir_ratio,
            ]
            profile = source.profile.copy()
            source_width = source.width
            source_height = source.height
            source_crs = source.crs.to_string()
            profile.update(
                driver="GTiff",
                count=len(output_arrays),
                dtype="float32",
                nodata=OUTPUT_NODATA,
                compress="deflate",
                predictor=3,
            )

            with rasterio.open(destination, "w", **profile) as output:
                for index, (description, values) in enumerate(
                    zip(FEATURE_BAND_DESCRIPTIONS, output_arrays), start=1
                ):
                    output.write(values.astype(np.float32, copy=False), index)
                    output.set_band_description(index, description.split(":", maxsplit=1)[0])
                    output.update_tags(index, description=description)
                output.update_tags(
                    source_phase3_stack=str(source_path.name),
                    feature_disclaimer=(
                        "NDVI, NDWI, red/blue iron-oxide proxy, and SWIR ratio are spectral "
                        "proxies only; they are not a direct manganese detector or reserve estimate."
                    ),
                    feature_processing="float32; nodata=-9999; guarded division",
                )
    except RasterioError as error:
        raise FeatureGenerationError(f"Could not read or write feature GeoTIFF: {error}") from error

    return FeatureGenerationResult(
        input_path=source_path,
        output_path=destination,
        width=source_width,
        height=source_height,
        crs=source_crs,
        feature_count=len(FEATURE_BAND_DESCRIPTIONS),
    )
