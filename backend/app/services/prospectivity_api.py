"""Read-only access to the validated GSI PU prospectivity result for the web API."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import struct
import zlib

import numpy as np
import rasterio
from rasterio.warp import transform, transform_bounds

from backend.app.services.raster import OUTPUT_NODATA


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREDICTIONS_DIRECTORY = PROJECT_ROOT / "backend" / "data" / "predictions"
PROSPECTIVITY_PATH = PREDICTIONS_DIRECTORY / "balaghat_manganese_prospectivity.tif"
SUMMARY_PATH = PREDICTIONS_DIRECTORY / "balaghat_manganese_prospectivity_summary.json"
CLASS_COLORS = ("#1d4ed8", "#38bdf8", "#facc15", "#f97316", "#dc2626")


class ProspectivityApiError(RuntimeError):
    """Raised when the validated prospectivity result cannot be safely served."""


def _require_files() -> None:
    if not PROSPECTIVITY_PATH.is_file() or not SUMMARY_PATH.is_file():
        raise ProspectivityApiError("Validated prospectivity raster or summary metadata is unavailable.")


@lru_cache(maxsize=1)
def _summary() -> dict:
    _require_files()
    try:
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProspectivityApiError("Prospectivity summary metadata is unreadable.") from error


def _classes() -> list[dict]:
    classes = _summary().get("classification", {}).get("classes", [])
    if len(classes) != 5:
        raise ProspectivityApiError("Prospectivity summary must define five display classes.")
    return classes


def _class_index(score: float) -> int:
    upper_bounds = np.asarray([entry["score_upper"] for entry in _classes()[:-1]], dtype=float)
    return int(np.searchsorted(upper_bounds, score, side="right"))


def prospectivity_metadata() -> dict:
    """Return only display and georeferencing metadata for the existing result."""
    _require_files()
    summary = _summary()
    with rasterio.open(PROSPECTIVITY_PATH) as raster:
        if raster.crs is None:
            raise ProspectivityApiError("Prospectivity raster has no CRS.")
        west, south, east, north = transform_bounds(raster.crs, "EPSG:4326", *raster.bounds, densify_pts=21)
        classes = [
            {
                "name": entry["class"],
                "lower": float(entry["score_lower"]),
                "upper": float(entry["score_upper"]),
                "pixel_count": int(entry["pixel_count"]),
                "color": CLASS_COLORS[index],
            }
            for index, entry in enumerate(_classes())
        ]
        return {
            "study_area": "Balaghat, Madhya Pradesh",
            "bounds_wgs84": {"west": west, "south": south, "east": east, "north": north},
            "raster": {
                "crs": raster.crs.to_string(), "width": raster.width, "height": raster.height,
                "resolution_meters": 20, "nodata": raster.nodata,
            },
            "model": "Random Forest PU ranking model",
            "authoritative_gsi_positive_points": 170,
            "feature_count": 10,
            "score_label": "Relative manganese prospectivity score",
            "score_interpretation": summary["score_interpretation"],
            "classes": classes,
            "disclaimer": (
                "Prospectivity ranking based on Sentinel-2 spectral proxies and authoritative GSI occurrences. "
                "It is not a confirmed reserve, ore-grade estimate, economic assessment, or geological confirmation."
            ),
        }


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _rgba_png(rgba: np.ndarray) -> bytes:
    height, width, channels = rgba.shape
    if channels != 4:
        raise ProspectivityApiError("Overlay requires an RGBA image.")
    rows = b"".join(b"\x00" + rgba[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) + _png_chunk(b"IDAT", zlib.compress(rows, 6)) + _png_chunk(b"IEND", b"")


@lru_cache(maxsize=1)
def classified_overlay_png() -> bytes:
    """Render the existing score raster as a transparent five-class PNG overlay."""
    _require_files()
    colors = np.asarray(
        [tuple(int(color[index:index + 2], 16) for index in (1, 3, 5)) for color in CLASS_COLORS], dtype=np.uint8
    )
    with rasterio.open(PROSPECTIVITY_PATH) as raster:
        scores = raster.read(1).astype(np.float32, copy=False)
    valid = np.isfinite(scores) & (scores != OUTPUT_NODATA)
    rgba = np.zeros((*scores.shape, 4), dtype=np.uint8)
    if valid.any():
        indexes = np.searchsorted(
            np.asarray([entry["score_upper"] for entry in _classes()[:-1]], dtype=float), scores[valid], side="right"
        )
        rgba[valid, :3] = colors[indexes]
        rgba[valid, 3] = 175
    return _rgba_png(rgba)


def prospectivity_value(latitude: float, longitude: float) -> dict:
    """Look up a single relative PU ranking score from WGS84 coordinates."""
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ProspectivityApiError("Latitude or longitude is outside valid WGS84 bounds.")
    _require_files()
    with rasterio.open(PROSPECTIVITY_PATH) as raster:
        x, y = transform("EPSG:4326", raster.crs, [longitude], [latitude])
        row, col = raster.index(x[0], y[0])
        if not (0 <= row < raster.height and 0 <= col < raster.width):
            return {"latitude": latitude, "longitude": longitude, "inside_study_area": False, "score": None, "prospectivity_class": None}
        score = float(raster.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])
    if not np.isfinite(score) or score == OUTPUT_NODATA:
        return {"latitude": latitude, "longitude": longitude, "inside_study_area": True, "score": None, "prospectivity_class": None}
    entry = _classes()[_class_index(score)]
    return {
        "latitude": latitude, "longitude": longitude, "inside_study_area": True,
        "score": score, "prospectivity_class": entry["class"],
    }
