"""STAC metadata search for Sentinel-2 L2A scenes.

This module intentionally stops at scene discovery and metadata inspection.
It does not download, read, resample, or process satellite raster assets.
"""

from __future__ import annotations

from datetime import date
from math import inf
from typing import Any, Sequence

from pystac import Item

from backend.app.config import SENTINEL2_L2A_COLLECTION
from backend.app.services.planetary_computer import open_planetary_computer_catalog


class Sentinel2SearchError(RuntimeError):
    """Raised for invalid Sentinel-2 search inputs or failed searches."""


class NoSuitableSceneError(Sentinel2SearchError):
    """Raised when no cloud-filtered Sentinel-2 L2A item is available."""


def _validate_bbox(bbox: Sequence[float]) -> list[float]:
    if len(bbox) != 4:
        raise Sentinel2SearchError("bbox must contain [west, south, east, north].")

    try:
        west, south, east, north = (float(value) for value in bbox)
    except (TypeError, ValueError) as error:
        raise Sentinel2SearchError("bbox values must be numeric.") from error

    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise Sentinel2SearchError(
            "bbox must have valid longitude/latitude bounds and west < east, south < north."
        )
    return [west, south, east, north]


def _validate_dates(start_date: str, end_date: str) -> tuple[str, str]:
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except ValueError as error:
        raise Sentinel2SearchError("Dates must use ISO format: YYYY-MM-DD.") from error

    if start > end:
        raise Sentinel2SearchError("start_date must be on or before end_date.")
    return start.isoformat(), end.isoformat()


def _cloud_cover(item: Item) -> float:
    """Return cloud cover for sorting; missing values are always ranked last."""
    cloud_cover = item.properties.get("eo:cloud_cover")
    try:
        return float(cloud_cover) if cloud_cover is not None else inf
    except (TypeError, ValueError):
        return inf


def _sort_key(item: Item) -> tuple[float, float]:
    """Sort cloud cover first, then acquisition time without timezone mixing."""
    acquisition_timestamp = item.datetime.timestamp() if item.datetime else inf
    return (_cloud_cover(item), acquisition_timestamp)


def _bbox_overlap_area(item: Item, aoi_bbox: Sequence[float]) -> float:
    """Return a simple WGS84 bbox-overlap area for scene-coverage ranking."""
    if item.bbox is None or len(item.bbox) < 4:
        return 0.0
    west, south, east, north = aoi_bbox
    item_west, item_south, item_east, item_north = item.bbox[:4]
    overlap_width = max(0.0, min(east, item_east) - max(west, item_west))
    overlap_height = max(0.0, min(north, item_north) - max(south, item_south))
    return overlap_width * overlap_height


def search_sentinel2(
    bbox: Sequence[float],
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 20.0,
) -> list[Item]:
    """Return cloud-filtered Sentinel-2 L2A STAC items, lowest cloud first.

    Parameters are intentionally limited to STAC discovery. Returned items
    include their STAC asset metadata but no imagery is downloaded.
    """
    valid_bbox = _validate_bbox(bbox)
    valid_start, valid_end = _validate_dates(start_date, end_date)

    try:
        cloud_limit = float(max_cloud_cover)
    except (TypeError, ValueError) as error:
        raise Sentinel2SearchError("max_cloud_cover must be a number from 0 to 100.") from error
    if not 0 <= cloud_limit <= 100:
        raise Sentinel2SearchError("max_cloud_cover must be between 0 and 100.")

    try:
        catalog = open_planetary_computer_catalog()
        search = catalog.search(
            collections=[SENTINEL2_L2A_COLLECTION],
            bbox=valid_bbox,
            datetime=f"{valid_start}/{valid_end}",
            query={"eo:cloud_cover": {"lte": cloud_limit}},
        )
        items = list(search.items())
    except Sentinel2SearchError:
        raise
    except Exception as error:
        raise Sentinel2SearchError(
            "Sentinel-2 STAC search failed. Check the date range, network, and service status."
        ) from error

    return sorted(items, key=_sort_key)


def select_best_scene(items: Sequence[Item], aoi_bbox: Sequence[float] | None = None) -> Item:
    """Select the best item from a non-empty search result.

    With no AOI argument (the Phase 2 behavior), this returns the lowest-cloud
    scene. When an AOI is supplied, it first maximizes footprint coverage of
    that AOI, then applies the same lowest-cloud tie-breaker. This prevents a
    tiny tile-edge overlap from being selected for raster processing.
    """
    if not items:
        raise NoSuitableSceneError(
            "No Sentinel-2 L2A scenes matched this AOI, date range, and cloud-cover limit."
        )
    if aoi_bbox is None:
        return min(items, key=_sort_key)

    valid_aoi_bbox = _validate_bbox(aoi_bbox)
    return min(items, key=lambda item: (-_bbox_overlap_area(item, valid_aoi_bbox), *_sort_key(item)))


def scene_summary(item: Item) -> dict[str, Any]:
    """Create JSON-friendly metadata for a selected STAC scene."""
    return {
        "item_id": item.id,
        "acquisition_datetime": item.datetime.isoformat() if item.datetime else None,
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "collection": item.collection_id,
        "geometry": item.geometry,
        "assets": [
            {
                "key": key,
                "title": asset.title,
                "roles": asset.roles or [],
                "media_type": asset.media_type,
            }
            for key, asset in item.assets.items()
        ],
    }
