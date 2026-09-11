"""Phase 3: create an AOI-only aligned Sentinel-2 spectral stack for Balaghat.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -m backend.scripts.prepare_balaghat_raster

The output is a six-band, float32 GeoTIFF aligned to B11's native 20 m grid.
No whole Sentinel-2 tile is downloaded.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, timedelta
from pathlib import Path

from backend.app.config import BALAGHAT_BBOX
from backend.app.services.raster import (
    REQUIRED_SPECTRAL_BANDS,
    RasterPreparationError,
    prepare_sentinel2_aoi_stack,
)
from backend.app.services.sentinel2 import (
    NoSuitableSceneError,
    Sentinel2SearchError,
    search_sentinel2,
    select_best_scene,
)


def parse_arguments() -> argparse.Namespace:
    today = date.today()
    parser = argparse.ArgumentParser(description="Prepare an aligned Balaghat Sentinel-2 raster stack.")
    parser.add_argument("--start-date", default=(today - timedelta(days=365)).isoformat())
    parser.add_argument("--end-date", default=today.isoformat())
    parser.add_argument("--max-cloud-cover", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=Path("backend/data/processed"))
    return parser.parse_args()


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> int:
    args = parse_arguments()
    try:
        items = search_sentinel2(
            BALAGHAT_BBOX,
            args.start_date,
            args.end_date,
            args.max_cloud_cover,
        )
        # The Phase 2 selector remains lowest-cloud by default. Supplying the
        # AOI here rejects a misleading tile-edge candidate before cloud cover
        # is used as the tie-breaker.
        item = select_best_scene(items, aoi_bbox=BALAGHAT_BBOX)
        output_path = args.output_dir / f"{safe_filename(item.id)}_balaghat_20m_stack.tif"
        result = prepare_sentinel2_aoi_stack(item, BALAGHAT_BBOX, output_path)
    except (Sentinel2SearchError, NoSuitableSceneError, RasterPreparationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("Phase 3 verification summary")
    print(f"Scene ID: {item.id}")
    print(f"Acquisition date: {item.datetime.isoformat() if item.datetime else 'unknown'}")
    print(f"Cloud cover: {item.properties.get('eo:cloud_cover')}%")
    print(f"Bands processed: {', '.join(result.bands)}")
    print(f"Target CRS: {result.crs}")
    print(f"Raster dimensions: {result.width} x {result.height} pixels (20 m target grid)")
    print(f"Output file: {result.output_path.resolve()}")
    print("Access method: AOI-only Cloud Optimized GeoTIFF window reads; no full tile download.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
