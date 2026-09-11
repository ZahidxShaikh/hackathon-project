"""Phase 2: find the lowest-cloud Sentinel-2 L2A scene for Balaghat.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -m backend.scripts.select_best_scene

This script prints STAC metadata only. It does not download any satellite asset.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from backend.app.config import BALAGHAT_BBOX
from backend.app.services.sentinel2 import (
    NoSuitableSceneError,
    Sentinel2SearchError,
    scene_summary,
    search_sentinel2,
    select_best_scene,
)


def parse_arguments() -> argparse.Namespace:
    today = date.today()
    default_start = today - timedelta(days=365)
    parser = argparse.ArgumentParser(
        description="Select the lowest-cloud Sentinel-2 L2A STAC scene for Balaghat."
    )
    parser.add_argument("--start-date", default=default_start.isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=today.isoformat(), help="YYYY-MM-DD")
    parser.add_argument(
        "--max-cloud-cover",
        type=float,
        default=20.0,
        help="Maximum scene cloud cover percentage (default: 20).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        items = search_sentinel2(
            bbox=BALAGHAT_BBOX,
            start_date=args.start_date,
            end_date=args.end_date,
            max_cloud_cover=args.max_cloud_cover,
        )
        best_scene = select_best_scene(items)
    except (Sentinel2SearchError, NoSuitableSceneError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    summary = scene_summary(best_scene)
    print(f"Search area (Balaghat bbox): {BALAGHAT_BBOX}")
    print(f"Search dates: {args.start_date} to {args.end_date}")
    print(f"Cloud-cover limit: {args.max_cloud_cover}%")
    print(f"Suitable scenes found: {len(items)}")
    print("\nSelected best scene (lowest cloud cover):")
    print(f"Item ID: {summary['item_id']}")
    print(f"Acquisition date/time: {summary['acquisition_datetime']}")
    print(f"Cloud cover: {summary['cloud_cover']}%")
    print(f"Collection: {summary['collection']}")
    print("\nAvailable assets/bands (metadata only; no assets downloaded):")
    for asset in summary["assets"]:
        title = asset["title"] or "(no title)"
        roles = ", ".join(asset["roles"]) or "(no roles)"
        media_type = asset["media_type"] or "(not specified)"
        print(f"- {asset['key']}: {title} | roles: {roles} | type: {media_type}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

