"""Create a separate Sentinel-2 stack and proxy features for GSI NUID 49367."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

from backend.app.services.gsi_study_aoi import (
    GSI_AOI_CONFIG_NAME,
    derive_gsi_study_aoi,
    generate_gsi_study_features,
    write_gsi_aoi_config,
)
from backend.app.services.raster import prepare_sentinel2_aoi_stack
from backend.app.services.sentinel2 import search_sentinel2, select_best_scene


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH = PROJECT_ROOT / "backend" / "data" / "ground_truth" / "manganese_positive.geojson"
GROUND_TRUTH_METADATA = PROJECT_ROOT / "backend" / "data" / "ground_truth" / "metadata.json"
OUTPUT_DIRECTORY = PROJECT_ROOT / "backend" / "data" / "processed" / "gsi_49367"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=(date.today() - timedelta(days=365)).isoformat())
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--max-cloud-cover", type=float, default=19.999, help="Strictly below 20 percent.")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    try:
        aoi = derive_gsi_study_aoi(GROUND_TRUTH, GROUND_TRUTH_METADATA)
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        aoi_config = write_gsi_aoi_config(
            aoi, PROJECT_ROOT / "backend" / "data" / "ground_truth" / GSI_AOI_CONFIG_NAME
        )
        items = search_sentinel2(aoi.bbox, args.start_date, args.end_date, args.max_cloud_cover)
        item = select_best_scene(items, aoi.bbox)
        cloud_cover = float(item.properties["eo:cloud_cover"])
        stack_path = OUTPUT_DIRECTORY / f"{item.id}_gsi_49367_20m_stack.tif"
        features_path = OUTPUT_DIRECTORY / f"{item.id}_gsi_49367_phase4_features.tif"
        result = prepare_sentinel2_aoi_stack(item, aoi.bbox, stack_path)
        generate_gsi_study_features(result.output_path, features_path)
        provenance = {
            "purpose": "Corrected AOI Sentinel-2 preprocessing for GSI NUID 49367; archived original Balaghat Phase 3/4 outputs are not modified.",
            "why_original_aoi_was_incorrect": "The original [80.10, 21.70, 80.30, 21.95] AOI has zero overlap with the GSI report study block west of it.",
            "ground_truth_source": "Geological Survey of India report NUID 49367, validated field/exploration samples",
            "aoi_config": str(aoi_config.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "corrected_aoi_bbox_epsg4326": list(aoi.bbox),
            "report_study_boundary_epsg4326": list(aoi.report_boundary),
            "valid_ground_truth_points_inside_aoi": aoi.valid_ground_truth_count,
            "scene_id": item.id,
            "acquisition_datetime": item.datetime.isoformat() if item.datetime else None,
            "cloud_cover_percent": cloud_cover,
            "collection": item.collection_id,
            "target_crs": result.crs,
            "target_resolution_m": 20,
            "raster_width": result.width,
            "raster_height": result.height,
            "bands_processed": list(result.bands),
            "stack_path": stack_path.name,
            "feature_path": features_path.name,
            "feature_bands": ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "NDMI", "Iron/Oxide Proxy", "SWIR Ratio"],
            "remote_access": "Signed Planetary Computer COG assets read only for the AOI window; no complete tile download.",
        }
        provenance_path = OUTPUT_DIRECTORY / "provenance.json"
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        (OUTPUT_DIRECTORY / "README.md").write_text(
            "# Corrected GSI study-area raster outputs\n\n"
            "These products are separate from the archived original Balaghat Phase 3/4 outputs. "
            "The prior AOI (80.10–80.30 E) did not overlap the authoritative GSI NUID 49367 study block. "
            "This AOI uses the GSI report boundary expanded by 0.005 degrees and is verified to contain all valid GSI samples. "
            "See `provenance.json` for the selected Sentinel-2 scene, acquisition date, cloud cover, CRS, grid, and output names.\n\n"
            "The spectral bands and indices are spectral/geological proxies only; they do not identify confirmed manganese deposits or reserves.\n",
            encoding="utf-8",
        )
    except Exception as error:
        print(f"GSI STUDY-AREA PREPARATION FAILED: {error}", file=sys.stderr)
        return 1

    print("GSI STUDY-AREA PREPARATION COMPLETE")
    print(f"Corrected AOI (EPSG:4326): {list(aoi.bbox)}")
    print(f"Valid GSI points in corrected AOI: {aoi.valid_ground_truth_count}")
    print(f"Scene ID: {item.id}")
    print(f"Acquisition: {item.datetime.isoformat() if item.datetime else 'unknown'}")
    print(f"Cloud cover: {cloud_cover}%")
    print(f"Stack: {stack_path}")
    print(f"Features: {features_path}")
    print(f"Grid: {result.crs}, {result.width} x {result.height}, 20 m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
