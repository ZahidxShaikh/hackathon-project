"""Generate the ten-band Phase 4 feature raster from the corrected local stack.

This script is deliberately local-only: it reads the already prepared GSI AOI
stack and never makes a STAC request or downloads Sentinel-2 imagery.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from backend.app.services.gsi_study_aoi import GSI_FEATURE_DESCRIPTIONS, generate_gsi_study_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIRECTORY = PROJECT_ROOT / "backend" / "data" / "processed" / "gsi_49367"
PROVENANCE_PATH = OUTPUT_DIRECTORY / "provenance.json"


def main() -> int:
    try:
        provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
        stack_path = OUTPUT_DIRECTORY / provenance["stack_path"]
        scene_id = provenance["scene_id"]
        output_path = OUTPUT_DIRECTORY / f"{scene_id}_gsi_49367_phase4_features.tif"
        generate_gsi_study_features(stack_path, output_path)
        provenance["feature_path"] = output_path.name
        provenance["feature_bands"] = [description.split(":", maxsplit=1)[0] for description in GSI_FEATURE_DESCRIPTIONS]
        provenance["feature_processing"] = "Local Phase 4 feature generation from corrected AOI stack; no satellite data downloaded."
        PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    except Exception as error:
        print(f"GSI PHASE 4 FEATURE GENERATION FAILED: {error}", file=sys.stderr)
        return 1
    print("GSI PHASE 4 FEATURE GENERATION COMPLETE")
    print(f"Feature raster: {output_path}")
    print(f"Bands ({len(GSI_FEATURE_DESCRIPTIONS)}): {', '.join(provenance['feature_bands'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
