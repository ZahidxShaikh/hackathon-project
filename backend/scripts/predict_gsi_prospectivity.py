"""Generate the final GSI-study-area PU manganese prospectivity-ranking raster."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from backend.app.services.ml.pu_prediction import predict_gsi_pu_prospectivity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GSI_OUTPUT_DIRECTORY = PROJECT_ROOT / "backend" / "data" / "processed" / "gsi_49367"
PREDICTION_DIRECTORY = PROJECT_ROOT / "backend" / "data" / "predictions"


def main() -> int:
    try:
        provenance = json.loads((GSI_OUTPUT_DIRECTORY / "provenance.json").read_text(encoding="utf-8"))
        result = predict_gsi_pu_prospectivity(
            GSI_OUTPUT_DIRECTORY / provenance["feature_path"],
            PROJECT_ROOT / "backend" / "models" / "manganese_pu_model.joblib",
            PREDICTION_DIRECTORY / "balaghat_manganese_prospectivity.tif",
            PREDICTION_DIRECTORY / "balaghat_manganese_prospectivity_summary.json",
        )
    except Exception as error:
        print(f"GSI PU PROSPECTIVITY PREDICTION FAILED: {error}", file=sys.stderr)
        return 1
    print("GSI PU PROSPECTIVITY PREDICTION COMPLETE")
    print(f"Output: {result.output_path}")
    print(f"Score min/max/mean/median: {result.minimum:.6f} / {result.maximum:.6f} / {result.mean:.6f} / {result.median:.6f}")
    print(f"Valid pixels: {result.valid_pixels}; NoData pixels: {result.nodata_pixels}")
    print(f"Classification summary: {result.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
