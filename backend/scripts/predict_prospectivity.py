"""Generate a prospectivity/proxy raster from a valid trained Phase 5B model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.app.services.ml.model import MODEL_PATH
from backend.app.services.ml.prediction import PredictionError, predict_prospectivity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Phase 4 feature GeoTIFF.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--output", type=Path,
        default=Path("backend/data/predictions/balaghat_manganese_prospectivity.tif"),
    )
    args = parser.parse_args()
    try:
        output = predict_prospectivity(args.input, args.model, args.output)
    except (PredictionError, OSError, ValueError) as error:
        print(f"PREDICTION STOPPED: {error}", file=sys.stderr)
        return 1
    print(f"Manganese prospectivity / spectral-geological proxy: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

