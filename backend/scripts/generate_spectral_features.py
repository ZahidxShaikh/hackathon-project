"""Phase 4: generate local spectral features from the Phase 3 GeoTIFF.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -m backend.scripts.generate_spectral_features

This script does not search for or download satellite imagery.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.app.services.features import FeatureGenerationError, generate_spectral_features


def latest_phase3_stack(processed_dir: Path) -> Path:
    candidates = sorted(
        processed_dir.glob("*_20m_stack.tif"), key=lambda path: path.stat().st_mtime
    )
    if not candidates:
        raise FeatureGenerationError(
            f"No Phase 3 '*_20m_stack.tif' file exists in {processed_dir}. Run Phase 3 first."
        )
    return candidates[-1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a local Sentinel-2 spectral feature stack.")
    parser.add_argument("--input", type=Path, help="Existing Phase 3 six-band GeoTIFF.")
    parser.add_argument("--output", type=Path, help="Optional output GeoTIFF path.")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        input_path = args.input or latest_phase3_stack(Path("backend/data/processed"))
        result = generate_spectral_features(input_path, args.output)
    except FeatureGenerationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("Phase 4 feature-generation summary")
    print(f"Input GeoTIFF: {result.input_path.resolve()}")
    print(f"Output GeoTIFF: {result.output_path.resolve()}")
    print(f"Feature bands written: {result.feature_count}")
    print(f"CRS: {result.crs}")
    print(f"Dimensions: {result.width} x {result.height}")
    print("Features are spectral proxies only, not a direct manganese detector.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

