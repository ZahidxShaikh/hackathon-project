"""Extract and validate field-coordinate evidence from the supplied GSI report."""

from __future__ import annotations

import sys
from pathlib import Path

from backend.app.services.ground_truth import GroundTruthExtractionError, write_ground_truth

PDF_PATH = Path("backend/data/raw/gsi_manganese_report/CRO-24228-2024/TEXT/20260724162936.935_CR_Final_Report_49367_2024_25_JBP.pdf")
OUTPUT_DIR = Path("backend/data/ground_truth")


def main() -> int:
    try:
        geojson, metadata, summary = write_ground_truth(PDF_PATH, OUTPUT_DIR)
    except GroundTruthExtractionError as error:
        print(f"EXTRACTION FAILED: {error}", file=sys.stderr)
        return 1
    print(f"GeoJSON: {geojson.resolve()}")
    print(f"Metadata: {metadata.resolve()}")
    print(f"Extracted: {summary['total_records_extracted']}")
    print(f"Valid: {summary['valid_records']}; suspicious: {summary['suspicious_records']}; out-of-study-area: {summary['out_of_study_area_records']}")
    print(f"Project AOI records: {summary['project_aoi_record_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

