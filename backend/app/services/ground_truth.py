"""Extract GSI field-sample locations from a supplied authoritative PDF report.

This module preserves reported coordinates verbatim (after decimal conversion)
and flags, rather than repairs, values outside the report's declared study area.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re

import pdfplumber


REPORT_BOUNDARY = {
    "south": 21 + 39 / 60 + 10 / 3600,  # 21°39'10" N
    "north": 21 + 45 / 60,              # 21°45'00" N
    "west": 79 + 37 / 60 + 45 / 3600,   # 79°37'45" E
    "east": 79 + 44 / 60 + 45 / 3600,   # 79°44'45" E
}
PROJECT_AOI = [80.10, 21.70, 80.30, 21.95]
NUID = "49367"
REPORT_TITLE = (
    "Reconnaissance survey for manganese in Sausar Group of rocks in Jogitola, "
    "Tikari and Dongargaon areas, Balaghat district, Madhya Pradesh"
)


class GroundTruthExtractionError(RuntimeError):
    """Raised when the report cannot be parsed as expected."""


def _sample_key(sample_id: str) -> tuple[str, int]:
    match = re.search(r"/(BRS|PTS|PCS|PS)[_-](\d+)", sample_id.replace(" ", ""), re.I)
    if not match:
        raise GroundTruthExtractionError(f"Could not identify sample type/number: {sample_id}")
    return match.group(1).upper(), int(match.group(2))


def _clean_sample_id(value: str) -> str:
    return re.sub(r"\s+", "", value).replace("-", "_")


def _float_or_none(value: object) -> float | None:
    try:
        return float(str(value).replace("\n", "").strip())
    except (TypeError, ValueError):
        return None


def _status(latitude: float, longitude: float) -> str:
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return "suspicious"
    if not (
        REPORT_BOUNDARY["south"] <= latitude <= REPORT_BOUNDARY["north"]
        and REPORT_BOUNDARY["west"] <= longitude <= REPORT_BOUNDARY["east"]
    ):
        return "out_of_study_area"
    return "valid"


def _extract_location_rows(pdf: pdfplumber.PDF) -> list[dict]:
    """Read Annexure I-IV reported coordinate tables (PDF pages 119-125)."""
    records: list[dict] = []
    # Zero-based indices corresponding to printed report pages 109–115.
    for page_index in range(118, 125):
        for table in pdf.pages[page_index].extract_tables():
            for row in table:
                if not row or not row[0] or "ME/JBP" not in row[0]:
                    continue
                sample_id = _clean_sample_id(row[0])
                try:
                    sample_type, _ = _sample_key(sample_id)
                    latitude, longitude = float(row[1]), float(row[2])
                except (IndexError, ValueError, GroundTruthExtractionError) as error:
                    raise GroundTruthExtractionError(
                        f"Could not parse coordinate row on PDF page {page_index + 1}: {row}"
                    ) from error
                records.append({
                    "sample_id": sample_id,
                    "sample_type": sample_type,
                    "latitude": latitude,
                    "longitude": longitude,
                    "elevation_m": _float_or_none(row[3] if len(row) > 3 else None),
                    "location": (row[4] if len(row) > 4 else "") or None,
                    "page": page_index + 1,
                })
    if len(records) != 172:
        raise GroundTruthExtractionError(f"Expected 172 Annexure I-IV records, extracted {len(records)}.")
    return records


def _extract_mno(pdf: pdfplumber.PDF) -> dict[tuple[str, int], float]:
    """Read MnO (%) from BRS, PTS, and PCS analytical tables where intact."""
    results: dict[tuple[str, int], float] = {}
    # Printed report pages 122–124 (BRS), 131–132 (PTS), 135 (PCS).
    for page_index in [131, 132, 133, 134, 140, 141, 144]:
        for table in pdf.pages[page_index].extract_tables():
            for row in table:
                if len(row) < 20 or not row[1] or "/" not in row[1]:
                    continue
                try:
                    sample_type, number = _sample_key(str(row[1]))
                except GroundTruthExtractionError:
                    continue
                mno = _float_or_none(row[19])  # MnO (%) column in the report table.
                if mno is not None:
                    results[(sample_type, number)] = mno
    return results


def extract_gsi_manganese_ground_truth(pdf_path: str | Path) -> tuple[dict, dict]:
    """Build GeoJSON and metadata dictionaries from the supplied GSI report PDF."""
    source = Path(pdf_path)
    if not source.is_file():
        raise GroundTruthExtractionError(f"GSI source PDF does not exist: {source}")
    with pdfplumber.open(source) as pdf:
        records = _extract_location_rows(pdf)
        mno_results = _extract_mno(pdf)

    features = []
    status_counts = {"valid": 0, "suspicious": 0, "out_of_study_area": 0}
    for record in records:
        sample_type, number = _sample_key(record["sample_id"])
        status = _status(record["latitude"], record["longitude"])
        status_counts[status] += 1
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [record["longitude"], record["latitude"]]},
            "properties": {
                "sample_id": record["sample_id"], "sample_type": sample_type,
                "latitude": record["latitude"], "longitude": record["longitude"],
                "Mn": None, "MnO": mno_results.get((sample_type, number)),
                "location": record["location"], "validation_status": status,
                "source_report": source.name, "page": record["page"], "NUID": NUID,
                "exploration_stage": "G4", "commodity": "Manganese",
                "elevation_m": record["elevation_m"],
            },
        })
    project_aoi_count = sum(
        PROJECT_AOI[0] <= item["properties"]["longitude"] <= PROJECT_AOI[2]
        and PROJECT_AOI[1] <= item["properties"]["latitude"] <= PROJECT_AOI[3]
        for item in features
    )
    metadata = {
        "source_organization": "Geological Survey of India (GSI)",
        "source_portal": "NGDR/Bhu-Chayan", "report_title": REPORT_TITLE,
        "NUID": NUID, "district": "Balaghat", "state": "Madhya Pradesh",
        "commodity": "Manganese", "extraction_date": datetime.now(timezone.utc).isoformat(),
        "original_source_filename": source.name, "coordinate_reference_system": "EPSG:4326",
        "coordinate_field_names": {"latitude": "Latitude", "longitude": "Longitude"},
        "total_records_extracted": len(features), "valid_records": status_counts["valid"],
        "suspicious_records": status_counts["suspicious"],
        "out_of_study_area_records": status_counts["out_of_study_area"],
        "project_aoi_bbox": PROJECT_AOI, "project_aoi_record_count": project_aoi_count,
        "report_study_boundary_wgs84": REPORT_BOUNDARY,
        "filtering_methodology": (
            "All coordinate records from Annexure I (BRS), II (PTS), III (PCS), and IV (PS) "
            "were retained. Coordinates are valid when within the report-declared boundary; "
            "valid global coordinates outside it are flagged out_of_study_area and never repaired. "
            "MnO values are joined only from parseable Annexure VI-VIII analytical tables."
        ),
        "limitation": (
            "The GSI report study block lies west of the existing satellite AOI; therefore zero "
            "records intersect the project AOI. These are field/exploration evidence, not economic reserves."
        ),
    }
    return {"type": "FeatureCollection", "name": "manganese_positive", "features": features}, metadata


def write_ground_truth(pdf_path: str | Path, output_dir: str | Path) -> tuple[Path, Path, dict]:
    geojson, metadata = extract_gsi_manganese_ground_truth(pdf_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    geojson_path = destination / "manganese_positive.geojson"
    metadata_path = destination / "metadata.json"
    geojson_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return geojson_path, metadata_path, metadata
