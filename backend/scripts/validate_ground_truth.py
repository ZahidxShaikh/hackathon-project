"""Validate extracted GSI ground-truth GeoJSON without changing its points."""

from __future__ import annotations

from collections import Counter
import json
import sys
from pathlib import Path


PATH = Path("backend/data/ground_truth/manganese_positive.geojson")


def main() -> int:
    try:
        document = json.loads(PATH.read_text(encoding="utf-8"))
        assert document["type"] == "FeatureCollection"
        features = document["features"]
        assert features
        coordinates = []
        sample_types = Counter()
        statuses = Counter()
        missing_mn = missing_mno = 0
        for feature in features:
            assert feature["type"] == "Feature" and feature["geometry"]["type"] == "Point"
            longitude, latitude = feature["geometry"]["coordinates"]
            properties = feature["properties"]
            assert -180 <= longitude <= 180 and -90 <= latitude <= 90
            assert longitude == properties["longitude"] and latitude == properties["latitude"]
            assert properties["commodity"] == "Manganese"
            assert properties["source_report"] and properties["NUID"] == "49367"
            coordinates.append((longitude, latitude))
            sample_types[properties["sample_type"]] += 1
            statuses[properties["validation_status"]] += 1
            missing_mn += properties["Mn"] is None
            missing_mno += properties["MnO"] is None
        duplicate_coordinates = len(coordinates) - len(set(coordinates))
    except (AssertionError, KeyError, OSError, json.JSONDecodeError) as error:
        print(f"GROUND-TRUTH VALIDATION FAILED: {error}", file=sys.stderr)
        return 1
    latitudes = [point[1] for point in coordinates]
    longitudes = [point[0] for point in coordinates]
    print("GROUND-TRUTH VALIDATION PASSED")
    print(f"total points: {len(features)}")
    print(f"valid points: {statuses['valid']}")
    print(f"suspicious points: {statuses['suspicious']}")
    print(f"out-of-study-area points: {statuses['out_of_study_area']}")
    print(f"minimum/maximum latitude: {min(latitudes)} / {max(latitudes)}")
    print(f"minimum/maximum longitude: {min(longitudes)} / {max(longitudes)}")
    print(f"sample-type counts: {dict(sample_types)}")
    print(f"missing Mn: {missing_mn}; missing MnO: {missing_mno}")
    print(f"duplicate coordinate pairs (retained as separate sample records): {duplicate_coordinates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

