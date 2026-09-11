"""Local API contract checks for the read-only Phase 7 prospectivity endpoints."""

from fastapi.testclient import TestClient

from backend.app.main import app


def main() -> int:
    client = TestClient(app)
    metadata = client.get("/api/prospectivity/metadata")
    assert metadata.status_code == 200, metadata.text
    payload = metadata.json()
    assert payload["raster"]["crs"] == "EPSG:32644"
    assert payload["raster"]["width"] == 660 and payload["raster"]["height"] == 599
    assert len(payload["classes"]) == 5
    overlay = client.get("/api/prospectivity/overlay.png")
    assert overlay.status_code == 200 and overlay.headers["content-type"] == "image/png"
    assert overlay.content.startswith(b"\x89PNG\r\n\x1a\n")
    bounds = payload["bounds_wgs84"]
    value = client.get(
        "/api/prospectivity/value",
        params={"latitude": (bounds["south"] + bounds["north"]) / 2, "longitude": (bounds["west"] + bounds["east"]) / 2},
    )
    assert value.status_code == 200, value.text
    point = value.json()
    assert point["inside_study_area"] and 0 <= point["score"] <= 1 and point["prospectivity_class"]
    raster = client.get("/api/prospectivity/raster")
    assert raster.status_code == 200 and len(raster.content) > 0
    print("PROSPECTIVITY API TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
