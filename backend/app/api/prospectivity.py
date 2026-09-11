"""Read-only API endpoints for the validated Phase 7 prospectivity result."""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

from backend.app.services.prospectivity_api import (
    PROSPECTIVITY_PATH,
    ProspectivityApiError,
    classified_overlay_png,
    prospectivity_metadata,
    prospectivity_value,
)


router = APIRouter(prefix="/api/prospectivity", tags=["prospectivity"])


def _failure(error: ProspectivityApiError) -> HTTPException:
    return HTTPException(status_code=503, detail=str(error))


@router.get("/metadata")
def metadata() -> dict:
    try:
        return prospectivity_metadata()
    except ProspectivityApiError as error:
        raise _failure(error) from error


@router.get("/overlay.png", response_class=Response)
def overlay() -> Response:
    try:
        return Response(content=classified_overlay_png(), media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})
    except ProspectivityApiError as error:
        raise _failure(error) from error


@router.get("/value")
def value(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
) -> dict:
    try:
        return prospectivity_value(latitude, longitude)
    except ProspectivityApiError as error:
        raise _failure(error) from error


@router.get("/raster")
def raster() -> FileResponse:
    try:
        # The path is fixed to the validated Phase 7 result; callers cannot
        # supply a filesystem path.
        prospectivity_metadata()
    except ProspectivityApiError as error:
        raise _failure(error) from error
    return FileResponse(PROSPECTIVITY_PATH, media_type="image/tiff", filename=PROSPECTIVITY_PATH.name)
