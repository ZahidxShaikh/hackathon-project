# Manganese Prospectivity Prototype

This project is a local prototype for **model-predicted manganese prospectivity / potential zones** using Sentinel-2 L2A imagery, known and verified manganese occurrence data, and machine learning. It does **not** detect, prove, or quantify manganese reserves. Any result must be validated by geological expertise and field investigation.

## Phase 1 status

Phase 1 is complete: the repository structure and dependency manifests are in place. Satellite access, raster processing, training, predictions, the API, and the dashboard deliberately have not been implemented yet; they will be added and tested one phase at a time.

The existing `backend/test_api.py` and `backend/search_satellite.py` files are retained as the already-tested Planetary Computer experiments. They will be organized into `backend/scripts/` during the relevant later phase, after re-verification.

## Project layout

```text
api/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI endpoints (later phases)
│   │   ├── services/     # STAC, raster, feature, and ML code
│   │   └── utils/        # Shared geospatial helpers
│   ├── data/
│   │   ├── raw/          # Verified input data only
│   │   ├── processed/    # Generated raster/features (ignored by Git)
│   │   └── training/     # Training datasets
│   ├── models/           # Generated trained models (ignored by Git)
│   ├── scripts/          # Reproducible command-line scripts
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── components/
    │   ├── pages/
    │   └── services/
    ├── package.json
    └── vite.config.js
```

## Prerequisites

- Python 3.11–3.13 is recommended for broad geospatial-package compatibility. The project should also be tested against your installed Python version before proceeding.
- Node.js 20 or newer for the React/Vite frontend.
- No Sentinel Hub key is required. Planetary Computer STAC access will be configured in Phase 2.

## Environment setup

From the project root in PowerShell:

```powershell
# Activate the existing virtual environment, or create it first if needed.
.\.venv\Scripts\Activate.ps1
# If activation is blocked by policy, use the interpreter directly instead:
# .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt

python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt

cd frontend
npm install
cd ..
```

## Phase 1 verification

Run these checks after installing dependencies:

```powershell
.\.venv\Scripts\python.exe -c "import fastapi, pystac_client, planetary_computer, rasterio, geopandas, sklearn; print('Backend dependencies ready')"
cd frontend
npm run build
cd ..
```

Expected result: Python prints `Backend dependencies ready`. The frontend build may not succeed until the Phase 13 React entry files are added; `package.json` and Vite configuration are intentionally prepared now.

## Development sequence

1. Project structure and environment setup — complete
2. Planetary Computer connection and Sentinel-2 scene selection — complete
3. Sentinel-2 AOI-only raster preparation — complete
4. Local spectral feature generation — complete
5. Ground-truth-ready ML training and prospectivity pipeline — Phase 5B complete; real training blocked pending verified labels
6. Spatially aware Random Forest training and probability raster
7. FastAPI
8. React + Leaflet dashboard and integration

## Data and scientific safeguards

- Use only verified occurrence/mine locations for positive labels.
- Demo data, if later introduced, will be visibly marked `DEMO ONLY` and never represented as real geology.
- Negative sampling and spatial train/test separation need careful design to avoid misleading results and spatial leakage.
- A prospectivity category is a model output, not a confirmed reserve or mining recommendation.

## Phase 2: select a Sentinel-2 scene

The Phase 2 script searches the public `sentinel-2-l2a` collection for the
Balaghat bounding box, uses a rolling 365-day date range by default, filters
for scenes with cloud cover at or below 20%, and selects the lowest-cloud
scene. It lists STAC asset metadata only; no image data is downloaded.

```powershell
.\.venv\Scripts\python.exe -m backend.scripts.select_best_scene
```

To select a date range explicitly:

```powershell
.\.venv\Scripts\python.exe -m backend.scripts.select_best_scene --start-date 2025-01-01 --end-date 2025-12-31 --max-cloud-cover 20
```

## Phase 3: prepare an aligned AOI raster stack

The Phase 3 script reuses the Phase 2 scene selector, opens only the Balaghat
AOI windows in the signed Sentinel-2 Cloud Optimized GeoTIFFs, and writes a
six-band `float32` GeoTIFF under `backend/data/processed/`. The target grid is
B11 at its native 20 m resolution: 10 m bands are averaged to 20 m, avoiding
an artificial resolution increase in B11/B12. The bands are B02, B03, B04,
B08, B11, and B12. Nodata pixels are written as `-9999`.

```powershell
.\.venv\Scripts\python.exe -m backend.scripts.prepare_balaghat_raster
```

## Phase 4: spectral feature generation

Phase 4 reads the already generated Phase 3 GeoTIFF locally; it does not make
a STAC request or download satellite imagery. It writes a ten-band float32
GeoTIFF in the same directory, retaining the input CRS, transform, extent,
20 m grid, and `-9999` nodata value. The added features are NDVI, NDWI, a
red/blue iron/oxide spectral proxy (`B04 / B02`), and a SWIR ratio (`B11 /
B12`). They are spectral proxies only, **not** direct manganese detection or
reserve confirmation.

```powershell
.\.venv\Scripts\python.exe -m backend.scripts.generate_spectral_features
.\.venv\Scripts\python.exe -m backend.scripts.test_phase4_features
```

## Phase 5B: ground-truth-ready ML pipeline

Phase 5B adds a modular Random Forest baseline without creating any labels.
Real training requires an authoritative dataset in `backend/data/training/` that
satisfies the contract documented in [backend/data/training/README.md](backend/data/training/README.md).
Required feature fields are `ndvi`, `ndmi`, `iron_oxide_proxy`, `swir_ratio`,
and B02/B03/B04/B08/B11/B12, alongside WGS84 `longitude`, `latitude`, and a
documented `label` (`1` authoritative manganese occurrence; `0` confirmed
background). The Phase 4 raster does not include NDMI, so prediction derives it
as `(B08-B11)/(B08+B11)` with guarded division.

```powershell
# Validate a real authoritative input (fails safely if missing/synthetic/invalid)
.\.venv\Scripts\python.exe -m backend.scripts.prepare_training_dataset --input backend\data\training\authoritative_points.geojson

# Train only with sufficient real labels; saves model and metadata on success
.\.venv\Scripts\python.exe -m backend.scripts.train_model --input backend\data\training\validated_training_dataset.csv --source-description "Authoritative source citation"

# Generate prospectivity/proxy output only after a valid model exists
.\.venv\Scripts\python.exe -m backend.scripts.predict_prospectivity --input backend\data\processed\YOUR_20m_features.tif

# Run isolated synthetic mechanics tests; this never writes project labels/models
.\.venv\Scripts\python.exe -m backend.scripts.test_phase5b
```

The output is explicitly **Manganese prospectivity / spectral-geological proxy**.
It is not a confirmed deposit, reserve estimate, ore grade, or economic result.
Metrics are only reported when both classes exist; ROC-AUC is omitted when it is
mathematically invalid. A reproducible stratified baseline split is included,
but spatially separated validation is required before operational interpretation.

## Corrected GSI study-area AOI

The archived original Phase 3/4 Balaghat outputs use `[80.10, 21.70, 80.30,
21.95]` (WGS84). The authoritative GSI NUID 49367 field samples lie west of
that AOI, so the two areas have no spatial overlap. They are retained unchanged
as reference outputs.

The separate configuration at
[gsi_study_aoi.json](backend/data/ground_truth/gsi_study_aoi.json) expands the
GSI report study boundary (`79.6291667–79.7458333 E`, `21.6527778–21.75 N`) by
0.005 degrees. It is used only for the corrected, separately stored GSI
products. The resulting bands and indices are spectral/geological proxies, not
direct manganese detection, reserves, ore grade, or economic assessment.

```powershell
.\.venv\Scripts\python.exe -m backend.scripts.prepare_gsi_study_area
.\.venv\Scripts\python.exe -m backend.scripts.validate_gsi_raster_overlap
```
