# Authoritative training-data contract

`training_samples.csv` is created only from valid GSI NUID 49367 field/
exploration samples and the already validated corrected-AOI Phase 4 raster.
It is an **authoritative-positive-only** preparation dataset: every row has
`label=1`, meaning authoritative manganese field/exploration evidence.
Unlabeled raster pixels are not included and are not called confirmed
non-manganese or label `0`.

The preparation retains separate samples with coincident coordinates when their
GSI sample IDs differ. A future train/validation design must group or otherwise
handle these locations to avoid spatial leakage. Model training is intentionally
blocked until a separately documented source supplies confirmed negative data
or an approved background-sampling protocol is established.

## Phase 6 positive-unlabeled workflow

`background_samples.csv` contains reproducibly sampled, valid Phase 4 raster
pixels outside a 300 m buffer around every authoritative GSI positive. They are
explicitly **unlabeled/background**, not confirmed non-manganese samples.
`pu_training_dataset.csv` combines those rows with the authoritative positives;
its `pu_training_target=0` on background rows is solely a surrogate target for
positive-unlabeled learning. `MnO` remains metadata and is never used as a
feature.

The Random Forest is a positive-vs-unlabeled spectral-proxy ranker. Spatial
GroupKFold blocks (1 km) are used for its initial validation. Resulting metrics
measure separation of known positives from unlabeled pixels; they are not
confirmed-deposit accuracy or calibrated manganese-occurrence probability.

```powershell
.\.venv\Scripts\python.exe -m backend.scripts.prepare_pu_dataset
.\.venv\Scripts\python.exe -m backend.scripts.train_pu_model
.\.venv\Scripts\python.exe -m backend.scripts.validate_pu_model
```

Create and validate the current dataset locally:

```powershell
.\.venv\Scripts\python.exe -m backend.scripts.prepare_gsi_training_dataset
.\.venv\Scripts\python.exe -m backend.scripts.validate_gsi_training_dataset
```

## Generic future training-data contract

Place a verified CSV, GeoJSON, GeoPackage, or Shapefile here only after an
authoritative source has been obtained. The generic pipeline will never create
labels.

Required fields (CSV uses EPSG:4326 longitude/latitude):

```text
longitude,latitude,label,ndvi,ndmi,iron_oxide_proxy,swir_ratio,B02,B03,B04,B08,B11,B12
```

- `label=1`: confirmed, authoritative manganese occurrence.
- `label=0`: confirmed non-manganese/background location, documented by source.
- Optional geology attributes must start with `geology_`; they are preserved for
  later feature selection but are not used by the Phase 5B baseline.
- Never use the synthetic test fixture as project ground truth.

Spatially separate training and validation data before operational reporting;
the included split is a reproducible baseline, not evidence of generalization.
