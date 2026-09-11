# GSI-study-area manganese prospectivity ranking

`balaghat_manganese_prospectivity.tif` is **Manganese prospectivity based on
Sentinel-2 spectral proxies and authoritative GSI manganese occurrences using a
positive-unlabeled Random Forest ranking model.**

Its single Float32 band preserves the corrected GSI-study-area Phase 4 feature
raster grid (EPSG:32644, 20 m). Valid pixels contain a relative
positive-vs-unlabeled ranking score from 0 to 1; `-9999` is NoData. This is not
a calibrated probability of manganese occurrence.

The companion `balaghat_manganese_prospectivity_summary.json` contains score
statistics and display-only Very Low/Low/Moderate/High/Very High classes based
on valid-pixel quintiles. These data-driven classes are not geological
concentration thresholds.

This output is not a confirmed manganese reserve, deposit, ore grade, mineral
tonnage, economic-viability assessment, or geological confirmation. Unlabeled
background samples can include unknown manganese occurrences, and the PU model
was evaluated only with spatial positive-vs-unlabeled surrogate metrics.

```powershell
.\.venv\Scripts\python.exe -m backend.scripts.predict_gsi_prospectivity
.\.venv\Scripts\python.exe -m backend.scripts.validate_gsi_prospectivity
```
