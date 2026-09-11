"""Configuration shared by backend modules.

Only STAC-discovery settings are defined in Phase 2. Raster and ML settings
will be introduced in their respective phases.
"""

PLANETARY_COMPUTER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SENTINEL2_L2A_COLLECTION = "sentinel-2-l2a"

# [west, south, east, north] for the small Balaghat study area used in this prototype.
BALAGHAT_BBOX = [80.10, 21.70, 80.30, 21.95]

