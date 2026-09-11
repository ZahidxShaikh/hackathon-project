from pystac_client import Client

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

# Balaghat, Madhya Pradesh
bbox = [80.10, 21.70, 80.30, 21.95]

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2025-01-01/2025-12-31",
    query={"eo:cloud_cover": {"lt": 20}}
)

items = list(search.items())

print("Images found:", len(items))

for item in items[:10]:
    print(
        item.id,
        "| Date:",
        item.datetime,
        "| Cloud:",
        item.properties.get("eo:cloud_cover")
    )