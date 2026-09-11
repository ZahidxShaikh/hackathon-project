const NOMINATIM_URL = "https://nominatim.openstreetmap.org";

function placeFromAddress(address = {}) {
  return {
    country: address.country || "Country unavailable",
    region: address.state || address.region || address.country_code?.toUpperCase() || "Region unavailable",
  };
}

export async function reverseGeocode(latitude, longitude) {
  const response = await fetch(`${NOMINATIM_URL}/reverse?format=jsonv2&lat=${latitude}&lon=${longitude}&zoom=5`, {
    headers: { Accept: "application/json" },
  });
  if (!response.ok) throw new Error("Location lookup failed.");
  const data = await response.json();
  return placeFromAddress(data.address);
}

export async function searchPlace(query) {
  const response = await fetch(`${NOMINATIM_URL}/search?format=jsonv2&limit=1&q=${encodeURIComponent(query)}`, {
    headers: { Accept: "application/json" },
  });
  if (!response.ok) throw new Error("Location search failed.");
  const results = await response.json();
  if (!results.length) throw new Error("No matching location was found.");
  return { latitude: Number(results[0].lat), longitude: Number(results[0].lon), label: results[0].display_name };
}
