const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function request(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(`Prospectivity API request failed (${response.status}).`);
  return response.json();
}

export const prospectivityApi = {
  metadata: () => request("/api/prospectivity/metadata"),
  value: (latitude, longitude) => request(`/api/prospectivity/value?latitude=${latitude}&longitude=${longitude}`),
  overlayUrl: `${API_BASE}/api/prospectivity/overlay.png`,
};
