import { useEffect, useMemo, useState } from "react";
import { CircleMarker, ImageOverlay, MapContainer, Rectangle, TileLayer, useMap, useMapEvents } from "react-leaflet";
import { prospectivityApi } from "./services/prospectivityApi";
import { reverseGeocode, searchPlace } from "./services/locationSearch";

const NAVIGATION = ["Explore", "Targets", "Analytics", "Methodology"];
const FEATURES = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "NDMI", "Iron/Oxide Proxy", "SWIR Ratio"];
const DATA_SOURCES = ["Sentinel-2 L2A", "GSI / NGDR"];

function buildRegionCatalog(metadata) {
  return [
    { id: "global", label: "Global", status: "Available", detail: "World map navigation", center: [18, 0], zoom: 2 },
    { id: "balaghat", label: "India · Balaghat Pilot", status: "Available", detail: `Validated pilot: ${metadata.study_area}`, center: [21.7, 79.69], zoom: 12, pilot: true },
    { id: "india", label: "India", status: "In Development", detail: "Pilot model currently limited to Balaghat", center: [20.59, 78.96], zoom: 5 },
    { id: "south-africa", label: "South Africa", status: "No Model Available", detail: "Data/model not yet available", center: [-30.56, 22.94], zoom: 5 },
    { id: "australia", label: "Australia", status: "No Model Available", detail: "Data/model not yet available", center: [-25.27, 133.78], zoom: 4 },
    { id: "gabon", label: "Gabon", status: "No Model Available", detail: "Data/model not yet available", center: [-0.8, 11.61], zoom: 6 },
    { id: "brazil", label: "Brazil", status: "No Model Available", detail: "Data/model not yet available", center: [-14.24, -51.93], zoom: 4 },
    { id: "ukraine", label: "Ukraine", status: "No Model Available", detail: "Data/model not yet available", center: [48.38, 31.17], zoom: 5 },
    { id: "ghana", label: "Ghana", status: "No Model Available", detail: "Data/model not yet available", center: [7.95, -1.02], zoom: 6 },
    { id: "china", label: "China", status: "No Model Available", detail: "Data/model not yet available", center: [35.86, 104.2], zoom: 4 },
  ];
}

function MapViewport({ view, pilotBounds }) {
  const map = useMap();
  useEffect(() => {
    if (view?.pilot) map.fitBounds(pilotBounds, { padding: [34, 34] });
    else if (view) map.flyTo(view.center, view.zoom, { duration: 0.7 });
  }, [map, view, pilotBounds]);
  return null;
}

function MapClicks({ onSelect }) {
  useMapEvents({ click: ({ latlng }) => onSelect(latlng.lat, latlng.lng) });
  return null;
}

function formatScore(value) { return typeof value === "number" ? value.toFixed(6) : "—"; }
function scoreOutOfHundred(value) { return typeof value === "number" ? `${(value * 100).toFixed(1)} / 100` : "—"; }
function DetailRows({ children }) { return <dl className="detail-rows">{children}</dl>; }

function RegionSelector({ regions, activeRegion, onChange }) {
  return <section className="region-selector">
    <div><p className="micro-label">EXPLORE REGION</p><h2>Global coverage, honest availability</h2></div>
    <select value={activeRegion.id} onChange={(event) => onChange(regions.find((region) => region.id === event.target.value))}>
      {regions.map((region) => <option key={region.id} value={region.id}>{region.label} — {region.status}</option>)}
    </select>
    <div className={`availability-pill ${activeRegion.status.toLowerCase().replaceAll(" ", "-")}`}><i />{activeRegion.status}<span>{activeRegion.detail}</span></div>
  </section>;
}

function LayerControl({ layers, setLayers, activeRegion }) {
  const pilotAvailable = activeRegion.pilot;
  return <div className="layer-control" aria-label="Map layers">
    <p>MAP LAYERS</p>
    <label><input type="checkbox" checked readOnly /> <span className="layer-dot base" />OpenStreetMap</label>
    <label className={!pilotAvailable ? "unavailable" : ""}><input type="checkbox" checked={layers.prospectivity} disabled={!pilotAvailable} onChange={(event) => setLayers({ ...layers, prospectivity: event.target.checked })} /> <span className="layer-dot prospectivity" />Manganese Prospectivity <em>{pilotAvailable ? "pilot available" : "no model"}</em></label>
    <label className="unavailable"><input type="checkbox" disabled /> <span className="layer-dot occurrences" />Known Manganese Occurrences <em>not exposed by API</em></label>
    <label><input type="checkbox" checked={layers.studyAreas} onChange={(event) => setLayers({ ...layers, studyAreas: event.target.checked })} /> <span className="layer-dot study" />Study Areas</label>
    <label><input type="checkbox" checked readOnly /> <span className="layer-dot boundaries" />Country Boundaries <em>OpenStreetMap</em></label>
  </div>;
}

function GlobalMap({ metadata, activeRegion, selectedLocation, onSelect }) {
  const boundsData = metadata.bounds_wgs84;
  const pilotBounds = useMemo(
    () => [[boundsData.south, boundsData.west], [boundsData.north, boundsData.east]],
    [boundsData.south, boundsData.west, boundsData.north, boundsData.east],
  );
  const [layers, setLayers] = useState({ prospectivity: true, studyAreas: true });
  const selectedClass = metadata.classes.find((item) => item.name === selectedLocation?.prospectivity_class);
  return <div className="map-card">
    <div className="map-card-head"><div><p className="micro-label">GLOBAL EXPLORATION MAP</p><h2>World manganese intelligence coverage</h2></div><span className="map-chip">OpenStreetMap base</span></div>
    <div className="map-wrap"><MapContainer className="map global-map" center={[18, 0]} zoom={2} minZoom={2} worldCopyJump scrollWheelZoom>
      <TileLayer attribution="© OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      {activeRegion.pilot && layers.prospectivity && <ImageOverlay url={prospectivityApi.overlayUrl} bounds={pilotBounds} opacity={0.81} />}
      {layers.studyAreas && <Rectangle bounds={pilotBounds} pathOptions={{ color: "#cf6935", weight: 2, fillColor: "#d8793f", fillOpacity: activeRegion.pilot ? 0.04 : 0, dashArray: "5 6" }} />}
      <MapViewport view={activeRegion} pilotBounds={pilotBounds} /><MapClicks onSelect={onSelect} />
      {selectedLocation && <CircleMarker center={[selectedLocation.latitude, selectedLocation.longitude]} radius={7} pathOptions={{ color: "#ffffff", weight: 3, fillColor: selectedClass?.color || "#142c4a", fillOpacity: 1 }} />}
    </MapContainer><LayerControl layers={layers} setLayers={setLayers} activeRegion={activeRegion} /></div>
    <div className="map-footer"><span>◎</span> Select Balaghat Pilot to view the validated raster overlay; all other regions remain unmodelled.</div>
  </div>;
}

function SelectedLocation({ metadata, selectedLocation }) {
  const selectedClass = metadata.classes.find((item) => item.name === selectedLocation?.prospectivity_class);
  return <section className="target-panel global-selected">
    <div className="card-head"><div><p className="micro-label">SELECTED LOCATION</p><h2>Location intelligence</h2></div><span className="target-icon">⌖</span></div>
    {!selectedLocation && <p className="empty-state">Click anywhere on the world map or search for a location.</p>}
    {selectedLocation?.loading && <p className="empty-state">Retrieving location and pilot coverage…</p>}
    {selectedLocation?.error && <p className="empty-state error-text">{selectedLocation.error}</p>}
    {selectedLocation && !selectedLocation.loading && !selectedLocation.error && <>
      <DetailRows><div><dt>Latitude</dt><dd>{selectedLocation.latitude.toFixed(6)}°</dd></div><div><dt>Longitude</dt><dd>{selectedLocation.longitude.toFixed(6)}°</dd></div><div><dt>Country</dt><dd>{selectedLocation.country || "Country unavailable"}</dd></div><div><dt>Region</dt><dd>{selectedLocation.region || "Region unavailable"}</dd></div></DetailRows>
      {selectedLocation.inside_study_area ? <>
        <div className="score-hero" style={{ borderColor: selectedClass?.color || "#94a3b8" }}><small>RELATIVE MANGANESE PROSPECTIVITY SCORE</small><strong>{scoreOutOfHundred(selectedLocation.score)}</strong><span>Positive-vs-unlabeled pilot ranking</span></div>
        <DetailRows><div><dt>Relative score</dt><dd>{formatScore(selectedLocation.score)}</dd></div><div><dt>Class</dt><dd className="class-value"><i style={{ background: selectedClass?.color || "#94a3b8" }} />{selectedLocation.prospectivity_class || "NoData"}</dd></div></DetailRows>
      </> : <div className="no-model-message"><strong>Prospectivity model is not currently available for this region.</strong><span>Only the validated Balaghat pilot currently has a prediction raster.</span></div>}
    </>}
  </section>;
}

function ProspectivityLegend({ metadata }) {
  return <section className="scale-card"><div><p className="micro-label">PILOT PROSPECTIVITY SCALE</p><h2>Relative ranking, low to high</h2></div><div className="gradient-legend"><div className="gradient-bar" /><div className="gradient-labels"><span>LOW</span><span>HIGH</span></div><div className="class-scale">{metadata.classes.map((item) => <div key={item.name}><i style={{ background: item.color }} /><strong>{item.name}</strong><small>{item.lower.toFixed(6)}–{item.upper.toFixed(6)}</small></div>)}</div></div><p className="scale-note">Relative prospectivity ranking — not a confirmed mineral reserve or ore-grade estimate.</p></section>;
}

function LocationSearch({ onSelect }) {
  const [query, setQuery] = useState("");
  const [coordinates, setCoordinates] = useState({ latitude: "", longitude: "" });
  const [error, setError] = useState(null);
  const submitCoordinates = (event) => { event.preventDefault(); const latitude = Number(coordinates.latitude); const longitude = Number(coordinates.longitude); if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) { setError("Enter valid latitude and longitude."); return; } setError(null); onSelect(latitude, longitude); };
  const submitSearch = async (event) => { event.preventDefault(); if (!query.trim()) return; setError(null); try { const place = await searchPlace(query); setCoordinates({ latitude: place.latitude.toFixed(6), longitude: place.longitude.toFixed(6) }); onSelect(place.latitude, place.longitude); } catch { setError("No matching location was found. Try a place name or coordinates."); } };
  return <section className="location-card global-search"><div><p className="micro-label">LOCATION SEARCH</p><h2>Explore a location</h2><p>Search the world map, enter coordinates, or click anywhere to inspect coverage.</p></div><form onSubmit={submitSearch} className="place-search"><label>Search place<input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="e.g. Balaghat, India" /></label><button className="primary-button">Search</button></form><form onSubmit={submitCoordinates} className="coordinate-form"><label>Latitude (°N)<input value={coordinates.latitude} onChange={(event) => setCoordinates({ ...coordinates, latitude: event.target.value })} inputMode="decimal" placeholder="21.700000" /></label><label>Longitude (°E)<input value={coordinates.longitude} onChange={(event) => setCoordinates({ ...coordinates, longitude: event.target.value })} inputMode="decimal" placeholder="79.690000" /></label><button className="outline-button">Evaluate</button></form>{error && <span className="form-error">{error}</span>}<small className="search-note">Place search and country labels use OpenStreetMap Nominatim when available. Scores are queried only from the existing pilot API.</small></section>;
}

function DataAvailability({ regions }) {
  return <section className="availability-card"><div><p className="micro-label">DATA AVAILABILITY</p><h2>Coverage status</h2></div><div className="availability-list">{regions.filter((region) => region.id !== "global").map((region) => <div key={region.id}><span className={`availability-symbol ${region.status.toLowerCase().replaceAll(" ", "-")}`}>●</span><strong>{region.label}</strong><small>{region.pilot ? "Pilot model available" : region.status === "In Development" ? "Pilot model only; regional coverage in development" : "Data/model not yet available"}</small></div>)}</div></section>;
}

function ExplorePage({ metadata, regions, activeRegion, setActiveRegion, selectedLocation, setSelectedLocation }) {
  const selectLocation = async (latitude, longitude) => {
    setSelectedLocation({ loading: true, latitude, longitude });
    const [placeResult, scoreResult] = await Promise.allSettled([reverseGeocode(latitude, longitude), prospectivityApi.value(latitude, longitude)]);
    const place = placeResult.status === "fulfilled" ? placeResult.value : {};
    const score = scoreResult.status === "fulfilled" ? scoreResult.value : { inside_study_area: false, score: null, prospectivity_class: null };
    setSelectedLocation({ latitude, longitude, ...place, ...score });
  };
  return <><RegionSelector regions={regions} activeRegion={activeRegion} onChange={setActiveRegion} /><section className="explore-layout"><GlobalMap metadata={metadata} activeRegion={activeRegion} selectedLocation={selectedLocation} onSelect={selectLocation} /><SelectedLocation metadata={metadata} selectedLocation={selectedLocation} /></section><ProspectivityLegend metadata={metadata} /><LocationSearch onSelect={selectLocation} /><DataAvailability regions={regions} /></>;
}

function TargetsPage() { const [filter, setFilter] = useState("Global"); return <section className="page-card targets-page"><p className="micro-label">GLOBAL EXPLORATION TARGETS</p><h2>Global Exploration Targets</h2><p className="page-subtitle">Only targets generated from available prediction data are eligible for display.</p><div className="filter-row">{["Global", "Country", "Region", "High Prospectivity", "Very High Prospectivity", "Greenfield", "Near Known Occurrences"].map((item) => <button key={item} onClick={() => setFilter(item)} className={filter === item ? "active-filter" : ""}>{item}</button>)}</div><div className="unavailable-card"><span>⌁</span><div><h3>Pilot-region targets not catalogued</h3><p>The existing API provides the Balaghat pilot raster and location-query results, but no extracted target catalogue, occurrence-proximity data, or CSV export. No global targets are displayed or fabricated.</p></div></div><button className="secondary-button" disabled title="No target-list export is exposed by the current API.">Download Top Targets CSV</button></section>; }

function AnalyticsPage({ metadata }) { const maxPixels = Math.max(...metadata.classes.map((item) => item.pixel_count)); return <section className="analytics-page"><div className="page-intro"><p className="micro-label">GLOBAL-READY ANALYTICS</p><h2>Exploration Analytics</h2><p className="page-subtitle">Only the validated Balaghat pilot distribution is currently available through the API.</p></div><div className="analytics-grid"><section className="page-card"><p className="micro-label">SPECTRAL SIGNATURES</p><h3>Pilot feature set</h3><div className="feature-tags">{FEATURES.map((feature) => <span key={feature}>{feature}</span>)}</div><p className="muted-copy">Ten validated Sentinel-2 spectral/geological proxy features are used for the Balaghat pilot model.</p></section><section className="page-card"><p className="micro-label">PROSPECTIVITY DISTRIBUTION</p><h3>Balaghat display-class pixels</h3><div className="bar-chart">{metadata.classes.map((item) => <div className="bar-row" key={item.name}><span>{item.name}</span><div className="bar-track"><i style={{ width: `${(item.pixel_count / maxPixels) * 100}%`, background: item.color }} /></div><small>{item.pixel_count.toLocaleString()}</small></div>)}</div><p className="muted-copy">Existing valid-pixel quantiles; not geological thresholds.</p></section><section className="page-card"><p className="micro-label">FEATURE IMPORTANCE</p><h3>Not available</h3><p className="muted-copy">Feature-importance data is not exposed by the current API.</p></section><section className="page-card"><p className="micro-label">KNOWN OCCURRENCE DISTRIBUTION</p><h3>Not available globally</h3><p className="muted-copy">The API provides 170 authoritative pilot GSI points as a count, but not occurrence geometries or global distribution data.</p></section><section className="page-card"><p className="micro-label">MODEL VALIDATION</p><h3>Not available in dashboard API</h3><p className="muted-copy">No ROC-AUC, PR-AUC, accuracy, or global validation statistic is rendered without an API-provided value.</p></section></div></section>; }

function MethodologyPage() { const cards = [["01", "Satellite Earth Observation", "Sentinel-2 Level-2A provides the currently connected Earth-observation source for the validated pilot."], ["02", "Spectral Feature Engineering", "The pilot uses six Sentinel-2 bands plus NDVI, NDMI, iron/oxide proxy, and SWIR ratio."], ["03", "Geological Data Integration", "The architecture is prepared to integrate verified geological data where available."], ["04", "Known Manganese Occurrences", "Authoritative GSI manganese field/exploration evidence provides the current pilot positive reference data."], ["05", "Machine Learning / PU Learning", "Random Forest positive-unlabeled learning ranks known pilot positives against an explicitly unlabeled background."], ["06", "Spatial Validation", "Validation is performed on the current pilot workflow; no unexposed metric is presented as global validation."], ["07", "Prospectivity Mapping", "The validated Balaghat prediction raster preserves the pilot study grid at 20 m resolution."], ["08", "Exploration Target Ranking", "Future target ranking will be displayed only where a connected dataset or prediction product exists."]]; return <section className="methodology-page"><div className="page-intro"><p className="micro-label">GLOBAL PLATFORM ARCHITECTURE</p><h2>Methodology</h2><p className="page-subtitle">GEOSPECTRA is designed for global expansion; Balaghat, Madhya Pradesh remains the current validated pilot.</p></div><div className="method-grid">{cards.map(([number, title, text]) => <section className="method-card" key={number}><span>{number}</span><h3>{title}</h3><p>{text}</p></section>)}</div><section className="limitations-card"><p className="micro-label">SCIENTIFIC LIMITATIONS</p><p>This system provides relative manganese prospectivity based on remote-sensing and geological proxy features. It is not a confirmed mineral reserve estimate, ore-grade estimate, economic assessment, or substitute for field investigation and geological confirmation.</p></section></section>; }

export default function App() {
  const [metadata, setMetadata] = useState(null); const [selectedLocation, setSelectedLocation] = useState(null); const [activePage, setActivePage] = useState("Explore"); const [error, setError] = useState(null);
  useEffect(() => { prospectivityApi.metadata().then(setMetadata).catch(() => setError("The prospectivity service is unavailable. Start the FastAPI backend and try again.")); }, []);
  const regions = useMemo(() => metadata ? buildRegionCatalog(metadata) : [], [metadata]); const [activeRegion, setActiveRegion] = useState(null);
  useEffect(() => { if (regions.length && !activeRegion) setActiveRegion(regions[0]); }, [regions, activeRegion]);
  if (error) return <main className="status-screen"><p className="micro-label">SERVICE UNAVAILABLE</p><h1>GEOSPECTRA</h1><p>{error}</p></main>;
  if (!metadata || !activeRegion) return <main className="status-screen"><div className="loading-dot" /><p>Loading global exploration intelligence…</p></main>;
  const topStats = [["STUDY COVERAGE", "Global"], ["KNOWN OCCURRENCES", `${metadata.authoritative_gsi_positive_points} pilot GSI points`], ["DATA SOURCES", DATA_SOURCES.join(" · ")], ["TARGETING MODEL", metadata.model], ["VALIDATION", "Pilot status only"]];
  return <main className="app-shell"><header className="topbar"><strong className="navbar-brand">MAGFINE</strong><span className="navbar-team">From Coder Life Team</span></header><div className="content-shell"><section className="title-row"><div><p className="micro-label">GLOBAL MANGANESE EXPLORATION INTELLIGENCE</p><h1>From Earth observation to global manganese exploration targets.</h1><p>Global-ready exploration intelligence with a current validated pilot in Balaghat, Madhya Pradesh, India.</p></div></section><section className="stat-grid">{topStats.map(([label, value]) => <article key={label}><p>{label}</p><strong>{value}</strong></article>)}</section><nav className="tabs" aria-label="Dashboard sections">{NAVIGATION.map((page) => <button key={page} className={activePage === page ? "active-tab" : ""} onClick={() => setActivePage(page)}>{page}</button>)}</nav>{activePage === "Explore" && <ExplorePage metadata={metadata} regions={regions} activeRegion={activeRegion} setActiveRegion={setActiveRegion} selectedLocation={selectedLocation} setSelectedLocation={setSelectedLocation} />}{activePage === "Targets" && <TargetsPage />}{activePage === "Analytics" && <AnalyticsPage metadata={metadata} />}{activePage === "Methodology" && <MethodologyPage />}<footer className="global-disclaimer"><strong>Exploration aid only</strong><span>Prospectivity ranking is an exploration aid and is not a confirmed mineral reserve, ore-grade estimate, economic assessment, or substitute for field investigation and geological confirmation.</span></footer></div></main>;
}
