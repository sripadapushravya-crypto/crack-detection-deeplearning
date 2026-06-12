const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function request(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

export function imageUrl(imageId) {
  return `${API_BASE}/api/predictions/${encodeURIComponent(imageId)}/image`;
}

export function artifactUrl(imageId, artifact) {
  return `${API_BASE}/api/predictions/${encodeURIComponent(imageId)}/${artifact}`;
}

export function projectArtifactUrl(projectId, imageId, artifact) {
  return `${API_BASE}/api/projects/${encodeURIComponent(projectId)}/images/${encodeURIComponent(imageId)}/${artifact}`;
}

export function getSummary() {
  return request("/api/summary");
}

export function getMethodology() {
  return request("/api/methodology");
}

export function getOptions() {
  return request("/api/options");
}

export function getPredictions(filters) {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") params.set(key, value);
  });
  return request(`/api/predictions?${params.toString()}`);
}

export async function uploadProject(name, files, scale = {}) {
  const formData = new FormData();
  formData.set("name", name || "Concrete Inspection Project");
  Array.from(files).forEach((file) => formData.append("files", file));

  // Optional calibration scale (absent => pixel-domain results).
  formData.set("scale_source", scale.source || "none");
  if (scale.source === "manual" && scale.scale_mm_per_px) {
    formData.set("scale_mm_per_px", scale.scale_mm_per_px);
  }
  if (scale.source === "aruco" && scale.marker_length_mm) {
    formData.set("marker_length_mm", scale.marker_length_mm);
    if (scale.aruco_dict) formData.set("aruco_dict", scale.aruco_dict);
  }
  if (scale.source === "geometry") {
    if (scale.distance_mm) formData.set("distance_mm", scale.distance_mm);
    if (scale.focal_length_mm) formData.set("focal_length_mm", scale.focal_length_mm);
    if (scale.sensor_width_mm) formData.set("sensor_width_mm", scale.sensor_width_mm);
  }

  const response = await fetch(`${API_BASE}/api/projects`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Upload failed: ${response.status}`);
  }
  return response.json();
}
