/*
 * Reference: a scale input + an mm-aware result panel.
 *
 * Two pieces to splice into your existing dashboard:
 *   1. <ScaleControls>  - lets the inspector choose how the scale is supplied,
 *      and produces the form fields appended to your upload POST.
 *   2. <InspectionResult> - renders the result panel from your screenshot, but
 *      prints millimetres when the record is calibrated and falls back to
 *      "N px (uncalibrated)" otherwise.
 *
 * The key helper is formatMeasure(): it decides px vs mm per field from the
 * record itself, so a calibrated upload shows mm and an uncalibrated one shows
 * px — no separate UI state needed. Styling is intentionally minimal; match it
 * to your existing panel classes.
 */
import React, { useState } from "react";

/* ---- px / mm formatting, driven by the record ---- */
function formatMeasure(px, mm, digits = 2) {
  if (mm !== null && mm !== undefined) return `${Number(mm).toFixed(digits)} mm`;
  if (px !== null && px !== undefined) return `${Number(px).toFixed(1)} px`;
  return "\u2014";
}

function formatArea(pct, mm2) {
  const p = pct !== null && pct !== undefined ? `${(pct * 100).toFixed(1)}%` : "\u2014";
  return mm2 !== null && mm2 !== undefined ? `${p} (${Number(mm2).toFixed(1)} mm\u00b2)` : p;
}

/* ====================================================================== */
/* 1. Scale input                                                          */
/* ====================================================================== */
export function ScaleControls({ value, onChange }) {
  const v = value; // { source, scale_mm_per_px, marker_length_mm, distance_mm, focal_length_mm, sensor_width_mm }
  const set = (patch) => onChange({ ...v, ...patch });

  return (
    <div className="scale-controls" style={{ display: "grid", gap: 8, maxWidth: 420 }}>
      <label>
        Scale source&nbsp;
        <select value={v.source} onChange={(e) => set({ source: e.target.value })}>
          <option value="none">None — report pixels</option>
          <option value="aruco">ArUco marker in frame</option>
          <option value="geometry">Camera geometry (GSD)</option>
          <option value="manual">Manual mm / pixel</option>
        </select>
      </label>

      {v.source === "aruco" && (
        <label>
          Marker side length (mm)&nbsp;
          <input
            type="number" step="0.1" value={v.marker_length_mm ?? ""}
            onChange={(e) => set({ marker_length_mm: e.target.value })}
          />
        </label>
      )}

      {v.source === "geometry" && (
        <>
          <label>Standoff distance (mm)&nbsp;
            <input type="number" step="1" value={v.distance_mm ?? ""}
              onChange={(e) => set({ distance_mm: e.target.value })} /></label>
          <label>Focal length (mm)&nbsp;
            <input type="number" step="0.1" value={v.focal_length_mm ?? ""}
              onChange={(e) => set({ focal_length_mm: e.target.value })} /></label>
          <label>Sensor width (mm)&nbsp;
            <input type="number" step="0.1" value={v.sensor_width_mm ?? ""}
              onChange={(e) => set({ sensor_width_mm: e.target.value })} /></label>
        </>
      )}

      {v.source === "manual" && (
        <label>
          mm per pixel&nbsp;
          <input
            type="number" step="0.001" value={v.scale_mm_per_px ?? ""}
            onChange={(e) => set({ scale_mm_per_px: e.target.value })}
          />
        </label>
      )}
    </div>
  );
}

/* Append the scale fields to the FormData you already POST on upload. */
export function appendScaleFields(formData, scale) {
  formData.append("scale_source", scale.source ?? "none");
  if (scale.source === "manual" && scale.scale_mm_per_px)
    formData.append("scale_mm_per_px", scale.scale_mm_per_px);
  if (scale.source === "aruco" && scale.marker_length_mm)
    formData.append("marker_length_mm", scale.marker_length_mm);
  if (scale.source === "geometry") {
    if (scale.distance_mm) formData.append("distance_mm", scale.distance_mm);
    if (scale.focal_length_mm) formData.append("focal_length_mm", scale.focal_length_mm);
    if (scale.sensor_width_mm) formData.append("sensor_width_mm", scale.sensor_width_mm);
  }
  return formData;
}

/* ====================================================================== */
/* 2. Result panel (mirrors your screenshot, mm-aware)                     */
/* ====================================================================== */
export function InspectionResult({ prediction, confidence, localization, fileName }) {
  const r = localization || {};
  const calibrated = r.scale_mm_per_px !== null && r.scale_mm_per_px !== undefined;

  const rows = [
    ["Prediction", prediction],
    ["Confidence", confidence !== undefined ? `${(confidence * 100).toFixed(1)}%` : "\u2014"],
    [
      "Severity",
      r.severity_label
        ? `${r.severity_label} (${calibrated ? "calibrated" : "uncalibrated"})`
        : "\u2014",
    ],
    ["Area", formatArea(r.crack_area_pct, r.crack_area_mm2)],
    ["Length", formatMeasure(r.crack_length_px, r.crack_length_mm, 1)],
    ["Mean Width", formatMeasure(r.mean_width_px, r.mean_width_mm)],
    ["Max Width", formatMeasure(r.max_width_px, r.max_width_mm)],
    // Scale row makes the measurement basis auditable at a glance:
    [
      "Scale",
      calibrated
        ? `${Number(r.scale_mm_per_px).toFixed(4)} mm/px \u00b7 ${r.scale_source}`
        : "none (pixel-domain)",
    ],
    ["Method", r.measurement_method || "heuristic"],
    ["File", fileName],
  ];

  return (
    <div className="inspection-result">
      {rows.map(([k, val]) => (
        <div className="result-row" key={k}
          style={{ display: "flex", gap: 16, padding: "8px 0", borderBottom: "1px solid #eee" }}>
          <div style={{ width: 120, color: "#6b7280" }}>{k}</div>
          <div style={{ fontWeight: 500 }}>{val ?? "\u2014"}</div>
        </div>
      ))}
    </div>
  );
}

/* ====================================================================== */
/* Example wiring: upload with scale, then render                          */
/* ====================================================================== */
export default function InspectUploader() {
  const [scale, setScale] = useState({ source: "none" });
  const [result, setResult] = useState(null);

  async function handleUpload(file) {
    const fd = new FormData();
    fd.append("file", file);
    appendScaleFields(fd, scale);
    const res = await fetch("/inspect", { method: "POST", body: fd });
    setResult(await res.json());
  }

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <ScaleControls value={scale} onChange={setScale} />
      <input type="file" accept="image/*"
        onChange={(e) => e.target.files[0] && handleUpload(e.target.files[0])} />
      {result && (
        <InspectionResult
          prediction={result.prediction}
          confidence={result.confidence}
          localization={result.localization}
          fileName={result.localization?.image_id}
        />
      )}
    </div>
  );
}
