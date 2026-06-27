import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Database,
  Filter,
  FolderOpen,
  Image as ImageIcon,
  RefreshCw,
  UploadCloud,
} from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";
import {
  artifactUrl,
  getMethodology,
  getOptions,
  getPredictions,
  getSummary,
  imageUrl,
  projectArtifactUrl,
  uploadProject,
} from "./api";

const emptyOptions = { surfaces: [], predicted_labels: [], actual_labels: [], severity_labels: [] };

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function count(value) {
  return Number(value || 0).toLocaleString();
}

function number(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function titleLabel(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function bestSplit(metrics) {
  const splits = metrics?.splits || {};
  return splits.test ? ["test", splits.test] : splits.validation ? ["validation", splits.validation] : ["train", splits.train];
}

function Stat({ label, value, icon: Icon }) {
  return (
    <section className="stat">
      <Icon size={18} />
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </section>
  );
}

function Pill({ label, value }) {
  return (
    <div className="pill">
      <span>{label}</span>
      <strong>{count(value)}</strong>
    </div>
  );
}

function RadarChart({ radar }) {
  const metrics = radar?.metrics || [];
  const models = radar?.models || [];
  const size = 290;
  const center = size / 2;
  const radius = 104;
  const colors = ["#237a57", "#b85c2f", "#3e6c9f"];

  if (!metrics.length || !models.length) {
    return <div className="emptyState compact">No radar metrics available</div>;
  }

  const point = (index, value = 100) => {
    const angle = -Math.PI / 2 + (index * Math.PI * 2) / metrics.length;
    const scaled = radius * (Number(value) / 100);
    return [center + Math.cos(angle) * scaled, center + Math.sin(angle) * scaled];
  };

  const polygonPoints = (model) =>
    metrics
      .map((metric, index) => point(index, model.metrics?.[metric] || 0).map((coord) => coord.toFixed(1)).join(","))
      .join(" ");

  return (
    <div className="radarWrap">
      <svg className="radarChart" viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Performance radar">
        {[0.25, 0.5, 0.75, 1].map((level) => (
          <polygon
            key={level}
            points={metrics.map((_, index) => point(index, level * 100).join(",")).join(" ")}
            className="radarGrid"
          />
        ))}
        {metrics.map((metric, index) => {
          const [x, y] = point(index, 100);
          const [labelX, labelY] = point(index, 118);
          return (
            <g key={metric}>
              <line x1={center} y1={center} x2={x} y2={y} className="radarAxis" />
              <text x={labelX} y={labelY} textAnchor="middle" dominantBaseline="middle">
                {titleLabel(metric)}
              </text>
            </g>
          );
        })}
        {models.map((model, index) => (
          <polygon
            key={model.model}
            points={polygonPoints(model)}
            fill={colors[index % colors.length]}
            stroke={colors[index % colors.length]}
            className="radarModel"
          />
        ))}
      </svg>
      <div className="radarLegend">
        {models.map((model, index) => (
          <span key={model.model}>
            <i style={{ background: colors[index % colors.length] }} />
            {model.model}
          </span>
        ))}
      </div>
    </div>
  );
}

function DashboardPage({ onOpenUpload }) {
  const [summary, setSummary] = useState(null);
  const [methodology, setMethodology] = useState(null);
  const [options, setOptions] = useState(emptyOptions);
  const [predictions, setPredictions] = useState({ total: 0, records: [] });
  const [selected, setSelected] = useState(null);
  const [previewMode, setPreviewMode] = useState("overlay");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState({
    limit: 60,
    surface: "",
    predicted_label: "",
    actual_label: "",
    min_confidence: "",
    sort_by: "confidence",
    direction: "desc",
  });

  async function loadAll() {
    setLoading(true);
    setError("");
    try {
      // Each call can fail independently — e.g. /api/options needs manifest.csv,
      // which is not deployed alongside the lightweight summary/metrics JSON.
      // Promise.all() would reject the WHOLE batch (including the good summary
      // and methodology responses) if any single one fails; Promise.allSettled()
      // lets the others through.
      const [summaryResult, optionsResult, methodologyResult] = await Promise.allSettled([
        getSummary(),
        getOptions(),
        getMethodology(),
      ]);

      const summaryPayload = summaryResult.status === "fulfilled" ? summaryResult.value : null;
      const optionsPayload = optionsResult.status === "fulfilled" ? optionsResult.value : null;
      const methodologyPayload = methodologyResult.status === "fulfilled" ? methodologyResult.value : null;

      setSummary(summaryPayload);
      setMethodology(methodologyPayload || summaryPayload?.methodology || null);
      setOptions(optionsPayload);

      // Surface a non-blocking notice only if the core summary data itself failed.
      if (summaryResult.status === "rejected") {
        setError(summaryResult.reason?.message || "Failed to load dashboard summary.");
      } else if (optionsResult.status === "rejected") {
        setError("");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadPredictions() {
    setLoading(true);
    try {
      const payload = await getPredictions(filters);
      setPredictions(payload);
      setSelected((current) => current || payload.records[0] || null);
      setError("");
    } catch (err) {
      // predictions.csv is intentionally not deployed alongside the lightweight
      // summary/metrics JSON, so a 404 here is expected on this deployment and
      // should not blank out the dashboard's error banner with a scary message.
      setPredictions({ total: 0, records: [] });
      setSelected(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadAll();
  }, []);

  useEffect(() => {
    loadPredictions();
  }, [filters]);

  const [splitName, splitMetrics] = bestSplit(summary?.metrics || {});
  const predictedLabels = summary?.summary?.predicted_labels || {};
  const actualLabels = summary?.summary?.actual_labels || {};
  const localization = summary?.summary?.localization || {};
  const artifacts = summary?.status || {};
  const methodologyPayload = methodology || summary?.methodology || {};
  const radar = methodologyPayload.performance_radar || {};
  const confusion = splitMetrics?.confusion_matrix || [
    [0, 0],
    [0, 0],
  ];

  const crackShare = useMemo(() => {
    const cracked = Number(predictedLabels.cracked || 0);
    const total = Number(summary?.summary?.rows || 0);
    return total ? pct(cracked / total) : "n/a";
  }, [predictedLabels, summary]);

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">SDNET2018 Local PoC</p>
          <h1>Concrete Crack Detection</h1>
        </div>
        <div className="topActions">
          <button className="iconButton" onClick={onOpenUpload} title="Upload inspection project">
            <UploadCloud size={18} />
            <span>Upload Project</span>
          </button>
          <button className="iconButton" onClick={loadAll} disabled={loading} title="Refresh dashboard data">
            <RefreshCw size={18} />
            <span>Refresh</span>
          </button>
        </div>
      </header>

      {error && (
        <section className="notice">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </section>
      )}

      <section className="statusStrip">
        {Object.entries(artifacts).map(([name, artifact]) => (
          <div className="statusItem" key={name}>
            {artifact.exists ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
            <span>{name}</span>
          </div>
        ))}
      </section>

      <section className="statsGrid">
        <Stat label="Images Processed" value={count(summary?.summary?.rows)} icon={Database} />
        <Stat label="Predicted Crack Share" value={crackShare} icon={Activity} />
        <Stat label="Localized Cracks" value={count(localization?.rows)} icon={ImageIcon} />
        <Stat label={`${splitName || "Model"} F1`} value={pct(splitMetrics?.f1)} icon={BarChart3} />
      </section>

      <section className="analysisBand">
        <div className="distribution">
          <h2>Distribution</h2>
          <div className="pillGrid">
            <Pill label="Predicted cracked" value={predictedLabels.cracked} />
            <Pill label="Predicted non-cracked" value={predictedLabels.non_cracked} />
            <Pill label="Actual cracked" value={actualLabels.cracked} />
            <Pill label="Actual non-cracked" value={actualLabels.non_cracked} />
          </div>
        </div>

        <div className="confusion">
          <h2>Confusion Matrix</h2>
          <div className="matrix" aria-label="Confusion matrix">
            <span></span>
            <span>Pred Non</span>
            <span>Pred Crack</span>
            <span>Actual Non</span>
            <strong>{count(confusion[0]?.[0])}</strong>
            <strong>{count(confusion[0]?.[1])}</strong>
            <span>Actual Crack</span>
            <strong>{count(confusion[1]?.[0])}</strong>
            <strong>{count(confusion[1]?.[1])}</strong>
          </div>
        </div>
      </section>

      <section className="methodologyBand">
        <div className="methodologyPanel">
          <div className="sectionHeader flatHeader">
            <div>
              <h2>CrackNet Methodology</h2>
              <p>{titleLabel(methodologyPayload.measurement_mode || "methodology adoption")}</p>
            </div>
          </div>
          <div className="stageRail">
            {(methodologyPayload.stages || []).map((stage) => (
              <article className={`stageChip ${stage.status}`} key={stage.name}>
                <span>{stage.order}</span>
                <strong>{stage.name}</strong>
                <small>{titleLabel(stage.status)}</small>
              </article>
            ))}
          </div>
          <div className="methodologyStatus">
            {(methodologyPayload.architectures || []).map((item) => (
              <div key={item.name}>
                <strong>{item.name}</strong>
                <span>{titleLabel(item.role)}</span>
                <small>{titleLabel(item.implementation_status)}</small>
              </div>
            ))}
          </div>
        </div>

        <div className="radarPanel">
          <div className="sectionHeader flatHeader">
            <div>
              <h2>Performance Radar</h2>
              <p>{radar.scale || "0-100 normalized score"}</p>
              {radar.basis && <p className="radarBasis">{radar.basis}</p>}
            </div>
          </div>
          <RadarChart radar={radar} />
        </div>
      </section>

      <section className="workspace">
        <aside className="filters">
          <h2>
            <Filter size={18} />
            Filters
          </h2>
          <label>
            Surface
            <select value={filters.surface} onChange={(event) => setFilters({ ...filters, surface: event.target.value })}>
              <option value="">All surfaces</option>
              {options.surfaces.map((surface) => (
                <option key={surface} value={surface}>
                  {surface}
                </option>
              ))}
            </select>
          </label>
          <label>
            Prediction
            <select
              value={filters.predicted_label}
              onChange={(event) => setFilters({ ...filters, predicted_label: event.target.value })}
            >
              <option value="">All predictions</option>
              {options.predicted_labels.map((label) => (
                <option key={label} value={label}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Actual Label
            <select value={filters.actual_label} onChange={(event) => setFilters({ ...filters, actual_label: event.target.value })}>
              <option value="">All labels</option>
              {options.actual_labels.map((label) => (
                <option key={label} value={label}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Minimum Confidence
            <input
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={filters.min_confidence}
              onChange={(event) => setFilters({ ...filters, min_confidence: event.target.value })}
              placeholder="0.75"
            />
          </label>
        </aside>

        <section className="tablePanel">
          <div className="sectionHeader">
            <div>
              <h2>Predictions</h2>
              <p>{count(predictions.total)} matching images</p>
            </div>
            <button className="iconButton" onClick={loadPredictions} disabled={loading} title="Refresh predictions">
              <RefreshCw size={16} />
              <span>Reload</span>
            </button>
          </div>
          <div className="tableWrap">
            <table>
              <thead>
                <tr>
                  <th>Image</th>
                  <th>Surface</th>
                  <th>Actual</th>
                  <th>Prediction</th>
                  <th>Crack Prob.</th>
                  <th>Severity</th>
                  <th>Area</th>
                  <th>Length</th>
                  <th>Width</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {predictions.records.map((row) => (
                  <tr
                    key={row.image_id}
                    className={selected?.image_id === row.image_id ? "active" : ""}
                    onClick={() => setSelected(row)}
                  >
                    <td>{row.image_id}</td>
                    <td>{row.surface}</td>
                    <td>{row.label || "unknown"}</td>
                    <td>
                      <span className={`badge ${row.predicted_label}`}>{row.predicted_label}</span>
                    </td>
                    <td>{pct(row.crack_probability)}</td>
                    <td>{row.severity_label || "n/a"}</td>
                    <td>{row.crack_area_pct !== undefined && row.crack_area_pct !== null ? pct(row.crack_area_pct) : "n/a"}</td>
                    <td>{row.crack_length_px ? `${number(row.crack_length_px, 0)} px` : "n/a"}</td>
                    <td>{row.max_width_px ? `${number(row.mean_width_px, 1)} / ${number(row.max_width_px, 1)} px` : "n/a"}</td>
                    <td>{pct(row.confidence)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <aside className="preview">
          <h2>Image Preview</h2>
          {selected ? (
            <>
              <div className="previewTabs" role="tablist" aria-label="Preview mode">
                {["original", "overlay", "heatmap", "mask"].map((mode) => (
                  <button
                    key={mode}
                    className={previewMode === mode ? "active" : ""}
                    onClick={() => setPreviewMode(mode)}
                    type="button"
                  >
                    {mode}
                  </button>
                ))}
              </div>
              <img
                src={
                  previewMode === "original"
                    ? imageUrl(selected.image_id)
                    : selected[`${previewMode}_path`]
                      ? artifactUrl(selected.image_id, previewMode)
                      : imageUrl(selected.image_id)
                }
                alt={`${previewMode} ${selected.image_id}`}
              />
              <dl>
                <div>
                  <dt>Image ID</dt>
                  <dd>{selected.image_id}</dd>
                </div>
                <div>
                  <dt>Prediction</dt>
                  <dd>{selected.predicted_label}</dd>
                </div>
                <div>
                  <dt>Confidence</dt>
                  <dd>{pct(selected.confidence)}</dd>
                </div>
                <div>
                  <dt>Severity</dt>
                  <dd>{selected.severity_label || "n/a"}</dd>
                </div>
                <div>
                  <dt>Crack Area</dt>
                  <dd>{selected.crack_area_pct !== undefined && selected.crack_area_pct !== null ? pct(selected.crack_area_pct) : "n/a"}</dd>
                </div>
                <div>
                  <dt>Length</dt>
                  <dd>{selected.crack_length_px ? `${number(selected.crack_length_px, 0)} px` : "n/a"}</dd>
                </div>
                <div>
                  <dt>Mean Width</dt>
                  <dd>{selected.mean_width_px ? `${number(selected.mean_width_px, 1)} px` : "n/a"}</dd>
                </div>
                <div>
                  <dt>Max Width</dt>
                  <dd>{selected.max_width_px ? `${number(selected.max_width_px, 1)} px` : "n/a"}</dd>
                </div>
                <div>
                  <dt>Components</dt>
                  <dd>{selected.component_count ?? "n/a"}</dd>
                </div>
                <div>
                  <dt>Method</dt>
                  <dd>{titleLabel(selected.segmentation_source || selected.measurement_method || "n/a")}</dd>
                </div>
                <div>
                  <dt>Source</dt>
                  <dd>{selected.relative_path}</dd>
                </div>
              </dl>
            </>
          ) : (
            <div className="emptyState">No prediction selected</div>
          )}
        </aside>
      </section>
    </main>
  );
}

function ProjectUploadPage({ onOpenDashboard }) {
  const [projectName, setProjectName] = useState("Concrete Inspection Project");
  const [files, setFiles] = useState([]);
  const [project, setProject] = useState(null);
  const [selected, setSelected] = useState(null);
  const [previewMode, setPreviewMode] = useState("overlay");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleUpload(event) {
    event.preventDefault();
    if (!files.length) {
      setError("Select at least one image.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const payload = await uploadProject(projectName, files);
      setProject(payload);
      setSelected(payload.records?.[0] || null);
      setPreviewMode("overlay");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const records = project?.records || [];
  const summary = project?.summary || {};

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Inspection Project</p>
          <h1>Upload Crack Images</h1>
        </div>
        <div className="topActions">
          <button className="iconButton" onClick={onOpenDashboard} title="Open full SDNET results">
            <FolderOpen size={18} />
            <span>Dataset Results</span>
          </button>
        </div>
      </header>

      {error && (
        <section className="notice">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </section>
      )}

      <section className="uploadWorkspace">
        <form className="uploadPanel" onSubmit={handleUpload}>
          <h2>Project Upload</h2>
          <label>
            Project Name
            <input value={projectName} onChange={(event) => setProjectName(event.target.value)} />
          </label>
          <label>
            Images
            <input
              type="file"
              accept="image/png,image/jpeg,image/bmp,image/tiff"
              multiple
              onChange={(event) => setFiles(Array.from(event.target.files || []))}
            />
          </label>
          <button className="primaryButton" type="submit" disabled={loading || !files.length}>
            <UploadCloud size={18} />
            <span>{loading ? "Processing..." : `Process ${count(files.length)} Images`}</span>
          </button>
        </form>

        <section className="projectPanel">
          <div className="sectionHeader">
            <div>
              <h2>{project?.name || "Project Results"}</h2>
              <p>{project?.project_id || "No project processed yet"}</p>
            </div>
          </div>

          <div className="statsGrid compactStats">
            <Stat label="Images" value={count(summary.image_count)} icon={Database} />
            <Stat label="Cracked" value={count(summary.predicted_cracked)} icon={AlertTriangle} />
            <Stat label="Non-cracked" value={count(summary.predicted_non_cracked)} icon={CheckCircle2} />
            <Stat label="Localized" value={count(summary.localized_cracks)} icon={ImageIcon} />
          </div>

          <div className="projectResults">
            <div className="tableWrap">
              <table>
                <thead>
                  <tr>
                    <th>Image</th>
                    <th>Prediction</th>
                    <th>Probability</th>
                    <th>Severity</th>
                    <th>Area</th>
                    <th>Length</th>
                    <th>Width</th>
                  </tr>
                </thead>
                <tbody>
                  {records.map((row) => (
                    <tr
                      key={row.image_id}
                      className={selected?.image_id === row.image_id ? "active" : ""}
                      onClick={() => setSelected(row)}
                    >
                      <td>{row.original_filename || row.image_id}</td>
                      <td>
                        <span className={`badge ${row.predicted_label}`}>{row.predicted_label || "error"}</span>
                      </td>
                      <td>{pct(row.crack_probability)}</td>
                      <td>{row.severity_label || "n/a"}</td>
                      <td>{row.crack_area_pct !== undefined && row.crack_area_pct !== null ? pct(row.crack_area_pct) : "n/a"}</td>
                      <td>{row.crack_length_px ? `${number(row.crack_length_px, 0)} px` : "n/a"}</td>
                      <td>{row.mean_width_px ? `${number(row.mean_width_px, 1)} px` : "n/a"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {!records.length && <div className="emptyState">Upload images to create a project</div>}
            </div>

            <aside className="preview">
              <h2>Project Image</h2>
              {selected && project ? (
                <>
                  <div className="previewTabs" role="tablist" aria-label="Project preview mode">
                    {["original", "overlay", "heatmap", "mask"].map((mode) => (
                      <button
                        key={mode}
                        className={previewMode === mode ? "active" : ""}
                        onClick={() => setPreviewMode(mode)}
                        type="button"
                      >
                        {mode}
                      </button>
                    ))}
                  </div>
                  <img
                    src={projectArtifactUrl(
                      project.project_id,
                      selected.image_id,
                      previewMode === "original" || selected[`${previewMode}_path`] ? previewMode : "original",
                    )}
                    alt={`${previewMode} ${selected.image_id}`}
                  />
                  <dl>
                    <div>
                      <dt>Prediction</dt>
                      <dd>{selected.predicted_label || "n/a"}</dd>
                    </div>
                    <div>
                      <dt>Confidence</dt>
                      <dd>{pct(selected.confidence)}</dd>
                    </div>
                    <div>
                      <dt>Severity</dt>
                      <dd>{selected.severity_label || "n/a"}</dd>
                    </div>
                    <div>
                      <dt>Area</dt>
                      <dd>{selected.crack_area_pct !== undefined && selected.crack_area_pct !== null ? pct(selected.crack_area_pct) : "n/a"}</dd>
                    </div>
                    <div>
                      <dt>Length</dt>
                      <dd>{selected.crack_length_px ? `${number(selected.crack_length_px, 0)} px` : "n/a"}</dd>
                    </div>
                    <div>
                      <dt>Mean Width</dt>
                      <dd>{selected.mean_width_px ? `${number(selected.mean_width_px, 1)} px` : "n/a"}</dd>
                    </div>
                    <div>
                      <dt>Max Width</dt>
                      <dd>{selected.max_width_px ? `${number(selected.max_width_px, 1)} px` : "n/a"}</dd>
                    </div>
                    <div>
                      <dt>Method</dt>
                      <dd>{titleLabel(selected.segmentation_source || selected.measurement_method || "n/a")}</dd>
                    </div>
                    <div>
                      <dt>File</dt>
                      <dd>{selected.original_filename || selected.relative_path}</dd>
                    </div>
                  </dl>
                </>
              ) : (
                <div className="emptyState">No project image selected</div>
              )}
            </aside>
          </div>
        </section>
      </section>
    </main>
  );
}

function App() {
  const [page, setPage] = useState("dashboard");
  if (page === "upload") {
    return <ProjectUploadPage onOpenDashboard={() => setPage("dashboard")} />;
  }
  return <DashboardPage onOpenUpload={() => setPage("upload")} />;
}

export default App;
