"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import type { ComparisonData, ComparisonRow, ModelDetail } from "@/lib/types";
import {
  chartSummary,
  exportDetailedCsv,
  exportSummaryCsv,
  fmtMs,
  fmtNum,
  fmtUsd,
} from "@/lib/format";

const COLORS = [
  "#c4a574", "#6d8dff", "#5cb88a", "#d4a24c", "#c76b6b",
  "#9b7ed9", "#4ecdc4", "#f07a6e", "#7eb8da", "#b8d46b",
];

type ChartScale = "robust" | "linear" | "log";
type SortKey = keyof ComparisonRow | "name" | "status" | "modality";
type SortDir = "asc" | "desc";

/** Cap axis so one outlier (e.g. Wav2Lip RTF) doesn't crush the rest. */
function axisCeiling(finite: number[], mode: ChartScale): { max: number; clippedIds: boolean } {
  if (!finite.length) return { max: 1, clippedIds: false };
  const sorted = [...finite].sort((a, b) => a - b);
  const rawMax = sorted[sorted.length - 1];
  if (mode === "linear") return { max: Math.max(rawMax, 1e-6), clippedIds: false };
  if (mode === "log") return { max: Math.max(rawMax, 1e-6), clippedIds: false };
  if (sorted.length === 1) return { max: Math.max(rawMax, 1e-6), clippedIds: false };
  const second = sorted[sorted.length - 2];
  const p75 = sorted[Math.floor((sorted.length - 1) * 0.75)];
  let max = Math.max(second * 1.25, p75 * 2.2, 1e-6);
  // If the leader is only mildly ahead, use true max.
  if (rawMax <= max * 1.08) return { max: rawMax, clippedIds: false };
  return { max, clippedIds: true };
}

function barHeightRatio(v: number, max: number, mode: ChartScale): number {
  if (mode === "log") {
    const lo = Math.log10(Math.max(max * 1e-4, 1e-4));
    const hi = Math.log10(Math.max(max, 1e-4));
    const x = Math.log10(Math.max(v, 1e-4));
    return Math.max(0.02, (x - lo) / Math.max(hi - lo, 1e-9));
  }
  return Math.min(1, v / max);
}

function drawBreakMark(
  c: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  dpr: number,
) {
  c.save();
  c.fillStyle = "#0b0f14";
  c.fillRect(x - 1 * dpr, y - 5 * dpr, w + 2 * dpr, 10 * dpr);
  c.strokeStyle = "#e8eef7";
  c.lineWidth = 1.5 * dpr;
  c.beginPath();
  c.moveTo(x - 2 * dpr, y + 3 * dpr);
  c.lineTo(x + w * 0.35, y - 3 * dpr);
  c.lineTo(x + w * 0.65, y + 3 * dpr);
  c.lineTo(x + w + 2 * dpr, y - 3 * dpr);
  c.stroke();
  c.restore();
}

function barChart(
  canvas: HTMLCanvasElement,
  rows: ComparisonRow[],
  key: keyof ComparisonRow,
  legendEl: HTMLElement | null,
  opts: { fmt?: (v: number) => string; scale?: ChartScale } = {},
) {
  const mode = opts.scale || "robust";
  const c = canvas.getContext("2d");
  if (!c) return;
  const dpr = window.devicePixelRatio || 1;
  const w = (canvas.width = Math.max(1, canvas.clientWidth) * dpr);
  const h = (canvas.height = 200 * dpr);
  c.clearRect(0, 0, w, h);
  const vals = rows.map((r) => Number(r[key]));
  const finite = vals.filter((v) => Number.isFinite(v) && v > 0);
  const { max, clippedIds } = axisCeiling(finite, mode);
  const n = rows.length;
  const padL = 10 * dpr, padR = 10 * dpr, padT = 18 * dpr, padB = 36 * dpr;
  const gap = 5 * dpr;
  const barW = Math.max(4 * dpr, (w - padL - padR - gap * Math.max(0, n - 1)) / n);
  const plotH = h - padT - padB;

  if (mode === "log" || clippedIds) {
    c.fillStyle = "#7e8aa0";
    c.font = `${9 * dpr}px DM Sans, sans-serif`;
    c.textAlign = "left";
    const note =
      mode === "log"
        ? "log scale"
        : "axis capped — zig-zag = outlier above scale";
    c.fillText(note, padL, 11 * dpr);
  }

  rows.forEach((r, i) => {
    const v = vals[i];
    const x = padL + i * (barW + gap);
    const color = COLORS[i % COLORS.length];
    if (!Number.isFinite(v) || v <= 0) {
      c.globalAlpha = 0.28;
      c.fillStyle = color;
      c.fillRect(x, h - padB - 3 * dpr, barW, 3 * dpr);
      c.globalAlpha = 1;
    } else {
      const clipped = mode === "robust" && v > max * 1.001;
      const ratio = barHeightRatio(Math.min(v, max), max, mode);
      const bh = Math.max(2 * dpr, ratio * plotH);
      c.fillStyle = color;
      c.fillRect(x, h - padB - bh, barW, bh);
      if (clipped) drawBreakMark(c, x, h - padB - bh, barW, dpr);
    }
    c.fillStyle = "#9aa6b8";
    c.font = `${9.5 * dpr}px DM Sans, sans-serif`;
    c.textAlign = "center";
    const label = (r.name || r.id).replace("SoulX-", "").split(/[\s-]/)[0].slice(0, 7);
    c.fillText(label, x + barW / 2, h - 12 * dpr);
  });
  if (legendEl) {
    legendEl.innerHTML = rows
      .map((r, i) => {
        const v = vals[i];
        const shown = Number.isFinite(v) ? (opts.fmt ? opts.fmt(v) : String(v)) : "—";
        const over = mode === "robust" && Number.isFinite(v) && v > max * 1.001;
        return `<span><i style="background:${COLORS[i % COLORS.length]}"></i>${r.name}: <b>${shown}${over ? " †" : ""}</b></span>`;
      })
      .join("");
  }
}

function groupedManualChart(
  canvas: HTMLCanvasElement,
  rows: ComparisonRow[],
  legendEl: HTMLElement | null,
) {
  const keys = [
    { k: "fidelity_overall" as const, label: "Fidelity", color: "#6d8dff" },
    { k: "uncanny_valley" as const, label: "Uncanny↑", color: "#5cb88a" },
    { k: "hosting_overall" as const, label: "Hosting", color: "#d4a24c" },
  ];
  const c = canvas.getContext("2d");
  if (!c) return;
  const dpr = window.devicePixelRatio || 1;
  const w = (canvas.width = Math.max(1, canvas.clientWidth) * dpr);
  const h = (canvas.height = 200 * dpr);
  c.clearRect(0, 0, w, h);
  const n = rows.length;
  const padL = 10 * dpr, padR = 10 * dpr, padT = 14 * dpr, padB = 36 * dpr;
  const groupGap = 8 * dpr;
  const groupW = (w - padL - padR - groupGap * Math.max(0, n - 1)) / n;
  const barGap = 2 * dpr;
  const barW = (groupW - barGap * (keys.length - 1)) / keys.length;
  rows.forEach((r, i) => {
    const gx = padL + i * (groupW + groupGap);
    keys.forEach((key, ki) => {
      const v = Number(r[key.k]);
      const x = gx + ki * (barW + barGap);
      if (!Number.isFinite(v)) {
        c.globalAlpha = 0.2;
        c.fillStyle = key.color;
        c.fillRect(x, h - padB - 3 * dpr, barW, 3 * dpr);
        c.globalAlpha = 1;
        return;
      }
      const bh = Math.max(2 * dpr, (v / 10) * (h - padT - padB));
      c.fillStyle = key.color;
      c.fillRect(x, h - padB - bh, barW, bh);
    });
    c.fillStyle = "#9aa6b8";
    c.font = `${9.5 * dpr}px DM Sans, sans-serif`;
    c.textAlign = "center";
    c.fillText(
      (r.name || r.id).replace("SoulX-", "").split(/[\s-]/)[0].slice(0, 7),
      gx + groupW / 2,
      h - 12 * dpr,
    );
  });
  if (legendEl) {
    legendEl.innerHTML = keys
      .map((k) => `<span><i style="background:${k.color}"></i>${k.label}</span>`)
      .join("");
  }
}

function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="metric">
      <dt>{label}</dt>
      <dd>
        {value}
        {hint ? <span className="metric-hint">{hint}</span> : null}
      </dd>
    </div>
  );
}

function compareRows(a: ComparisonRow, b: ComparisonRow, key: SortKey, dir: SortDir): number {
  const mul = dir === "asc" ? 1 : -1;
  const av = a[key as keyof ComparisonRow];
  const bv = b[key as keyof ComparisonRow];
  if (typeof av === "string" || typeof bv === "string" || key === "name" || key === "status" || key === "modality") {
    const as = String(av ?? "");
    const bs = String(bv ?? "");
    return as.localeCompare(bs) * mul;
  }
  const an = Number(av);
  const bn = Number(bv);
  const aOk = Number.isFinite(an);
  const bOk = Number.isFinite(bn);
  if (!aOk && !bOk) return 0;
  if (!aOk) return 1;
  if (!bOk) return -1;
  if (an === bn) return String(a.name).localeCompare(String(b.name));
  return (an - bn) * mul;
}

export function Dashboard({ data }: { data: ComparisonData }) {
  const router = useRouter();
  const uid = useId();
  const [sortKey, setSortKey] = useState<SortKey>("fidelity_overall");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [chartScale, setChartScale] = useState<ChartScale>("robust");
  const [dlOpen, setDlOpen] = useState(false);

  const sortedRows = useMemo(() => {
    return [...data.rows].sort((a, b) => compareRows(a, b, sortKey, sortDir));
  }, [data.rows, sortKey, sortDir]);

  const details = useMemo(() => {
    const list = data.details?.length
      ? [...data.details]
      : (data.rows as ModelDetail[]);
    return list.sort((a, b) => {
      const fa = Number(a.fidelity_overall);
      const fb = Number(b.fidelity_overall);
      const aOk = Number.isFinite(fa);
      const bOk = Number.isFinite(fb);
      if (aOk && bOk && fa !== fb) return fb - fa;
      if (a.id === "flashhead") return -1;
      if (b.id === "flashhead") return 1;
      return a.name.localeCompare(b.name);
    });
  }, [data]);

  const rtfRef = useRef<HTMLCanvasElement>(null);
  const spgRef = useRef<HTMLCanvasElement>(null);
  const costRef = useRef<HTMLCanvasElement>(null);
  const manRef = useRef<HTMLCanvasElement>(null);
  const legRtf = useRef<HTMLDivElement>(null);
  const legSpg = useRef<HTMLDivElement>(null);
  const legCost = useRef<HTMLDivElement>(null);
  const legMan = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const rows = sortedRows;
    const paint = () => {
      if (rtfRef.current) {
        barChart(rtfRef.current, rows, "realtime_factor", legRtf.current, {
          fmt: (v) => `${v.toFixed(2)}×`,
          scale: chartScale,
        });
      }
      if (spgRef.current) {
        barChart(spgRef.current, rows, "sessions_per_gpu", legSpg.current, {
          fmt: (v) => v.toFixed(2),
          scale: chartScale,
        });
      }
      if (costRef.current) {
        barChart(costRef.current, rows, "usd_per_session_hour_gpu", legCost.current, {
          fmt: (v) => `$${v.toFixed(3)}`,
          scale: chartScale,
        });
      }
      if (manRef.current) groupedManualChart(manRef.current, rows, legMan.current);
    };
    paint();
    window.addEventListener("resize", paint);
    return () => window.removeEventListener("resize", paint);
  }, [sortedRows, chartScale]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      // Costs / gen / busy: lower is better → default asc; scores / RTF → desc
      const ascDefault = new Set<SortKey>([
        "gen_ms_avg",
        "busy_ratio",
        "usd_per_session_hour_gpu",
        "name",
        "status",
        "modality",
      ]);
      setSortDir(ascDefault.has(key) ? "asc" : "desc");
    }
  }

  function SortTh({
    label,
    k,
    numeric,
  }: {
    label: string;
    k: SortKey;
    numeric?: boolean;
  }) {
    const active = sortKey === k;
    const aria = active
      ? sortDir === "asc"
        ? "ascending"
        : "descending"
      : "none";
    return (
      <th scope="col" className={numeric ? "num" : undefined} aria-sort={aria}>
        <button
          type="button"
          className={`sort-btn ${active ? "active" : ""}`}
          onClick={() => toggleSort(k)}
        >
          <span>{label}</span>
          <span className="sort-ind" aria-hidden="true">
            {active ? (sortDir === "asc" ? "▲" : "▼") : "↕"}
          </span>
        </button>
      </th>
    );
  }

  async function logout() {
    await fetch("/api/logout", { method: "POST" });
    router.replace("/login");
    router.refresh();
  }

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to main content
      </a>

      <div className="shell">
        <header className="topbar">
          <div>
            <p className="eyebrow">Confidential bench report</p>
            <div className="brand" id="site-title">
              Avatar model analytics
            </div>
          </div>
          <div className="meta">
            {data.updated_at && (
              <time dateTime={data.updated_at}>Synced {data.updated_at}</time>
            )}
            <div className="dl-wrap">
              <button
                type="button"
                className="btn"
                aria-expanded={dlOpen}
                aria-controls={`${uid}-dl`}
                onClick={() => setDlOpen((v) => !v)}
              >
                Download data
              </button>
              {dlOpen && (
                <div className="dl-menu" id={`${uid}-dl`} role="menu">
                  <button
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      exportSummaryCsv(data);
                      setDlOpen(false);
                    }}
                  >
                    Summary CSV (matrix)
                  </button>
                  <button
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      exportDetailedCsv(data);
                      setDlOpen(false);
                    }}
                  >
                    Detailed CSV (notes + findings)
                  </button>
                  <button
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      const blob = new Blob([JSON.stringify(data, null, 2)], {
                        type: "application/json",
                      });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement("a");
                      a.href = url;
                      a.download = "avatar-model-comparison.json";
                      a.click();
                      URL.revokeObjectURL(url);
                      setDlOpen(false);
                    }}
                  >
                    Full JSON
                  </button>
                </div>
              )}
            </div>
            <button type="button" className="btn" onClick={logout}>
              Sign out
            </button>
          </div>
        </header>

        <nav className="toc" aria-label="Page sections">
          <a href="#overview">Overview</a>
          <a href="#matrix">Matrix</a>
          <a href="#charts">Charts</a>
          <a href="#method">Method</a>
          <a href="#reviews">Model reviews</a>
        </nav>

        <main id="main">
          <section id="overview" className="section" aria-labelledby="overview-h">
            <h1 id="overview-h">Model comparison</h1>
            <p className="lead">
              Side-by-side evaluation of open-source talking-head backends for
              conversational use on a single GPU. Automated metrics come from an
              isolated local harness; fidelity and uncanny scores are manual;
              hosting scores are curated operations ratings.
            </p>
            {data.hardware_note && (
              <p className="note-banner" role="note">
                {data.hardware_note}
              </p>
            )}

            <div className="cat-strip" role="list">
              {(data.categories || []).map((c) => (
                <article className="cat-card" role="listitem" key={c.id}>
                  <span className="auto">{c.automated ? "Automated" : "Manual"}</span>
                  <h2>{c.label}</h2>
                  <p>{c.description}</p>
                </article>
              ))}
            </div>
          </section>

          <section id="matrix" className="section" aria-labelledby="matrix-h">
            <div className="section-head">
              <h2 id="matrix-h">Comparison matrix</h2>
              <p className="sub">
                Click a column header to sort (click again to reverse). Higher
                realtime factor and sessions/GPU are better. Lower $/sess-hr is
                better. Uncanny↑: 10 = most natural. Sorted by{" "}
                <strong>{String(sortKey).replace(/_/g, " ")}</strong>{" "}
                ({sortDir === "asc" ? "low → high" : "high → low"}).
              </p>
            </div>
            <div className="tbl-wrap" tabIndex={0} role="region" aria-labelledby="matrix-h">
              <table className="tbl">
                <caption className="sr-only">
                  Sortable talking-head model metrics including realtime factor,
                  generation time, sessions per GPU, cost, fidelity, and hosting
                </caption>
                <thead>
                  <tr>
                    <SortTh label="Model" k="name" />
                    <SortTh label="Status" k="status" />
                    <SortTh label="Modality" k="modality" />
                    <SortTh label="RT factor" k="realtime_factor" numeric />
                    <SortTh label="Gen ms" k="gen_ms_avg" numeric />
                    <SortTh label="Busy ratio" k="busy_ratio" numeric />
                    <SortTh label="Sessions/GPU" k="sessions_per_gpu" numeric />
                    <SortTh label="$/sess-hr" k="usd_per_session_hour_gpu" numeric />
                    <SortTh label="Fidelity" k="fidelity_overall" numeric />
                    <SortTh label="Uncanny↑" k="uncanny_valley" numeric />
                    <SortTh label="Composite" k="composite_stability" numeric />
                    <SortTh label="Lips" k="lip_sync" numeric />
                    <SortTh label="Identity" k="identity" numeric />
                    <SortTh label="Hosting" k="hosting_overall" numeric />
                  </tr>
                </thead>
                <tbody>
                  {sortedRows.map((r) => {
                    const st = r.status || "empty";
                    return (
                      <tr key={r.id}>
                        <th scope="row">
                          <a href={`#model-${r.id}`}>{r.name}</a>
                        </th>
                        <td>
                          <span className={`pill ${st}`}>{st}</span>
                        </td>
                        <td>{(r.modality || "–").replace(/_/g, " ")}</td>
                        <td className="num">{fmtNum(r.realtime_factor)}</td>
                        <td className="num">{fmtNum(r.gen_ms_avg, 0)}</td>
                        <td className="num">{fmtNum(r.busy_ratio)}</td>
                        <td className="num">{fmtNum(r.sessions_per_gpu, 2)}</td>
                        <td className="num">{fmtUsd(r.usd_per_session_hour_gpu)}</td>
                        <td className="num">{fmtNum(r.fidelity_overall, 1)}</td>
                        <td className="num">{fmtNum(r.uncanny_valley, 1)}</td>
                        <td className="num">{fmtNum(r.composite_stability, 1)}</td>
                        <td className="num">{fmtNum(r.lip_sync, 1)}</td>
                        <td className="num">{fmtNum(r.identity, 1)}</td>
                        <td className="num">{fmtNum(r.hosting_overall, 1)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>

          <section id="charts" className="section" aria-labelledby="charts-h">
            <div className="section-head charts-head">
              <div>
                <h2 id="charts-h">Charts</h2>
                <p className="sub">
                  Bars follow the matrix sort order. Default scale is{" "}
                  <strong>robust</strong> (caps extreme outliers like Wav2Lip RTF
                  or EchoMimic cost so the rest stay readable). † in the legend
                  means the true value is above the capped axis.
                </p>
              </div>
              <div className="scale-toggle" role="group" aria-label="Chart scale">
                {(
                  [
                    ["robust", "Robust"],
                    ["linear", "Linear"],
                    ["log", "Log"],
                  ] as const
                ).map(([id, label]) => (
                  <button
                    key={id}
                    type="button"
                    className={`btn ${chartScale === id ? "btn-active" : ""}`}
                    aria-pressed={chartScale === id}
                    onClick={() => setChartScale(id)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
            <div className="chart-grid">
              <figure className="chart-card">
                <figcaption className="tag">Realtime factor (higher = faster than playback)</figcaption>
                <canvas
                  ref={rtfRef}
                  height={200}
                  role="img"
                  aria-label={chartSummary(sortedRows, "realtime_factor", "Realtime factor")}
                />
                <div className="legend" ref={legRtf} />
              </figure>
              <figure className="chart-card">
                <figcaption className="tag">Sessions per GPU</figcaption>
                <canvas
                  ref={spgRef}
                  height={200}
                  role="img"
                  aria-label={chartSummary(sortedRows, "sessions_per_gpu", "Sessions per GPU")}
                />
                <div className="legend" ref={legSpg} />
              </figure>
              <figure className="chart-card">
                <figcaption className="tag">GPU cost $/session-hour (lower better)</figcaption>
                <canvas
                  ref={costRef}
                  height={200}
                  role="img"
                  aria-label={chartSummary(sortedRows, "usd_per_session_hour_gpu", "Cost per session-hour")}
                />
                <div className="legend" ref={legCost} />
              </figure>
              <figure className="chart-card">
                <figcaption className="tag">Fidelity / uncanny / hosting (1–10)</figcaption>
                <canvas
                  ref={manRef}
                  height={200}
                  role="img"
                  aria-label="Grouped scores for fidelity, uncanny valley, and hosting per model"
                />
                <div className="legend" ref={legMan} />
              </figure>
            </div>
          </section>

          <section id="method" className="section" aria-labelledby="method-h">
            <h2 id="method-h">How cost is calculated</h2>
            <div className="method-panel">
              <p>
                <strong>$/sess-hr</strong> estimates GPU-only cost to run one
                concurrent talking session for one hour from harness measurements —
                not a full cloud invoice.
              </p>
              <ol className="method-steps">
                <li>
                  <strong>busy_ratio</strong> — continuous models:{" "}
                  <code>chunk_gen_ms ÷ chunk_duration</code>; clip models:{" "}
                  <code>utterance_render_ms ÷ audio_duration</code>.
                </li>
                <li>
                  <strong>Effective sessions per GPU</strong> with 15% headroom:{" "}
                  <code>eff_sessions = 0.85 ÷ busy_ratio</code>, also capped by
                  VRAM. Values below 1 mean slower than realtime.
                </li>
                <li>
                  <strong>GPU $/session-hour</strong> ={" "}
                  <code>$0.44 ÷ eff_sessions</code> (RTX 4090-class pod rate used
                  for this project).
                </li>
              </ol>
              <div className="method-grid">
                <div>
                  <h3 className="tag">Includes</h3>
                  <ul>
                    <li>Measured generation / render times</li>
                    <li>busy_ratio and VRAM use</li>
                    <li>Fixed GPU hourly rate ($0.44)</li>
                  </ul>
                </div>
                <div>
                  <h3 className="tag">Excludes</h3>
                  <ul>
                    <li>TTS / LLM / network invoices</li>
                    <li>TensorRT or multi-tenant packing</li>
                    <li>LiveKit and orchestration overhead</li>
                  </ul>
                </div>
              </div>
            </div>
          </section>

          <section id="reviews" className="section" aria-labelledby="reviews-h">
            <div className="section-head">
              <h2 id="reviews-h">Model reviews</h2>
              <p className="sub">
                Findings from local Demo runs with full-image framing. Sorted by
                fidelity score. Jump from the matrix via each model name.
              </p>
            </div>

            <div className="review-list">
              {details.map((d, idx) => (
                <article
                  key={d.id}
                  id={`model-${d.id}`}
                  className={`review-card ${d.id === "flashhead" ? "featured" : ""}`}
                  aria-labelledby={`model-h-${d.id}`}
                >
                  <header className="review-head">
                    <div>
                      <p className="eyebrow">
                        #{idx + 1}
                        {d.family ? ` · ${d.family.replace(/_/g, " ")}` : ""}
                        {d.modality ? ` · ${d.modality.replace(/_/g, " ")}` : ""}
                      </p>
                      <h3 id={`model-h-${d.id}`}>{d.name}</h3>
                      {d.verdict && <p className="verdict">{d.verdict}</p>}
                    </div>
                    <div className="review-badges">
                      <span className={`pill ${d.status || "empty"}`}>{d.status}</span>
                      {d.repo && (
                        <a
                          className="btn btn-link"
                          href={d.repo}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          Repository
                          <span className="sr-only"> for {d.name} (opens in new tab)</span>
                        </a>
                      )}
                    </div>
                  </header>

                  <dl className="metric-grid">
                    <Metric label="Realtime factor" value={`${fmtNum(d.realtime_factor)}×`} />
                    <Metric label="Gen time" value={fmtMs(d.gen_ms_avg)} />
                    <Metric label="Busy ratio" value={fmtNum(d.busy_ratio)} />
                    <Metric label="Sessions / GPU" value={fmtNum(d.sessions_per_gpu, 3)} />
                    <Metric label="GPU $/sess-hr" value={fmtUsd(d.usd_per_session_hour_gpu)} />
                    <Metric label="VRAM used" value={d.vram_used_gb != null ? `${fmtNum(d.vram_used_gb)} GB` : "–"} />
                    <Metric label="Fidelity" value={fmtNum(d.fidelity_overall, 1)} hint="/10" />
                    <Metric label="Uncanny↑" value={fmtNum(d.uncanny_valley, 1)} hint="/10" />
                    <Metric label="Composite" value={fmtNum(d.composite_stability, 1)} hint="/10" />
                    <Metric label="Lip sync" value={fmtNum(d.lip_sync, 1)} hint="/10" />
                    <Metric label="Identity" value={fmtNum(d.identity, 1)} hint="/10" />
                    <Metric label="Hosting" value={fmtNum(d.hosting_overall, 1)} hint="/10" />
                  </dl>

                  {(d.pros?.length || d.cons?.length) ? (
                    <div className="pros-cons">
                      <div>
                        <h4>Pros</h4>
                        <ul>
                          {(d.pros || []).map((p) => (
                            <li key={p}>{p}</li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <h4>Cons</h4>
                        <ul>
                          {(d.cons || []).map((c) => (
                            <li key={c}>{c}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  ) : null}

                  <div className="notes-block">
                    {d.realtime_claim && (
                      <p>
                        <strong>Claimed realtime:</strong> {d.realtime_claim}
                      </p>
                    )}
                    {d.catalog_notes && (
                      <p>
                        <strong>Overview:</strong> {d.catalog_notes}
                      </p>
                    )}
                    {d.fidelity_notes && (
                      <p>
                        <strong>Fidelity notes:</strong> {d.fidelity_notes}
                      </p>
                    )}
                    {d.hosting_notes && (
                      <p>
                        <strong>Hosting notes:</strong> {d.hosting_notes}
                      </p>
                    )}
                    {d.automated_notes && (
                      <p>
                        <strong>Harness notes:</strong> {d.automated_notes}
                      </p>
                    )}
                    {d.cost_notes && (
                      <p>
                        <strong>Cost notes:</strong> {d.cost_notes}
                      </p>
                    )}
                    {(d.chunk_seconds != null || d.needs_composite != null) && (
                      <p className="meta-line">
                        {d.chunk_seconds != null && (
                          <>
                            Chunk {fmtNum(d.chunk_seconds)} s
                            {d.chunk_frames != null ? ` (${d.chunk_frames} frames)` : ""}
                            {" · "}
                          </>
                        )}
                        Composite required: {d.needs_composite ? "yes" : "no"}
                        {d.hardware?.gpu ? ` · ${(d.hardware as { gpu: string }).gpu}` : ""}
                        {d.hardware?.os ? ` / ${(d.hardware as { os: string }).os}` : ""}
                      </p>
                    )}
                  </div>
                </article>
              ))}
            </div>
          </section>
        </main>

        <footer className="site-foot">
          <p>
            Internal evaluation report. Metrics reflect this project&apos;s local
            harness and scoring protocol — not third-party vendor claims alone.
          </p>
        </footer>
      </div>
    </>
  );
}
