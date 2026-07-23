import type { ComparisonData, ComparisonRow, ModelDetail } from "@/lib/types";

export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v == null || !Number.isFinite(Number(v))) return "–";
  const n = Number(v);
  return Number.isInteger(n) ? String(n) : n.toFixed(digits);
}

export function fmtMs(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(Number(v))) return "–";
  const n = Number(v);
  return n >= 1000 ? `${(n / 1000).toFixed(2)} s` : `${Math.round(n)} ms`;
}

export function fmtUsd(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(Number(v))) return "–";
  return `$${Number(v).toFixed(3)}`;
}

function csvEscape(v: unknown): string {
  if (v == null) return "";
  const s = String(v);
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

/** Excel-friendly UTF-8 CSV with BOM. */
export function downloadCsv(filename: string, headers: string[], rows: unknown[][]) {
  const lines = [
    headers.map(csvEscape).join(","),
    ...rows.map((r) => r.map(csvEscape).join(",")),
  ];
  const bom = "\uFEFF";
  const blob = new Blob([bom + lines.join("\r\n")], {
    type: "text/csv;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function exportSummaryCsv(data: ComparisonData) {
  const headers = [
    "id",
    "name",
    "status",
    "modality",
    "realtime_factor",
    "gen_ms_avg",
    "busy_ratio",
    "sessions_per_gpu",
    "vram_used_gb",
    "ttfw_ms_avg",
    "usd_per_session_hour_gpu",
    "fidelity_overall",
    "uncanny_valley",
    "composite_stability",
    "lip_sync",
    "identity",
    "hosting_overall",
    "languages_display",
    "languages_count",
    "languages_labels",
    "license_spdx",
    "license_commercial_ok",
  ];
  const rows = data.rows.map((r) =>
    headers.map((h) => {
      if (h === "languages_labels") return (r.languages_labels || []).join(" | ");
      return (r as Record<string, unknown>)[h] ?? "";
    }),
  );
  downloadCsv("avatar-model-summary.csv", headers, rows);
}

export function exportDetailedCsv(data: ComparisonData) {
  const details = data.details?.length ? data.details : (data.rows as ModelDetail[]);
  const headers = [
    "id",
    "name",
    "status",
    "family",
    "modality",
    "realtime_claim",
    "repo",
    "realtime_factor",
    "gen_ms_avg",
    "busy_ratio",
    "sessions_per_gpu",
    "vram_used_gb",
    "chunk_seconds",
    "ttfw_ms_avg",
    "first_audio_ms_avg",
    "usd_per_session_hour_gpu",
    "fidelity_overall",
    "uncanny_valley",
    "composite_stability",
    "lip_sync",
    "identity",
    "hosting_overall",
    "languages_display",
    "languages_count",
    "languages_labels",
    "languages_notes",
    "license_spdx",
    "license_commercial_ok",
    "license_summary",
    "license_caveats",
    "verdict",
    "pros",
    "cons",
    "fidelity_notes",
    "hosting_notes",
    "automated_notes",
    "catalog_notes",
    "hardware_gpu",
    "hardware_os",
    "updated_at",
  ];
  const rows = details.map((d) => [
    d.id,
    d.name,
    d.status ?? "",
    d.family ?? "",
    d.modality ?? "",
    d.realtime_claim ?? "",
    d.repo ?? "",
    d.realtime_factor ?? "",
    d.gen_ms_avg ?? "",
    d.busy_ratio ?? "",
    d.sessions_per_gpu ?? "",
    d.vram_used_gb ?? "",
    d.chunk_seconds ?? "",
    d.ttfw_ms_avg ?? "",
    d.first_audio_ms_avg ?? "",
    d.usd_per_session_hour_gpu ?? "",
    d.fidelity_overall ?? "",
    d.uncanny_valley ?? "",
    d.composite_stability ?? "",
    d.lip_sync ?? "",
    d.identity ?? "",
    d.hosting_overall ?? "",
    d.languages_display ?? "",
    d.languages_count ?? "",
    (d.languages_labels || []).join(" | "),
    d.languages_notes ?? "",
    d.license_spdx ?? "",
    d.license_commercial_ok ?? "",
    d.license_summary ?? "",
    (d.license_caveats || []).join(" | "),
    d.verdict ?? "",
    (d.pros || []).join(" | "),
    (d.cons || []).join(" | "),
    d.fidelity_notes ?? "",
    d.hosting_notes ?? "",
    d.automated_notes ?? "",
    d.catalog_notes ?? "",
    (d.hardware as { gpu?: string } | undefined)?.gpu ?? "",
    (d.hardware as { os?: string } | undefined)?.os ?? "",
    d.updated_at ?? "",
  ]);
  downloadCsv("avatar-model-detailed.csv", headers, rows);
}

export function chartSummary(
  rows: ComparisonRow[],
  key: keyof ComparisonRow,
  label: string,
): string {
  const parts = rows.map((r) => {
    const v = r[key];
    const shown =
      v == null || !Number.isFinite(Number(v)) ? "n/a" : String(Number(v).toFixed(2));
    return `${r.name} ${shown}`;
  });
  return `${label}: ${parts.join("; ")}`;
}
