export type Category = {
  id: string;
  label: string;
  description: string;
  automated: boolean;
  score_scale: string;
};

export type ComparisonRow = {
  id: string;
  name: string;
  status?: string;
  modality?: string;
  needs_composite?: boolean;
  realtime_factor?: number | null;
  gen_ms_avg?: number | null;
  busy_ratio?: number | null;
  sessions_per_gpu?: number | null;
  vram_used_gb?: number | null;
  ttfw_ms_avg?: number | null;
  first_audio_ms_avg?: number | null;
  usd_per_session_hour_gpu?: number | null;
  gpu_usd_per_hr?: number | null;
  fidelity_overall?: number | null;
  uncanny_valley?: number | null;
  composite_stability?: number | null;
  lip_sync?: number | null;
  identity?: number | null;
  hosting_overall?: number | null;
  manual_complete?: boolean;
};

export type ModelDetail = ComparisonRow & {
  family?: string;
  output?: string;
  realtime_claim?: string;
  repo?: string;
  catalog_notes?: string;
  chunk_frames?: number | null;
  chunk_seconds?: number | null;
  prep_ms?: number | null;
  video_pipeline_ms_avg?: number | null;
  automated_notes?: string;
  cost_notes?: string;
  fidelity_notes?: string;
  hosting_notes?: string;
  hosting_breakdown?: Record<string, number | null | undefined>;
  hardware?: Record<string, unknown>;
  verdict?: string;
  pros?: string[];
  cons?: string[];
  updated_at?: string | null;
};

export type ComparisonData = {
  categories: Category[];
  models: { id: string; name: string; notes?: string; status?: string }[];
  rows: ComparisonRow[];
  details?: ModelDetail[];
  hardware_note?: string;
  updated_at?: string;
};
