"""Bench results I/O + merge of live /stats into the active model's JSON."""

from __future__ import annotations

import json
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "bench"
MODELS_PATH = BENCH / "models.json"
RESULTS_DIR = BENCH / "results"
GPU_USD_PER_HR = 0.44
# Leave ~15% GPU headroom so a session spike doesn't drop frames.
TARGET_GPU_UTIL = 0.85
VRAM_UTIL = 0.90


def load_catalog() -> dict:
    return json.loads(MODELS_PATH.read_text(encoding="utf-8"))


def result_path(model_id: str) -> Path:
    return RESULTS_DIR / f"{model_id}.json"


def load_result(model_id: str) -> dict | None:
    p = result_path(model_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_result(model_id: str, data: dict) -> dict:
    data = dict(data)
    data["model_id"] = model_id
    data["updated_at"] = time.strftime("%Y-%m-%d")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path(model_id).write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
    return data


def clear_result(model_id: str) -> dict:
    """Reset a model's results JSON to empty template (invalid / failed runs)."""
    tpl_path = RESULTS_DIR / "_template.json"
    if tpl_path.exists():
        data = json.loads(tpl_path.read_text(encoding="utf-8"))
    else:
        data = {
            "automated": {},
            "manual": {"fidelity": {}, "hosting": {}},
            "clips": [],
            "status": "empty",
        }
    data["model_id"] = model_id
    data["status"] = "empty"
    data["updated_at"] = None
    data["automated"] = {
        k: (0.44 if k == "gpu_usd_per_hr" else None)
        for k in (
            "prep_ms", "chunk_frames", "chunk_seconds", "gen_ms_avg",
            "realtime_factor", "busy_ratio", "sessions_per_gpu", "vram_used_gb",
            "ttfw_ms_avg", "first_audio_ms_avg", "video_pipeline_ms_avg",
            "gpu_usd_per_hr", "usd_per_session_hour_gpu",
        )
    }
    data["automated"]["notes"] = ""
    data["automated"]["cost_notes"] = ""
    data["manual"] = {
        "fidelity": {
            "identity": None, "skin_teeth": None, "lip_sync": None,
            "blinks_idle": None, "head_motion": None, "composite_stability": None,
            "uncanny_valley": None, "overall": None, "notes": "",
        },
        "hosting": {
            "self_host": None, "windows_ok": None, "deps_fragility": None,
            "license_ok": None, "ops_ease": None, "overall": None, "notes": "",
        },
    }
    data["clips"] = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = dict(data)
    out["model_id"] = model_id
    result_path(model_id).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def list_results() -> list[dict]:
    catalog = load_catalog()
    out = []
    for m in catalog["models"]:
        r = load_result(m["id"]) or {
            "model_id": m["id"],
            "status": "empty",
            "automated": {},
            "manual": {},
        }
        out.append({"meta": m, "result": r})
    return out


def effective_sessions_per_gpu(
    busy_ratio: float | None,
    vram_used: float | None = None,
    vram_total: float | None = None,
) -> float | None:
    """Continuous session capacity (can be < 1 if slower than realtime).

    busy_ratio = GPU_seconds / speech_seconds (or chunk_gen / chunk_dur).
      0.5 → ~1.7 sessions/GPU at 85% util
      2.0 → ~0.43 (needs >1 GPU for one realtime session)

    Previously we floored to an integer ≥1, which made FlashHead and Ditto
    both report exactly 1 session → identical $0.44/sess-hr.
    """
    by_time = None
    if busy_ratio and busy_ratio > 0:
        by_time = TARGET_GPU_UTIL / float(busy_ratio)
    by_vram = None
    if vram_used and vram_total and vram_used > 0:
        by_vram = (float(vram_total) * VRAM_UTIL) / float(vram_used)
    if by_time is not None and by_vram is not None:
        return round(min(by_time, by_vram), 3)
    if by_time is not None:
        return round(by_time, 3)
    if by_vram is not None:
        return round(by_vram, 3)
    return None


def sessions_per_gpu(
    busy_ratio: float | None,
    vram_used: float | None = None,
    vram_total: float | None = None,
) -> float | None:
    """Alias — returns effective (fractional) sessions per GPU."""
    return effective_sessions_per_gpu(busy_ratio, vram_used, vram_total)


def usd_per_session_hour_gpu(
    busy_ratio: float | None,
    vram_used: float | None = None,
    vram_total: float | None = None,
    gpu_usd_per_hr: float = GPU_USD_PER_HR,
) -> float | None:
    """GPU $/session-hour = pod_rate / effective_sessions_per_gpu.

    If effective sessions < 1 (model slower than realtime), cost rises above
    the raw pod rate — you need more than one GPU-hour per session-hour.
    """
    spg = effective_sessions_per_gpu(busy_ratio, vram_used, vram_total)
    if spg is None or spg <= 0:
        return None
    return round(float(gpu_usd_per_hr) / spg, 3)


def merge_live_stats(model_id: str, stats: dict) -> dict:
    """Update automated fields from a live /stats snapshot (non-destructive)."""
    existing = load_result(model_id) or {
        "model_id": model_id,
        "status": "partial",
        "hardware": {},
        "modality": {},
        "automated": {},
        "manual": {"fidelity": {}, "hosting": {}},
        "clips": [],
    }
    auto = dict(existing.get("automated") or {})
    gen = stats.get("gen_ms_avg")
    chunk_s = stats.get("chunk_seconds")
    busy = stats.get("busy_ratio")
    rtf = stats.get("realtime_factor")
    vram_u = stats.get("vram_used_gb")
    vram_t = stats.get("vram_total_gb")
    gpu_hr = float(auto.get("gpu_usd_per_hr") or GPU_USD_PER_HR)
    spg = effective_sessions_per_gpu(busy, vram_u, vram_t)
    cost = usd_per_session_hour_gpu(busy, vram_u, vram_t, gpu_hr)

    if gen:
        auto["gen_ms_avg"] = gen
    if chunk_s:
        auto["chunk_seconds"] = chunk_s
        auto["chunk_frames"] = round(chunk_s * 25)
    if busy is not None:
        auto["busy_ratio"] = busy
    if rtf is not None:
        auto["realtime_factor"] = rtf
    if vram_u is not None:
        auto["vram_used_gb"] = vram_u
    auto["gpu_usd_per_hr"] = gpu_hr
    if spg is not None:
        auto["sessions_per_gpu"] = spg
    if cost is not None:
        auto["usd_per_session_hour_gpu"] = cost
        auto["cost_notes"] = (
            f"$/sess-hr = ${gpu_hr}/hr ÷ {spg} eff. sessions "
            f"(0.85/busy_ratio, capped by VRAM). "
            f"<1 session means slower-than-realtime."
        )

    turns = stats.get("turns") or []
    if turns:
        def avg(k):
            vals = [t[k] for t in turns if t.get(k) is not None]
            return round(sum(vals) / len(vals)) if vals else None

        fa = avg("first_audio_ms")
        if fa is not None:
            auto["first_audio_ms_avg"] = fa

    existing["automated"] = auto
    hw = dict(existing.get("hardware") or {})
    if vram_t:
        hw["vram_gb"] = vram_t
    existing["hardware"] = hw
    if existing.get("status") == "empty":
        existing["status"] = "partial"
    return save_result(model_id, existing)


def load_findings() -> dict:
    p = BENCH / "findings.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def comparison_matrix() -> dict:
    """Flat matrix for charts + rich per-model details for the public report."""
    findings = load_findings()
    rows = []
    details = []
    for item in list_results():
        m, r = item["meta"], item["result"]
        auto = dict(r.get("automated") or {})
        # Recompute cost from busy_ratio so older captures (floored to 1)
        # refresh on Analytics reload without a full re-bench.
        busy = auto.get("busy_ratio")
        vram_u = auto.get("vram_used_gb")
        vram_t = (r.get("hardware") or {}).get("vram_gb")
        gpu_hr = float(auto.get("gpu_usd_per_hr") or GPU_USD_PER_HR)
        if busy:
            spg = effective_sessions_per_gpu(busy, vram_u, vram_t)
            cost = usd_per_session_hour_gpu(busy, vram_u, vram_t, gpu_hr)
            if spg is not None:
                auto["sessions_per_gpu"] = spg
            if cost is not None:
                auto["usd_per_session_hour_gpu"] = cost
        fid = (r.get("manual") or {}).get("fidelity") or {}
        host = (r.get("manual") or {}).get("hosting") or {}
        hw = dict(r.get("hardware") or {})
        find = findings.get(m["id"]) or {}
        row = {
            "id": m["id"],
            "name": m["name"],
            "status": r.get("status") or m.get("status"),
            "modality": m.get("modality"),
            "needs_composite": m.get("needs_composite"),
            "realtime_factor": auto.get("realtime_factor"),
            "gen_ms_avg": auto.get("gen_ms_avg"),
            "busy_ratio": auto.get("busy_ratio"),
            "sessions_per_gpu": auto.get("sessions_per_gpu"),
            "vram_used_gb": auto.get("vram_used_gb"),
            "ttfw_ms_avg": auto.get("ttfw_ms_avg"),
            "first_audio_ms_avg": auto.get("first_audio_ms_avg"),
            "usd_per_session_hour_gpu": auto.get("usd_per_session_hour_gpu"),
            "gpu_usd_per_hr": auto.get("gpu_usd_per_hr") or GPU_USD_PER_HR,
            "fidelity_overall": fid.get("overall"),
            "uncanny_valley": fid.get("uncanny_valley"),
            "composite_stability": fid.get("composite_stability"),
            "lip_sync": fid.get("lip_sync"),
            "identity": fid.get("identity"),
            "hosting_overall": host.get("overall"),
            "manual_complete": fid.get("overall") is not None,
        }
        rows.append(row)
        details.append({
            **row,
            "family": m.get("family"),
            "output": m.get("output"),
            "realtime_claim": m.get("realtime_claim"),
            "repo": m.get("repo"),
            "catalog_notes": m.get("notes"),
            "chunk_frames": auto.get("chunk_frames"),
            "chunk_seconds": auto.get("chunk_seconds"),
            "prep_ms": auto.get("prep_ms"),
            "video_pipeline_ms_avg": auto.get("video_pipeline_ms_avg"),
            "automated_notes": auto.get("notes") or "",
            "cost_notes": auto.get("cost_notes") or "",
            "fidelity_notes": fid.get("notes") or "",
            "hosting_notes": host.get("notes") or "",
            "hosting_breakdown": {
                "self_host": host.get("self_host"),
                "windows_ok": host.get("windows_ok"),
                "deps_fragility": host.get("deps_fragility"),
                "license_ok": host.get("license_ok"),
                "ops_ease": host.get("ops_ease"),
            },
            "hardware": hw,
            "verdict": find.get("verdict") or "",
            "pros": list(find.get("pros") or []),
            "cons": list(find.get("cons") or []),
            "updated_at": r.get("updated_at"),
        })
    return {
        "categories": load_catalog()["categories"],
        "models": load_catalog()["models"],
        "rows": rows,
        "details": details,
        "hardware_note": "Primary harness: RTX 4090-class GPU, Windows, torch.compile off unless noted.",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
