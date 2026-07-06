#!/usr/bin/env bash
# Run on RunPod when git pull has no updates — applies FLP patches + env, then starts worker.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
FLP_ROOT="${FLP_ROOT:-/workspace/FasterLivePortrait}"

echo "==> Applying FasterLivePortrait compatibility patches…"
if [[ -f "$SCRIPT_DIR/patch_flp_compat.py" ]]; then
  python3 "$SCRIPT_DIR/patch_flp_compat.py" "$FLP_ROOT"
else
  echo "WARN: patch_flp_compat.py missing — run git pull on your PC and push first."
  echo "      Applying inline patches…"
  python3 - <<'PY'
import re, sys
from pathlib import Path
root = Path("/workspace/FasterLivePortrait")

def patch_file(path, fn):
    if path.is_file():
        fn(path)

def torch_load(p):
    t = p.read_text()
    if "weights_only=False" not in t:
        t2, n = re.subn(r'torch\.load\(([^)]+)\)',
            lambda m: m.group(0) if "weights_only" in m.group(0) else m.group(0)[:-1]+", weights_only=False)", t)
        if n: p.write_text(t2); print("patched torch.load", p)

def hubert(p):
    t = p.read_text()
    m = "output_attentions = False  # patched for transformers sdpa"
    if m in t: return
    b = re.search(r'(\s*)self\.config\.output_attentions\s*=\s*True\s*\n\s*output_attentions\s*=\s*output_attentions if output_attentions is not None else self\.config\.output_attentions', t)
    if b:
        t = t[:b.start()] + f"{b.group(1)}{m}" + t[b.end():]
        p.write_text(t); print("patched hubert", p)

def dit(p):
    t = p.read_text()
    o = "HubertModel.from_pretrained(audio_encoder_path)"
    n = 'HubertModel.from_pretrained(audio_encoder_path, attn_implementation="eager")'
    if n not in t and o in t:
        p.write_text(t.replace(o, n)); print("patched dit", p)

def realtime(p):
    t = p.read_text()
    o = 'realtime = kwargs.get("realtime", False)'
    n = 'realtime = kwargs.pop("realtime", False)'
    if o in t and n not in t:
        p.write_text(t.replace(o, n)); print("patched realtime pop", p)

def warping_cpu(p):
    t = p.read_text()
    marker = "# patched: warping_spade 5D GridSample"
    if marker in t: return
    o = ("        self.debug = kwargs.get(\"debug\", False)\n"
         "        providers = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']\n")
    n = ("        self.debug = kwargs.get(\"debug\", False)\n"
         f"        {marker}\n"
         "        if \"warping_spade\" in str(model_path):\n"
         "            providers = [\"CPUExecutionProvider\"]\n"
         "        else:\n"
         "            providers = [\"CUDAExecutionProvider\", \"CPUExecutionProvider\"]\n")
    if o in t:
        p.write_text(t.replace(o, n)); print("patched warping_spade CPU", p)

patch_file(root / "src/pipelines/joyvasa_audio_to_motion_pipeline.py", torch_load)
patch_file(root / "src/models/JoyVASA/hubert.py", hubert)
patch_file(root / "src/models/JoyVASA/dit_talking_head.py", dit)
patch_file(root / "src/pipelines/faster_live_portrait_pipeline.py", realtime)
patch_file(root / "src/models/predictor.py", warping_cpu)
print("inline patches done")
PY
fi

echo "==> Fixing onnxruntime (CUDA 12)…"
if [[ -f "$SCRIPT_DIR/fix_onnx_cuda.sh" ]]; then
  bash "$SCRIPT_DIR/fix_onnx_cuda.sh"
else
  python3 -m pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
  python3 -m pip install --force-reinstall --no-cache-dir "onnxruntime-gpu==1.20.2"
  python3 -m pip install "numpy>=1.22,<2.5"
fi

echo "==> Updating .env…"
bash "$SCRIPT_DIR/apply_liveportrait_env.sh"

echo "==> Starting worker…"
set -a
# shellcheck disable=SC1091
source <(grep -v '^#' .env | sed 's/^/export /')
set +a
python3 main.py
