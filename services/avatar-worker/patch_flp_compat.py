#!/usr/bin/env python3
"""Patch FasterLivePortrait for RunPod: torch.load, HuBERT/transformers sdpa."""
from __future__ import annotations

import re
import sys
from pathlib import Path

FLP_ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/FasterLivePortrait")


def patch_torch_load(path: Path) -> bool:
    if not path.is_file():
        print(f"skip (missing): {path}")
        return False
    text = path.read_text(encoding="utf-8")
    if "weights_only=False" in text:
        print(f"torch.load already patched: {path}")
        return False
    new_text, n = re.subn(
        r"torch\.load\(([^)]+)\)",
        lambda m: (
            m.group(0)
            if "weights_only" in m.group(0)
            else m.group(0)[:-1] + ", weights_only=False)"
        ),
        text,
    )
    if n == 0:
        print(f"no torch.load in {path}")
        return False
    path.write_text(new_text, encoding="utf-8")
    print(f"patched {n} torch.load call(s) in {path}")
    return True


def patch_hubert(path: Path) -> bool:
    if not path.is_file():
        print(f"skip (missing): {path}")
        return False
    text = path.read_text(encoding="utf-8")
    marker = "output_attentions = False  # patched for transformers sdpa"
    if marker in text:
        print(f"hubert already patched: {path}")
        return False

    block = re.search(
        r"(\s*)self\.config\.output_attentions\s*=\s*True\s*\n"
        r"\s*output_attentions\s*=\s*output_attentions if output_attentions is not None "
        r"else self\.config\.output_attentions",
        text,
    )
    if block:
        indent = block.group(1)
        replacement = f"{indent}{marker}"
        text = text[: block.start()] + replacement + text[block.end() :]
        path.write_text(text, encoding="utf-8")
        print(f"patched hubert forward in {path}")
        return True

    # Fallback: drop config mutation only.
    new_text, n = re.subn(r"\s*self\.config\.output_attentions\s*=\s*True\s*\n", "\n", text, count=1)
    if n:
        path.write_text(new_text, encoding="utf-8")
        print(f"patched hubert (fallback) in {path}")
        return True

    print(f"hubert patch pattern not found: {path}")
    return False


def patch_dit_hubert_load(path: Path) -> bool:
    if not path.is_file():
        print(f"skip (missing): {path}")
        return False
    text = path.read_text(encoding="utf-8")
    old = "HubertModel.from_pretrained(audio_encoder_path)"
    new = 'HubertModel.from_pretrained(audio_encoder_path, attn_implementation="eager")'
    if new in text:
        print(f"dit hubert load already patched: {path}")
        return False
    if old not in text:
        print(f"HubertModel.from_pretrained not found in {path}")
        return False
    text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    n = text.count(new)
    print(f"patched {n} HubertModel.from_pretrained call(s) in {path}")
    return True


def patch_run_with_pkl(path: Path) -> bool:
    """FLP passes realtime positionally and again via **kwargs — breaks on Python 3."""
    if not path.is_file():
        print(f"skip (missing): {path}")
        return False
    text = path.read_text(encoding="utf-8")
    if 'kwargs.pop("realtime"' in text:
        print(f"run_with_pkl already patched: {path}")
        return False
    old = 'realtime = kwargs.get("realtime", False)'
    new = 'realtime = kwargs.pop("realtime", False)'
    if old not in text:
        print(f"run_with_pkl pattern not found: {path}")
        return False
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"patched run_with_pkl realtime kwarg in {path}")
    return True


def patch_warping_spade_cpu(path: Path) -> bool:
    """ONNX CUDA cannot run 5D GridSample in warping_spade; CPU ort can."""
    if not path.is_file():
        print(f"skip (missing): {path}")
        return False
    text = path.read_text(encoding="utf-8")
    marker = "# patched: warping_spade 5D GridSample"
    if marker in text:
        print(f"predictor already patched: {path}")
        return False
    old = (
        "        self.debug = kwargs.get(\"debug\", False)\n"
        "        providers = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']\n"
    )
    new = (
        "        self.debug = kwargs.get(\"debug\", False)\n"
        f"        {marker}\n"
        "        if \"warping_spade\" in str(model_path):\n"
        "            providers = [\"CPUExecutionProvider\"]\n"
        "        else:\n"
        "            providers = [\"CUDAExecutionProvider\", \"CPUExecutionProvider\"]\n"
    )
    if old not in text:
        print(f"predictor providers pattern not found: {path}")
        return False
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"patched warping_spade CPU provider in {path}")
    return True


def apply_patches(root: Path) -> None:
    patch_torch_load(root / "src/pipelines/joyvasa_audio_to_motion_pipeline.py")
    patch_hubert(root / "src/models/JoyVASA/hubert.py")
    patch_dit_hubert_load(root / "src/models/JoyVASA/dit_talking_head.py")
    patch_run_with_pkl(root / "src/pipelines/faster_live_portrait_pipeline.py")
    patch_warping_spade_cpu(root / "src/models/predictor.py")


def main() -> None:
    apply_patches(FLP_ROOT)
    print("FLP compatibility patches applied.")


if __name__ == "__main__":
    main()
