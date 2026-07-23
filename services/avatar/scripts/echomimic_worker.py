"""EchoMimicV3-Flash warm JSONL worker (image + audio → mp4).

Loads WanFunInpaintAudioPipeline once; prepare/infer/quit over stdin.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor


def _reply(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def get_sample_size(pil_img, sample_size):
    w, h = pil_img.size
    ori_a = w * h
    default_a = sample_size[0] * sample_size[1]
    if default_a < ori_a:
        ratio_a = math.sqrt(ori_a / sample_size[0] / sample_size[1])
        w = w / ratio_a // 16 * 16
        h = h / ratio_a // 16 * 16
    else:
        w = w // 16 * 16
        h = h // 16 * 16
    return [int(h), int(w)]


def loudness_norm(audio_array, sr=16000, lufs=-23):
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > 100:
            return audio_array
        return pyln.normalize.loudness(audio_array, loudness, lufs)
    except Exception:
        return audio_array


def get_audio_embed(mel_input, wav2vec_feature_extractor, audio_encoder, video_length, sr=16000, device="cpu"):
    audio_feature = np.squeeze(
        wav2vec_feature_extractor(mel_input, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device).unsqueeze(0)
    with torch.no_grad():
        embeddings = audio_encoder(
            audio_feature, seq_len=int(video_length), output_hidden_states=True
        )
    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")
    return audio_emb.cpu().detach()


def main() -> int:
    root = Path(os.environ.get("ECHOMIMIC_ROOT", ".")).resolve()
    flash = Path(os.environ.get("ECHOMIMIC_FLASH_DIR", root / "flash")).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))

    model_name = str(flash / "Wan2.1-Fun-V1.1-1.3B-InP")
    wav2vec_dir = str(flash / "chinese-wav2vec2-base")
    transformer_path = str(flash / "transformer" / "diffusion_pytorch_model.safetensors")
    config_path = str(root / "config" / "config.yaml")
    steps = int(os.environ.get("ECHOMIMIC_STEPS", "8") or "8")
    size = int(os.environ.get("ECHOMIMIC_SIZE", "512") or "512")
    sample_size = [size, size]
    fps = 25
    weight_dtype = torch.bfloat16
    prompt_default = os.environ.get(
        "ECHOMIMIC_PROMPT", "A person is speaking naturally to the camera."
    )

    for p, label in (
        (model_name, "Wan base"),
        (wav2vec_dir, "wav2vec"),
        (transformer_path, "flash transformer"),
        (config_path, "config"),
    ):
        if not Path(p).exists():
            _reply({"ok": False, "cmd": "ready", "error": f"missing {label}: {p}"})
            return 1

    t0 = time.time()
    try:
        from src.dist import set_multi_gpus_devices
        from src.wan_vae import AutoencoderKLWan
        from src.wan_image_encoder import CLIPModel
        from src.wan_text_encoder import WanT5EncoderModel
        from src.wan_transformer3d_audio_2512 import (
            WanTransformerAudioMask3DModel as WanTransformer,
        )
        from src.pipeline_wan_fun_inpaint_audio_2512 import WanFunInpaintAudioPipeline
        from src.utils import filter_kwargs, get_image_to_video_latent2, save_videos_grid
        from src.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from src.cache_utils import get_teacache_coefficients
        from src.wav2vec2 import Wav2Vec2Model

        device = set_multi_gpus_devices(1, 1)
        config = OmegaConf.load(config_path)

        audio_encoder = Wav2Vec2Model.from_pretrained(
            wav2vec_dir, local_files_only=True
        ).to("cpu")
        audio_encoder.feature_extractor._freeze_parameters()
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec_dir, local_files_only=True
        )

        transformer = WanTransformer.from_pretrained(
            os.path.join(
                model_name,
                config["transformer_additional_kwargs"].get("transformer_subpath", "transformer"),
            ),
            transformer_additional_kwargs=OmegaConf.to_container(
                config["transformer_additional_kwargs"]
            ),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        from safetensors.torch import load_file

        state_dict = load_file(transformer_path)
        transformer.load_state_dict(state_dict, strict=False)

        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
            additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
        ).to(weight_dtype)

        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(
                model_name,
                config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
            )
        )
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(
                model_name,
                config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
            ),
            additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        ).eval()

        clip_image_encoder = CLIPModel.from_pretrained(
            os.path.join(
                model_name,
                config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"),
            )
        ).to(weight_dtype).eval()

        sched_cfg = OmegaConf.to_container(config["scheduler_kwargs"])
        sched_cfg["shift"] = 1
        scheduler = FlowUniPCMultistepScheduler(
            **filter_kwargs(FlowUniPCMultistepScheduler, sched_cfg)
        )

        pipeline = WanFunInpaintAudioPipeline(
            transformer=transformer,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            clip_image_encoder=clip_image_encoder,
        )
        pipeline.to(device=device)

        coefficients = get_teacache_coefficients(model_name)
        if coefficients is not None:
            pipeline.transformer.enable_teacache(
                coefficients, steps, 0.1, num_skip_start_steps=5, offload=False
            )
    except Exception as e:
        _reply({"ok": False, "cmd": "ready", "error": str(e), "trace": traceback.format_exc()})
        return 1

    prepared_image: str | None = None
    _reply({
        "ok": True,
        "cmd": "ready",
        "load_s": round(time.time() - t0, 1),
        "steps": steps,
        "size": size,
    })

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _reply({"ok": False, "error": f"bad json: {e}"})
            continue

        cmd = req.get("cmd")
        if cmd == "quit":
            _reply({"ok": True, "cmd": "quit"})
            return 0

        if cmd == "prepare":
            try:
                src = Path(req["source_path"]).resolve()
                if not src.is_file():
                    raise FileNotFoundError(str(src))
                prepared_image = str(src)
                _reply({"ok": True, "cmd": "prepare", "source": prepared_image})
            except Exception as e:
                _reply({
                    "ok": False,
                    "cmd": "prepare",
                    "error": str(e),
                    "trace": traceback.format_exc(),
                })
            continue

        if cmd == "infer":
            try:
                if not prepared_image:
                    raise RuntimeError("Call prepare first")
                audio_path = Path(req["audio_path"]).resolve()
                output_path = Path(req["output_path"]).resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                prompt = req.get("prompt") or prompt_default
                max_frames = int(os.environ.get("ECHOMIMIC_MAX_FRAMES", "81") or "81")

                ref_image = Image.open(prepared_image).convert("RGB")
                mel_input, sr = librosa.load(str(audio_path), sr=16000)
                mel_input = loudness_norm(mel_input, sr)
                audio_dur = len(mel_input) / float(sr)
                video_length_actual = min(int(audio_dur * fps), max_frames)
                ratio = vae.config.temporal_compression_ratio
                video_length_actual = (
                    int((video_length_actual - 1) // ratio * ratio) + 1
                    if video_length_actual != 1
                    else 1
                )
                mel_input = mel_input[: int(video_length_actual / fps * sr)]

                t1 = time.time()
                audio_feature = get_audio_embed(
                    mel_input,
                    wav2vec_feature_extractor,
                    audio_encoder,
                    video_length_actual,
                    sr=16000,
                    device="cpu",
                )
                audio_embeds = audio_feature.to(device=device, dtype=weight_dtype)
                indices = (torch.arange(2 * 2 + 1) - 2) * 1
                center_indices = torch.arange(0, video_length_actual, 1).unsqueeze(1) + indices.unsqueeze(0)
                center_indices = torch.clamp(center_indices, min=0, max=audio_embeds.shape[0] - 1)
                audio_embeds = audio_embeds[center_indices].unsqueeze(0).to(device=device)

                sample_h, sample_w = get_sample_size(ref_image, sample_size)
                input_video, input_video_mask, clip_image = get_image_to_video_latent2(
                    ref_image,
                    None,
                    video_length=video_length_actual,
                    sample_size=[sample_h, sample_w],
                )

                generator = torch.Generator(device=device).manual_seed(
                    int(os.environ.get("ECHOMIMIC_SEED", "43") or "43")
                )
                sample = pipeline(
                    prompt,
                    num_frames=video_length_actual,
                    negative_prompt=os.environ.get(
                        "ECHOMIMIC_NEG",
                        "Bad hands. Bad fingers. Unclear gestures, broken hands.",
                    ),
                    audio_embeds=audio_embeds,
                    audio_scale=1.0,
                    ip_mask=None,
                    use_un_ip_mask=False,
                    height=sample_h,
                    width=sample_w,
                    generator=generator,
                    neg_scale=1.0,
                    neg_steps=0,
                    use_dynamic_cfg=False,
                    use_dynamic_acfg=False,
                    guidance_scale=float(os.environ.get("ECHOMIMIC_GUIDANCE", "6.0") or "6.0"),
                    audio_guidance_scale=float(
                        os.environ.get("ECHOMIMIC_AUDIO_GUIDANCE", "3.0") or "3.0"
                    ),
                    num_inference_steps=steps,
                    video=input_video,
                    mask_video=input_video_mask,
                    clip_image=clip_image,
                    cfg_skip_ratio=0.0,
                    shift=5.0,
                ).videos

                tmp = output_path.with_suffix(".tmp.mp4")
                save_videos_grid(sample[:, :, :video_length_actual], str(tmp), fps=fps)

                # Re-encode to plain mp4v for OpenCV read (drop audio track)
                frames: list[np.ndarray] = []
                cap = cv2.VideoCapture(str(tmp))
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    frames.append(fr)
                cap.release()
                tmp.unlink(missing_ok=True)
                if not frames:
                    raise RuntimeError("EchoMimic produced no frames")
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, h))
                for fr in frames:
                    writer.write(fr)
                writer.release()

                _reply({
                    "ok": True,
                    "cmd": "infer",
                    "ms": int((time.time() - t1) * 1000),
                    "video_path": str(output_path),
                    "n_frames": len(frames),
                })
            except Exception as e:
                _reply({
                    "ok": False,
                    "cmd": "infer",
                    "error": str(e),
                    "trace": traceback.format_exc(),
                })
            continue

        _reply({"ok": False, "error": f"unknown cmd: {cmd}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
