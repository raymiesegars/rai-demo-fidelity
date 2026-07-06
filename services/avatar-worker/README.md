# GPU lip-sync upgrade (RunPod)

When `AVATAR_MODE=gpu`, install FasterLivePortrait + JoyVASA on the pod:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login --token $HUGGINGFACE_TOKEN

git clone https://github.com/warmshao/FasterLivePortrait.git /workspace/FasterLivePortrait
cd /workspace/FasterLivePortrait
pip install -r requirements.txt

huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints
huggingface-cli download jdh-algo/JoyVASA --local-dir ./checkpoints/JoyVASA
huggingface-cli download TencentGameMate/chinese-hubert-base --local-dir ./checkpoints/chinese-hubert-base
```

Use **lip-only** animation region when source is `alan-loop.mp4`.

Until GPU pipeline is wired, `AVATAR_MODE=mock` streams the ping-pong loop while agent TTS audio plays separately — sufficient for end-to-end demo validation.
