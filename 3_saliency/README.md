# Saliency Prediction (Assignment 3)

This folder implements the saliency prediction experiment per `AGENT.md`.
No command-line flags are required; configuration lives in code defaults.

## Files
- `dataset_utils.py` — load paired Stimuli / FIXATIONMAPS, augment, and build dataloaders.
- `model.py` — UNet-style decoder with ResNet34 encoder (ImageNet init).
- `metrics.py` — CC, KL divergence, NSS, MAE.
- `train.py` — end-to-end training; saves best checkpoint, curves, and qualitative examples.
- `predict.py` — loads checkpoint and runs on test split, writing metrics and sample outputs.

## How to run (uv friendly)
1) Train + auto test (will auto-extract the zips if only `dataset/3-Saliency-*.zip` exist):
   ```bash
   uv run 3_saliency/train.py
   ```
   - Saves best checkpoint to `3_saliency/models/saliency_unet.pth`
   - Writes curves and validation examples to `3_saliency/reports/`
   - If the test zip/folder exists, also saves `test_metrics.txt` under the same reports folder.

2) Optional: standalone test/inference (e.g., different checkpoint/path):
   ```bash
   uv run 3_saliency/predict.py
   ```

Outputs land in `3_saliency/models/` and `3_saliency/reports/`.
