"""
Inference / evaluation script for saliency model.
Loads the saved checkpoint and runs on the test set, writing metrics and example outputs.
"""

from pathlib import Path

import dataset_utils
import matplotlib.pyplot as plt
import metrics
import torch
from model import load_model
from torch.utils.data import DataLoader


def run_inference(
    checkpoint: str = "3_saliency/models/saliency_unet.pth",
    test_root: str = "dataset/3-Saliency-TestSet.zip",
    image_size=(256, 256),
    batch_size: int = 6,
    num_workers: int = 4,
    out_dir: str = "3_saliency/reports/test_examples",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds = dataset_utils.SaliencyDataset(
        test_root, resize=image_size, augment=False
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = load_model(checkpoint, map_location=device)
    model.to(device)
    model.eval()

    total = 0
    metric_accum = {"cc": 0.0, "kld": 0.0, "nss": 0.0, "mae": 0.0}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader):
            xb = xb.to(device)
            yb = yb.to(device)
            pred = torch.sigmoid(model(xb))

            # metrics
            batch_metrics = metrics.compute_all_metrics(pred, yb)
            bs = xb.size(0)
            total += bs
            for k in metric_accum:
                metric_accum[k] += batch_metrics[k] * bs

            # save a few examples
            if batch_idx < 2:
                mean = (
                    torch.tensor([0.485, 0.456, 0.406])
                    .view(3, 1, 1)
                    .to(device)
                )
                std = (
                    torch.tensor([0.229, 0.224, 0.225])
                    .view(3, 1, 1)
                    .to(device)
                )
                xb_vis = (xb * std + mean).clamp(0, 1)
                for i in range(min(bs, 4)):
                    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                    axes[0].imshow(xb_vis[i].cpu().permute(1, 2, 0))
                    axes[0].set_title("Input")
                    axes[1].imshow(yb[i].cpu().squeeze(), cmap="hot")
                    axes[1].set_title("GT")
                    axes[2].imshow(pred[i].cpu().squeeze(), cmap="hot")
                    axes[2].set_title("Pred")
                    for ax in axes:
                        ax.axis("off")
                    plt.tight_layout()
                    plt.savefig(
                        out_dir / f"batch{batch_idx}_sample{i}.png", dpi=150
                    )
                    plt.close()

    for k in metric_accum:
        metric_accum[k] /= max(total, 1)

    print("Test metrics:", metric_accum)
    with open(out_dir / "metrics.txt", "w") as f:
        for k, v in metric_accum.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"Saved report to {out_dir}")


if __name__ == "__main__":
    run_inference()
