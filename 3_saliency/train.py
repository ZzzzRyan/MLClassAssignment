"""
End-to-end training script for saliency prediction.

Run directly: `python 3_saliency/train.py`
No command-line arguments are required; tweak the Config dataclass below if needed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import dataset_utils
import matplotlib.pyplot as plt
import metrics
import torch
import torch.nn.functional as F
from model import UNetSaliency, count_parameters
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader


@dataclass
class Config:
    train_root: str = "dataset/3-Saliency-TrainSet.zip"
    test_root: str = "dataset/3-Saliency-TestSet.zip"
    image_size: tuple = (256, 256)
    batch_size: int = 6
    epochs: int = 25
    lr: float = 1e-4
    weight_decay: float = 1e-4
    val_split: float = 0.1
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_path: Path = Path("3_saliency/models/saliency_unet.pth")
    report_dir: Path = Path("3_saliency/reports")
    examples_dir: Path = Path("3_saliency/reports/examples")
    use_amp: bool = torch.cuda.is_available()
    augment: bool = True


def mixed_precision_enabled(cfg: Config) -> bool:
    return cfg.use_amp and cfg.device.startswith("cuda")


def save_curves(history: Dict[str, List[float]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["cc"], label="val CC")
    plt.plot(epochs, history["mae"], label="val MAE")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved curves to {out_path}")


def save_predictions(
    model: UNetSaliency,
    loader: DataLoader,
    out_dir: Path,
    device: torch.device,
    max_batches: int = 2,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for b_idx, (xb, yb) in enumerate(loader):
            if b_idx >= max_batches:
                break
            xb = xb.to(device)
            yb = yb.to(device)
            pred = torch.sigmoid(model(xb))
            for i in range(min(len(xb), 4)):
                img = xb[i].detach().cpu()
                gt = yb[i].detach().cpu()
                pr = pred[i].detach().cpu()

                # denormalize image for visualization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_vis = (img * std + mean).clamp(0, 1)

                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                axes[0].imshow(img_vis.permute(1, 2, 0))
                axes[0].set_title("Input")
                axes[1].imshow(gt.squeeze(), cmap="hot")
                axes[1].set_title("GT")
                axes[2].imshow(pr.squeeze(), cmap="hot")
                axes[2].set_title("Pred")
                for ax in axes:
                    ax.axis("off")
                plt.tight_layout()
                out_path = out_dir / f"batch{b_idx}_sample{i}.png"
                plt.savefig(out_path, dpi=150)
                plt.close()


def compute_loss(
    pred_logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    pred = torch.sigmoid(pred_logits)
    l1 = F.l1_loss(pred, target)
    return 0.7 * bce + 0.3 * l1


def evaluate(model: UNetSaliency, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total = 0
    metric_accum: Dict[str, float] = {
        "cc": 0.0,
        "kld": 0.0,
        "nss": 0.0,
        "mae": 0.0,
    }

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = compute_loss(logits, yb)
            total_loss += float(loss.item()) * xb.size(0)
            total += xb.size(0)
            pred = torch.sigmoid(logits)
            batch_metrics = metrics.compute_all_metrics(pred, yb)
            for k in metric_accum:
                metric_accum[k] += batch_metrics[k] * xb.size(0)

    avg_loss = total_loss / max(total, 1)
    for k in metric_accum:
        metric_accum[k] /= max(total, 1)
    return avg_loss, metric_accum


def train():
    cfg = Config()
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = dataset_utils.get_dataloaders(
        cfg.train_root,
        resize=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        val_split=cfg.val_split,
        augment=cfg.augment,
    )

    # Model
    model = UNetSaliency(pretrained=True).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    scaler = GradScaler() if mixed_precision_enabled(cfg) else None

    history = {"train_loss": [], "val_loss": [], "cc": [], "mae": []}
    best_cc = -1e9
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                with autocast():
                    logits = model(xb)
                    loss = compute_loss(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb)
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            count += xb.size(0)

        train_loss = running_loss / max(count, 1)

        val_loss, val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["cc"].append(val_metrics["cc"])
        history["mae"].append(val_metrics["mae"])

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} - train_loss {train_loss:.4f} "
            f"val_loss {val_loss:.4f} CC {val_metrics['cc']:.4f} MAE {val_metrics['mae']:.4f}"
        )

        # Save best
        if val_metrics["cc"] > best_cc:
            best_cc = val_metrics["cc"]
            best_state = model.state_dict()
            cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                },
                cfg.model_path,
            )
            print(f"  -> Saved new best model to {cfg.model_path}")

    # Reports
    if best_state is not None:
        model.load_state_dict(best_state)

    save_curves(history, cfg.report_dir)
    save_predictions(model, val_loader, cfg.examples_dir, device)
    print("Training complete. Best CC:", best_cc)

    # Optional test evaluation (auto-extracts if zip exists)
    try:
        test_ds = dataset_utils.SaliencyDataset(
            cfg.test_root, resize=cfg.image_size, augment=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,  # avoid Windows multiprocessing pickle issue
            pin_memory=True,
        )
        test_loss, test_metrics = evaluate(model, test_loader, device)
        print(
            f"Test -> loss {test_loss:.4f} CC {test_metrics['cc']:.4f} "
            f"MAE {test_metrics['mae']:.4f} NSS {test_metrics['nss']:.4f} KLD {test_metrics['kld']:.4f}"
        )
        cfg.report_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg.report_dir / "test_metrics.txt", "w") as f:
            for k, v in test_metrics.items():
                f.write(f"{k}: {v:.6f}\n")
    except FileNotFoundError as e:
        print(f"Skip test evaluation: {e}")


if __name__ == "__main__":
    train()
