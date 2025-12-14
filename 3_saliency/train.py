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
    batch_size: int = 12  # Larger batch for more stable gradients
    epochs: int = 40  # More epochs for better convergence
    lr: float = 5e-4  # Higher initial LR for OneCycleLR
    weight_decay: float = 5e-5  # Reduced weight decay
    val_split: float = 0.10  # Smaller validation, more training data
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_path: Path = Path("3_saliency/models/saliency_unet.pth")
    report_dir: Path = Path("3_saliency/reports")
    examples_dir: Path = Path("3_saliency/reports/examples")
    use_amp: bool = torch.cuda.is_available()
    augment: bool = True  # Data augmentation for better generalization


def mixed_precision_enabled(cfg: Config) -> bool:
    return cfg.use_amp and cfg.device.startswith("cuda")


def save_curves(history: Dict[str, List[float]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves
    axes[0, 0].plot(epochs, history["train_loss"], label="train", linewidth=2)
    axes[0, 0].plot(epochs, history["val_loss"], label="val", linewidth=2)
    axes[0, 0].set_xlabel("Epoch", fontsize=12)
    axes[0, 0].set_ylabel("Loss", fontsize=12)
    axes[0, 0].set_title("Training & Validation Loss", fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # CC curve
    axes[0, 1].plot(
        epochs, history["cc"], label="val CC", color="green", linewidth=2
    )
    axes[0, 1].set_xlabel("Epoch", fontsize=12)
    axes[0, 1].set_ylabel("CC Score", fontsize=12)
    axes[0, 1].set_title(
        "Correlation Coefficient (Higher is Better)", fontsize=14
    )
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # MAE curve
    axes[1, 0].plot(
        epochs, history["mae"], label="val MAE", color="orange", linewidth=2
    )
    axes[1, 0].set_xlabel("Epoch", fontsize=12)
    axes[1, 0].set_ylabel("MAE Score", fontsize=12)
    axes[1, 0].set_title("Mean Absolute Error (Lower is Better)", fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # KLD curve
    axes[1, 1].plot(
        epochs, history["kld"], label="val KLD", color="red", linewidth=2
    )
    axes[1, 1].set_xlabel("Epoch", fontsize=12)
    axes[1, 1].set_ylabel("KLD Score", fontsize=12)
    axes[1, 1].set_title("KL Divergence (Lower is Better)", fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "training_curves.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved enhanced curves to {out_path}")


def save_predictions(
    model: UNetSaliency,
    loader: DataLoader,
    out_dir: Path,
    device: torch.device,
    max_batches: int = 2,
):
    """Save prediction visualizations with overlay effects."""
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
                img_vis = (
                    (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
                )

                # Convert saliency maps to numpy
                gt_map = gt.squeeze().numpy()
                pr_map = pr.squeeze().numpy()

                # Create overlays with hot colormap
                hot_cmap = plt.get_cmap("hot")
                gt_colored = hot_cmap(gt_map)[..., :3]  # Remove alpha channel
                pr_colored = hot_cmap(pr_map)[..., :3]

                # Blend: 60% original image + 40% saliency heatmap
                alpha = 0.4
                gt_overlay = img_vis * (1 - alpha) + gt_colored * alpha
                pr_overlay = img_vis * (1 - alpha) + pr_colored * alpha

                # Create figure with 5 subplots
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                # Row 1: Original, GT heatmap, GT overlay
                axes[0, 0].imshow(img_vis)
                axes[0, 0].set_title(
                    "Original Image", fontsize=12, fontweight="bold"
                )
                axes[0, 0].axis("off")

                axes[0, 1].imshow(gt_map, cmap="hot")
                axes[0, 1].set_title(
                    "Ground Truth Saliency", fontsize=12, fontweight="bold"
                )
                axes[0, 1].axis("off")

                axes[0, 2].imshow(gt_overlay)
                axes[0, 2].set_title(
                    "GT Overlay", fontsize=12, fontweight="bold"
                )
                axes[0, 2].axis("off")

                # Row 2: Original (duplicate for symmetry), Pred heatmap, Pred overlay
                axes[1, 0].imshow(img_vis)
                axes[1, 0].set_title(
                    "Original Image", fontsize=12, fontweight="bold"
                )
                axes[1, 0].axis("off")

                axes[1, 1].imshow(pr_map, cmap="hot")
                axes[1, 1].set_title(
                    "Predicted Saliency", fontsize=12, fontweight="bold"
                )
                axes[1, 1].axis("off")

                axes[1, 2].imshow(pr_overlay)
                axes[1, 2].set_title(
                    "Prediction Overlay", fontsize=12, fontweight="bold"
                )
                axes[1, 2].axis("off")

                plt.tight_layout()
                out_path = out_dir / f"batch{b_idx}_sample{i}_overlay.png"
                plt.savefig(out_path, dpi=200, bbox_inches="tight")
                plt.close()
                # print(f"Saved overlay visualization: {out_path.name}")


def cc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Correlation Coefficient loss (maximize CC = minimize 1-CC).

    Returns 1-CC so the loss is always positive:
    - CC = 1.0 (perfect) → loss = 0.0
    - CC = 0.0 (no correlation) → loss = 1.0
    - CC = -1.0 (inverse) → loss = 2.0
    """
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    p_mean = p.mean(dim=1, keepdim=True)
    t_mean = t.mean(dim=1, keepdim=True)
    p_centered = p - p_mean
    t_centered = t - t_mean
    numerator = (p_centered * t_centered).sum(dim=1)
    denominator = torch.sqrt(
        (p_centered.pow(2).sum(dim=1) + 1e-8)
        * (t_centered.pow(2).sum(dim=1) + 1e-8)
    )
    cc = numerator / denominator
    return 1.0 - cc.mean()  # minimize (1-CC) = maximize CC, always positive


def kld_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL Divergence loss."""
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    p = p / (p.sum(dim=1, keepdim=True) + 1e-8)
    t = t / (t.sum(dim=1, keepdim=True) + 1e-8)
    kld = (t * torch.log((t + 1e-8) / (p + 1e-8))).sum(dim=1)
    return kld.mean()


def compute_loss(
    pred_logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Combined loss: BCE + CC + KLD with balanced weights.

    Simplified and balanced design for better convergence:
    - BCE: pixel-level accuracy
    - CC: spatial correlation (primary metric)
    - KLD: distribution similarity
    """
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    pred = torch.sigmoid(pred_logits)
    cc = cc_loss(pred, target)
    kld = kld_loss(pred, target)
    # Balanced combination: equal emphasis on BCE and CC
    return 2.0 * bce + 1.5 * cc + 0.5 * kld


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
    # Use OneCycleLR for better convergence with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # 30% warmup
        div_factor=25,  # initial_lr = max_lr/25
        final_div_factor=1e4,  # final_lr = max_lr/1e4
    )
    scaler = GradScaler() if mixed_precision_enabled(cfg) else None

    history = {
        "train_loss": [],
        "val_loss": [],
        "cc": [],
        "mae": [],
        "kld": [],
    }
    best_cc = -1e9
    best_state = None
    patience = 12  # Increased patience for better exploration
    min_delta = 0.0005  # Minimum improvement threshold
    patience_counter = 0

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

            # OneCycleLR steps after each batch
            scheduler.step()

        train_loss = running_loss / max(count, 1)

        val_loss, val_metrics = evaluate(model, val_loader, device)
        # OneCycleLR scheduler.step() called per batch, not per epoch

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["cc"].append(val_metrics["cc"])
        history["mae"].append(val_metrics["mae"])
        history["kld"].append(val_metrics["kld"])

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{cfg.epochs} - LR {current_lr:.2e} - "
            f"train_loss {train_loss:.4f} val_loss {val_loss:.4f} "
            f"CC {val_metrics['cc']:.4f} MAE {val_metrics['mae']:.4f} KLD {val_metrics['kld']:.4f}"
        )

        # Save best and early stopping with minimum improvement threshold
        if val_metrics["cc"] > best_cc + min_delta:
            best_cc = val_metrics["cc"]
            best_state = model.state_dict()
            patience_counter = 0
            cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "best_cc": best_cc,
                    "epoch": epoch,
                },
                cfg.model_path,
            )
            print(
                f"  -> Saved new best model (CC: {best_cc:.4f}) to {cfg.model_path}"
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

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
