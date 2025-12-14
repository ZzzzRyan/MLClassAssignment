"""
医学图像分类训练脚本
用于眼底图像的患病/正常二分类任务
"""

import argparse
from pathlib import Path

import dataset_utils
import numpy as np
import torch
import torch.nn.functional as F
from evaluation import (
    compute_metrics,
    find_best_threshold,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_training_history,
    print_metrics_report,
    save_prediction_results,
)
from model import count_parameters, create_model
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import amp
from torch.utils.data import DataLoader, TensorDataset


def get_data_augmentation():
    """获取数据增强变换"""
    import torchvision.transforms as T

    train_transform = T.Compose(
        [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ]
    )
    return train_transform


def augment_batch(X: torch.Tensor, transform) -> torch.Tensor:
    """对batch数据进行增强"""
    import torchvision.transforms.functional as TF

    augmented = []
    for img in X:
        # 转换为 PIL 进行增强
        img_pil = TF.to_pil_image(img)
        img_aug = transform(img_pil)
        img_tensor = TF.to_tensor(img_aug)
        augmented.append(img_tensor)
    return torch.stack(augmented)


def get_class_weights(
    y: np.ndarray, neg_multiplier: float = 1.0
) -> torch.Tensor:
    """计算类别权重以处理类别不平衡，同时允许放大正常类权重以降低误报"""
    class_counts = np.bincount(y)
    total = len(y)
    class_weights = total / (len(class_counts) * class_counts)
    if len(class_weights) > 0:
        class_weights[0] *= neg_multiplier
    return torch.FloatTensor(class_weights)


def evaluate(model, dataloader, device, threshold=0.5, return_probs=False):
    """评估模型"""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=1)
            if probs.shape[1] == 2:
                pos_prob = probs[:, 1].cpu().numpy()
                preds = (pos_prob >= threshold).astype(np.int64)
            else:
                pos_prob = probs.max(dim=1)[0].cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(yb.numpy())
            all_probs.append(pos_prob)  # 患病概率或最大概率

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    if return_probs:
        return all_labels, all_preds, all_probs
    return all_labels, all_preds


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    scaler,
    class_weights=None,
    label_smoothing=0.1,
    fp_penalty=0.0,
    use_augmentation=False,
):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    transform = get_data_augmentation() if use_augmentation else None

    for xb, yb in dataloader:
        # 数据增强
        if transform is not None:
            xb = augment_batch(xb, transform)

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with amp.autocast(device.type):
                logits = model(xb)
                ce_loss = F.cross_entropy(
                    logits,
                    yb,
                    weight=class_weights,
                    label_smoothing=label_smoothing,
                )
                probs = F.softmax(logits, dim=1)
                pos_probs = probs[:, 1]
                normal_mask = (yb == 0).float()
                fp_loss = (pos_probs * normal_mask).mean()
                loss = ce_loss + fp_penalty * fp_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            ce_loss = F.cross_entropy(
                logits,
                yb,
                weight=class_weights,
                label_smoothing=label_smoothing,
            )
            probs = F.softmax(logits, dim=1)
            pos_probs = probs[:, 1]
            normal_mask = (yb == 0).float()
            fp_loss = (pos_probs * normal_mask).mean()
            loss = ce_loss + fp_penalty * fp_loss
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total_samples += xb.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def save_example_images(X, y_true, y_pred, y_prob, out_dir: Path, n=16):
    """保存部分预测样本示例"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 选择一些正确和错误的预测样本
    correct_idx = np.where(y_pred == y_true)[0]
    wrong_idx = np.where(y_pred != y_true)[0]

    n_correct = min(n // 2, len(correct_idx))
    n_wrong = min(n // 2, len(wrong_idx))

    rng = np.random.RandomState(42)
    selected_correct = (
        rng.choice(correct_idx, size=n_correct, replace=False)
        if n_correct > 0
        else []
    )
    selected_wrong = (
        rng.choice(wrong_idx, size=n_wrong, replace=False)
        if n_wrong > 0
        else []
    )

    selected = list(selected_correct) + list(selected_wrong)

    for i, idx in enumerate(selected):
        # X 形状为 (C, H, W)
        img = X[idx].transpose(1, 2, 0)  # 转换为 (H, W, C)
        img = (img * 255).astype(np.uint8)
        im = Image.fromarray(img)

        status = "correct" if y_pred[idx] == y_true[idx] else "wrong"
        true_label = "disease" if y_true[idx] == 1 else "normal"
        pred_label = "disease" if y_pred[idx] == 1 else "normal"
        prob = y_prob[idx]

        fn = (
            out_dir
            / f"{status}_{i}_true_{true_label}_pred_{pred_label}_prob_{prob:.3f}.jpg"
        )
        im.save(fn)


def main():
    parser = argparse.ArgumentParser(
        description="Medical Image Classification Training"
    )
    parser.add_argument(
        "--train-zip", default="dataset/2-MedImage-TrainSet.zip"
    )
    parser.add_argument("--test-zip", default="dataset/2-MedImage-TestSet.zip")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--val-split", type=float, default=0.15, help="Validation split ratio"
    )
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Epochs to train only classifier",
    )
    parser.add_argument(
        "--neg-weight-multiplier",
        type=float,
        default=2.0,
        help="Extra multiplier applied to normal-class weight to penalize false positives",
    )
    parser.add_argument(
        "--fp-penalty",
        type=float,
        default=0.3,
        help="Penalty strength for predicting high disease probability on normal samples",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.85,
        help="Minimum precision when auto-selecting decision threshold",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help="Manual decision threshold (skip auto tuning)",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.1,
        help="Minimum threshold value when sweeping",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.99,
        help="Maximum threshold value when sweeping",
    )
    parser.add_argument(
        "--model-type", default="resnet18", choices=["resnet18", "simple"]
    )
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument(
        "--no-pretrained", dest="pretrained", action="store_false"
    )
    parser.add_argument("--augmentation", action="store_true", default=True)
    parser.add_argument(
        "--no-augmentation", dest="augmentation", action="store_false"
    )
    parser.add_argument("--out", default="2_medimage/models/best_model.pth")
    parser.add_argument("--report-dir", default="2_medimage/reports")
    parser.add_argument("--resize", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据集
    print("Loading datasets...")
    X_train, y_train, train_files = dataset_utils.load_dataset_from_zip(
        args.train_zip, resize=tuple(args.resize)
    )
    X_test, y_test, test_files = dataset_utils.load_dataset_from_zip(
        args.test_zip, resize=tuple(args.resize)
    )

    # 划分验证集
    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=args.val_split,
        stratify=y_train,
        random_state=42,
    )

    X_train_split = X_train[train_idx]
    y_train_split = y_train[train_idx]
    X_val_split = X_train[val_idx]
    y_val_split = y_train[val_idx]

    print(f"Training set: {len(y_train_split)} images")
    print(
        f"  Class distribution: {dataset_utils.get_class_distribution(y_train_split)}"
    )
    print(f"Validation set: {len(y_val_split)} images")
    print(
        f"  Class distribution: {dataset_utils.get_class_distribution(y_val_split)}"
    )
    print(f"Test set: {len(y_test)} images")
    print(
        f"  Class distribution: {dataset_utils.get_class_distribution(y_test)}"
    )

    # 转换为 PyTorch tensors
    X_train_t = torch.from_numpy(X_train_split).float()
    y_train_t = torch.from_numpy(y_train_split).long()
    X_val_t = torch.from_numpy(X_val_split).float()
    y_val_t = torch.from_numpy(y_val_split).long()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    # 计算类别权重
    class_weights = get_class_weights(
        y_train_split, neg_multiplier=args.neg_weight_multiplier
    ).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 创建模型
    model = create_model(
        model_type=args.model_type, num_classes=2, pretrained=args.pretrained
    )
    print(f"Model: {args.model_type}, Parameters: {count_parameters(model):,}")
    model = model.to(device)

    # 微调策略：先冻结backbone，只训练分类器
    if args.pretrained and args.warmup_epochs > 0:
        print(
            f"\nWarmup phase: Freezing backbone for {args.warmup_epochs} epochs"
        )
        for param in model.backbone.parameters():
            param.requires_grad = False
        # 只解冻分类器
        for param in model.backbone.fc.parameters():
            param.requires_grad = True

    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # 混合精度训练
    scaler = amp.GradScaler("cuda") if device.type == "cuda" else None

    # 训练历史记录
    history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
    }

    best_auc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        # 解冻backbone（warmup后）
        if args.pretrained and epoch == args.warmup_epochs + 1:
            print("\nUnfreezing backbone, switching to full fine-tuning")
            for param in model.backbone.parameters():
                param.requires_grad = True
            # 重新创建优化器以包含所有参数
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr * 0.1,  # 降低学习率
                weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=5
            )

        # 训练
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
            fp_penalty=args.fp_penalty,
            use_augmentation=args.augmentation,
        )

        # 在验证集上评估
        y_val, y_val_pred, y_val_prob = evaluate(
            model, val_loader, device, return_probs=True
        )
        val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob)

        scheduler.step(val_metrics["auc"])

        # 记录历史
        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_auc"].append(val_metrics["auc"])

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:2d}/{args.epochs}: "
            f"loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, val_auc={val_metrics['auc']:.4f}, "
            f"lr={current_lr:.2e}"
        )

        # 保存最佳模型
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            print(f"  -> New best model! AUC: {best_auc:.4f}")
        else:
            epochs_no_improve += 1

        # 早停
        if epochs_no_improve >= args.early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    # 加载最佳模型进行最终评估
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 最终评估
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    # 使用验证集确定最佳阈值
    best_threshold = args.decision_threshold
    y_val_full, _, y_val_prob_full = evaluate(
        model, val_loader, device, threshold=0.5, return_probs=True
    )
    if best_threshold is None:
        best_threshold, tuned_val_metrics = find_best_threshold(
            y_val_full,
            y_val_prob_full,
            target_precision=args.target_precision,
            thresholds=np.linspace(
                args.threshold_min, args.threshold_max, 200
            ),
        )
        print(
            f"Selected decision threshold {best_threshold:.3f} "
            f"(val precision={tuned_val_metrics['precision']:.4f}, "
            f"recall={tuned_val_metrics['recall']:.4f})"
        )
    else:
        tuned_preds = (y_val_prob_full >= best_threshold).astype(int)
        tuned_val_metrics = compute_metrics(
            y_val_full, tuned_preds, y_val_prob_full
        )
        print(
            f"Using user-defined threshold {best_threshold:.3f} "
            f"(val precision={tuned_val_metrics['precision']:.4f}, "
            f"recall={tuned_val_metrics['recall']:.4f})"
        )

    y_true, y_pred, y_prob = evaluate(
        model,
        test_loader,
        device,
        threshold=best_threshold,
        return_probs=True,
    )
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics_report(metrics)

    # 保存模型
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "decision_threshold": best_threshold,
            "metrics": metrics,
        },
        out_path,
    )
    print(f"Saved best model to {out_path}")

    # 生成报告图表
    report_dir = Path(args.report_dir)
    figures_dir = report_dir / "figures"

    print("\nGenerating evaluation figures...")

    # 1. 训练曲线
    plot_training_history(history, figures_dir / "training_curves.png")

    # 2. ROC曲线
    plot_roc_curve(y_true, y_prob, figures_dir / "roc_curve.png")

    # 3. 混淆矩阵
    plot_confusion_matrix(y_true, y_pred, figures_dir / "confusion_matrix.png")

    # 4. PR曲线
    plot_precision_recall_curve(y_true, y_prob, figures_dir / "pr_curve.png")

    # 5. 保存预测结果（兼容MATLAB格式）
    save_prediction_results(
        y_true, y_pred, y_prob, report_dir / "predictions.txt"
    )

    # 6. 保存示例图像
    examples_dir = report_dir / "examples"
    save_example_images(X_test, y_true, y_pred, y_prob, examples_dir)
    print(f"Saved example images to {examples_dir}")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
