"""
评估模块 - 包含ROC曲线、AUC和其他性能指标的计算与可视化
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as skm


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    计算分类性能指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测为患病(类别1)的概率

    Returns:
        包含各种指标的字典
    """
    # 基本指标
    accuracy = skm.accuracy_score(y_true, y_pred)
    precision = skm.precision_score(y_true, y_pred, zero_division=0)
    recall = skm.recall_score(y_true, y_pred, zero_division=0)  # 灵敏度/召回率
    f1 = skm.f1_score(y_true, y_pred, zero_division=0)

    # 特异度 (Specificity) = TN / (TN + FP)
    tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # AUC
    try:
        auc = skm.roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    # F2分数 (更重视召回率，适合医疗场景)
    f2 = skm.fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,  # 灵敏度
        "sensitivity": recall,
        "specificity": specificity,
        "f1_score": f1,
        "f2_score": f2,
        "auc": auc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: Optional[Path] = None,
    title: str = "ROC Curve",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    绘制ROC曲线

    Args:
        y_true: 真实标签
        y_prob: 预测为患病的概率
        out_path: 保存路径（可选）
        title: 图表标题

    Returns:
        fpr, tpr, auc
    """
    fpr, tpr, thresholds = skm.roc_curve(y_true, y_prob)
    auc = skm.auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color="red", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot(
        [0, 1], [0, 1], color="blue", lw=2, linestyle="--", label="Random"
    )

    # 找到最佳阈值点（Youden's J statistic）
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    plt.scatter(
        fpr[best_idx],
        tpr[best_idx],
        marker="o",
        color="green",
        s=100,
        label=f"Best threshold = {best_threshold:.3f}",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Specificity (False Positive Rate)", fontsize=12)
    plt.ylabel("Sensitivity (True Positive Rate)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved ROC curve to {out_path}")

    plt.close()
    return fpr, tpr, auc


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Optional[Path] = None,
    class_names: list = ["Normal", "Disease"],
) -> np.ndarray:
    """
    绘制混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        out_path: 保存路径
        class_names: 类别名称

    Returns:
        混淆矩阵
    """
    cm = skm.confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 16},
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {out_path}")

    plt.close()
    return cm


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    绘制精确率-召回率曲线

    Args:
        y_true: 真实标签
        y_prob: 预测为患病的概率
        out_path: 保存路径

    Returns:
        precision, recall, ap (Average Precision)
    """
    precision, recall, _ = skm.precision_recall_curve(y_true, y_prob)
    ap = skm.average_precision_score(y_true, y_prob)

    plt.figure(figsize=(8, 8))
    plt.plot(
        recall,
        precision,
        color="blue",
        lw=2,
        label=f"PR curve (AP = {ap:.4f})",
    )
    plt.axhline(y=np.mean(y_true), color="red", linestyle="--", label="Random")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved PR curve to {out_path}")

    plt.close()
    return precision, recall, ap


def plot_training_history(
    history: dict,
    out_path: Optional[Path] = None,
) -> None:
    """
    绘制训练历史曲线

    Args:
        history: 包含 epochs, train_loss, train_acc, val_loss, val_acc 的字典
        out_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    axes[0].plot(
        history["epochs"],
        history["train_loss"],
        label="Train Loss",
        marker=".",
    )
    if "val_loss" in history and history["val_loss"]:
        axes[0].plot(
            history["epochs"],
            history["val_loss"],
            label="Val Loss",
            marker=".",
        )
    axes[0].set_title("Loss Curve", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy 曲线
    axes[1].plot(
        history["epochs"],
        history["train_acc"],
        label="Train Accuracy",
        color="orange",
        marker=".",
    )
    if "val_acc" in history and history["val_acc"]:
        axes[1].plot(
            history["epochs"],
            history["val_acc"],
            label="Val Accuracy",
            color="green",
            marker=".",
        )
    axes[1].set_title("Accuracy Curve", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {out_path}")

    plt.close()


def save_prediction_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    out_path: Path,
) -> None:
    """
    保存预测结果到txt文件（兼容ROC文件夹中的MATLAB代码格式）

    格式：
    第一列：测试样本序号
    第二列：ground truth (1=患病, 0=正常)
    第三列：预测是否正确 (1=正确, 0=错误)
    第四列：预测患病概率

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率
        out_path: 保存路径
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for i in range(len(y_true)):
            sample_id = i + 1
            gt = int(y_true[i])
            correct = 1 if y_pred[i] == y_true[i] else 0
            prob = float(y_prob[i])
            f.write(f"{sample_id}\t{gt}\t{correct}\t{prob:.6f}\n")

    print(f"Saved prediction results to {out_path}")


def print_metrics_report(metrics: dict) -> None:
    """打印性能指标报告"""
    print("\n" + "=" * 50)
    print("          Classification Performance Report")
    print("=" * 50)
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f} (Sensitivity)")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  F1 Score:    {metrics['f1_score']:.4f}")
    print(f"  F2 Score:    {metrics['f2_score']:.4f}")
    print(f"  AUC:         {metrics['auc']:.4f}")
    print("-" * 50)
    print(f"  TP: {metrics['tp']:4d}  |  FN: {metrics['fn']:4d}")
    print(f"  FP: {metrics['fp']:4d}  |  TN: {metrics['tn']:4d}")
    print("=" * 50 + "\n")


def apply_threshold(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    """根据阈值将概率转换为类别预测"""
    return (y_prob >= threshold).astype(int)


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_precision: Optional[float] = None,
    maximize_metric: str = "f1_score",
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, dict]:
    """
    在给定的概率上搜索最佳决策阈值。

    Args:
        y_true: 真实标签
        y_prob: 预测概率（类别1）
        target_precision: 目标精确率（若提供，则仅考虑达到该精确率的阈值）
        maximize_metric: 在满足目标精确率下最大化的指标
        thresholds: 自定义阈值列表（默认0.05到0.95）

    Returns:
        (best_threshold, metrics_at_threshold)
    """

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)

    best_threshold = 0.5
    best_metrics = None
    best_score = -np.inf

    for thr in thresholds:
        y_pred = apply_threshold(y_prob, thr)
        metrics = compute_metrics(y_true, y_pred, y_prob)

        if (
            target_precision is not None
            and metrics["precision"] < target_precision
        ):
            continue

        score = metrics.get(maximize_metric, metrics["f1_score"])
        if score > best_score:
            best_score = score
            best_threshold = thr
            best_metrics = metrics

    if best_metrics is None:
        # 未达到目标精确率，选择精确率最高的阈值
        best_precision = -np.inf
        for thr in thresholds:
            y_pred = apply_threshold(y_prob, thr)
            metrics = compute_metrics(y_true, y_pred, y_prob)
            if metrics["precision"] > best_precision:
                best_precision = metrics["precision"]
                best_threshold = thr
                best_metrics = metrics

    return best_threshold, best_metrics
