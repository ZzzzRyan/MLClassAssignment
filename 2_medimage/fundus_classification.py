# -*- coding: utf-8 -*-
"""
医学图像分类（彩色眼底图像）
使用 ResNet18 进行迁移学习
满足以下要求：
1. 自动加载 2-MedImage-TrainSet 与 2-MedImage-TestSet（含 disease/normal 子目录）
2. 输出准确率、精确率、召回率、F1、ROC 曲线、AUC、PR 曲线
3. 可视化成功案例（绿色标题）和失败案例（红色标题）
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# ================================
# 一、配置区（可根据需要修改）
# ================================
TRAIN_DIR = "dataset/2-MedImage-TrainSet_extracted/2-MedImage-TrainSet"
TEST_DIR = "dataset/2-MedImage-TestSet_extracted/2-MedImage-TestSet"
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 20
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "2_medimage/models/fundus_resnet18.pth"
FIGURE_DIR = "2_medimage/reports/figures"
EXAMPLE_DIR = "2_medimage/reports/examples"


# ================================
# 二、数据集定义
# ================================
class FundusDataset(Dataset):
    """自定义数据集：读取 disease/normal 子文件夹"""

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.img_paths = []
        self.labels = []

        # disease = 1, normal = 0
        for label_name, label in [("disease", 1), ("normal", 0)]:
            folder = os.path.join(root, label_name)
            for fname in os.listdir(folder):
                if fname.lower().endswith(".jpg"):
                    self.img_paths.append(os.path.join(folder, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx], img_path


# ================================
# 三、数据增强与 DataLoader
# ================================
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

train_ds = FundusDataset(TRAIN_DIR, train_transform)
test_ds = FundusDataset(TEST_DIR, test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print("训练集样本数：", len(train_ds))
print("测试集样本数：", len(test_ds))

# ================================
# 四、构建 ResNet18 模型
# ================================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ================================
# 五、训练函数
# ================================
def train_one_epoch():
    model.train()
    total_loss = 0

    for imgs, labels, _ in tqdm(train_loader, desc="训练中"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(train_ds)


# ================================
# 六、训练主循环
# ================================
for epoch in range(1, EPOCHS + 1):
    loss = train_one_epoch()
    print(f"Epoch {epoch}/{EPOCHS} —— Loss：{loss:.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print("模型已保存为：", MODEL_PATH)


# ================================
# 七、测试评估
# ================================
def evaluate():
    model.eval()
    labels_all = []
    preds_all = []
    probs_all = []
    paths_all = []

    with torch.no_grad():
        for imgs, labels, paths in tqdm(test_loader, desc="测试中"):
            imgs = imgs.to(DEVICE)

            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = np.argmax(outputs.cpu().numpy(), axis=1)

            labels_all.extend(labels.numpy())
            preds_all.extend(preds)
            probs_all.extend(probs)
            paths_all.extend(paths)

    return (
        np.array(labels_all),
        np.array(preds_all),
        np.array(probs_all),
        paths_all,
    )


labels, preds, probs, paths = evaluate()

# 指标计算
acc = accuracy_score(labels, preds)
prec = precision_score(labels, preds)
rec = recall_score(labels, preds)
f1 = f1_score(labels, preds)
cm = confusion_matrix(labels, preds)

# ROC
fpr, tpr, _ = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)

# PR
prec_curve, rec_curve, _ = precision_recall_curve(labels, probs)
ap = average_precision_score(labels, probs)

print("========== 测试集性能指标 ==========")
print(f"Accuracy：{acc:.4f}")
print(f"Precision：{prec:.4f}")
print(f"Recall：{rec:.4f}")
print(f"F1 Score：{f1:.4f}")
print(f"ROC-AUC：{roc_auc:.4f}")
print(f"AP（PR-AUC）：{ap:.4f}")
print("Confusion Matrix:\n", cm)


# ================================
# 八、绘制 ROC 与 PR 曲线
# ================================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "--")
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(rec_curve, prec_curve, label=f"AP={ap:.4f}")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig(f"{FIGURE_DIR}/roc_pr_curves.png")
print("ROC 与 PR 曲线已保存至：", f"{FIGURE_DIR}/roc_pr_curves.png")


# ================================
# 九、展示成功与失败案例
# ================================
def show_cases(indices, title_color, case_type):
    plt.figure(figsize=(12, 8))
    count = min(8, len(indices))
    for i, idx in enumerate(indices[:count]):
        img = Image.open(paths[idx]).convert("RGB")
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.axis("off")

        true = "disease" if labels[idx] == 1 else "normal"
        pred = "disease" if preds[idx] == 1 else "normal"
        title = f"{case_type}\nTrue:{true}\nPred:{pred}"

        plt.title(title, color=title_color, fontsize=9)
    plt.tight_layout(pad=2.0)
    # plt.show()
    plt.savefig(f"{EXAMPLE_DIR}/{case_type.lower()}_cases.png")
    print(
        f"{case_type} 案例已保存至：",
        f"{EXAMPLE_DIR}/{case_type.lower()}_cases.png",
    )


# 成功案例
success_idx = [i for i in range(len(labels)) if labels[i] == preds[i]]
show_cases(success_idx, "green", "Success")

# 失败案例
fail_idx = [i for i in range(len(labels)) if labels[i] != preds[i]]
show_cases(fail_idx, "red", "Failure")
