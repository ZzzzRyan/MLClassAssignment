"""
医学图像数据集工具模块
用于加载和预处理眼底图像数据
"""

import os
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def ensure_extracted(zip_path: str, out_dir: str) -> str:
    """确保zip文件已解压"""
    out_dir = str(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
    return out_dir


def _label_from_filename(fname: str) -> int:
    """
    从文件名解析标签
    disease开头 -> 1 (患病)
    normal开头 -> 0 (正常)
    """
    base = os.path.basename(fname).lower()
    if base.startswith("disease"):
        return 1
    elif base.startswith("normal"):
        return 0
    else:
        raise ValueError(f"Cannot parse label from filename: {fname}")


def _label_from_folder(folder_path: str) -> int:
    """
    从文件夹名解析标签
    disease文件夹 -> 1 (患病)
    normal文件夹 -> 0 (正常)
    """
    folder_name = os.path.basename(folder_path).lower()
    if "disease" in folder_name:
        return 1
    elif "normal" in folder_name:
        return 0
    else:
        return -1  # 未知标签


def load_images_from_folder(
    folder: str, resize: Optional[Tuple[int, int]] = (224, 224)
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    从文件夹加载图像

    Args:
        folder: 图像文件夹路径
        resize: 目标尺寸 (height, width)

    Returns:
        X: 图像数组 (N, C, H, W) 归一化到 [0, 1]
        y: 标签数组
        filenames: 文件名列表
    """
    files = []
    for root, dirs, filenames in os.walk(folder):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                files.append(os.path.join(root, f))

    X: List[np.ndarray] = []
    y: List[int] = []
    fnames: List[str] = []

    for fp in tqdm(
        sorted(files), desc=f"Loading images from {Path(folder).name}"
    ):
        try:
            # 首先尝试从文件夹名获取标签
            parent_folder = os.path.dirname(fp)
            label = _label_from_folder(parent_folder)

            # 如果文件夹名无法确定标签，尝试从文件名获取
            if label == -1:
                label = _label_from_filename(fp)

            # 加载彩色图像
            im = Image.open(fp).convert("RGB")
            if resize is not None:
                im = im.resize(resize, Image.Resampling.LANCZOS)

            # 转换为 numpy 数组并归一化
            arr = np.asarray(im, dtype=np.float32) / 255.0
            # 转换为 (C, H, W) 格式
            arr = arr.transpose(2, 0, 1)

            X.append(arr)
            y.append(label)
            fnames.append(os.path.basename(fp))

        except Exception as e:
            print(f"Warning: failed to load {fp}: {e}")

    if not X:
        return (
            np.zeros((0, 3, resize[0], resize[1]), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            [],
        )

    Xmat = np.stack(X, axis=0)
    yarr = np.array(y, dtype=np.int64)
    return Xmat, yarr, fnames


def load_dataset_from_zip(
    zip_path: str,
    extracted_subdir: Optional[str] = None,
    resize: Optional[Tuple[int, int]] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    从zip文件加载数据集

    Args:
        zip_path: zip文件路径
        extracted_subdir: 解压目录（可选）
        resize: 目标尺寸

    Returns:
        X: 图像数组 (N, C, H, W)
        y: 标签数组
        filenames: 文件名列表
    """
    zip_path = str(zip_path)
    if extracted_subdir is None:
        extracted_subdir = os.path.join(
            os.path.dirname(zip_path), Path(zip_path).stem + "_extracted"
        )
    extracted = ensure_extracted(zip_path, extracted_subdir)
    return load_images_from_folder(extracted, resize=resize)


def get_class_distribution(y: np.ndarray) -> dict:
    """获取类别分布"""
    unique, counts = np.unique(y, return_counts=True)
    class_names = {0: "normal", 1: "disease"}
    return {
        class_names.get(int(u), str(u)): int(c) for u, c in zip(unique, counts)
    }


if __name__ == "__main__":
    print("dataset_utils module for medical image classification.")
    print("Use load_dataset_from_zip() from your training script.")
