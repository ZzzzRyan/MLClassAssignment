import os
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def ensure_extracted(zip_path: str, out_dir: str) -> str:
    out_dir = str(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
    return out_dir


def _label_from_filename(fname: str) -> int:
    # filename starts with the label digit per README; try to parse leading digits
    base = os.path.basename(fname)
    # find first run of digits
    digits = ""
    for ch in base:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    if not digits:
        raise ValueError(f"Cannot parse label from filename: {fname}")
    return int(digits)


def load_images_from_folder(
    folder: str, resize: Optional[Tuple[int, int]] = (28, 28)
) -> Tuple[np.ndarray, np.ndarray]:
    files = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            if f.lower().endswith(".bmp"):
                files.append(os.path.join(root, f))

    X: List[np.ndarray] = []
    y: List[int] = []
    for fp in tqdm(
        sorted(files), desc=f"Loading images from {Path(folder).name}"
    ):
        try:
            im = Image.open(fp).convert("L")
            if resize is not None:
                im = im.resize(resize, Image.Resampling.LANCZOS)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            X.append(arr.ravel())
            y.append(_label_from_filename(fp))
        except Exception as e:
            # skip problematic files but warn
            print(f"Warning: failed to load {fp}: {e}")

    if not X:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(
            (0,), dtype=np.int64
        )

    Xmat = np.stack(X, axis=0)
    yarr = np.array(y, dtype=np.int64)
    return Xmat, yarr


def load_dataset_from_zip(
    zip_path: str,
    extracted_subdir: Optional[str] = None,
    resize: Optional[Tuple[int, int]] = (28, 28),
) -> Tuple[np.ndarray, np.ndarray]:
    zip_path = str(zip_path)
    if extracted_subdir is None:
        extracted_subdir = os.path.join(
            os.path.dirname(zip_path), Path(zip_path).stem + "_extracted"
        )
    extracted = ensure_extracted(zip_path, extracted_subdir)
    return load_images_from_folder(extracted, resize=resize)


if __name__ == "__main__":
    # simple smoke test if run directly (won't run heavy processing automatically)
    print(
        "dataset_utils module. Use load_dataset_from_zip() from your training script."
    )
