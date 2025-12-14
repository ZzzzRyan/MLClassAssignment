"""
Dataset utilities for saliency prediction.
- Pairs input RGB images with grayscale saliency/ground-truth maps.
- Provides a PyTorch Dataset with optional augmentation and resizing.
"""

import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

# Default image size for training/inference
DEFAULT_SIZE: Tuple[int, int] = (256, 256)


def ensure_extracted(
    zip_path: str | Path, out_dir: str | Path | None = None
) -> str:
    """Ensure a zip is extracted. Returns the extraction directory as str."""
    zip_path = Path(zip_path)
    out_dir = (
        Path(out_dir)
        if out_dir is not None
        else zip_path.with_name(zip_path.stem + "_extracted")
    )
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
    return str(out_dir)


def _index_files(folder: Path) -> dict:
    """Index files by relative stem for quick pairing."""
    index = {}
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
        }:
            rel = path.relative_to(folder)
            key = rel.with_suffix("")  # drop extension
            index[key] = path
    return index


def _resolve_root(root: str | Path) -> Path:
    """Return a folder that contains Stimuli/FIXATIONMAPS, auto-extracting zip if needed."""
    root_path = Path(root)

    # If given a zip file, extract then infer subfolder
    if root_path.is_file() and root_path.suffix.lower() == ".zip":
        extracted_dir = Path(ensure_extracted(root_path))
        candidate = extracted_dir / root_path.stem
        if (candidate / "Stimuli").exists():
            return candidate
        return extracted_dir

    # If already a folder with Stimuli
    if (root_path / "Stimuli").exists():
        return root_path

    # Try to find a sibling zip and extract
    root_zip = root_path.with_suffix(".zip")
    if root_zip.exists():
        extracted_dir = Path(ensure_extracted(root_zip))
        candidate = extracted_dir / root_zip.stem
        if (candidate / "Stimuli").exists():
            return candidate
        if (extracted_dir / "Stimuli").exists():
            return extracted_dir

    raise FileNotFoundError(
        f"Cannot locate Stimuli/FIXATIONMAPS under {root_path}. "
        f"If you only have the zip, place it at {root_zip} and rerun."
    )


def collect_image_pairs(root: str | Path) -> List[Tuple[Path, Path]]:
    """Collect paired (image, saliency_map) paths under a dataset root.

    Expected structure:
    root/
      Stimuli/<Category>/<image>.jpg
      FIXATIONMAPS/<Category>/<image>.jpg
    """
    root = _resolve_root(root)
    stimuli_dir = root / "Stimuli"
    gt_dir = root / "FIXATIONMAPS"
    if not stimuli_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(
            f"Expect subfolders 'Stimuli' and 'FIXATIONMAPS' under {root}"
        )

    stim_index = _index_files(stimuli_dir)
    gt_index = _index_files(gt_dir)

    pairs: List[Tuple[Path, Path]] = []
    missing = 0
    for key, stim_path in stim_index.items():
        if key in gt_index:
            pairs.append((stim_path, gt_index[key]))
        else:
            missing += 1
    if missing:
        print(
            f"Warning: {missing} stimuli images have no matching GT; skipped."
        )
    if not pairs:
        raise RuntimeError(f"No image/GT pairs found under {root}")
    return pairs


class SaliencyDataset(Dataset):
    """Torch Dataset for saliency prediction."""

    def __init__(
        self,
        root: str | Path,
        resize: Tuple[int, int] = DEFAULT_SIZE,
        augment: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.pairs = collect_image_pairs(root)
        if limit is not None:
            self.pairs = self.pairs[:limit]
        self.resize = resize
        self.augment = augment

        # Transforms: keep augmentations lightweight to avoid altering saliency targets drastically.
        # Build list conditionally to avoid lambda (pickle issue on Windows multiprocessing)
        img_transforms: list = [
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
        ]
        if augment:
            img_transforms.append(T.RandomHorizontalFlip())
            img_transforms.append(
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
            )
        img_transforms.extend(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.img_transform = T.Compose(img_transforms)

        map_transforms: list = [
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
        ]
        if augment:
            map_transforms.append(T.RandomHorizontalFlip())
        map_transforms.append(T.ToTensor())
        self.map_transform = T.Compose(map_transforms)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, gt_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        gt = Image.open(gt_path).convert("L")

        img_t = self.img_transform(image)
        gt_t = self.map_transform(gt)
        return img_t, gt_t


def get_dataloaders(
    root: str,
    resize: Tuple[int, int] = DEFAULT_SIZE,
    batch_size: int = 8,
    num_workers: int = 4,
    val_split: float = 0.1,
    seed: int = 42,
    augment: bool = True,
):
    """Create train/val dataloaders from a single dataset root."""
    ds = SaliencyDataset(root, resize=resize, augment=augment)
    n_val = max(1, int(len(ds) * val_split))
    n_train = len(ds) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=generator
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    root = "dataset/3-Saliency-TrainSet_extracted/3-Saliency-TrainSet"
    print(f"Dataset size preview: {len(SaliencyDataset(root, limit=10))}")
