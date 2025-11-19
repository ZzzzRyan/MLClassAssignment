import argparse
import os
import random
from pathlib import Path

import dataset_utils
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_model import SimpleCNN, count_parameters
from torch import amp
from torch.utils.data import DataLoader, TensorDataset


def to_device(tensor, device):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        return [t.to(device) for t in tensor]
    return tensor.to(device)


def save_examples(X, y_true, y_pred, out_dir: Path, resize=(28, 28), n=12):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(y_true), size=n, replace=False)
    for i, j in enumerate(idx):
        arr = (X[j].reshape(resize) * 255.0).astype("uint8")
        im = Image.fromarray(arr, mode="L")
        fn = (
            out_dir
            / f"sample_{i}_pred{int(y_pred[j])}_true{int(y_true[j])}.png"
        )
        im.save(fn)


def evaluate(model, dataloader, device):
    model.eval()
    ys = []
    ys_pred = []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            ys_pred.append(preds)
            ys.append(yb.numpy())
    ys = np.concatenate(ys, axis=0)
    ys_pred = np.concatenate(ys_pred, axis=0)
    return ys, ys_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-zip", default="dataset/1-Digit-TrainSet.zip")
    parser.add_argument("--test-zip", default="dataset/1-Digit-TestSet.zip")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="1_digit/models/pytorch_model.pth")
    parser.add_argument("--examples-dir", default="1_digit/reports/examples")
    parser.add_argument("--resize", type=int, nargs=2, default=(28, 28))
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading datasets...")
    X_train, y_train = dataset_utils.load_dataset_from_zip(
        args.train_zip, resize=tuple(args.resize)
    )
    X_test, y_test = dataset_utils.load_dataset_from_zip(
        args.test_zip, resize=tuple(args.resize)
    )

    # reshape to NCHW tensors
    h, w = args.resize
    X_train_t = torch.from_numpy(X_train.reshape(-1, 1, h, w)).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_test_t = torch.from_numpy(X_test.reshape(-1, 1, h, w)).float()
    y_test_t = torch.from_numpy(y_test).long()

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = SimpleCNN(num_classes=10)
    print(f"Model parameters: {count_parameters(model):,}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=4, gamma=0.5
    )

    scaler = amp.GradScaler("cuda") if device.type == "cuda" else None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            if scaler is not None:
                with amp.autocast(device.type):
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        scheduler.step()
        train_acc = correct / total
        avg_loss = total_loss / total
        print(
            f"Epoch {epoch}/{args.epochs}: loss={avg_loss:.4f} train_acc={train_acc:.4f}"
        )

    # final evaluation
    import sklearn.metrics as skm

    ys, ys_pred = evaluate(model, test_loader, device)
    acc = skm.accuracy_score(ys, ys_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(skm.classification_report(ys, ys_pred))

    # save model
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "args": vars(args)}, out_path
    )
    print(f"Saved PyTorch model to {out_path}")

    # save some example images with predictions
    examples_dir = Path(args.examples_dir)
    examples_dir.mkdir(parents=True, exist_ok=True)
    # generate predictions for first N test samples and save
    n_examples = 12
    X_test_np = X_test.reshape(-1, h, w)
    save_examples(
        X_test_np, y_test, ys_pred, examples_dir, resize=(h, w), n=n_examples
    )
    print(f"Saved {n_examples} example images to {examples_dir}")


if __name__ == "__main__":
    main()
