"""
Metrics for saliency map evaluation.

Implements common objectives mentioned in the assignment:
- Correlation Coefficient (CC)
- KL Divergence (KLD)
- Normalized Scanpath Saliency (NSS)
- Mean Absolute Error (MAE)
"""

from typing import Dict

import torch

EPS = 1e-8


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


def correlation_coefficient(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Pearson correlation coefficient per sample, averaged over batch."""
    p = _flatten(pred)
    t = _flatten(target)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    denom = torch.sqrt(
        (p.pow(2).sum(dim=1) + EPS) * (t.pow(2).sum(dim=1) + EPS)
    )
    cc = num / denom
    return cc.mean()


def kl_divergence(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence between normalized maps (lower is better)."""
    p = _flatten(pred)
    t = _flatten(target)
    p = p / (p.sum(dim=1, keepdim=True) + EPS)
    t = t / (t.sum(dim=1, keepdim=True) + EPS)
    kl_pt = (t * (torch.log(t + EPS) - torch.log(p + EPS))).sum(dim=1)
    kl_tp = (p * (torch.log(p + EPS) - torch.log(t + EPS))).sum(dim=1)
    return 0.5 * (kl_pt + kl_tp).mean()


def normalized_scanpath_saliency(
    pred: torch.Tensor, fixation: torch.Tensor
) -> torch.Tensor:
    """NSS assumes fixation map is binary; higher is better."""
    p = _flatten(pred)
    f = _flatten(fixation)
    p = (p - p.mean(dim=1, keepdim=True)) / (p.std(dim=1, keepdim=True) + EPS)
    # treat fixation map as weights; if zero fixations, return zero to avoid nan
    mask_sum = f.sum(dim=1, keepdim=True)
    score = (p * f).sum(dim=1) / (mask_sum + EPS)
    return score.mean()


def mean_absolute_error(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return torch.abs(pred - target).mean()


def compute_all_metrics(
    pred: torch.Tensor, target: torch.Tensor
) -> Dict[str, float]:
    """Return a dict of scalar metrics for logging."""
    with torch.no_grad():
        return {
            "cc": float(correlation_coefficient(pred, target).item()),
            "kld": float(kl_divergence(pred, target).item()),
            "nss": float(normalized_scanpath_saliency(pred, target).item()),
            "mae": float(mean_absolute_error(pred, target).item()),
        }


if __name__ == "__main__":
    # simple smoke test
    a = torch.rand(2, 1, 16, 16)
    b = torch.rand(2, 1, 16, 16)
    print(compute_all_metrics(a, b))
