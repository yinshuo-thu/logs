"""Settings and training primitives for the 36-hour TW/HK experiment."""
from dataclasses import asdict, dataclass
import json

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Experiment:
    seed: int = 20261121
    first_report_hour: int = 36
    goal_ic: float = 0.30
    quantity_history_days: int = 5
    joint_stop_hour: int = 22
    selection_stop_hour: int = 33
    val_days: int = 15
    test_days: int = 15
    grid_seconds: int = 5
    min_cross_section: int = 20
    ple_bins: int = 16
    width: int = 128
    layers: int = 4
    state: int = 16
    expand: int = 2
    tokens: int = 256
    warmup_tokens: int = 32
    microbatch_trials: tuple = (4, 8, 16)
    joint_lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    joint_epochs: int = 3
    max_joint_epochs: int = 5
    return_horizons: tuple = (10, 60, 120, 300)
    head_weights: tuple = (1.0, 0.25, 0.25, 0.15, 0.10, 0.15)
    mask_fraction: float = 0.15
    reconstruct_batch_fraction: float = 0.25
    finetune_lr_trials: tuple = (1e-5, 3e-5)
    ic_weight_trials: tuple = (0.0, 0.15)
    head_lr: float = 3e-4
    probe_lr: float = 1e-3
    lora_rank: int = 4
    lora_alpha: int = 8
    finetune_epochs: int = 3
    loader_workers_per_rank: int = 1
    prefetch_factor: int = 1
    gpu_allocator_cap: float = 0.82
    gpu_trial_peak_limit: float = 0.80
    pin_memory: bool = False
    persistent_workers: bool = False


CFG = Experiment()


def configure_gpu(local_rank):
    """One process per GPU; call before model/optimizer construction."""
    torch.cuda.set_device(local_rank)
    torch.cuda.set_per_process_memory_fraction(CFG.gpu_allocator_cap, local_rank)


def mid_move_bp(mid, previous, valid_pair):
    """Inputs share shape. Caller resets valid_pair across gaps/symbols/sessions."""
    mid, previous = mid.double(), previous.double()
    valid = valid_pair.bool() & mid.isfinite() & previous.isfinite() & (mid > 0) & (previous > 0)
    one = torch.ones_like(mid)
    move = 1e4 * torch.log(torch.where(valid, mid, one) / torch.where(valid, previous, one))
    return move.float(), valid


def price_from_open_bp(price, open_price, open_known):
    """Broadcast an as-of, same-day security open to the input price fields."""
    return mid_move_bp(price, open_price, open_known)


def quantity_from_history(quantity, past_daily_medians):
    """For one security: q[..., F], refs[D,F] in calendar order, strictly before today.
    Caller supplies a row per market trading day; missing values are NaN.
    Returns normalized values, observation mask, and usable history counts per field.
    """
    if past_daily_medians.ndim != 2 or quantity.shape[-1] != past_daily_medians.shape[1]:
        raise ValueError("expected quantity[..., fields] and history[days, fields]")
    history = past_daily_medians[-CFG.quantity_history_days:].to(quantity.device, torch.float64)
    if history.shape[0] == 0:
        valid = torch.zeros_like(quantity, dtype=torch.bool)
        return torch.zeros_like(quantity, dtype=torch.float32), valid, valid.new_zeros(quantity.shape[-1], dtype=torch.long)
    present = history.isfinite() & (history >= 0)
    ref = torch.nanquantile(history.masked_fill(~present, float("nan")), 0.5, dim=0)
    valid = quantity.isfinite() & (quantity >= 0) & ref.isfinite() & (ref > 0)
    q = torch.where(valid, quantity, 0).double()
    divisor = torch.where(ref.isfinite() & (ref > 0), ref, 1.)
    return torch.log1p(q / divisor).float(), valid, present.sum(dim=0)


def target_loss(pred, target, valid, scale, ic_weight=0.0):
    """[time groups, stocks], exactly one market per call; pred is in scaled units.

    scale is a fixed positive train-only scalar for this market's 60s label.
    Keep each complete timestamp group on one rank when using the IC term.
    """
    if pred.shape != target.shape or valid.shape != pred.shape or pred.ndim != 2:
        raise ValueError("expected aligned [groups, stocks] tensors")
    if not scale > 0:
        raise ValueError("train-only return scale must be positive")
    mask = valid.bool() & target.isfinite()
    if not torch.isfinite(pred[mask]).all():
        raise FloatingPointError("nonfinite model score")
    p = torch.where(mask, pred.float(), 0.0)
    y = torch.where(mask, target.float() / scale, 0.0)
    count = mask.sum(1, keepdim=True)
    huber = F.huber_loss(p, y, reduction="none").sum() / count.sum().clamp_min(1)
    if not ic_weight:
        return huber
    pc = (p - p.sum(1, keepdim=True) / count.clamp_min(1)) * mask
    yc = (y - y.sum(1, keepdim=True) / count.clamp_min(1)) * mask
    pp, yy = pc.square().sum(1), yc.square().sum(1)
    eligible = (count[:, 0] >= CFG.min_cross_section) & (pp > 1e-10) & (yy > 1e-10)
    corr = (pc * yc).sum(1) / (pp.clamp_min(1e-10) * yy.clamp_min(1e-10)).sqrt()
    penalty = ((1 - corr) * eligible).sum() / eligible.sum().clamp_min(1)
    return huber + ic_weight * penalty


if __name__ == "__main__":
    print(json.dumps(asdict(CFG), indent=2))
