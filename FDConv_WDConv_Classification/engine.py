"""
Training & evaluation engine with comprehensive metric tracking.

Designed for Colab T4 GPU. Every metric that could appear in a paper table
or figure is computed and returned.
"""

import os
import json
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm.auto import tqdm

import pickle
import sys


# ==============================================================================
#  Seeding
# ==============================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
#  Dataset: ImageNet 64x64 (pickle format)
# ==============================================================================

class ImageNet64Dataset(Dataset):
    """ImageNet-1K downsampled to 64x64.

    Directory layout:
        data_dir/
            train/
                train_data_batch_1 ... train_data_batch_10
            val/
                val_data

    Each pickle: {"data": (N,12288) uint8, "labels": list[int] 1-indexed}
    """

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, data_dir: str, train: bool = True, num_classes: int = None,
                 max_train_files: int = None):
        self.train = train

        if train:
            subdir = os.path.join(data_dir, "train")
            files = sorted(f for f in os.listdir(subdir)
                           if f.startswith("train_data_batch"))
            if max_train_files is not None:
                files = files[:max_train_files]
            paths = [os.path.join(subdir, f) for f in files]
        else:
            subdir = os.path.join(data_dir, "val")
            paths = [os.path.join(subdir, "val_data")]

        split = "train" if train else "val"
        arrays_x, arrays_y = [], []
        for path in tqdm(paths, desc=f"Loading {split}", unit="file"):
            with open(path, "rb") as f:
                d = pickle.load(f)
            arrays_x.append(d["data"])
            arrays_y.append(np.array(d["labels"], dtype=np.int64))

        self.images = np.concatenate(arrays_x, axis=0)
        self.labels = np.concatenate(arrays_y, axis=0)

        # Convert 1-indexed to 0-indexed
        if self.labels.min() >= 1:
            self.labels -= 1

        # Subset to first N classes (for faster experiments)
        if num_classes is not None and num_classes < 1000:
            mask = self.labels < num_classes
            self.images = self.images[mask]
            self.labels = self.labels[mask]

        print(f"  {split}: {len(self.labels):,} images, "
              f"{len(np.unique(self.labels))} classes, "
              f"labels [{self.labels.min()}..{self.labels.max()}]")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].reshape(3, 64, 64)
        img = torch.from_numpy(img.copy()).float() / 255.0

        if self.train:
            if random.random() > 0.5:
                img = img.flip(-1)
            pad = 4
            img = F.pad(img, [pad] * 4, mode="reflect")
            i = random.randint(0, 2 * pad)
            j = random.randint(0, 2 * pad)
            img = img[:, i:i + 64, j:j + 64]

        img = (img - self.MEAN) / self.STD
        return img, self.labels[idx]


def build_datasets(data_dir, num_classes=None, max_train_files=None):
    train_ds = ImageNet64Dataset(data_dir, train=True, num_classes=num_classes,
                                 max_train_files=max_train_files)
    val_ds = ImageNet64Dataset(data_dir, train=False, num_classes=num_classes)
    return train_ds, val_ds


def build_dataloaders(train_ds, val_ds, batch_size, num_workers=4, seed=42):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"Train: {len(train_ds):,} images ({len(train_loader)} batches @ bs={batch_size})")
    print(f"Val:   {len(val_ds):,} images ({len(val_loader)} batches @ bs={batch_size})")
    return train_loader, val_loader


# ==============================================================================
#  Model building
# ==============================================================================

def make_resnet18_64(num_classes=1000):
    """ResNet-18 adapted for 64x64 input (3x3 stem, no maxpool)."""
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def count_params(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def replace_convs(module, conv_class, convert_fn_name, kernel_num=4,
                  _log=None, _prefix=""):
    """Recursively replace Conv2d with FDConv or WDConv, copying weights."""
    if _log is None:
        _log = []
    for name, child in list(module.named_children()):
        full_name = f"{_prefix}.{name}" if _prefix else name
        if isinstance(child, nn.Conv2d) and type(child) is nn.Conv2d:
            ks = child.kernel_size[0]
            if ks not in [1, 3]:
                continue
            new_conv = conv_class(
                child.in_channels, child.out_channels, ks,
                stride=child.stride, padding=child.padding,
                dilation=child.dilation, groups=child.groups,
                bias=child.bias is not None,
                kernel_num=kernel_num, convert_param=False,
            )
            with torch.no_grad():
                new_conv.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_conv.bias.data.copy_(child.bias.data)
                if hasattr(new_conv, "KSM_Global"):
                    getattr(new_conv, convert_fn_name)(True)
                    mode = "full"
                else:
                    mode = "pass-through"
            setattr(module, name, new_conv)
            _log.append((full_name, child.in_channels, child.out_channels, ks, mode))
        else:
            replace_convs(child, conv_class, convert_fn_name, kernel_num, _log, full_name)
    return _log


def build_all_models(num_classes, kernel_num, seed, fdconv_cls, wdconv_cls):
    """Build Baseline, FDConv, and WDConv ResNet-18 models.

    Returns:
        model_dict:       {"Baseline": model, "FDConv": model, "WDConv": model}
        replacement_logs: {"FDConv": [...], "WDConv": [...]}
        model_infos:      {"Baseline": {...}, "FDConv": {...}, "WDConv": {...}}
    """
    torch.manual_seed(seed)

    model_dict = {
        "Baseline": make_resnet18_64(num_classes),
        "FDConv": make_resnet18_64(num_classes),
        "WDConv": make_resnet18_64(num_classes),
    }

    log_fd = replace_convs(model_dict["FDConv"], fdconv_cls, "convert2dftweight", kernel_num)
    log_wd = replace_convs(model_dict["WDConv"], wdconv_cls, "convert2dwtweight", kernel_num)

    replacement_logs = {"FDConv": log_fd, "WDConv": log_wd}

    fd_full = sum(1 for *_, m in log_fd if m == "full")
    wd_full = sum(1 for *_, m in log_wd if m == "full")
    print(f"FDConv: {fd_full} full + {len(log_fd) - fd_full} pass-through")
    print(f"WDConv: {wd_full} full + {len(log_wd) - wd_full} pass-through")

    model_infos = {}
    for name, model in model_dict.items():
        total = count_params(model)
        trainable = count_params(model, only_trainable=True)
        log = replacement_logs.get(name)
        info = {
            "variant": name,
            "total_params": total,
            "total_params_M": round(total / 1e6, 3),
            "trainable_params": trainable,
            "trainable_params_M": round(trainable / 1e6, 3),
        }
        if log is not None:
            info["replaced_layers"] = [
                {"name": e[0], "in_ch": e[1], "out_ch": e[2],
                 "kernel_size": e[3], "mode": e[4]}
                for e in log
            ]
            info["full_dynamic_layers"] = sum(1 for e in log if e[4] == "full")
            info["passthrough_layers"] = sum(1 for e in log if e[4] == "pass-through")
        else:
            info["replaced_layers"] = []
            info["full_dynamic_layers"] = 0
            info["passthrough_layers"] = 0
        model_infos[name] = info
        print(f"  {name}: {total:,} params ({total / 1e6:.2f}M)")

    return model_dict, replacement_logs, model_infos


# ==============================================================================
#  Checkpoints
# ==============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt.get("metrics", {})


# ==============================================================================
#  Training
# ==============================================================================

def compute_grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch,
                    accum_steps=1):
    """Train for one epoch with optional gradient accumulation.

    Returns dict with all training metrics.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0
    grad_norm_sum = 0.0
    grad_steps = 0
    optimizer_steps = 0
    loss_log = []

    t0 = time.perf_counter()
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"  Train E{epoch+1}", unit="batch",
                dynamic_ncols=True, leave=True)

    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        scaled_loss = loss / accum_steps
        scaled_loss.backward()

        total_loss += loss.item()
        loss_log.append(loss.item())

        with torch.no_grad():
            _, pred5 = logits.topk(5, dim=1)
            correct1 += (pred5[:, 0] == labels).sum().item()
            correct5 += (pred5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

        if (i + 1) % accum_steps == 0:
            optimizer_steps += 1
            cur_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if optimizer_steps % 50 == 0:
                grad_norm_sum += cur_norm.item()
                grad_steps += 1
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (i + 1) % 10 == 0:
            avg = total_loss / (i + 1)
            acc = correct1 / total * 100
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix_str(f"loss={avg:.3f}  acc={acc:.1f}%  lr={lr:.5f}")

    # Handle leftover micro-batches
    if len(dataloader) % accum_steps != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    pbar.close()
    elapsed_sec = time.perf_counter() - t0
    avg_grad_norm = grad_norm_sum / max(grad_steps, 1)

    return {
        "avg_loss": total_loss / len(dataloader),
        "train_top1": correct1 / total * 100,
        "train_top5": correct5 / total * 100,
        "grad_norm": avg_grad_norm,
        "elapsed_sec": elapsed_sec,
        "throughput": total / elapsed_sec,
        "loss_log": loss_log,
    }


# ==============================================================================
#  Evaluation
# ==============================================================================

@torch.inference_mode()
def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0
    num_batches = 0

    t0 = time.perf_counter()
    pbar = tqdm(dataloader, desc="  Val", unit="batch",
                dynamic_ncols=True, leave=True)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        num_batches += 1

        _, pred5 = logits.topk(5, dim=1)
        correct1 += (pred5[:, 0] == labels).sum().item()
        correct5 += (pred5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

        if num_batches % 10 == 0:
            t1 = correct1 / total * 100
            t5 = correct5 / total * 100
            pbar.set_postfix_str(f"top1={t1:.1f}%  top5={t5:.1f}%")

    pbar.close()
    elapsed_sec = time.perf_counter() - t0

    return {
        "val_loss": total_loss / num_batches,
        "val_top1": correct1 / total * 100,
        "val_top5": correct5 / total * 100,
        "elapsed_sec": elapsed_sec,
        "throughput": total / elapsed_sec,
    }


# ==============================================================================
#  Optimizer & Scheduler
# ==============================================================================

def build_optimizer_and_scheduler(model, lr, momentum, weight_decay,
                                  warmup_epochs, num_epochs, steps_per_epoch):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01,
        total_iters=steps_per_epoch * warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps_per_epoch * (num_epochs - warmup_epochs),
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine],
        milestones=[steps_per_epoch * warmup_epochs],
    )
    return optimizer, scheduler


# ==============================================================================
#  Benchmarking
# ==============================================================================

@torch.inference_mode()
def benchmark_inference(model, device, batch_size=64, img_size=64,
                        warmup=10, runs=50):
    """Benchmark inference speed. Returns dict with timing info."""
    model.eval()
    model.to(device)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)

    for _ in range(warmup):
        _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # ms
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
        "throughput_img_s": float(batch_size / (np.mean(times) / 1000)),
        "batch_size": batch_size,
    }


# ==============================================================================
#  JSON I/O
# ==============================================================================

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path, default=None):
    if default is None:
        default = []
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


# ==============================================================================
#  Full training pipeline for one variant
# ==============================================================================

def train_variant(name, model, train_loader, val_loader, device, output_dir,
                  lr=0.1, momentum=0.9, weight_decay=1e-4, warmup_epochs=1,
                  num_epochs=10, accum_steps=1, seed=42):
    """Train a single model variant with checkpoint resume and rich metrics."""
    variant_dir = os.path.join(output_dir, name)
    os.makedirs(variant_dir, exist_ok=True)

    metrics_path = os.path.join(variant_dir, "metrics.json")
    losses_path = os.path.join(variant_dir, "losses.json")
    best_path = os.path.join(variant_dir, "best.pth")
    last_path = os.path.join(variant_dir, "last.pth")

    model.to(device)

    steps_per_epoch = len(train_loader) // accum_steps
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr, momentum, weight_decay, warmup_epochs, num_epochs,
        steps_per_epoch,
    )

    # Resume
    start_epoch = 0
    epoch_metrics = load_json(metrics_path, default=[])
    best_top1 = max((m["val_top1"] for m in epoch_metrics), default=0.0)
    cumulative_time = sum(m.get("epoch_time_sec", 0) for m in epoch_metrics)

    if os.path.exists(last_path):
        start_epoch, _ = load_checkpoint(last_path, model, optimizer, scheduler)
        start_epoch += 1
        print(f"  Resumed from last.pth (epoch {start_epoch})")
        print(f"  Previous best top-1: {best_top1:.2f}%")

    if start_epoch >= num_epochs:
        print(f"  Already trained {start_epoch} epochs, target is {num_epochs}. Skipping.")
        return epoch_metrics

    all_losses = load_json(losses_path, default=[])

    micro_bs = train_loader.batch_size
    print(f"  Effective batch: {micro_bs * accum_steps}  "
          f"(micro={micro_bs} x accum={accum_steps})")
    print(f"  Optimizer steps/epoch: {steps_per_epoch}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\n--- {name} Epoch {epoch + 1}/{num_epochs} ---")

        train_out = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch,
            accum_steps=accum_steps,
        )
        all_losses.extend(train_out["loss_log"])

        print(f"  Train: loss={train_out['avg_loss']:.4f}  "
              f"top1={train_out['train_top1']:.2f}%  "
              f"top5={train_out['train_top5']:.2f}%  "
              f"grad_norm={train_out['grad_norm']:.2f}  "
              f"time={train_out['elapsed_sec'] / 60:.1f}m  "
              f"throughput={train_out['throughput']:.0f} img/s")

        val_out = evaluate(model, val_loader, device)

        print(f"  Val:   loss={val_out['val_loss']:.4f}  "
              f"top1={val_out['val_top1']:.2f}%  "
              f"top5={val_out['val_top5']:.2f}%  "
              f"time={val_out['elapsed_sec']:.1f}s  "
              f"throughput={val_out['throughput']:.0f} img/s")

        epoch_time = train_out["elapsed_sec"] + val_out["elapsed_sec"]
        cumulative_time += epoch_time

        record = {
            "epoch": epoch + 1,
            "train_loss": round(train_out["avg_loss"], 4),
            "train_top1": round(train_out["train_top1"], 2),
            "train_top5": round(train_out["train_top5"], 2),
            "grad_norm": round(train_out["grad_norm"], 4),
            "train_throughput": round(train_out["throughput"], 1),
            "val_loss": round(val_out["val_loss"], 4),
            "val_top1": round(val_out["val_top1"], 2),
            "val_top5": round(val_out["val_top5"], 2),
            "val_throughput": round(val_out["throughput"], 1),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "train_time_sec": round(train_out["elapsed_sec"], 1),
            "val_time_sec": round(val_out["elapsed_sec"], 1),
            "epoch_time_sec": round(epoch_time, 1),
            "cumulative_time_sec": round(cumulative_time, 1),
        }
        epoch_metrics.append(record)

        save_json(epoch_metrics, metrics_path)
        save_json(all_losses, losses_path)
        save_checkpoint(model, optimizer, scheduler, epoch, val_out, last_path)

        if val_out["val_top1"] > best_top1:
            best_top1 = val_out["val_top1"]
            save_checkpoint(model, optimizer, scheduler, epoch, val_out, best_path)
            print(f"  >> New best! top1={best_top1:.2f}%")

    print(f"\n  {name} done. Best top-1: {best_top1:.2f}%  "
          f"Total time: {cumulative_time / 60:.1f}m")
    return epoch_metrics
