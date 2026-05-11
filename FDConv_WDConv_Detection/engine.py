import os
import json
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from metrics import evaluate


def seed_everything(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    if scaler is not None and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    return ckpt.get('epoch', 0), ckpt.get('metrics', [])


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler,
                    warmup_iters=0, warmup_factor=0.001, max_norm=5.0,
                    use_amp=True):
    model.train()
    loss_accum = {}
    loss_log = []
    n_batches = 0
    nan_count = 0

    # Warmup scheduler (only for requested iterations)
    lr_scheduler_warmup = None
    if warmup_iters > 0:
        lr_scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor,
            total_iters=warmup_iters,
        )

    pbar = tqdm(data_loader, desc=f'Epoch {epoch}', leave=False)
    for i, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        if use_amp:
            with autocast(dtype=torch.bfloat16):
                loss_dict = model(images, targets)
                total_loss = sum(loss_dict.values())

            if not torch.isfinite(total_loss):
                nan_count += 1
                if nan_count <= 5:
                    print(f'WARNING: non-finite loss at iter {i}, skipping')
                elif nan_count == 6:
                    print(f'WARNING: suppressing further NaN warnings...')
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            total_loss = sum(loss_dict.values())

            if not torch.isfinite(total_loss):
                nan_count += 1
                if nan_count <= 5:
                    print(f'WARNING: non-finite loss at iter {i}, skipping')
                elif nan_count == 6:
                    print(f'WARNING: suppressing further NaN warnings...')
                continue

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if lr_scheduler_warmup is not None and i < warmup_iters:
            lr_scheduler_warmup.step()

        loss_val = total_loss.item()
        loss_log.append(loss_val)
        for k, v in loss_dict.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v.item()
        n_batches += 1

        pbar.set_postfix(loss=f'{loss_val:.4f}',
                         lr=f'{optimizer.param_groups[0]["lr"]:.6f}')

    if nan_count > 0:
        print(f'  Total NaN iterations this epoch: {nan_count}/{len(data_loader)}')

    avg = {k: v / max(n_batches, 1) for k, v in loss_accum.items()}
    avg['total'] = sum(avg.values())
    return avg, loss_log


def train_variant(name, model, train_loader, test_loader, device,
                  output_dir, lr=0.005, num_epochs=12, max_norm=5.0,
                  warmup_iters=500, use_amp=True):
    run_dir = os.path.join(output_dir, name)
    os.makedirs(run_dir, exist_ok=True)

    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[8, 11], gamma=0.1)
    scaler = GradScaler() if use_amp else None

    start_epoch = 0
    all_metrics = []
    all_losses = []
    best_map = -1.0

    last_ckpt = os.path.join(run_dir, 'last.pth')
    if os.path.exists(last_ckpt):
        print(f'  Resuming from {last_ckpt}')
        start_epoch, all_metrics = load_checkpoint(
            last_ckpt, model, optimizer, scheduler, scaler)
        start_epoch += 1
        losses_path = os.path.join(run_dir, 'losses.json')
        if os.path.exists(losses_path):
            all_losses = load_json(losses_path)
        if all_metrics:
            best_map = max(m.get('val_mAP_50', 0) for m in all_metrics)

    cumulative_time = sum(m.get('epoch_time_sec', 0) for m in all_metrics)

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()

        wi = warmup_iters if epoch == 0 else 0
        avg_losses, epoch_loss_log = train_one_epoch(
            model, optimizer, train_loader, device, epoch, scaler,
            warmup_iters=wi, max_norm=max_norm, use_amp=use_amp)
        all_losses.extend(epoch_loss_log)

        if epoch >= 1:
            scheduler.step()

        eval_result = evaluate(model, test_loader, device)

        epoch_time = time.time() - t0
        cumulative_time += epoch_time

        metrics = {
            'epoch': epoch,
            'train_loss': avg_losses.get('total', 0),
            'train_loss_cls': avg_losses.get('loss_classifier', 0),
            'train_loss_bbox': avg_losses.get('loss_box_reg', 0),
            'train_loss_rpn_cls': avg_losses.get('loss_objectness', 0),
            'train_loss_rpn_bbox': avg_losses.get('loss_rpn_box_reg', 0),
            'val_mAP': eval_result.get('mAP', 0),
            'val_mAP_50': eval_result.get('mAP_50', 0),
            'val_mAP_75': eval_result.get('mAP_75', 0),
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time_sec': round(epoch_time, 1),
            'cumulative_time_sec': round(cumulative_time, 1),
        }
        if 'per_class_ap50' in eval_result:
            metrics['per_class_ap50'] = eval_result['per_class_ap50']

        all_metrics.append(metrics)

        print(f'  Epoch {epoch}: loss={metrics["train_loss"]:.4f}  '
              f'mAP@50={metrics["val_mAP_50"]:.4f}  '
              f'mAP@75={metrics["val_mAP_75"]:.4f}  '
              f'lr={metrics["lr"]:.6f}  time={epoch_time:.0f}s')

        ckpt_state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'metrics': all_metrics,
        }
        if scaler is not None:
            ckpt_state['scaler'] = scaler.state_dict()
        save_checkpoint(ckpt_state, last_ckpt)

        if metrics['val_mAP_50'] > best_map:
            best_map = metrics['val_mAP_50']
            save_checkpoint(ckpt_state, os.path.join(run_dir, 'best.pth'))
            print(f'  New best mAP@50: {best_map:.4f}')

        save_json(all_metrics, os.path.join(run_dir, 'metrics.json'))
        save_json(all_losses, os.path.join(run_dir, 'losses.json'))

    return all_metrics
