import time
import torch
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision


@torch.inference_mode()
def evaluate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type='bbox', class_metrics=True)
    for images, targets in tqdm(data_loader, desc='Evaluating', leave=False):
        images = [img.to(device) for img in images]
        outputs = model(images)
        preds = [{
            'boxes': o['boxes'].cpu(),
            'scores': o['scores'].cpu(),
            'labels': o['labels'].cpu(),
        } for o in outputs]
        tgts = [{
            'boxes': t['boxes'].cpu(),
            'labels': t['labels'].cpu(),
        } for t in targets]
        metric.update(preds, tgts)
    result = metric.compute()
    out = {
        'mAP': float(result['map']),
        'mAP_50': float(result['map_50']),
        'mAP_75': float(result['map_75']),
    }
    if 'map_per_class' in result and result['map_per_class'].numel() > 0:
        out['per_class_ap50'] = result['map_per_class'].tolist()
    return out


@torch.inference_mode()
def benchmark_fps(model, device, warmup=10, runs=50):
    model.eval()
    dummy = [torch.randn(3, 800, 1333, device=device)]
    for _ in range(warmup):
        model(dummy)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(dummy)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return runs / elapsed
