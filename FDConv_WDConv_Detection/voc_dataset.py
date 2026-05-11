import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import functional as F_tv

VOC_CLASSES = [
    '__background__',
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]

CLASS_TO_IDX = {name: i for i, name in enumerate(VOC_CLASSES)}


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1).item() < self.p:
            image = F_tv.hflip(image)
            w = image.shape[-1]
            boxes = target['boxes']
            boxes = boxes[:, [2, 1, 0, 3]]
            boxes[:, 0] = w - boxes[:, 0]
            boxes[:, 2] = w - boxes[:, 2]
            target['boxes'] = boxes
        return image, target


class VOCDetectionDataset(Dataset):
    def __init__(self, root, year='2007', split='trainval', augment=False):
        self.voc = torchvision.datasets.VOCDetection(
            root, year=year, image_set=split, download=False,
        )
        self.augment = augment
        self.flip = RandomHorizontalFlip(p=0.5) if augment else None

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        image, annotation = self.voc[idx]
        image = F_tv.to_tensor(image)

        objs = annotation['annotation']['object']
        if isinstance(objs, dict):
            objs = [objs]

        boxes = []
        labels = []
        iscrowd = []
        for obj in objs:
            cls_name = obj['name']
            if cls_name not in CLASS_TO_IDX:
                continue
            bbox = obj['bndbox']
            x1 = float(bbox['xmin'])
            y1 = float(bbox['ymin'])
            x2 = float(bbox['xmax'])
            y2 = float(bbox['ymax'])
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_TO_IDX[cls_name])
            difficult = int(obj.get('difficult', '0'))
            iscrowd.append(difficult)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx], dtype=torch.int64),
            'area': area,
            'iscrowd': iscrowd,
        }

        if self.flip is not None:
            image, target = self.flip(image, target)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
