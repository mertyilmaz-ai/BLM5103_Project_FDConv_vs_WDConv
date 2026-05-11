import io
import sys
import contextlib
import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def get_variant_config(variant, kernel_num=64):
    if variant == 'FDConv':
        sys.path.insert(0, '/content/FDConv_repo/FDConv_detection/mmdet_custom')
        from FDConv import FDConv
        return FDConv, 'convert2dftweight', {
            'kernel_num': kernel_num,
            'use_fdconv_if_c_gt': 16,
            'use_fdconv_if_k_in': [1, 3],
            'use_fbm_if_k_in': [3],
            'convert_param': False,
            'fbm_cfg': {
                'k_list': [2, 4, 8],
                'lowfreq_att': False,
                'fs_feat': 'feat',
                'act': 'sigmoid',
                'spatial': 'conv',
                'spatial_group': 1,
                'spatial_kernel': 3,
                'init': 'zero',
            },
        }
    elif variant == 'WDConv':
        from wdconv import WDConv
        return WDConv, 'convert2dwtweight', {
            'kernel_num': kernel_num,
            'use_wdconv_if_c_gt': 16,
            'use_wdconv_if_k_in': [1, 3],
            'use_wbm_if_k_in': [3],
            'convert_param': False,
            'wbm_cfg': {
                'spatial_group': 1,
                'spatial_kernel': 3,
                'act': 'sigmoid',
                'init': 'zero',
                'include_LL': False,
            },
        }
    else:
        raise ValueError(f'Unknown variant: {variant}')


def replace_convs(module, conv_class, convert_fn_name, variant_kwargs,
                  log=None, prefix=''):
    if log is None:
        log = []
    for name, child in list(module.named_children()):
        full = f'{prefix}.{name}' if prefix else name
        if isinstance(child, nn.Conv2d) and type(child) is nn.Conv2d:
            ks = child.kernel_size[0]
            if ks not in [1, 3]:
                continue
            new_conv = conv_class(
                child.in_channels, child.out_channels, ks,
                stride=child.stride, padding=child.padding,
                dilation=child.dilation, groups=child.groups,
                bias=child.bias is not None,
                **variant_kwargs,
            )
            with torch.no_grad():
                new_conv.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_conv.bias.data.copy_(child.bias.data)
                if hasattr(new_conv, 'KSM_Global'):
                    getattr(new_conv, convert_fn_name)(True)
                    mode = 'full'
                else:
                    mode = 'pass-through'
            setattr(module, name, new_conv)
            log.append((full, child.in_channels, child.out_channels, ks, mode))
        else:
            replace_convs(child, conv_class, convert_fn_name, variant_kwargs,
                          log, full)
    return log


def build_model(num_classes, variant='Baseline', kernel_num=64,
                trainable_layers=5, pretrained=True):
    from torchvision.models import ResNet50_Weights
    backbone = resnet_fpn_backbone(
        backbone_name='resnet50',
        weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
        trainable_layers=trainable_layers,
    )

    log = []
    if variant in ('FDConv', 'WDConv'):
        conv_class, convert_fn_name, variant_kwargs = get_variant_config(
            variant, kernel_num)
        with contextlib.redirect_stdout(io.StringIO()):
            log = replace_convs(backbone.body, conv_class, convert_fn_name,
                                variant_kwargs)

    model = FasterRCNN(backbone, num_classes=num_classes)
    return model, log
