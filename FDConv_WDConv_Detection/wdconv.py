"""
WDConv: Wavelet Dynamic Convolution.

A wavelet-domain variant of FDConv (CVPR 2025). Replaces 2D FFT with 2D Haar
Discrete Wavelet Transform. Key differences:

  * WDW (Wavelet Disjoint Weights) — kernel weights live in wavelet domain.
    Four oriented subbands (LL, LH, HL, HH) give spatially-localized bases
    that FFT destroys.
  * WBM (Wavelet Band Modulation) — replaces FBM. Modulates each wavelet
    subband (horizontal detail, vertical detail, diagonal detail) with
    input-adaptive attention.
  * KSM modules (Global + Local) reused unchanged from FDConv.

Reference: FDConv — Frequency Dynamic Convolution for Dense Image
Prediction (Chen et al., CVPR 2025).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.layers import trunc_normal_


# ═══════════════════════════════════════════════════════════════════════════════
#  Haar Wavelet Transforms
# ═══════════════════════════════════════════════════════════════════════════════

_SQRT2 = math.sqrt(2.0)


def haar_dwt2d(x: torch.Tensor):
    """Single-level 2D Haar DWT. x: (..., H, W) with H,W even.
    Returns (LL, LH, HL, HH) each (..., H/2, W/2)."""
    # Horizontal pass
    xl = (x[..., :, 0::2] + x[..., :, 1::2]) / _SQRT2
    xh = (x[..., :, 0::2] - x[..., :, 1::2]) / _SQRT2
    # Vertical pass
    LL = (xl[..., 0::2, :] + xl[..., 1::2, :]) / _SQRT2
    LH = (xl[..., 0::2, :] - xl[..., 1::2, :]) / _SQRT2
    HL = (xh[..., 0::2, :] + xh[..., 1::2, :]) / _SQRT2
    HH = (xh[..., 0::2, :] - xh[..., 1::2, :]) / _SQRT2
    return LL, LH, HL, HH


def haar_idwt2d(LL, LH, HL, HH):
    """Inverse 2D Haar DWT."""
    h2, w2 = LL.shape[-2], LL.shape[-1]
    batch_shape = LL.shape[:-2]
    xl = torch.empty(*batch_shape, h2 * 2, w2, dtype=LL.dtype, device=LL.device)
    xh = torch.empty_like(xl)
    xl[..., 0::2, :] = (LL + LH) / _SQRT2
    xl[..., 1::2, :] = (LL - LH) / _SQRT2
    xh[..., 0::2, :] = (HL + HH) / _SQRT2
    xh[..., 1::2, :] = (HL - HH) / _SQRT2
    x = torch.empty(*batch_shape, h2 * 2, w2 * 2, dtype=LL.dtype, device=LL.device)
    x[..., :, 0::2] = (xl + xh) / _SQRT2
    x[..., :, 1::2] = (xl - xh) / _SQRT2
    return x


def haar_dwt2d_pack(x: torch.Tensor) -> torch.Tensor:
    """DWT then pack into standard layout: [[LL,LH],[HL,HH]] → (H,W)."""
    LL, LH, HL, HH = haar_dwt2d(x)
    h2, w2 = LL.shape[-2], LL.shape[-1]
    out = torch.empty(*x.shape[:-2], h2 * 2, w2 * 2, dtype=x.dtype, device=x.device)
    out[..., :h2, :w2] = LL
    out[..., :h2, w2:] = LH
    out[..., h2:, :w2] = HL
    out[..., h2:, w2:] = HH
    return out


def haar_idwt2d_unpack(packed: torch.Tensor) -> torch.Tensor:
    """Unpack [[LL,LH],[HL,HH]] layout then inverse DWT."""
    h, w = packed.shape[-2], packed.shape[-1]
    h2, w2 = h // 2, w // 2
    LL = packed[..., :h2, :w2]
    LH = packed[..., :h2, w2:]
    HL = packed[..., h2:, :w2]
    HH = packed[..., h2:, w2:]
    return haar_idwt2d(LL, LH, HL, HH)


def get_wavelet_indices(H: int, W: int):
    """Return (2, H*W) index tensor sorted by subband priority.
    Priority: LL (approx) first, then LH+HL (details), then HH (diag)."""
    h2, w2 = H // 2, W // 2
    priority = torch.empty(H, W, dtype=torch.float32)
    priority[:h2, :w2] = 0.0   # LL
    priority[:h2, w2:] = 1.0   # LH
    priority[h2:, :w2] = 1.0   # HL
    priority[h2:, w2:] = 2.0   # HH
    flat = priority.flatten()
    sorted_idx = torch.argsort(flat, stable=True)
    rows = sorted_idx // W
    cols = sorted_idx % W
    return torch.stack([rows, cols], dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Attention modules (from FDConv, reproduced for self-containedness)
# ═══════════════════════════════════════════════════════════════════════════════

class StarReLU(nn.Module):
    """StarReLU: s * relu(x)^2 + b"""
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class KernelSpatialModulation_Global(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1,
                 reduction=0.0625, kernel_num=4, min_channel=16,
                 temp=1.0, kernel_temp=None, kernel_att_init=None,
                 att_multi=2.0, ksm_only_kernel_att=False, att_grid=1,
                 stride=1, spatial_freq_decompose=False, act_type='sigmoid'):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.act_type = act_type
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = temp
        self.kernel_temp = kernel_temp
        self.ksm_only_kernel_att = ksm_only_kernel_att
        self.att_multi = att_multi

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = StarReLU()
        self.spatial_freq_decompose = spatial_freq_decompose

        if ksm_only_kernel_att:
            self.func_channel = self.skip
        else:
            ch_out = in_planes * 2 if (spatial_freq_decompose and kernel_size > 1) else in_planes
            self.channel_fc = nn.Conv2d(attention_channel, ch_out, 1, bias=True)
            self.func_channel = self.get_channel_attention

        if (in_planes == groups and in_planes == out_planes) or ksm_only_kernel_att:
            self.func_filter = self.skip
        else:
            f_out = out_planes * 2 if spatial_freq_decompose else out_planes
            self.filter_fc = nn.Conv2d(attention_channel, f_out, 1, stride=stride, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1 or ksm_only_kernel_att:
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for attr in ('spatial_fc', 'kernel_fc', 'channel_fc'):
            if hasattr(self, attr) and isinstance(getattr(self, attr), nn.Conv2d):
                nn.init.normal_(getattr(self, attr).weight, std=1e-6)

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        out = self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1))
        if self.act_type == 'sigmoid':
            return torch.sigmoid(out / self.temperature) * self.att_multi
        return 1 + torch.tanh_(out / self.temperature)

    def get_filter_attention(self, x):
        out = self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1))
        if self.act_type == 'sigmoid':
            return torch.sigmoid(out / self.temperature) * self.att_multi
        return 1 + torch.tanh_(out / self.temperature)

    def get_spatial_attention(self, x):
        out = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        if self.act_type == 'sigmoid':
            return torch.sigmoid(out / self.temperature) * self.att_multi
        return 1 + torch.tanh_(out / self.temperature)

    def get_kernel_attention(self, x):
        out = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        if self.act_type == 'softmax':
            return F.softmax(out / self.kernel_temp, dim=1)
        elif self.act_type == 'sigmoid':
            return torch.sigmoid(out / self.kernel_temp) * 2 / out.size(1)
        return (1 + torch.tanh(out / self.kernel_temp)) / out.size(1)

    def forward(self, x, use_checkpoint=False):
        avg_x = self.relu(self.bn(self.fc(x)))
        return (self.func_channel(avg_x), self.func_filter(avg_x),
                self.func_spatial(avg_x), self.func_kernel(avg_x))


class KernelSpatialModulation_Local(nn.Module):
    def __init__(self, channel=None, kernel_num=1, out_n=1, k_size=3, use_global=False):
        super().__init__()
        self.kn = kernel_num
        self.out_n = out_n
        self.channel = channel
        if channel is not None:
            k_size = round((math.log2(channel) / 2) + 0.5) // 2 * 2 + 1
        self.conv = nn.Conv1d(1, kernel_num * out_n, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        nn.init.constant_(self.conv.weight, 1e-6)
        self.use_global = use_global
        if self.use_global:
            self.complex_weight = nn.Parameter(
                torch.randn(1, self.channel // 2 + 1, 2, dtype=torch.float32) * 1e-6)
        self.norm = nn.LayerNorm(self.channel)

    def forward(self, x, x_std=None):
        x = x.squeeze(-1).transpose(-1, -2)
        b, _, c = x.shape
        if self.use_global:
            x_rfft = torch.fft.rfft(x.float(), dim=-1)
            x_real = x_rfft.real * self.complex_weight[..., 0][None]
            x_imag = x_rfft.imag * self.complex_weight[..., 1][None]
            x = x + torch.fft.irfft(
                torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1)), dim=-1)
        x = self.norm(x)
        att_logit = self.conv(x)
        att_logit = att_logit.reshape(x.size(0), self.kn, self.out_n, c)
        att_logit = att_logit.permute(0, 1, 3, 2)
        return att_logit


# ═══════════════════════════════════════════════════════════════════════════════
#  Wavelet Band Modulation (WBM) — replaces FBM
# ═══════════════════════════════════════════════════════════════════════════════

class WaveletBandModulation(nn.Module):
    """Modulates each DWT subband with input-adaptive attention.
    LH → horizontal detail, HL → vertical detail, HH → diagonal detail.
    Optionally modulates LL (approximation) too."""

    def __init__(self, in_channels, spatial_group=1, spatial_kernel=3,
                 act='sigmoid', init='zero', include_LL=False, **kwargs):
        super().__init__()
        if spatial_group > 64:
            spatial_group = in_channels
        self.spatial_group = spatial_group
        self.act = act
        self.include_LL = include_LL
        n_bands = 4 if include_LL else 3
        self.weight_convs = nn.ModuleList([
            nn.Conv2d(in_channels, spatial_group, spatial_kernel,
                      padding=spatial_kernel // 2, groups=spatial_group, bias=True)
            for _ in range(n_bands)
        ])
        if init == 'zero':
            for c in self.weight_convs:
                nn.init.normal_(c.weight, std=1e-6)
                if c.bias is not None:
                    c.bias.data.zero_()

    def sp_act(self, w):
        if self.act == 'sigmoid':
            return w.sigmoid() * 2
        elif self.act == 'tanh':
            return 1 + w.tanh()
        raise NotImplementedError

    def forward(self, x, att_feat=None):
        if att_feat is None:
            att_feat = x
        x = x.to(torch.float32)
        b, c, h, w = x.shape
        # Pad to even dims
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            att_feat = F.pad(att_feat, (0, pad_w, 0, pad_h), mode='reflect')
        hp, wp = x.shape[-2], x.shape[-1]

        LL, LH, HL, HH = haar_dwt2d(x)
        subbands = [LL, LH, HL, HH]

        # Reconstruct each subband in isolation at full resolution
        bands = []
        for i in range(4):
            zeros = [torch.zeros_like(s) for s in subbands]
            zeros[i] = subbands[i]
            bands.append(haar_idwt2d(*zeros))

        # Modulate detail bands (LH, HL, HH), pass LL through or modulate
        out = torch.zeros(b, c, hp, wp, device=x.device, dtype=x.dtype)
        idx = 0
        for i in range(1, 4):  # LH, HL, HH
            att = self.sp_act(self.weight_convs[idx](att_feat))
            band = bands[i]
            tmp = (att.reshape(b, self.spatial_group, -1, hp, wp) *
                   band.reshape(b, self.spatial_group, -1, hp, wp))
            out = out + tmp.reshape(b, c, hp, wp)
            idx += 1

        if self.include_LL:
            att = self.sp_act(self.weight_convs[idx](att_feat))
            band = bands[0]
            tmp = (att.reshape(b, self.spatial_group, -1, hp, wp) *
                   band.reshape(b, self.spatial_group, -1, hp, wp))
            out = out + tmp.reshape(b, c, hp, wp)
        else:
            out = out + bands[0]

        if pad_h or pad_w:
            out = out[..., :h, :w]
        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  WDConv — main module
# ═══════════════════════════════════════════════════════════════════════════════

class WDConv(nn.Conv2d):
    """Wavelet Dynamic Convolution. Drop-in replacement for nn.Conv2d / FDConv."""

    def __init__(self,
                 *args,
                 reduction=0.0625,
                 kernel_num=4,
                 use_wdconv_if_c_gt=16,
                 use_wdconv_if_k_in=[1, 3],
                 use_wdconv_if_stride_in=[1],
                 use_wbm_if_k_in=[3],
                 use_wbm_for_stride=False,
                 kernel_temp=1.0,
                 temp=None,
                 att_multi=2.0,
                 param_ratio=1,
                 param_reduction=1.0,
                 ksm_only_kernel_att=False,
                 att_grid=1,
                 use_ksm_local=True,
                 ksm_local_act='sigmoid',
                 ksm_global_act='sigmoid',
                 spatial_freq_decompose=False,
                 convert_param=True,
                 linear_mode=False,
                 wbm_cfg=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if wbm_cfg is None:
            wbm_cfg = {
                'spatial_group': 1,
                'spatial_kernel': 3,
                'act': 'sigmoid',
                'init': 'zero',
                'include_LL': False,
            }
        self.use_wdconv_if_c_gt = use_wdconv_if_c_gt
        self.use_wdconv_if_k_in = use_wdconv_if_k_in
        self.use_wdconv_if_stride_in = use_wdconv_if_stride_in
        self.kernel_num = kernel_num
        self.param_ratio = param_ratio
        self.param_reduction = param_reduction
        self.use_ksm_local = use_ksm_local
        self.att_multi = att_multi
        self.use_wbm_if_k_in = use_wbm_if_k_in
        self.ksm_local_act = ksm_local_act
        self.ksm_global_act = ksm_global_act

        if self.kernel_num is None:
            self.kernel_num = self.out_channels // 2
            kernel_temp = math.sqrt(self.kernel_num * self.param_ratio)
        if temp is None:
            temp = kernel_temp

        # Gate: only activate WDConv for sufficiently large layers
        if (min(self.in_channels, self.out_channels) <= self.use_wdconv_if_c_gt
                or self.kernel_size[0] not in self.use_wdconv_if_k_in):
            return

        # ── sqrt(N) normalization ──────────────────────────────────────────
        # Haar DWT is orthonormal: ||DWT(w)|| ≈ ||w||.  FDConv's rfft2 is
        # un-normalised: ||rfft2(w)|| ≈ sqrt(N)*||w||.  Both divide by the
        # same `scale`, so dwt_weight ends up ~sqrt(N) smaller than
        # dft_weight.  The gradient, however, is NOT damped by 1/N (no
        # irfft2 in the backward path), so the effective learning rate on
        # dwt_weight is ~N times that of dft_weight → explosion.
        #
        # Fix: store dwt_weight *= sqrt(N)  and  alpha /= sqrt(N).
        # Forward output is unchanged; gradient magnitude matches FDConv.
        _H = self.out_channels * self.kernel_size[0]
        _W = self.in_channels * self.kernel_size[1]
        self._sqrtN = math.sqrt(_H * _W)

        self.alpha = (min(self.out_channels, self.in_channels) // 2
                      * self.kernel_num * self.param_ratio
                      / param_reduction) / self._sqrtN

        self.KSM_Global = KernelSpatialModulation_Global(
            self.in_channels, self.out_channels, self.kernel_size[0],
            groups=self.groups, temp=temp, kernel_temp=kernel_temp,
            reduction=reduction,
            kernel_num=self.kernel_num * self.param_ratio,
            kernel_att_init=None, att_multi=att_multi,
            ksm_only_kernel_att=ksm_only_kernel_att,
            act_type=self.ksm_global_act, att_grid=att_grid,
            stride=self.stride, spatial_freq_decompose=spatial_freq_decompose)

        if self.kernel_size[0] in use_wbm_if_k_in or (use_wbm_for_stride and self.stride[0] > 1):
            self.WBM = WaveletBandModulation(self.in_channels, **wbm_cfg)

        if self.use_ksm_local:
            self.KSM_Local = KernelSpatialModulation_Local(
                channel=self.in_channels, kernel_num=1,
                out_n=int(self.out_channels * self.kernel_size[0] * self.kernel_size[1]))

        self.linear_mode = linear_mode
        self.convert2dwtweight(convert_param)

    # ── Weight conversion: spatial → wavelet domain ──────────────────────────

    def convert2dwtweight(self, convert_param):
        d1, d2 = self.out_channels, self.in_channels
        k1, k2 = self.kernel_size[0], self.kernel_size[1]
        H, W = d1 * k1, d2 * k2

        # Haar DWT needs even dimensions
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(
                f"WDConv requires (out_ch*k1, in_ch*k2) to be even. "
                f"Got ({H}, {W}). Use use_wdconv_if_c_gt to skip small layers.")

        weight = self.weight.permute(0, 2, 1, 3).reshape(H, W)
        dwt_weight = haar_dwt2d_pack(weight)  # (H, W) real wavelet coefficients

        # Priority-sorted indices
        indices_all = get_wavelet_indices(H, W)  # (2, H*W)

        scale = max(min(self.out_channels, self.in_channels) // 2, 1)

        # Optionally keep only a fraction of coefficients
        if self.param_reduction < 1:
            num_keep = int(indices_all.size(1) * self.param_reduction)
            indices_all = indices_all[:, :num_keep]

        # Trim to be divisible by kernel_num
        total = indices_all.size(1)
        per_kernel = total // self.kernel_num
        kept = per_kernel * self.kernel_num
        indices_all = indices_all[:, :kept]

        # Build learnable wavelet weight tensor
        # Multiply by sqrt(N) so dwt_weight has same magnitude as FDConv's
        # dft_weight (see alpha comment in __init__).
        sqrtN = math.sqrt(H * W)
        if self.param_reduction < 1:
            flat = dwt_weight[indices_all[0], indices_all[1]]
            dwt_weight_param = flat[None].repeat(self.param_ratio, 1) * sqrtN / scale
        else:
            dwt_weight_param = dwt_weight[None].repeat(self.param_ratio, 1, 1) * sqrtN / scale

        if convert_param:
            self.dwt_weight = nn.Parameter(dwt_weight_param, requires_grad=True)
            del self.weight
        else:
            if self.linear_mode:
                assert k1 == 1 and k2 == 1
                self.weight = nn.Parameter(self.weight.squeeze(), requires_grad=True)

        # Index partition across kernel slots
        idx_partitioned = indices_all.reshape(2, self.kernel_num, -1)
        indices = torch.stack([idx_partitioned] * self.param_ratio, dim=0)
        self.register_buffer('indices', indices, persistent=False)

    def get_WDW(self):
        """Derive wavelet weights from self.weight (for convert_param=False)."""
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        weight = self.weight.reshape(d1, d2, k1, k2).permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        dwt = haar_dwt2d_pack(weight)
        scale = max(min(self.out_channels, self.in_channels) // 2, 1)
        sqrtN = math.sqrt(d1 * k1 * d2 * k2)
        return dwt[None].repeat(self.param_ratio, 1, 1) * sqrtN / scale

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x):
        if (min(self.in_channels, self.out_channels) <= self.use_wdconv_if_c_gt
                or self.kernel_size[0] not in self.use_wdconv_if_k_in):
            return super().forward(x)

        batch_size, in_planes, height, width = x.size()

        # ─ Attention ──────────────────────────────────────────────────────────
        global_x = F.adaptive_avg_pool2d(x, 1)
        channel_att, filter_att, spatial_att, kernel_att = self.KSM_Global(global_x)

        if self.use_ksm_local:
            hr = self.KSM_Local(global_x)
            hr = hr.reshape(batch_size, 1, self.in_channels, self.out_channels,
                            self.kernel_size[0], self.kernel_size[1])
            hr = hr.permute(0, 1, 3, 2, 4, 5)
            if self.ksm_local_act == 'sigmoid':
                hr_att = hr.sigmoid() * self.att_multi
            else:
                hr_att = 1 + hr.tanh()
        else:
            hr_att = 1

        # ─ Wavelet Disjoint Weight assembly ───────────────────────────────────
        H = self.out_channels * self.kernel_size[0]
        W = self.in_channels * self.kernel_size[1]

        # Force float32 for coefficient assembly — mirrors FDConv's implicit float32 DFT_map.
        # Under bfloat16 autocast, using x.dtype (bfloat16) here corrupts gradients of
        # dwt_weight via 7-bit mantissa quantization, causing exponential loss explosion.
        DWT_map = torch.zeros((batch_size, H, W), device=x.device, dtype=torch.float32)
        kernel_att = kernel_att.reshape(batch_size, self.param_ratio, self.kernel_num, -1)

        dwt_w = (self.dwt_weight if hasattr(self, 'dwt_weight') else self.get_WDW()).float()

        for i in range(self.param_ratio):
            indices = self.indices[i]  # (2, kernel_num, coeff_per_kernel)
            if self.param_reduction < 1:
                w = dwt_w[i].reshape(self.kernel_num, -1)[None]
                DWT_map[:, indices[0, :, :], indices[1, :, :]] += (w * kernel_att[:, i].float())
            else:
                w = dwt_w[i][indices[0, :, :], indices[1, :, :]][None] * self.alpha
                DWT_map[:, indices[0, :, :], indices[1, :, :]] += (w * kernel_att[:, i].float())

        # Inverse wavelet → spatial kernel weights (computed in float32 for stability)
        adaptive_weights = haar_idwt2d_unpack(DWT_map)
        adaptive_weights = adaptive_weights.reshape(
            batch_size, 1, self.out_channels, self.kernel_size[0],
            self.in_channels, self.kernel_size[1])
        adaptive_weights = adaptive_weights.permute(0, 1, 2, 4, 3, 5)
        # Cast back to input dtype so conv2d receives consistent types
        adaptive_weights = adaptive_weights.to(x.dtype)

        # ─ Wavelet Band Modulation on input features ──────────────────────────
        if hasattr(self, 'WBM'):
            x = self.WBM(x)

        # ─ Dynamic convolution ────────────────────────────────────────────────
        kernel_area = self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        feat_area = (in_planes + self.out_channels) * height * width

        if kernel_area < feat_area:
            agg = spatial_att * channel_att * filter_att * adaptive_weights * hr_att
            agg = torch.sum(agg, dim=1)
            agg = agg.view(-1, self.in_channels // self.groups,
                           self.kernel_size[0], self.kernel_size[1])
            x_flat = x.reshape(1, -1, height, width)
            output = F.conv2d(x_flat, weight=agg, bias=None,
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
            output = output.view(batch_size, self.out_channels,
                                 output.size(-2), output.size(-1))
        else:
            agg = spatial_att * adaptive_weights * hr_att
            agg = torch.sum(agg, dim=1)
            if not isinstance(channel_att, float):
                x = x * channel_att.view(batch_size, -1, 1, 1)
            agg = agg.view(-1, self.in_channels // self.groups,
                           self.kernel_size[0], self.kernel_size[1])
            x_flat = x.reshape(1, -1, height, width)
            output = F.conv2d(x_flat, weight=agg, bias=None,
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
            if isinstance(filter_att, float):
                output = output.view(batch_size, self.out_channels,
                                     output.size(-2), output.size(-1))
            else:
                output = (output.view(batch_size, self.out_channels,
                                      output.size(-2), output.size(-1))
                          * filter_att.view(batch_size, -1, 1, 1))

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output


# ═══════════════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    torch.manual_seed(42)

    # 1. Haar DWT round-trip
    x = torch.randn(2, 64, 16, 16)
    LL, LH, HL, HH = haar_dwt2d(x)
    recon = haar_idwt2d(LL, LH, HL, HH)
    err = (recon - x).abs().max().item()
    print(f"[test] Haar DWT/IDWT round-trip error: {err:.2e}")
    assert err < 1e-5

    # 2. Pack/unpack round-trip
    packed = haar_dwt2d_pack(x)
    recon2 = haar_idwt2d_unpack(packed)
    err2 = (recon2 - x).abs().max().item()
    print(f"[test] Pack/Unpack round-trip error:   {err2:.2e}")
    assert err2 < 1e-5

    # 3. WDConv forward
    m = WDConv(in_channels=64, out_channels=128, kernel_size=3, padding=1,
               bias=True, kernel_num=4)
    y = m(x)
    print(f"[test] WDConv forward: {x.shape} → {y.shape}")
    assert y.shape == (2, 128, 16, 16)

    # 4. Small channels → pass-through (standard Conv2d)
    m2 = WDConv(in_channels=8, out_channels=8, kernel_size=3, padding=1, kernel_num=4)
    y2 = m2(torch.randn(1, 8, 8, 8))
    print(f"[test] WDConv pass-through (small ch): {y2.shape}")
    assert y2.shape == (1, 8, 8, 8)

    # 5. 1x1 conv
    m3 = WDConv(in_channels=64, out_channels=64, kernel_size=1, kernel_num=4)
    m3.eval()
    y3 = m3(torch.randn(1, 64, 16, 16))
    print(f"[test] WDConv 1x1: {y3.shape}")
    assert y3.shape == (1, 64, 16, 16)

    print("\nAll tests passed!")
