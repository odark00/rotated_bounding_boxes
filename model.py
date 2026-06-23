"""
RTMDet-Rotated style model for parcel 3D detection.

Architecture:
  Backbone : ResNet50 (grayscale) → C3/C4/C5 multi-scale features
  Neck     : lightweight FPN → P3/P4/P5 (strides 8/16/32), all 256-ch
  Head     : shared depthwise-conv towers per level, predicts per anchor:
               cls       (1)  — foreground logit
               cx, cy    (2)  — box centre in absolute pixels
               w, h      (2)  — box width/height in pixels
               sin_a,    (2)  — normalised (sin, cos) of rotation angle
               cos_a
               depth     (1)  — metric depth (m)
               height3d  (1)  — metric 3D height of parcel (m)

At inference with camera intrinsics (fx, fy):
  W_3d = w_px  * depth / fx
  L_3d = h_px  * depth / fy
  H_3d = height3d  (direct prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
class GrayscaleResNet(nn.Module):
    """ResNet50 adapted for 1-channel input; returns (C3, C4, C5) feature maps."""

    def __init__(self):
        super().__init__()
        r = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        w = r.conv1.weight.data.mean(dim=1, keepdim=True)
        r.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        r.conv1.weight.data = w

        self.stem   = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1 = r.layer1   # stride 4,  256-ch
        self.layer2 = r.layer2   # stride 8,  512-ch  → C3
        self.layer3 = r.layer3   # stride 16, 1024-ch → C4
        self.layer4 = r.layer4   # stride 32, 2048-ch → C5

    def forward(self, x):
        x  = self.stem(x)
        x  = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


# ---------------------------------------------------------------------------
# FPN neck
# ---------------------------------------------------------------------------
class FPN(nn.Module):
    def __init__(self, in_chs=(512, 1024, 2048), out_ch=256):
        super().__init__()
        self.lat    = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_chs])
        self.smooth = nn.ModuleList([nn.Conv2d(out_ch, out_ch, 3, 1, 1) for _ in in_chs])

    def forward(self, feats):
        lats = [l(f) for l, f in zip(self.lat, feats)]
        for i in range(len(lats) - 2, -1, -1):
            lats[i] = lats[i] + F.interpolate(lats[i+1], size=lats[i].shape[-2:], mode='nearest')
        return [s(l) for s, l in zip(self.smooth, lats)]   # P3, P4, P5


# ---------------------------------------------------------------------------
# Head building block
# ---------------------------------------------------------------------------
class DWConvBNSiLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        return F.silu(self.bn(self.pw(self.dw(x))))


# ---------------------------------------------------------------------------
# Per-level RTMDet head
# ---------------------------------------------------------------------------
class RTMDetHead(nn.Module):
    """
    One head instance per FPN level.

    All spatial outputs are (B, 1, H, W) in decoded, absolute units:
      cls       — raw logit (no activation; apply sigmoid externally)
      cx, cy    — absolute pixel coords
      w, h      — pixel dimensions (positive)
      sin_a     — sin(angle), normalised so sin²+cos²=1
      cos_a     — cos(angle)
      depth     — meters in [depth_min, depth_max]
      height3d  — meters in [h3d_min, h3d_max]
    """

    def __init__(self, ch=256, num_convs=2,
                 depth_range=(1.2, 1.5), height3d_range=(0.04, 0.30)):
        super().__init__()
        self.dmin, self.dmax = depth_range
        self.hmin, self.hmax = height3d_range

        self.cls_tower = nn.Sequential(*[DWConvBNSiLU(ch) for _ in range(num_convs)])
        self.reg_tower = nn.Sequential(*[DWConvBNSiLU(ch) for _ in range(num_convs)])

        self.cls_out    = nn.Conv2d(ch, 1, 1)
        self.box_out    = nn.Conv2d(ch, 4, 1)   # (dx_off, dy_off, log_w, log_h)
        self.angle_out  = nn.Conv2d(ch, 2, 1)   # (raw_sin, raw_cos)
        self.depth_out  = nn.Conv2d(ch, 1, 1)
        self.height_out = nn.Conv2d(ch, 1, 1)

        # low initial false-positive rate
        nn.init.constant_(self.cls_out.bias, -4.6)

    def forward(self, feat, stride):
        B, C, H, W = feat.shape
        dev = feat.device

        cls_f = self.cls_tower(feat)
        reg_f = self.reg_tower(feat)

        # anchor grid centres in absolute pixels
        gy, gx = torch.meshgrid(
            torch.arange(H, device=dev, dtype=feat.dtype) + 0.5,
            torch.arange(W, device=dev, dtype=feat.dtype) + 0.5,
            indexing='ij',
        )
        ax = (gx * stride).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        ay = (gy * stride).unsqueeze(0).unsqueeze(0)

        box = self.box_out(reg_f)                       # (B,4,H,W)
        # centre stays within ±0.5 stride of anchor
        cx = ax + torch.tanh(box[:, 0:1]) * stride
        cy = ay + torch.tanh(box[:, 1:2]) * stride
        # w,h: exp with clamp for stability
        w  = torch.exp(box[:, 2:3].clamp(-4, 4)) * stride
        h  = torch.exp(box[:, 3:4].clamp(-4, 4)) * stride

        ar = self.angle_out(reg_f)                      # (B,2,H,W)
        norm = ar.norm(dim=1, keepdim=True).clamp(min=1e-6)
        sin_a = ar[:, 0:1] / norm
        cos_a = ar[:, 1:2] / norm

        depth    = self.dmin + (self.dmax - self.dmin) * self.depth_out(reg_f).sigmoid()
        height3d = self.hmin + (self.hmax - self.hmin) * self.height_out(reg_f).sigmoid()

        return dict(
            cls=self.cls_out(cls_f),
            cx=cx, cy=cy, w=w, h=h,
            sin_a=sin_a, cos_a=cos_a,
            depth=depth, height3d=height3d,
        )


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class RTMDetRotated(nn.Module):
    """
    RTMDet-Rotated detector.

    forward() returns a list of 3 dicts (one per FPN level P3/P4/P5).
    Each dict has keys: cls, cx, cy, w, h, sin_a, cos_a, depth, height3d
    all shaped (B, 1, H_l, W_l).
    """

    STRIDES = [8, 16, 32]

    def __init__(self, d_model=256, num_head_convs=2,
                 depth_range=(1.2, 1.5), height3d_range=(0.04, 0.30)):
        super().__init__()
        self.backbone = GrayscaleResNet()
        self.neck     = FPN(out_ch=d_model)
        self.heads    = nn.ModuleList([
            RTMDetHead(ch=d_model, num_convs=num_head_convs,
                       depth_range=depth_range, height3d_range=height3d_range)
            for _ in self.STRIDES
        ])

    def forward(self, images):
        feats = self.neck(self.backbone(images))
        return [head(f, s) for head, f, s in zip(self.heads, feats, self.STRIDES)]
