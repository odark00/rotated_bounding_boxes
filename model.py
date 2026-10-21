"""
RTMDet-Rotated style model for parcel 3D detection.

Architecture:
  Backbone : ResNet (grayscale) -> C3/C4/C5 multi-scale features
  Neck     : lightweight FPN -> P3/P4/P5 (strides 8/16/32), `d_model`-ch
  Head     : shared depthwise-conv towers per level, predicts per anchor:
               cls         (1)  - foreground logit
               cx, cy      (2)  - box centre in absolute pixels
               w, h        (2)  - box LONG/SHORT edge in pixels (w >= h by
                                  construction, see dataset._rbox_from_top_face)
               sin_a,      (2)  - normalised (sin, cos) of 2*rotation angle (see
               cos_a              losses.py: doubled to remove the discontinuity
                                  at the rectangle's 180-degree wrap point)
               type_logit  (1)  - P(palletised shipping unit) vs loose parcel
               W0 L0 H0    (3)  - METRIC long/short/height if it is a parcel
               W1 L1 H1    (3)  - ... if it is a palletised unit
               depth_canon (1)  - CANONICAL depth, direct estimate (see below)
               log_sig_dir (1)  - Laplacian log-scale of that direct depth
               log_sig_geo (1)  - ... of the geometric depth (see below)
               log_sig_dim (1)  - ... of the metric dimensions

Why `depth_canon` / `h3d_px` instead of metric depth / height
-------------------------------------------------------------
The TASLER cameras span 11 lens models with fx from 411 to 1100 px (a 2.68x
spread) at a fixed 640x640 render size. A parcel at 4 m under a 1100 px lens
projects to exactly the same pixels as a parcel at 1.5 m under a 411 px lens,
so *metric* depth and *metric* height are simply not functions of the image:
asking a conv net to regress them forces it to output a dataset average.
Measured on the training labels, log(H3d_metres) is predicted from apparent
size with R^2 = 0.000 — i.e. no signal at all — which is why the previous
model's H MAE sat at ~37% of the median height.

Both become well posed once expressed in units the image actually determines --
an "apparent size in pixels", which is what a camera measures:

  depth_canon = depth * FX_REF / fx      (depth under a reference lens)
  w3d_px      = W_3d * fx / depth        (apparent long  dimension, pixels)
  l3d_px      = L_3d * fy / depth        (apparent short dimension, pixels)
  h3d_px      = H_3d * fy / depth        (apparent height,          pixels)

and the metric answer is then recovered analytically from the known
intrinsics, with no learned scale factor anywhere:

  depth = depth_canon * fx / FX_REF
  W_3d  = w3d_px * depth / fx
  L_3d  = l3d_px * depth / fy
  H_3d  = h3d_px * depth / fy

Note that the metric dimensions come from their own heads, not from the 2D box
edges `w`/`h`. Reusing the box edges (W_3d = w_px * depth / fx) assumes the
parcel's top face is fronto-parallel and its whole edge sits at the centre
depth; measured against the labels that approximation costs a mean 8.8 mm and
up to 51 cm on W, a systematic floor no amount of training removes. Separating
them also lets `w`/`h` stay a pure 2D detection quantity -- what NMS, IoU
matching and the angle share -- while the dimensions reported to the user come
from heads supervised directly on the metric labels.

Revision 2026-09-14: metric dimensions, object type, uncertainty
----------------------------------------------------------------
The `w3d_px * depth / fx` recovery above made every metric size the PRODUCT of
two learned quantities, so the whole absolute-scale ambiguity of monocular
depth landed on the sizes too. It was fine in-distribution (6% median error)
and failed the moment the scale cue was uncertain: on the real `packstuecke/`
frames the pixel extents were right (0.95 of the human 2D box) but depth_canon
was half what a EUR-pallet unit needs, and the predicted long edge came out at
0.59 m - almost exactly the geometric mean (0.56 m) of the parcel prior
(0.24 m) and the unit prior (1.15 m). A single log-space regression head,
unsure which of two size modes it is looking at, averages them.

The head now follows the standard multi-dataset recipe (Cube R-CNN):

* **Metric dimensions come from their own head**, as a log residual over a
  per-type prior:  dim = prior[type] * exp(delta[type]). A pallet is ~1.2 m
  under any lens, so metric size is a property of the object, not of the
  camera - it no longer needs depth at all, and it is fx-free.
* **Object type is an explicit, class-balanced classification**, with one set
  of residuals per type. Picking the mode by classification and regressing only
  within it is what stops the head averaging between modes.
* **Depth is fused from two estimates** (MonoFlex / GUPNet): the direct
  canonical head, and a geometric one that follows from the metric size and the
  2D box, depth_geo = W * fx / w_px + H / 2 (the box is the TOP face, which is
  half a height nearer the camera than the centre the depth label refers to).
  Each carries a learned Laplacian uncertainty and they are combined by
  inverse uncertainty, so whichever is more reliable on a given object wins.

`decode_3d()` below is the single implementation of the metric conversion;
predict.py and evaluate.py both call it so inference and eval cannot drift.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# Reference focal length (px, at model input resolution) defining the
# "canonical" camera that `depth_canon` is expressed in. Chosen as roughly the
# median fx of the TASLER camera set, so canonical and metric depth agree for a
# typical lens. Stored in the checkpoint hparams; changing it changes the
# meaning of the depth head, so it must match between training and inference.
FX_REF = 800.0


# ---------------------------------------------------------------------------
# Object types and their size priors
# ---------------------------------------------------------------------------
TYPE_NAMES = ("parcel", "palletised_load")

# Geometric-mean (W_long, L_short, H) per type, in metres, measured on the
# training split of TASLER_MULTI + TASLER + TASLER_PACK (val excluded). The
# heads predict log residuals over these, so they only set the starting point
# and the scale of the residual - not a hard limit. Stored in the checkpoint.
SIZE_PRIORS = ((0.2355, 0.1708, 0.1528),     # parcel           (n=53434)
               (1.1461, 0.8981, 1.0469))     # palletised_load  (n=2388)

LOG_SIG_CLAMP = (-5.0, 5.0)


# ---------------------------------------------------------------------------
# Metric decoding (shared by training-time logging, predict.py and evaluate.py)
# ---------------------------------------------------------------------------
def _xp(x):
    """numpy for arrays/floats, torch for tensors - keeps one implementation."""
    return torch if torch.is_tensor(x) else np


def _floor(x, lo):
    return x.clamp(min=lo) if torch.is_tensor(x) else np.maximum(x, lo)


def geometric_depth(W, L, H, w_px, h_px, fx, fy):
    """Metric centre depth implied by a metric size and its 2D top-face box.

    The 2D box is the projection of the TOP face, so its extents give the depth
    of that face; the label (and the direct head) refer to the box centre, half
    a height further away along a near-nadir optical axis.
    """
    top = 0.5 * (W * fx / _floor(w_px, 1e-3) + L * fy / _floor(h_px, 1e-3))
    return top + 0.5 * H


def decode_3d(dets, fx, fy, fx_ref=FX_REF):
    """
    Convert one image's decoded head outputs into metric quantities.

    Args:
        dets   : dict with numpy arrays (or tensors) for keys w, h, type_logit,
                 W0 L0 H0 W1 L1 H1, depth_canon, log_sig_dir, log_sig_geo
        fx, fy : intrinsics **at model input resolution**
        fx_ref : the FX_REF the checkpoint was trained with

    Returns:
        dict(depth, W, L, H, p_unit, depth_dir, depth_geo) - metres. W is the
        longer horizontal dimension and L the shorter, matching the long-edge
        box convention. W/L/H do not depend on fx at all; depth does, as it
        physically must.
    """
    xp = _xp(dets["w"])
    p_unit = 1.0 / (1.0 + xp.exp(-dets["type_logit"]))
    unit = p_unit >= 0.5
    W = xp.where(unit, dets["W1"], dets["W0"])
    L = xp.where(unit, dets["L1"], dets["L0"])
    H = xp.where(unit, dets["H1"], dets["H0"])

    depth_dir = dets["depth_canon"] * fx / fx_ref
    depth_geo = geometric_depth(W, L, H, dets["w"], dets["h"], fx, fy)

    # inverse-uncertainty fusion in log space (MonoFlex): a Laplacian scale is
    # the expected absolute log error, so 1/sigma weights each estimate by how
    # reliable the head says it is on this particular object
    wd = xp.exp(-dets["log_sig_dir"])
    wg = xp.exp(-dets["log_sig_geo"])
    depth = xp.exp((wd * xp.log(_floor(depth_dir, 1e-3))
                    + wg * xp.log(_floor(depth_geo, 1e-3))) / (wd + wg))
    return dict(depth=depth, W=W, L=L, H=H, p_unit=p_unit,
                depth_dir=depth_dir, depth_geo=depth_geo)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
_BACKBONE_FACTORY = {
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.DEFAULT, (128, 256, 512)),
    "resnet34": (tvm.resnet34, tvm.ResNet34_Weights.DEFAULT, (128, 256, 512)),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.DEFAULT, (512, 1024, 2048)),
}


class GrayscaleResNet(nn.Module):
    """ResNet adapted for 1-channel input; returns (C3, C4, C5) feature maps."""

    def __init__(self, backbone: str = "resnet50"):
        super().__init__()
        if backbone not in _BACKBONE_FACTORY:
            raise ValueError(f"Unknown backbone {backbone!r}; choose from {list(_BACKBONE_FACTORY)}")
        ctor, weights, self.out_channels = _BACKBONE_FACTORY[backbone]

        r = ctor(weights=weights)
        w = r.conv1.weight.data.mean(dim=1, keepdim=True)
        r.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        r.conv1.weight.data = w

        self.stem   = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1 = r.layer1   # stride 4
        self.layer2 = r.layer2   # stride 8,  -> C3
        self.layer3 = r.layer3   # stride 16, -> C4
        self.layer4 = r.layer4   # stride 32, -> C5

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
    def __init__(self, in_chs=(512, 1024, 2048), out_ch=128):
        # in_chs default matches resnet50; pass GrayscaleResNet.out_channels
        # for other backbones (e.g. resnet34 -> (128, 256, 512)).
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

    All spatial outputs are (B, 1, H, W) in decoded units:
      cls         - raw logit (no activation; apply sigmoid externally)
      cx, cy      - absolute pixel coords
      w, h        - pixel long/short edge (positive)
      sin_a       - sin(2*angle), normalised so sin^2+cos^2=1
      cos_a       - cos(2*angle)
      type_logit  - raw logit, P(palletised unit)
      W0..H1      - metric dims per type: prior * exp(residual), metres
      depth_canon - canonical depth, log-uniform in [d_min, d_max]
      log_sig_*   - Laplacian log-scales (direct depth, geometric depth, dims)
    """

    def __init__(self, ch=128, num_convs=2, depth_canon_range=(0.4, 28.0),
                 size_priors=SIZE_PRIORS):
        super().__init__()
        # (n_types, 3) log priors; residuals are predicted around these
        self.register_buffer("log_prior", torch.tensor(size_priors).log())
        # log-uniform parameterisation: the head's sigmoid spans log-depth, so
        # resolution is uniform in *relative* error, matching the log-space
        # depth loss in losses.py (a 5 cm error at 1 m and a 50 cm error at
        # 10 m are equally penalised).
        self.register_buffer("log_dmin", torch.tensor(float(depth_canon_range[0])).log())
        self.register_buffer("log_dspan", torch.tensor(
            float(depth_canon_range[1]) / float(depth_canon_range[0])).log())

        self.cls_tower = nn.Sequential(*[DWConvBNSiLU(ch) for _ in range(num_convs)])
        self.reg_tower = nn.Sequential(*[DWConvBNSiLU(ch) for _ in range(num_convs)])
        # 3D gets its own tower rather than sharing the 2D regression one: what
        # decides an object's metric size and depth (texture scale, a pallet's
        # slats, a forklift's forks) is a different feature from where its
        # edges are, and sharing one tower makes the two compete.
        self.d3_tower = nn.Sequential(*[DWConvBNSiLU(ch) for _ in range(num_convs)])

        n_types = self.log_prior.shape[0]
        self.cls_out    = nn.Conv2d(ch, 1, 1)
        self.box_out    = nn.Conv2d(ch, 4, 1)   # (dx_off, dy_off, log_w, log_h)
        self.angle_out  = nn.Conv2d(ch, 2, 1)   # (raw_sin, raw_cos)
        self.type_out   = nn.Conv2d(ch, 1, 1)   # -> P(palletised unit)
        self.dim_out    = nn.Conv2d(ch, 3 * n_types, 1)  # log residuals per type
        self.depth_out  = nn.Conv2d(ch, 1, 1)   # -> depth_canon (direct)
        self.sig_out    = nn.Conv2d(ch, 3, 1)   # log-sigma: dir, geo, dim

        # low initial false-positive rate
        nn.init.constant_(self.cls_out.bias, -4.6)
        # residuals start at the prior; uncertainties start at sigma = 1
        nn.init.zeros_(self.dim_out.weight)
        nn.init.zeros_(self.dim_out.bias)
        nn.init.zeros_(self.sig_out.bias)

    def forward(self, feat, stride):
        B, C, H, W = feat.shape
        dev = feat.device

        cls_f = self.cls_tower(feat)
        reg_f = self.reg_tower(feat)
        d3_f = self.d3_tower(feat)

        # anchor grid centres in absolute pixels
        gy, gx = torch.meshgrid(
            torch.arange(H, device=dev, dtype=feat.dtype) + 0.5,
            torch.arange(W, device=dev, dtype=feat.dtype) + 0.5,
            indexing='ij',
        )
        ax = (gx * stride).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        ay = (gy * stride).unsqueeze(0).unsqueeze(0)

        box = self.box_out(reg_f)                       # (B,4,H,W)
        # centre stays within +-0.5 stride of anchor
        cx = ax + torch.tanh(box[:, 0:1]) * stride
        cy = ay + torch.tanh(box[:, 1:2]) * stride
        # w,h: exp with clamp for stability
        w  = torch.exp(box[:, 2:3].clamp(-4, 4)) * stride
        h  = torch.exp(box[:, 3:4].clamp(-4, 4)) * stride

        ar = self.angle_out(reg_f)                      # (B,2,H,W)
        norm = ar.norm(dim=1, keepdim=True).clamp(min=1e-6)
        sin_a = ar[:, 0:1] / norm
        cos_a = ar[:, 1:2] / norm

        depth_canon = torch.exp(self.log_dmin + self.log_dspan * self.depth_out(d3_f).sigmoid())

        # metric dims: prior * exp(residual), one (W, L, H) triple per type
        n_types = self.log_prior.shape[0]
        delta = self.dim_out(d3_f).clamp(-3, 3).view(B, n_types, 3, H, W)
        dims = torch.exp(self.log_prior.view(1, n_types, 3, 1, 1) + delta)

        sig = self.sig_out(d3_f).clamp(*LOG_SIG_CLAMP)

        out = dict(
            cls=self.cls_out(cls_f),
            cx=cx, cy=cy, w=w, h=h,
            sin_a=sin_a, cos_a=cos_a,
            type_logit=self.type_out(d3_f),
            depth_canon=depth_canon,
            log_sig_dir=sig[:, 0:1], log_sig_geo=sig[:, 1:2], log_sig_dim=sig[:, 2:3],
        )
        for t in range(n_types):
            out[f"W{t}"] = dims[:, t, 0:1]
            out[f"L{t}"] = dims[:, t, 1:2]
            out[f"H{t}"] = dims[:, t, 2:3]
        return out


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class RTMDetRotated(nn.Module):
    """
    RTMDet-Rotated detector.

    forward() returns a list of 3 dicts (one per FPN level P3/P4/P5).
    Each dict has the keys in OUT_KEYS, all shaped (B, 1, H_l, W_l).
    """

    STRIDES = [8, 16, 32]
    OUT_KEYS = ['cls', 'cx', 'cy', 'w', 'h', 'sin_a', 'cos_a', 'type_logit',
                'W0', 'L0', 'H0', 'W1', 'L1', 'H1',
                'depth_canon', 'log_sig_dir', 'log_sig_geo', 'log_sig_dim']

    def __init__(self, d_model=128, num_head_convs=2, backbone="resnet50",
                 depth_canon_range=(0.4, 28.0), size_priors=SIZE_PRIORS):
        super().__init__()
        self.backbone = GrayscaleResNet(backbone=backbone)
        self.neck     = FPN(in_chs=self.backbone.out_channels, out_ch=d_model)
        self.heads    = nn.ModuleList([
            RTMDetHead(ch=d_model, num_convs=num_head_convs,
                       depth_canon_range=depth_canon_range,
                       size_priors=size_priors)
            for _ in self.STRIDES
        ])

    def forward(self, images):
        feats = self.neck(self.backbone(images))
        return [head(f, s) for head, f, s in zip(self.heads, feats, self.STRIDES)]
