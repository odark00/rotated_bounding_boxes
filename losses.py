"""
RTMDet-Rotated training module.

Assignment  : top-k (k=9) anchor points per GT by centre distance, per FPN level,
              gated by object size so each GT trains the level matching its scale.
Losses      : Focal (cls, normalised by #positives)
            + GWD (oriented box: centre + size + angle jointly)
            + L1 (centre / log-size, for early-training stability)
            + elongation-weighted angle L1
            + class-balanced BCE on the object type (parcel / palletised unit)
            + Laplacian NLL on the metric dimensions of the GT type
            + Laplacian NLL on the direct canonical depth
            + Laplacian NLL on the geometric depth (trains its uncertainty only)
            + log-space L1 on the uncertainty-fused depth

Uncertainty-weighted 3D losses follow Cube R-CNN / MonoDLE:
    L = sqrt(2) * |log y_hat - log y| * exp(-s) + s,     s = log sigma
which lets the head say "this one is ambiguous" (a near-square top face, a
heavily truncated unit) instead of dragging its estimate for every other
object towards the compromise. A plain L1 term rides along at a small weight so
a head that inflates its sigma cannot switch its own gradient off.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import math

from model import RTMDetRotated, FX_REF, SIZE_PRIORS, geometric_depth

_SQRT2 = math.sqrt(2.0)


def laplace_nll(err, log_sig):
    """err: absolute log error (>= 0). Sum-reduced; caller normalises."""
    return (_SQRT2 * err * torch.exp(-log_sig) + log_sig).sum()


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Binary focal loss, **sum**-reduced (caller divides by #positives)."""
    p   = logits.sigmoid()
    ce  = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t) ** gamma * ce).sum()


# ---------------------------------------------------------------------------
# Gaussian Wasserstein Distance loss for oriented boxes
# ---------------------------------------------------------------------------
def _box_gaussian(w, h, sin2a, cos2a):
    """
    Covariance of the Gaussian equivalent to an oriented box, built straight from
    the doubled angle. Using (sin 2t, cos 2t) rather than t avoids ever halving
    the angle (atan2 then /2), which is non-differentiable at the wrap point:
        cos^2 t = (1 + cos2t)/2,  sin^2 t = (1 - cos2t)/2,  cos t sin t = sin2t/2
    Returns the three distinct entries of the symmetric 2x2 covariance.
    """
    ww = w * w / 4.0
    hh = h * h / 4.0
    a11 = ww * (1 + cos2a) / 2 + hh * (1 - cos2a) / 2
    a22 = ww * (1 - cos2a) / 2 + hh * (1 + cos2a) / 2
    a12 = (ww - hh) * sin2a / 2
    return a11, a12, a22


def gwd_loss(pred, gt, tau=1.0, eps=1e-7):
    """
    Gaussian Wasserstein Distance between predicted and GT oriented boxes.

    pred / gt : dicts of (P,) tensors with keys cx, cy, w, h, sin2a, cos2a,
                with cx/cy/w/h already normalised by image size.

    Why GWD rather than L1 on (cx, cy, w, h) + a separate angle term: it scores
    the box as a whole, so centre, size and angle trade off against each other
    the way IoU does, and -- crucially for this dataset -- a near-square box has
    a near-isotropic covariance, so its (genuinely ambiguous) angle barely moves
    the loss. 41% of TASLER parcels are near-square, and under the old separate
    angle L1 every one of them contributed a large, essentially unlearnable
    gradient.

    Closed form for 2x2 covariances:
        GWD^2 = |mu1-mu2|^2 + Tr(S1) + Tr(S2)
                - 2 sqrt( Tr(S1 S2) + 2 sqrt(det S1 det S2) )
    """
    a11, a12, a22 = _box_gaussian(pred['w'], pred['h'], pred['sin2a'], pred['cos2a'])
    b11, b12, b22 = _box_gaussian(gt['w'], gt['h'], gt['sin2a'], gt['cos2a'])

    dmu = (pred['cx'] - gt['cx']) ** 2 + (pred['cy'] - gt['cy']) ** 2

    tr_a = a11 + a22
    tr_b = b11 + b22
    tr_ab = a11 * b11 + 2 * a12 * b12 + a22 * b22
    det_a = (a11 * a22 - a12 * a12).clamp(min=eps)
    det_b = (b11 * b22 - b12 * b12).clamp(min=eps)

    inner = (tr_ab + 2 * torch.sqrt(det_a * det_b)).clamp(min=eps)
    gwd2 = (dmu + tr_a + tr_b - 2 * torch.sqrt(inner)).clamp(min=eps)

    # bounded, smooth, 0 at a perfect match and -> 1 as the boxes diverge
    return 1.0 - 1.0 / (tau + torch.log1p(torch.sqrt(gwd2)))


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------
class RTMDetLitModule(pl.LightningModule):

    STRIDES = [8, 16, 32]
    # Per-level object-size gating (max(w_px, h_px) at model resolution), so a
    # GT box is only assigned at the FPN level whose stride matches its scale
    # instead of at all 3 levels independently.
    #
    # Ranges are set from the actual label distribution (median long edge 41 px,
    # p90 119 px): the old (0,64)/(64,128)/(128,inf) split sent 83% of all
    # objects to P3 and starved P4/P5 at 9%/10%, so two thirds of the head
    # capacity trained on almost nothing. (0,48)/(48,96)/(96,inf) gives a
    # 65/24/11 split.
    LEVEL_RANGES = [(0, 48), (48, 96), (96, float('inf'))]

    def __init__(self, lr=1e-3, weight_decay=5e-2, backbone="resnet50", num_head_convs=2,
                 d_model=128, depth_canon_range=(0.4, 28.0), fx_ref=FX_REF,
                 top_k=9, img_size=640, epochs=50,
                 w_cls=1.0, w_gwd=2.0, w_l1=2.0, w_angle=1.0, w_depth=2.0, w_dim=1.0,
                 w_type=1.0, w_fused=0.5, w_plain=0.25,
                 size_priors=SIZE_PRIORS):
        super().__init__()
        self.save_hyperparameters()
        self.model = RTMDetRotated(backbone=backbone, num_head_convs=num_head_convs,
                                   d_model=d_model, depth_canon_range=depth_canon_range,
                                   size_priors=tuple(map(tuple, size_priors)))
        self.top_k = top_k
        self.img_size = img_size

    def forward(self, x):
        return self.model(x)

    # ------------------------------------------------------------------
    def _anchors(self, H, W, stride, device, dtype=torch.float32):
        """Centre pixel coords of every grid cell: (H*W, 2)."""
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype) + 0.5,
            torch.arange(W, device=device, dtype=dtype) + 0.5,
            indexing='ij',
        )
        return torch.stack([gx.flatten(), gy.flatten()], -1) * stride

    def _assign(self, gt_rboxes, anchors):
        """
        Top-k centre-distance assignment, fully vectorised.

        Each GT nominates its `top_k` nearest anchors that lie within
        1.5 * max(w, h) of its centre; each nominated anchor then goes to
        whichever nominating GT is closest. A GT with no anchor inside its
        radius still keeps its single nearest anchor, so tiny objects are never
        dropped.

        This replaces a Python double loop whose per-element `.item()` calls
        forced a GPU sync per GT per level -- ~400 syncs per step at this
        dataset's ~17 objects/image.

        gt_rboxes : (N, 5) = (cx, cy, w, h, angle)
        anchors   : (M, 2) in pixels
        Returns   : (M,) long, -1 = background, 0..N-1 = GT index
        """
        N, M = gt_rboxes.shape[0], anchors.shape[0]
        if N == 0:
            return anchors.new_full((M,), -1, dtype=torch.long)

        d = torch.cdist(anchors, gt_rboxes[:, :2])            # (M, N)
        radius = 1.5 * gt_rboxes[:, 2:4].amax(dim=1)          # (N,)
        ok = d < radius.unsqueeze(0)                          # (M, N)

        # GTs with nothing inside the radius fall back to their nearest anchor
        empty = ~ok.any(dim=0)                                # (N,)
        if empty.any():
            nearest = d.argmin(dim=0)                         # (N,)
            cols = torch.arange(N, device=d.device)
            ok[nearest[empty], cols[empty]] = True

        cand = d.masked_fill(~ok, float('inf'))
        k = min(self.top_k, M)
        topv, topi = cand.topk(k, dim=0, largest=False)       # (k, N)

        is_cand = torch.zeros_like(ok)
        is_cand.scatter_(0, topi, topv.isfinite())

        masked = d.masked_fill(~is_cand, float('inf'))
        best, assign = masked.min(dim=1)                      # (M,), (M,)
        return torch.where(best.isfinite(), assign, torch.full_like(assign, -1))

    # ------------------------------------------------------------------
    def compute_loss(self, level_outs, targets):
        dev = level_outs[0]['cls'].device
        S = float(self.img_size)
        fx_ref = self.hparams.fx_ref

        zero = lambda: torch.zeros((), device=dev)
        l_cls, l_gwd, l_l1, l_ang, l_dep, l_dim, l_fused = (zero() for _ in range(7))
        depth_rel_err = zero()
        dim_rel_err = zero()
        # type BCE is accumulated per class and averaged per class at the end:
        # parcels outnumber units ~22:1 at object level, and a plain mean would
        # let "always parcel" score 96% while being useless on exactly the
        # objects the type head exists for
        type_sum = [zero(), zero()]
        type_cnt = [0, 0]
        type_correct = [0, 0]
        real_unit_cnt = real_unit_correct = 0
        n_pos_total = 0
        # positives that carry metric 3D labels. Real frames (has_3d=False) have
        # only 2D boxes and a type, so the size / depth terms are averaged over
        # these alone - averaging over all positives would quietly shrink the 3D
        # loss every time a batch happens to hold real frames.
        n_pos_3d = 0

        for lvl, (out, stride) in enumerate(zip(level_outs, self.STRIDES)):
            B, _, H, W = out['cls'].shape
            anc = self._anchors(H, W, stride, dev)
            lo, hi = self.LEVEL_RANGES[lvl]

            for b in range(B):
                tgt = targets[b]
                gt_all = tgt['rboxes'].to(dev, non_blocking=True)        # (N,5)
                K = tgt['K'].to(dev, non_blocking=True)
                fx, fy = K[0, 0], K[1, 1]

                # only assign objects whose pixel size matches this level's scale
                long_edge = gt_all[:, 2:4].amax(dim=1)
                keep = (long_edge >= lo) & (long_edge < hi)
                gt = gt_all[keep]

                assign = self._assign(gt, anc)
                cls_flat = out['cls'][b, 0].flatten()
                pos = assign >= 0
                l_cls = l_cls + focal_loss(cls_flat, pos.to(cls_flat.dtype))

                n_p = int(pos.sum())
                if n_p == 0:
                    continue
                n_pos_total += n_p

                gi = assign[pos]
                g = gt[gi]                                               # (P,5)

                def _get(key):
                    return out[key][b, 0].flatten()[pos]

                pcx, pcy, pw, ph = _get('cx'), _get('cy'), _get('w'), _get('h')
                psin, pcos = _get('sin_a'), _get('cos_a')

                # --- GWD on the oriented box (normalised to image size) ---
                gsin, gcos = torch.sin(2 * g[:, 4]), torch.cos(2 * g[:, 4])
                l_gwd = l_gwd + gwd_loss(
                    dict(cx=pcx / S, cy=pcy / S, w=pw / S, h=ph / S, sin2a=psin, cos2a=pcos),
                    dict(cx=g[:, 0] / S, cy=g[:, 1] / S, w=g[:, 2] / S, h=g[:, 3] / S,
                         sin2a=gsin, cos2a=gcos),
                ).sum()

                # --- plain L1 alongside GWD: GWD's gradient is weak when the
                # boxes barely overlap, which is every anchor early in training ---
                l_l1 = l_l1 + (
                    (pcx - g[:, 0]).abs() + (pcy - g[:, 1]).abs()
                ).div(S).sum() + (
                    (pw.clamp(min=1e-3).log() - g[:, 2].clamp(min=1e-3).log()).abs() +
                    (ph.clamp(min=1e-3).log() - g[:, 3].clamp(min=1e-3).log()).abs()
                ).sum()

                # --- angle, weighted by how well-defined it is ---
                # A square top face has no meaningful orientation; elongation
                # (w-h)/(w+h) goes to 0 there and to 1 for a long thin box, so
                # ambiguous targets stop injecting noise into the angle head.
                elong = ((g[:, 2] - g[:, 3]) / (g[:, 2] + g[:, 3]).clamp(min=1e-6))
                l_ang = l_ang + (elong * ((psin - gsin).abs() + (pcos - gcos).abs())).sum()

                # --- 3D: type, metric dims, depth ------------------------------
                sizes = tgt['sizes'].to(dev, non_blocking=True)[keep][gi]   # (P,3) H,W,L
                gt_depth = tgt['depths'][:, 0].to(dev, non_blocking=True)[keep][gi]
                gt_type = tgt['obj_type'].to(dev, non_blocking=True)[keep][gi]  # (P,)

                gt_W = torch.maximum(sizes[:, 1], sizes[:, 2]).clamp(min=1e-3)
                gt_L = torch.minimum(sizes[:, 1], sizes[:, 2]).clamp(min=1e-3)
                gt_H = sizes[:, 0].clamp(min=1e-3)
                gt_dc = (gt_depth * fx_ref / fx).clamp(min=1e-3)

                # type: class-balanced BCE (see type_sum above)
                logit = _get('type_logit')
                bce = F.binary_cross_entropy_with_logits(
                    logit, gt_type.to(logit.dtype), reduction='none')
                with torch.no_grad():
                    pred_unit = logit >= 0
                for c in (0, 1):
                    mc = gt_type == c
                    nc = int(mc.sum())
                    if nc:
                        type_sum[c] = type_sum[c] + bce[mc].sum()
                        type_cnt[c] += nc
                        type_correct[c] += int((pred_unit[mc] == bool(c)).sum())

                if not tgt.get('has_3d', True):
                    # real frame: detection, box, angle and type are trained above;
                    # there is no metric size or depth to supervise
                    real_unit_cnt += int((gt_type == 1).sum())
                    real_unit_correct += int(pred_unit[gt_type == 1].sum())
                    continue
                n_pos_3d += n_p

                # metric dims of the GT type only: the classifier picks the mode,
                # the regressor never has to straddle two of them
                unit = gt_type == 1
                pW = torch.where(unit, _get('W1'), _get('W0'))
                pL = torch.where(unit, _get('L1'), _get('L0'))
                pH = torch.where(unit, _get('H1'), _get('H0'))
                s_dim = _get('log_sig_dim')
                for pd_, gd in ((pW, gt_W), (pL, gt_L), (pH, gt_H)):
                    e = (pd_.log() - gd.log()).abs()
                    l_dim = l_dim + laplace_nll(e, s_dim) + self.hparams.w_plain * e.sum()

                # direct canonical depth
                p_dc = _get('depth_canon').clamp(min=1e-3)
                s_dir = _get('log_sig_dir')
                e_dir = (p_dc.log() - gt_dc.log()).abs()
                l_dep = l_dep + laplace_nll(e_dir, s_dir) + self.hparams.w_plain * e_dir.sum()

                # geometric depth from the (detached) metric size and 2D box. Its
                # value is already fully supervised through the dims and box
                # losses, so it is detached here and this term trains only its
                # uncertainty - re-supervising the dims through depth would
                # re-couple size to scale, which is the defect being removed
                s_geo = _get('log_sig_geo')
                with torch.no_grad():
                    d_geo = geometric_depth(pW, pL, pH, pw, ph, fx, fy)
                    p_dc_geo = (d_geo * fx_ref / fx).clamp(min=1e-3)
                e_geo = (p_dc_geo.log() - gt_dc.log()).abs()
                l_dep = l_dep + laplace_nll(e_geo, s_geo)

                # the fused estimate the user actually gets
                wd, wg = torch.exp(-s_dir), torch.exp(-s_geo)
                log_fused = (wd * p_dc.log() + wg * p_dc_geo.log()) / (wd + wg)
                l_fused = l_fused + (log_fused - gt_dc.log()).abs().sum()

                with torch.no_grad():
                    depth_rel_err = depth_rel_err + (
                        (log_fused.exp() - gt_dc).abs() / gt_dc).sum()
                    dim_rel_err = dim_rel_err + ((pW - gt_W).abs() / gt_W).sum()

        # Normalise every term by the number of positive anchors (RetinaNet /
        # FCOS convention). The previous code used a mean over *all* anchors for
        # cls and a mean-per-image for the rest, so an image with 40 parcels and
        # one with 2 contributed equally and the cls term was scaled by the
        # background count rather than by the signal.
        norm = max(n_pos_total, 1)
        norm3d = max(n_pos_3d, 1)
        l_cls, l_gwd, l_l1 = l_cls / norm, l_gwd / norm, l_l1 / norm
        l_ang = l_ang / norm
        l_dep, l_dim, l_fused = l_dep / norm3d, l_dim / norm3d, l_fused / norm3d

        present = [c for c in (0, 1) if type_cnt[c]]
        l_type = (sum(type_sum[c] / type_cnt[c] for c in present) / len(present)
                  if present else zero())

        hp = self.hparams
        loss = (hp.w_cls * l_cls + hp.w_gwd * l_gwd + hp.w_l1 * l_l1 +
                hp.w_angle * l_ang + hp.w_depth * l_dep + hp.w_dim * l_dim +
                hp.w_type * l_type + hp.w_fused * l_fused)

        logs = dict(
            loss=loss.detach(),
            cls=l_cls.detach(),
            gwd=l_gwd.detach(),
            l1=l_l1.detach(),
            angle=l_ang.detach(),
            depth=l_dep.detach(),
            dims=l_dim.detach(),
            type=l_type.detach(),
            fused=l_fused.detach(),
            depth_rel=(depth_rel_err / norm3d).detach(),
            W_rel=(dim_rel_err / norm3d).detach(),
            acc_parcel=torch.tensor(type_correct[0] / max(type_cnt[0], 1), device=dev),
            acc_unit=torch.tensor(type_correct[1] / max(type_cnt[1], 1), device=dev),
            acc_unit_real=torch.tensor(real_unit_correct / max(real_unit_cnt, 1), device=dev),
            n_pos=torch.tensor(float(n_pos_total), device=dev),
        )
        return loss, logs

    # ------------------------------------------------------------------
    def training_step(self, batch, _):
        imgs, targets = batch
        loss, logs = self.compute_loss(self(imgs), targets)
        for k, v in logs.items():
            self.log(f"train/{k}", v, prog_bar=(k == "loss"), batch_size=imgs.size(0))
        return loss

    def validation_step(self, batch, _):
        imgs, targets = batch
        loss, logs = self.compute_loss(self(imgs), targets)
        for k, v in logs.items():
            self.log(f"val/{k}", v, prog_bar=(k in ("loss", "depth_rel")),
                     batch_size=imgs.size(0))
        return loss

    def configure_optimizers(self):
        # No weight decay on norms/biases — decaying BatchNorm scales fights the
        # head's calibrated cls bias and costs accuracy for nothing.
        decay, no_decay, backbone = [], [], []
        bb_ids = {id(p) for p in self.model.backbone.parameters()}
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in bb_ids:
                backbone.append(p)
            elif p.ndim <= 1:
                no_decay.append(p)
            else:
                decay.append(p)

        wd = self.hparams.weight_decay
        lr = self.hparams.lr
        opt = torch.optim.AdamW([
            {"params": backbone, "lr": lr * 0.1, "weight_decay": wd},
            {"params": decay, "lr": lr, "weight_decay": wd},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ])

        warmup = max(1, min(5, self.hparams.epochs // 10))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            [torch.optim.lr_scheduler.LinearLR(opt, 0.05, 1.0, total_iters=warmup),
             torch.optim.lr_scheduler.CosineAnnealingLR(
                 opt, T_max=max(1, self.hparams.epochs - warmup), eta_min=lr * 0.01)],
            milestones=[warmup],
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
