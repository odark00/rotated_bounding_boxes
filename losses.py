"""
RTMDet-Rotated training module.

Assignment  : top-k (k=9) anchor points per GT by centre distance per FPN level.
Losses      : Focal (cls)  +  L1 (box, angle, depth, height3d)
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from model import RTMDetRotated


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Binary focal loss. logits/targets: (N,) floats."""
    p   = logits.sigmoid()
    ce  = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t) ** gamma * ce).mean()


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------
class RTMDetLitModule(pl.LightningModule):

    STRIDES = [8, 16, 32]

    def __init__(self, lr=1e-4, weight_decay=1e-4,
                 depth_range=(1.2, 1.5), height3d_range=(0.04, 0.30),
                 top_k=9, img_size=640):
        super().__init__()
        self.save_hyperparameters()
        self.model    = RTMDetRotated(depth_range=depth_range, height3d_range=height3d_range)
        self.top_k    = top_k
        self.img_size = img_size

    def forward(self, x):
        return self.model(x)

    # ------------------------------------------------------------------
    def _anchors(self, H, W, stride, device):
        """Centre pixel coords of every grid cell: (H*W, 2)."""
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32) + 0.5,
            torch.arange(W, device=device, dtype=torch.float32) + 0.5,
            indexing='ij',
        )
        return torch.stack([gx.flatten(), gy.flatten()], -1) * stride

    def _assign(self, gt_rboxes, anchors):
        """
        Top-k centre-distance assignment.

        gt_rboxes : (N, 5) = (cx, cy, w, h, angle)
        anchors   : (M, 2) in pixels
        Returns   : (M,) long, -1 = background, 0..N-1 = GT index
        """
        N = gt_rboxes.shape[0]
        M = anchors.shape[0]
        out = anchors.new_full((M,), -1, dtype=torch.long)

        if N == 0:
            return out

        dists = torch.cdist(anchors, gt_rboxes[:, :2])   # (M, N)

        for n in range(N):
            d  = dists[:, n]
            r  = 1.5 * max(gt_rboxes[n, 2].item(), gt_rboxes[n, 3].item())
            ok = d < r
            if ok.sum() == 0:
                ok = (d == d.min())

            cand = d.clone().masked_fill(~ok, float('inf'))
            k    = min(self.top_k, ok.sum().item())
            _, idx = cand.topk(k, largest=False)

            for i in idx:
                i = i.item()
                if out[i] < 0 or d[i] < dists[i, out[i]]:
                    out[i] = n

        return out

    # ------------------------------------------------------------------
    def compute_loss(self, level_outs, targets):
        dev = level_outs[0]['cls'].device
        S   = float(self.img_size)

        loss_cls   = torch.zeros(1, device=dev)
        loss_box   = torch.zeros(1, device=dev)
        loss_angle = torch.zeros(1, device=dev)
        loss_depth = torch.zeros(1, device=dev)
        loss_hgt   = torch.zeros(1, device=dev)
        n_pos      = 0

        for out, stride in zip(level_outs, self.STRIDES):
            B, _, H, W = out['cls'].shape
            anc = self._anchors(H, W, stride, dev)   # (M, 2)

            for b in range(B):
                gt = targets[b]['rboxes'].to(dev)    # (N, 5)
                assign = self._assign(gt, anc)       # (M,)

                # --- cls loss (all anchors) ---
                cls_flat = out['cls'][b, 0].flatten()          # (M,)
                cls_tgt  = (assign >= 0).float()
                loss_cls = loss_cls + focal_loss(cls_flat, cls_tgt)

                pos = assign >= 0
                if pos.sum() == 0:
                    continue

                n_pos   += pos.sum().item()
                gi       = assign[pos]               # matched GT indices (P,)
                gt_pos   = gt[gi]                    # (P, 5)

                # predicted values at positive anchors
                def _get(key):
                    return out[key][b, 0].flatten()[pos]   # (P,)

                pcx, pcy, pw, ph = _get('cx'), _get('cy'), _get('w'), _get('h')
                psin, pcos       = _get('sin_a'), _get('cos_a')
                pdep, phgt       = _get('depth'), _get('height3d')

                # box L1 (normalised by image size)
                loss_box = loss_box + (
                    F.l1_loss(pcx / S, gt_pos[:, 0] / S) +
                    F.l1_loss(pcy / S, gt_pos[:, 1] / S) +
                    F.l1_loss(pw  / S, gt_pos[:, 2] / S) +
                    F.l1_loss(ph  / S, gt_pos[:, 3] / S)
                )

                # angle L1 on (sin, cos)
                gsin = torch.sin(gt_pos[:, 4])
                gcos = torch.cos(gt_pos[:, 4])
                loss_angle = loss_angle + F.l1_loss(psin, gsin) + F.l1_loss(pcos, gcos)

                # depth & 3D height  (sizes[:,0] = d01 = box height)
                gt_dep = targets[b]['depths'][gi, 0].to(dev)    # (P,)
                gt_hgt = targets[b]['sizes'][gi, 0].to(dev)     # (P,) d01 = vertical height
                loss_depth = loss_depth + F.l1_loss(pdep, gt_dep)
                loss_hgt   = loss_hgt   + F.l1_loss(phgt, gt_hgt)

        # average over batch × levels (focal loss is already mean-reduced)
        n_lvl = len(self.STRIDES)
        B     = level_outs[0]['cls'].shape[0]
        norm  = float(B * n_lvl)

        loss_cls   = loss_cls   / norm
        loss_box   = loss_box   / norm
        loss_angle = loss_angle / norm
        loss_depth = loss_depth / norm
        loss_hgt   = loss_hgt   / norm

        loss = (1.0 * loss_cls   +
                5.0 * loss_box   +
                2.0 * loss_angle +
                3.0 * loss_depth +
                2.0 * loss_hgt)

        logs = dict(
            loss=loss.detach().squeeze(),
            cls =loss_cls.detach().squeeze(),
            box =loss_box.detach().squeeze(),
            angle=loss_angle.detach().squeeze(),
            depth=loss_depth.detach().squeeze(),
            height=loss_hgt.detach().squeeze(),
        )
        return loss, logs

    # ------------------------------------------------------------------
    def training_step(self, batch, _):
        imgs, targets = batch
        loss, logs = self.compute_loss(self.model(imgs), targets)
        for k, v in logs.items():
            self.log(f"train/{k}", v, prog_bar=(k == "loss"), batch_size=imgs.size(0))
        return loss

    def validation_step(self, batch, _):
        imgs, targets = batch
        loss, logs = self.compute_loss(self.model(imgs), targets)
        for k, v in logs.items():
            self.log(f"val/{k}", v, prog_bar=(k == "loss"), batch_size=imgs.size(0))
        return loss

    def configure_optimizers(self):
        backbone_params = list(self.model.backbone.parameters())
        bids = {id(p) for p in backbone_params}
        other = [p for p in self.model.parameters() if id(p) not in bids]

        opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.hparams.lr * 0.1},
            {"params": other,           "lr": self.hparams.lr},
        ], weight_decay=self.hparams.weight_decay)

        warmup = torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, total_iters=10)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=190)
        sched  = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[10])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
