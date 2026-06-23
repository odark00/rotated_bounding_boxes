import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from model import DETR3D
from matcher import HungarianMatcher3D
from geometry import lift_uv_depth_to_camera, build_corners_from_geometry


def _ortho6d(r):
    """Normalise 6D rotation to orthogonal representation for loss."""
    a1 = F.normalize(r[..., :3], dim=-1)
    a2 = r[..., 3:6]
    a2 = F.normalize(a2 - (a1 * a2).sum(-1, keepdim=True) * a1, dim=-1)
    return torch.cat([a1, a2], dim=-1)


class Parcel3DLitModule(pl.LightningModule):
    def __init__(self, num_queries=10, lr=1e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = DETR3D(num_queries=num_queries)
        self.matcher = HungarianMatcher3D()
        self.class_weight = torch.tensor([0.1, 1.0])

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, outputs, targets):
        indices = self.matcher(outputs, targets)

        # --- Classification loss ---
        pred_logits = outputs["pred_logits"]   # (B, Q, 2)
        B, Q, _ = pred_logits.shape

        target_classes = torch.zeros(B, Q, dtype=torch.long, device=pred_logits.device)
        for b, (qi, ti) in enumerate(indices):
            target_classes[b, qi] = 1

        loss_ce = F.cross_entropy(
            pred_logits.reshape(-1, 2),
            target_classes.reshape(-1),
            weight=self.class_weight.to(pred_logits.device),
        )

        # --- Regression losses on matched queries ---
        loss_center = pred_logits.new_zeros(())
        loss_size   = pred_logits.new_zeros(())
        loss_rot    = pred_logits.new_zeros(())
        loss_corner = pred_logits.new_zeros(())
        n_total = 0

        for b, (qi, ti) in enumerate(indices):
            if qi.numel() == 0:
                continue

            dev = pred_logits.device
            K        = targets[b]["K"].to(dev)                      # (3,3)
            img_size = targets[b]["img_size"].float().to(dev)        # [W,H]

            pred_uv_norm = outputs["pred_uv"][b, qi]                # (n,2)
            pred_depth   = outputs["pred_depth"][b, qi]             # (n,1)
            pred_size    = outputs["pred_size"][b, qi]              # (n,3)
            pred_rot6d   = outputs["pred_rot6d"][b, qi]             # (n,6)

            # UV [0,1] → pixel coords at model resolution
            pred_uv_px = pred_uv_norm * img_size                    # (n,2)

            tc       = targets[b]["centers"][ti].to(dev)            # (n,3) Blender frame
            ts       = targets[b]["sizes"][ti].to(dev)              # (n,3)
            tr6d     = targets[b]["rot6d"][ti].to(dev)              # (n,6)
            tcorners = targets[b]["boxes_3d"][ti].to(dev)           # (n,8,3)

            # Lift predicted UV+depth to 3D center (Blender camera frame)
            pred_center = lift_uv_depth_to_camera(
                pred_uv_px, pred_depth, K, blender_convention=True
            )                                                        # (n,3)

            loss_center = loss_center + F.l1_loss(pred_center, tc, reduction="sum")
            loss_size   = loss_size   + F.l1_loss(pred_size, ts, reduction="sum")
            loss_rot    = loss_rot    + F.l1_loss(_ortho6d(pred_rot6d), _ortho6d(tr6d), reduction="sum")

            pred_corners = build_corners_from_geometry(
                pred_uv_px, pred_depth, pred_size, pred_rot6d, K,
                blender_convention=True
            )                                                        # (n,8,3)
            loss_corner = loss_corner + F.l1_loss(pred_corners, tcorners, reduction="sum")
            n_total += qi.numel()

        n_total = max(n_total, 1)
        loss_center = loss_center / n_total
        loss_size = loss_size / n_total
        loss_rot = loss_rot / n_total
        loss_corner = loss_corner / (n_total * 8)

        loss = (2.0 * loss_ce +
                5.0 * loss_center +
                2.0 * loss_size +
                1.0 * loss_rot +
                5.0 * loss_corner)

        return loss, {
            "loss": loss.detach(),
            "ce": loss_ce.detach(),
            "center": loss_center.detach(),
            "size": loss_size.detach(),
            "rot": loss_rot.detach(),
            "corner": loss_corner.detach(),
        }

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        out = self.model(imgs)
        loss, logs = self.compute_loss(out, targets)
        for k, v in logs.items():
            self.log(f"train/{k}", v, prog_bar=(k == "loss"), batch_size=imgs.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        out = self.model(imgs)
        loss, logs = self.compute_loss(out, targets)
        for k, v in logs.items():
            self.log(f"val/{k}", v, prog_bar=(k == "loss"), batch_size=imgs.size(0))
        return loss

    def configure_optimizers(self):
        backbone_params = list(self.model.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]

        optim = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.hparams.lr * 0.1},
            {"params": other_params, "lr": self.hparams.lr},
        ], weight_decay=self.hparams.weight_decay)

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100)
        return {"optimizer": optim, "lr_scheduler": sched}