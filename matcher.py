import torch
from torch import nn
from scipy.optimize import linear_sum_assignment


class HungarianMatcher3D(nn.Module):
    """
    Hungarian matcher for the new DETR3D model whose heads output:
        pred_logits  (B, Q, 2)
        pred_uv      (B, Q, 2)   normalised pixel coords [0,1]
        pred_depth   (B, Q, 1)   metric depth
        pred_size    (B, Q, 3)   metric size (w, h, d)
        pred_rot6d   (B, Q, 6)   continuous 6-D rotation

    Targets per sample must contain:
        uv           (N, 2)   normalised pixel coords of GT centres
        depths       (N,)     metric depth of GT centres
        sizes        (N, 3)   metric size
        rot6d        (N, 6)   6-D rotation (converted from GT rotation matrix)
    """

    def __init__(
        self,
        cost_class:  float = 1.0,
        cost_uv:     float = 5.0,
        cost_depth:  float = 3.0,
        cost_size:   float = 2.0,
        cost_rot:    float = 1.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_uv    = cost_uv
        self.cost_depth = cost_depth
        self.cost_size  = cost_size
        self.cost_rot   = cost_rot

    # ------------------------------------------------------------------
    @staticmethod
    def _rot6d_to_ortho6d(r: torch.Tensor) -> torch.Tensor:
        """
        Normalise the first two columns of the implied rotation so that
        the L2 distance between 6-D vectors is a meaningful rotation cost.

        r : (..., 6)
        """
        a1 = r[..., :3]
        a2 = r[..., 3:6]
        b1 = torch.nn.functional.normalize(a1, dim=-1)
        b2 = torch.nn.functional.normalize(
            a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1
        )
        return torch.cat([b1, b2], dim=-1)   # (..., 6)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, outputs: dict, targets: list) -> list:
        """
        Returns a list of (row_indices, col_indices) tuples, one per batch item.
        row_indices → matched query indices
        col_indices → matched target indices
        """
        B, Q = outputs["pred_logits"].shape[:2]
        indices = []

        for b in range(B):
            tgt = targets[b]
            n   = tgt["uv_norm"].shape[0]

            if n == 0:
                indices.append((
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.long),
                ))
                continue

            # ---- classification cost  (Q, N) -------------------------
            prob        = outputs["pred_logits"][b].softmax(-1)   # (Q, 2)
            cost_class  = -prob[:, 1].unsqueeze(1).expand(Q, n)   # (Q, N)

            # ---- UV cost  (Q, N) -------------------------------------
            pred_uv  = outputs["pred_uv"][b]                      # (Q, 2)
            tgt_uv   = tgt["uv_norm"].to(pred_uv)                 # (N, 2)
            cost_uv  = torch.cdist(pred_uv, tgt_uv, p=1)          # (Q, N)

            # ---- depth cost  (Q, N) ----------------------------------
            pred_depth = outputs["pred_depth"][b]                  # (Q, 1)
            tgt_depth  = tgt["depths"].to(pred_depth)              # (N, 1)
            cost_depth = torch.cdist(pred_depth, tgt_depth, p=1)   # (Q, N)

            # ---- size cost  (Q, N) -----------------------------------
            pred_size = outputs["pred_size"][b]                    # (Q, 3)
            tgt_size  = tgt["sizes"].to(pred_size)                 # (N, 3)
            cost_size = torch.cdist(pred_size, tgt_size, p=1)      # (Q, N)

            # ---- rotation cost  (Q, N) -------------------------------
            # Normalise both sides to proper ortho-6D before L2 distance
            pred_rot6d = self._rot6d_to_ortho6d(outputs["pred_rot6d"][b])  # (Q, 6)
            tgt_rot6d  = self._rot6d_to_ortho6d(tgt["rot6d"].to(pred_rot6d))  # (N, 6)
            cost_rot   = torch.cdist(pred_rot6d, tgt_rot6d, p=2)   # (Q, N)

            # ---- combined cost matrix  (Q, N) ------------------------
            C = (
                self.cost_class * cost_class +
                self.cost_uv    * cost_uv    +
                self.cost_depth * cost_depth +
                self.cost_size  * cost_size  +
                self.cost_rot   * cost_rot
            ).cpu()

            # ---- Hungarian assignment ---------------------------------
            row, col = linear_sum_assignment(C.numpy())
            indices.append((
                torch.as_tensor(row, dtype=torch.long),
                torch.as_tensor(col, dtype=torch.long),
            ))

        return indices