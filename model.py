import torch
from torch import nn
import torchvision.models as tvm


class GrayscaleBackbone(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        w = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
        resnet.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        resnet.conv1.weight.data = w
        self.body = nn.Sequential(*list(resnet.children())[:-2])
        self.proj = nn.Conv2d(2048, out_channels, 1)

    def forward(self, x):
        return self.proj(self.body(x))


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_h=40, max_w=40):
        super().__init__()
        self.row = nn.Embedding(max_h, d_model // 2)
        self.col = nn.Embedding(max_w, d_model // 2)

    def forward(self, x):
        B, C, H, W = x.shape
        i = torch.arange(H, device=x.device)
        j = torch.arange(W, device=x.device)
        pe = torch.cat([
            self.row(i)[:, None, :].expand(H, W, -1),
            self.col(j)[None, :, :].expand(H, W, -1),
        ], dim=-1)
        return pe.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)


class MLP(nn.Module):
    def __init__(self, in_dim, hid, out_dim, n_layers):
        super().__init__()
        dims = [in_dim] + [hid] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class DETR3D(nn.Module):
    """
    Predicts geometric primitives that get LIFTED to 3D using camera intrinsics.

    Outputs per query:
      - logits       : (B, Q, 2)   bg / parcel
      - uv_norm      : (B, Q, 2)   center pixel in [0,1] (relative to image)
      - depth        : (B, Q, 1)   metric depth (camera +z forward)
      - log_size     : (B, Q, 3)   log of (w,h,d) → predicted size = size_prior * exp(log_size)
      - rot6d        : (B, Q, 6)   continuous 6D rotation (Zhou et al. 2019)
    """

    def __init__(self, num_queries=10, d_model=256, nhead=8, num_layers=6,
                 size_prior=(0.25, 0.25, 0.15),
                 depth_range=(0.5, 3.0)):
        super().__init__()
        self.backbone = GrayscaleBackbone(d_model)
        self.pos_enc = PositionalEncoding2D(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=1024, dropout=0.1, batch_first=True,
        )
        self.query_embed = nn.Embedding(num_queries, d_model)

        self.class_head = nn.Linear(d_model, 2)
        self.uv_head    = MLP(d_model, d_model, 2, 3)   # sigmoid -> normalized pixel
        self.depth_head = MLP(d_model, d_model, 1, 3)   # depth (m)
        self.size_head  = MLP(d_model, d_model, 3, 3)   # log residual
        self.rot_head   = MLP(d_model, d_model, 6, 3)   # 6D rotation

        self.register_buffer("size_prior", torch.tensor(size_prior))
        self.depth_min, self.depth_max = depth_range
        self.num_queries = num_queries

    def forward(self, images):
        feat = self.backbone(images)
        pe = self.pos_enc(feat)
        src = (feat + pe).flatten(2).transpose(1, 2)
        B = src.size(0)
        tgt = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        hs = self.transformer(src, tgt)

        uv_norm = self.uv_head(hs).sigmoid()                     # [0,1]
        depth_raw = self.depth_head(hs)
        # bounded depth via sigmoid range
        depth = self.depth_min + (self.depth_max - self.depth_min) * depth_raw.sigmoid()

        log_size = self.size_head(hs)                            # residual in log space
        size = self.size_prior * torch.exp(log_size.clamp(-1.5, 1.5))

        return {
            "pred_logits": self.class_head(hs),
            "pred_uv":     uv_norm,
            "pred_depth":  depth,
            "pred_size":   size,
            "pred_rot6d":  self.rot_head(hs),
        }