"""
ViT-S per-frame encoder + Temporal Transformer for video deepfake detection.

Architecture
============
1. FrameEncoder  – ViT-S/16 (timm, ImageNet-pretrained)
                   Encodes every frame independently → [B*T, D]   (D = 384)
2. TemporalTransformer – small nn.TransformerEncoder
                   Operates over the sequence of frame embeddings → [B, D]
3. ClassificationHead – Linear(D, num_classes)

Expected input shape: [B, T, C, H, W]  where H = W = 224.
Output shape:         [B, num_classes]
"""

import math
import torch
import torch.nn as nn
import timm


class FrameEncoder(nn.Module):
    """
    Wraps a timm ViT-S/16 to act as a per-frame feature extractor.
    The classification head is removed; forward_features() returns the
    CLS token embedding of shape [B, D].
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=0,          # removes classifier head
        )
        self.embed_dim = self.vit.embed_dim  # 384

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, C, H, W] → [N, D]"""
        return self.vit(x)


class TemporalTransformer(nn.Module):
    """
    Small TransformerEncoder that models temporal relations among frame
    embeddings.  Uses a learnable [CLS]_time token prepended to the
    sequence and learnable temporal positional embeddings.
    """

    def __init__(
        self,
        d_model: int = 384,
        nhead: int = 6,
        dim_feedforward: int = 1536,
        num_layers: int = 4,
        dropout: float = 0.1,
        t_max: int = 64,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        self.t_max = t_max

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.pos_embed = nn.Parameter(torch.zeros(1, t_max + 1, d_model))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, t_max, d_model))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:            [B, T, D]
        padding_mask: [B, T]  bool — True for padded positions (optional).
        Returns:      [B, D]
        """
        B, T, D = x.shape

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)       # [B, 1, D]
            x = torch.cat([cls, x], dim=1)                # [B, T+1, D]
            x = x + self.pos_embed[:, : T + 1, :]

            if padding_mask is not None:
                cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        else:
            x = x + self.pos_embed[:, :T, :]

        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)

        if self.use_cls_token:
            return x[:, 0]                                # CLS token
        else:
            if padding_mask is not None:
                valid = (~padding_mask).unsqueeze(-1).float()  # [B, T, 1]
                return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            return x.mean(dim=1)


class ViTS_TemporalTransformer(nn.Module):
    """
    Full video-level deepfake detector:
        FrameEncoder (ViT-S) → TemporalTransformer → Linear head
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        temporal_layers: int = 4,
        temporal_heads: int = 6,
        temporal_ff: int = 1536,
        temporal_dropout: float = 0.1,
        t_max: int = 64,
        use_cls_token: bool = True,
    ):
        super().__init__()

        self.frame_encoder = FrameEncoder(pretrained=pretrained)
        d = self.frame_encoder.embed_dim                  # 384

        self.temporal = TemporalTransformer(
            d_model=d,
            nhead=temporal_heads,
            dim_feedforward=temporal_ff,
            num_layers=temporal_layers,
            dropout=temporal_dropout,
            t_max=t_max,
            use_cls_token=use_cls_token,
        )

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(d, num_classes),
        )

    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B*T, C, H, W] → [B*T, D]"""
        return self.frame_encoder(x)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:            [B, T, C, H, W]
        padding_mask: [B, T]  bool  (optional)
        Returns:      [B, num_classes]
        """
        B, T, C, H, W = x.shape

        # 1. Per-frame encoding
        x = x.reshape(B * T, C, H, W)
        frame_feats = self.encode_frames(x)               # [B*T, D]
        frame_feats = frame_feats.reshape(B, T, -1)       # [B, T, D]

        # 2. Temporal modelling
        pooled = self.temporal(frame_feats, padding_mask)  # [B, D]

        # 3. Classification
        logits = self.head(pooled)                         # [B, num_classes]
        return logits
