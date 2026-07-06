from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from timee.model.layers import (
    ClassEmbedding,
    CLSTokens,
    DecoderMLP,
    InstanceNorm,
    Patch,
    PatchEncoder,
    TransformerEncoderLayer,
)


class TimeeModel(nn.Module):
    """TIMEE: in-context learning (ICL) model for time series classification.

    Architecture:
      1. Patch the input time series (non-overlapping patches).
      2. Normalize each series (arcsinh instance norm) and across series.
      3. Embed patches linearly; add patch-level and series-level stats embeddings.
      4. Inject class embeddings into training-set patches.
      5. Run n_horizontal_layers of temporal self-attention (RoPE), followed by
         n_vertical_layers of cross-series attention.
      6. Compress each series to a fixed-size representation (num_cls_tokens CLS
         tokens) via a final attention layer.
      7. Run n_icl_layers of ICL attention over the compressed representations.
      8. Decode test-set representations to class logits via an MLP.

    Default arguments match the released checkpoint.
    """

    def __init__(
        self,
        n_max_classes: int = 10,
        patch_size: int = 16,
        patch_stride: int = 16,
        d_model: int = 128,
        n_horizontal_layers: int = 5,
        n_vertical_layers: int = 5,
        n_icl_layers: int = 2,
        n_heads: int = 4,
        d_kv: int = 32,
        d_kv_icl: int = 64,
        mlp_hidden_dim: int = 512,
        decoder_mlp_hidden_dim: int = 512,
        dropout_rate: float = 0.1,
        num_cls_tokens: int = 4,
    ) -> None:
        super().__init__()

        self.n_max_classes = n_max_classes
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.n_horizontal_layers = n_horizontal_layers
        self.n_vertical_layers = n_vertical_layers
        self.n_icl_layers = n_icl_layers
        self.n_heads = n_heads
        self.d_kv = d_kv
        self.d_kv_icl = d_kv_icl
        self.mlp_hidden_dim = mlp_hidden_dim
        self.decoder_mlp_hidden_dim = decoder_mlp_hidden_dim
        self.dropout_rate = dropout_rate
        self.num_cls_tokens = num_cls_tokens
        self.icl_dim = d_model * num_cls_tokens

        self.instance_norm_sequence = InstanceNorm(use_arcsinh=True)
        self.instance_norm_series = InstanceNorm()

        self.patch = Patch(patch_size=patch_size, patch_stride=patch_stride)
        self.patch_encoder = PatchEncoder(patch_size=patch_size, d_model=d_model)

        self.patch_stats_projection = nn.Linear(2, d_model, bias=False)
        self.series_stats_projection = nn.Linear(2, d_model, bias=False)

        self.cls_tokens = CLSTokens(num_cls_tokens, d_model)

        layer_kwargs = dict(
            d_model=d_model,
            n_heads=n_heads,
            d_kv=d_kv,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout_rate,
            activation="gelu",
        )

        self.encoder_horizontal = nn.ModuleList(
            [
                TransformerEncoderLayer(**layer_kwargs, use_rope=True)
                for _ in range(n_horizontal_layers)
            ]
        )
        self.encoder_vertical = nn.ModuleList(
            [
                TransformerEncoderLayer(**layer_kwargs, use_rope=False)
                for _ in range(n_vertical_layers)
            ]
        )
        self.last_horizontal_encoder = TransformerEncoderLayer(**layer_kwargs, use_rope=True)
        self.cls_tokens_ln = nn.LayerNorm(d_model)

        self.class_embedding_for_patches = ClassEmbedding(
            num_embeddings=n_max_classes,
            embedding_dim=d_model,
        )
        self.class_embedding_icl = ClassEmbedding(
            num_embeddings=n_max_classes,
            embedding_dim=self.icl_dim,
        )

        self.icl_block = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=self.icl_dim,
                    n_heads=n_heads,
                    d_kv=d_kv_icl,
                    mlp_hidden_dim=mlp_hidden_dim,
                    dropout=dropout_rate,
                    activation="gelu",
                    use_rope=False,
                )
                for _ in range(n_icl_layers)
            ]
        )

        self.decoder_mlp_ln = nn.LayerNorm(self.icl_dim)
        self.decoder_mlp = DecoderMLP(
            d_model=self.icl_dim,
            hidden_dim=decoder_mlp_hidden_dim,
            output_dim=n_max_classes,
            activation="gelu",
            dropout=dropout_rate,
        )

    def _compute_stats_embeddings(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scale-aware stat embeddings from raw (un-normalized) x (B, N, S).

        Returns (patch_stats, series_stats):
          patch_stats  (B, N, P, 2): arcsinh(mean, std) per patch, z-scored within each series.
          series_stats (B, N, 1, 2): arcsinh(mean, std) per series, z-scored within the episode.
        """
        x_BNPQ = self.patch(x)
        raw_mean = torch.nan_to_num(torch.nanmean(x_BNPQ, dim=-1, keepdim=True), nan=0.0)
        raw_std = ((x_BNPQ - raw_mean).nan_to_num(0.0) ** 2).mean(dim=-1, keepdim=True).sqrt()
        patch_stats = torch.arcsinh(torch.cat([raw_mean, raw_std], dim=-1))
        p_mean = patch_stats.mean(dim=2, keepdim=True)
        p_std = patch_stats.std(dim=2, correction=0, keepdim=True).clamp(min=1e-5)
        patch_stats = (patch_stats - p_mean) / p_std

        series_mean = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
        series_std = ((x - series_mean).nan_to_num(0.0) ** 2).mean(dim=-1, keepdim=True).sqrt()
        series_stats = torch.arcsinh(torch.cat([series_mean, series_std], dim=-1))
        ep_mean = series_stats.mean(dim=1, keepdim=True)
        ep_std = series_stats.std(dim=1, keepdim=True).clamp(min=1e-5)
        series_stats = ((series_stats - ep_mean) / ep_std).unsqueeze(2)

        return patch_stats, series_stats

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        eval_pos: int,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[float], torch.Tensor]:
        """
        Args:
            x:        (B, N, S) float — N series per episode (support first, then queries).
            y:        (B, N) int — class labels; query positions (>= eval_pos) are ignored.
            eval_pos: number of support series; x[:, :eval_pos] are labeled,
                      x[:, eval_pos:] are the queries to classify.
            labels:   optional ground-truth for computing cross-entropy loss on queries.
        Returns:
            (loss, logits): loss is None when labels is None;
                            logits shape is (B, N − eval_pos, n_max_classes).
        """
        x_BNS = x
        y_BN = y
        n_train = eval_pos
        batch_size, num_series, _ = x_BNS.shape

        patch_stats_BNPC, series_stats_BN12 = self._compute_stats_embeddings(x_BNS)

        # Normalize: instance norm per series, then cross-series norm.
        x_BNS, _ = self.instance_norm_sequence(x_BNS)
        x_BSN = rearrange(x_BNS, "b n s -> b s n")
        x_BSN, _ = self.instance_norm_series(x_BSN, eval_pos=n_train)
        x_BNS = rearrange(x_BSN, "b s n -> b n s")

        x_BNPQ = self.patch(x_BNS)
        x_BNPQ = torch.nan_to_num(x_BNPQ, nan=0.0)
        x_BNPD = self.patch_encoder(x_BNPQ)

        x_BNPD = x_BNPD + self.patch_stats_projection(patch_stats_BNPC)
        x_BNPD = x_BNPD + self.series_stats_projection(
            series_stats_BN12.expand(-1, -1, x_BNPD.shape[2], -1)
        )

        # Inject class embeddings into train patches.
        class_embed_BN1D = self.class_embedding_for_patches(y_BN[:, :n_train], n_train).unsqueeze(2)
        x_BNPD = torch.cat([x_BNPD[:, :n_train] + class_embed_BN1D, x_BNPD[:, n_train:]], dim=1)

        x_BNPD = torch.cat([self.cls_tokens(batch_size, num_series), x_BNPD], dim=2)

        num_patches_plus_cls = x_BNPD.shape[2]
        num_patches = num_patches_plus_cls - self.num_cls_tokens

        # Mask convention: True = attend, False = blocked.
        # Horizontal mask is fully open — each series attends to all its own patches.
        horizontal_mask = torch.ones(
            (batch_size * num_series, num_patches_plus_cls, num_patches_plus_cls),
            device=x_BNPD.device,
            dtype=torch.bool,
        )
        # Queries cannot attend to other queries; they can only see support series (and themselves).
        vertical_mask = torch.ones(
            (batch_size * num_patches_plus_cls, num_series, num_series),
            device=x_BNPD.device,
            dtype=torch.bool,
        )
        vertical_mask[:, :, n_train:] = False
        vertical_mask |= torch.eye(num_series, device=x_BNPD.device, dtype=torch.bool)

        position_ids = (
            torch.arange(num_patches, device=x_BNPD.device)
            .unsqueeze(0)
            .repeat(batch_size * num_series, 1)
        )

        for layer in self.encoder_horizontal:
            x_BNPD = rearrange(x_BNPD, "b n p d -> (b n) p d")
            x_BNPD = layer(x=x_BNPD, attention_mask=horizontal_mask, position_ids=position_ids)
            x_BNPD = rearrange(x_BNPD, "(b n) p d -> b n p d", b=batch_size, n=num_series)

        for layer in self.encoder_vertical:
            x_BNPD = rearrange(x_BNPD, "b n p d -> (b p) n d")
            x_BNPD = layer(x=x_BNPD, attention_mask=vertical_mask, position_ids=None)
            x_BNPD = rearrange(x_BNPD, "(b p) n d -> b n p d", b=batch_size, p=num_patches_plus_cls)

        # No position attends to CLS tokens as keys, so CLS tokens aggregate only from patches.
        compression_mask = torch.ones(
            (batch_size * num_series, num_patches_plus_cls, num_patches_plus_cls),
            device=x_BNPD.device,
            dtype=torch.bool,
        )
        compression_mask[:, :, : self.num_cls_tokens] = False

        x_BNPD = rearrange(x_BNPD, "b n p d -> (b n) p d")
        x_BNPD = self.last_horizontal_encoder(
            x=x_BNPD,
            attention_mask=compression_mask,
            position_ids=position_ids,
        )
        x_BNPD = rearrange(x_BNPD, "(b n) p d -> b n p d", b=batch_size, n=num_series)

        row_repr_BNDc = self.cls_tokens_ln(x_BNPD[:, :, : self.num_cls_tokens]).flatten(-2, -1)

        # ICL: cross-series attention over compressed representations.
        # Same masking policy as vertical layers — queries attend to support + self only.
        icl_mask = torch.ones(
            (batch_size, num_series, num_series),
            device=x_BNPD.device,
            dtype=torch.bool,
        )
        icl_mask[:, :, n_train:] = False
        icl_mask |= torch.eye(num_series, device=x_BNPD.device, dtype=torch.bool)

        class_embed_icl = self.class_embedding_icl(y_BN[:, :n_train], n_train)
        row_repr_BNDc = torch.cat(
            [row_repr_BNDc[:, :n_train] + class_embed_icl, row_repr_BNDc[:, n_train:]], dim=1
        )

        for layer in self.icl_block:
            row_repr_BNDc = layer(x=row_repr_BNDc, attention_mask=icl_mask, position_ids=None)

        logits = self.decoder_mlp(self.decoder_mlp_ln(row_repr_BNDc[:, n_train:]))

        loss = None
        if labels is not None:
            labels_test = labels.to(torch.long)[:, n_train:]
            loss = nn.CrossEntropyLoss()(
                rearrange(logits, "b n c -> (b n) c"),
                rearrange(labels_test, "b n -> (b n)"),
            )

        return loss, logits
