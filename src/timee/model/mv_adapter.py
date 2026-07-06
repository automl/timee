from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from timee.model.model import TimeeModel


class VariateAttentionPool(nn.Module):
    """Slot-wise attention pooling over M variates.

    For each of the C CLS token slots, a learnable query attends over M variate
    embeddings independently, then the C slot outputs are concatenated.

    Zero-init query => uniform softmax => mean pooling at init.
    """

    def __init__(self, num_cls_tokens: int, d_model: int) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.zeros(num_cls_tokens, d_model))
        self.scale = d_model**-0.5

    def forward(self, cls_BNMCD: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("bnmcd,cd->bnmc", cls_BNMCD, self.q) * self.scale
        weights = scores.softmax(dim=2)
        pooled = torch.einsum("bnmc,bnmcd->bncd", weights, cls_BNMCD)
        return pooled.flatten(-2, -1)


class TimeeMultivariateModel(TimeeModel):
    """Multivariate extension of TimeeModel via per-variate encoding + attention pooling.

    Accepts (B, N, S, M) input. Each variate is encoded independently through the
    shared UV encoder. CLS tokens are then pooled across M variates using
    VariateAttentionPool before the ICL phase.

    Variate batching is adaptive: starts by folding all M variates into the batch dim;
    halves the chunk size on CUDA OOM until falling back to serial (m_chunk=1).

    Load a pretrained UV checkpoint with strict=False, then fine-tune
    variate_pool + ICL block + decoder on multivariate data.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.variate_pool = VariateAttentionPool(self.num_cls_tokens, self.d_model)

    def _encode_variate_chunk(
        self,
        x_BNS: torch.Tensor,
        y_BN: torch.Tensor,
        n_train: int,
    ) -> torch.Tensor:
        """Run UV encoder on one batch of series; returns (B, N, num_cls_tokens, d_model)."""
        batch_size, num_series, _ = x_BNS.shape

        patch_stats_BNPC, series_stats_BN12 = self._compute_stats_embeddings(x_BNS)

        # Normalize
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

        class_embed_BN1D = self.class_embedding_for_patches(y_BN[:, :n_train], n_train).unsqueeze(2)
        x_BNPD = torch.cat([x_BNPD[:, :n_train] + class_embed_BN1D, x_BNPD[:, n_train:]], dim=1)

        x_BNPD = torch.cat([self.cls_tokens(batch_size, num_series), x_BNPD], dim=2)

        num_patches_plus_cls = x_BNPD.shape[2]
        num_patches = num_patches_plus_cls - self.num_cls_tokens

        h_mask = torch.ones(
            (batch_size * num_series, num_patches_plus_cls, num_patches_plus_cls),
            device=x_BNPD.device,
            dtype=torch.bool,
        )
        v_mask = torch.ones(
            (batch_size * num_patches_plus_cls, num_series, num_series),
            device=x_BNPD.device,
            dtype=torch.bool,
        )
        v_mask[:, :, n_train:] = False
        v_mask |= torch.eye(num_series, device=x_BNPD.device, dtype=torch.bool)

        position_ids = (
            torch.arange(num_patches, device=x_BNPD.device)
            .unsqueeze(0)
            .repeat(batch_size * num_series, 1)
        )

        for layer in self.encoder_horizontal:
            x_BNPD = rearrange(x_BNPD, "b n p d -> (b n) p d")
            x_BNPD = layer(x=x_BNPD, attention_mask=h_mask, position_ids=position_ids)
            x_BNPD = rearrange(x_BNPD, "(b n) p d -> b n p d", b=batch_size, n=num_series)

        for layer in self.encoder_vertical:
            x_BNPD = rearrange(x_BNPD, "b n p d -> (b p) n d")
            x_BNPD = layer(x=x_BNPD, attention_mask=v_mask, position_ids=None)
            x_BNPD = rearrange(x_BNPD, "(b p) n d -> b n p d", b=batch_size, p=num_patches_plus_cls)

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

        return self.cls_tokens_ln(x_BNPD[:, :, : self.num_cls_tokens])

    def _encode_variates(
        self,
        x_BNSM: torch.Tensor,
        y_BN: torch.Tensor,
        n_train: int,
        m_chunk: int,
    ) -> torch.Tensor:
        """Encode all M variates in chunks of m_chunk; returns (B, N, M, C, D)."""
        B, N, S, M = x_BNSM.shape
        cls_list = []
        for start in range(0, M, m_chunk):
            end = min(start + m_chunk, M)
            mb = end - start
            x_flat = rearrange(x_BNSM[:, :, :, start:end], "b n s m -> (b m) n s")
            y_flat = y_BN.repeat_interleave(mb, dim=0)
            cls_chunk = rearrange(
                self._encode_variate_chunk(x_flat, y_flat, n_train),
                "(b m) n c d -> b n m c d",
                b=B,
                m=mb,
            )
            cls_list.append(cls_chunk)
        return torch.cat(cls_list, dim=2)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        eval_pos: int,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[float], torch.Tensor]:
        if x.ndim == 3:
            x = x.unsqueeze(-1)

        B, N, S, M = x.shape
        n_train = eval_pos

        m_chunk = M
        cls_BNMCD = None
        while m_chunk >= 1:
            try:
                cls_BNMCD = self._encode_variates(x, y, n_train, m_chunk)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                m_chunk //= 2

        if cls_BNMCD is None:
            raise RuntimeError("CUDA OOM even with a single variate per chunk -- reduce batch size")

        row_repr_BNDc = self.variate_pool(cls_BNMCD)

        icl_mask = torch.ones((B, N, N), device=x.device, dtype=torch.bool)
        icl_mask[:, :, n_train:] = False
        icl_mask |= torch.eye(N, device=x.device, dtype=torch.bool)

        class_embed_icl = self.class_embedding_icl(y[:, :n_train], n_train)
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
