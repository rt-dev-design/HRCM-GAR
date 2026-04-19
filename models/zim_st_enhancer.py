import torch
import torch.nn as nn
from typing import Optional


class PositionalEncodingLearned(nn.Module):
    """
    Simple learned positional encoding per sequence length.
    """
    def __init__(self, length: int, dim: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(length, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected either (L, B, D) or (B, L, D) - we'll support (L, B, D)
        L, B, D = x.shape
        return x + self.pe[:L].unsqueeze(1)


class SpatialTemporalEnhancer(nn.Module):
    """
    Spatial-Temporal enhancement module for video features.

    Input: x shaped (T, S, D) or (B, T, S, D).

    Pipeline (sequence-first style for transformer internals):
      - Time Transformer Encoder: processes tokens along time for each (batch, space)
      - Space Transformer Encoder: processes tokens along space for each (batch, time)
      - Spatial Decoder: for each time t, decode spatial tokens (seq len S) using
        the temporal tokens at time t (seq len S) as memory (keys/values).
      - Temporal Decoder: symmetrical: for each space s, decode temporal tokens
        (seq len T) using spatial tokens at space s (seq len T) as memory.
      - Fuse decoded branches by elementwise addition and return same shape as input.

    All Transformer blocks are "sequence-first" style: (seq_len, batch, dim).

    Note: To keep the implementation straight-forward and explicit, the cross-decode
    stages loop over time/space to call the TransformerDecoder. This is easy to reason
    about and works well for moderate T and S. If you need full vectorization for
    large T and S, that can be done but complicates the code.
    """

    def __init__(
        self,
        d_model,
        nhead,
        norm_first,
        dim_feedforward,
        activation,
        dropout,
        num_time_enc_layers,
        num_space_enc_layers,
        num_time_dec_layers,
        num_space_dec_layers,
        use_time_positional,
        max_time,
        use_space_positional,
        max_space,
    ):
        super().__init__()
        self.d_model = d_model

        def make_encoder(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, norm_first=norm_first,
                dim_feedforward=dim_feedforward, dropout=dropout,
                activation=activation, batch_first=False
            )
            return nn.TransformerEncoder(layer, num_layers=num_layers)
        self.time_encoder = make_encoder(num_time_enc_layers)
        self.space_encoder = make_encoder(num_space_enc_layers)

        def make_decoder(num_layers):
            layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, norm_first=norm_first,
                dim_feedforward=dim_feedforward, dropout=dropout, 
                activation=activation, batch_first=False, 
            )
            return nn.TransformerDecoder(layer, num_layers=num_layers)
        self.temporal_decoder = make_decoder(num_time_dec_layers)
        self.spatial_decoder = make_decoder(num_space_dec_layers)

        # Optional learned positional encodings for time and space (sequence-first shapes)
        self.use_time_positional = use_time_positional
        self.use_space_positional = use_space_positional
        if use_time_positional:
            self.time_pos = PositionalEncodingLearned(max_time, d_model)
        if use_space_positional:
            self.space_pos = PositionalEncodingLearned(max_space, d_model)

        # final projection (optional) to refine fused tokens
        # self.output_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor with shape (T, S, D) or (B, T, S, D)
        returns Tensor with same shape as input (B, T, S, D) or (T, S, D) if no batch
        """
        has_batch = (x.dim() == 4)
        if not has_batch:
            # add batch dim
            x = x.unsqueeze(0)  # (1, T, S, D)

        B, T, S, D = x.shape
        assert D == self.d_model, f"input dim {D} must match module dim {self.d_model}"

        # --------------------------------------------------
        # 1) Time encoder: operate sequence-first with seq len T
        #    For each (b, s) we have a sequence of length T
        #    Prepare shape (T, B*S, D)
        # --------------------------------------------------
        xt = x.permute(1, 0, 2, 3).contiguous()  # (T, B, S, D)
        xt = xt.view(T, B * S, D)  # (T, B*S, D)
        if self.use_time_positional:
            xt = self.time_pos(xt)  # adds learned positional encodings along T

        enriched_temporal = self.time_encoder(xt)  # (T, B*S, D)
        # reshape back to (T, B, S, D) for convenient indexing
        enriched_temporal = enriched_temporal.view(T, B, S, D)

        # --------------------------------------------------
        # 2) Space encoder: operate sequence-first with seq len S
        #    For each (b, t) we have a sequence of length S
        #    Prepare shape (S, B*T, D)
        # --------------------------------------------------
        xs = x.permute(2, 0, 1, 3).contiguous()  # (S, B, T, D)
        xs = xs.view(S, B * T, D)  # (S, B*T, D)
        if self.use_space_positional:
            xs = self.space_pos(xs)

        enriched_spatial = self.space_encoder(xs)  # (S, B*T, D)
        enriched_spatial = enriched_spatial.view(S, B, T, D)

        # For symmetry and easier indexing also represent enriched_spatial as (S, B, T, D)
        # enriched_temporal is (T, B, S, D)

        # --------------------------------------------------
        # 3) Spatial decoding: for each time t, decode spatial tokens (seq len S)
        #    using temporal tokens at time t (the t-th temporal token across all S)
        #    as memory (seq len S). Both tgt and memory become shape (S, B, D) for
        #    the decoder call.
        # --------------------------------------------------
        decoded_spatial = torch.zeros_like(enriched_spatial)  # (S, B, T, D)
        for t in range(T):
            # tgt: spatial tokens for time t -> shape (S, B, D)
            tgt = enriched_spatial[:, :, t, :].contiguous()

            # memory: take temporal tokens at time t for all spatial positions ->
            # enriched_temporal[t] has shape (B, S, D) -> transpose to (S, B, D)
            memory = enriched_temporal[t].permute(1, 0, 2).contiguous()

            # run decoder: both are sequence-first
            out = self.spatial_decoder(tgt, memory)  # (S, B, D)
            decoded_spatial[:, :, t, :] = out

        # --------------------------------------------------
        # 4) Temporal decoding: symmetrical counterpart. For each space s, decode
        #    temporal tokens (seq len T) using spatial tokens at space s (seq len T)
        #    as memory.
        # --------------------------------------------------
        decoded_temporal = torch.zeros_like(enriched_temporal)  # (T, B, S, D)
        for s in range(S):
            tgt = enriched_temporal[:, :, s, :].contiguous()  # (T, B, D)
            memory = enriched_spatial[s].permute(1, 0, 2).contiguous()  # (T, B, D)
            out = self.temporal_decoder(tgt, memory)  # (T, B, D)
            decoded_temporal[:, :, s, :] = out

        # --------------------------------------------------
        # 5) Fuse: decoded_spatial is (S, B, T, D), decoded_temporal is (T, B, S, D)
        #    We'll reorder both to (B, T, S, D) and add them.
        # --------------------------------------------------
        decoded_spatial = decoded_spatial.permute(1, 2, 0, 3).contiguous()  # (B, T, S, D)
        decoded_temporal = decoded_temporal.permute(1, 0, 2, 3).contiguous()  # (B, T, S, D)

        fused = decoded_spatial + decoded_temporal  # (B, T, S, D)
        # fused = self.output_proj(fused)  # refine

        if not has_batch:
            fused = fused.squeeze(0)  # remove batch dim
        return fused


if __name__ == "__main__":
    # quick smoke test
    B, T, S, D = 2, 5, 6, 64
    x = torch.randn(B, T, S, D)
    model = SpatialTemporalEnhancer(dim=D, max_time=32, max_space=32)
    out = model(x)
    print("in:", x.shape, "out:", out.shape)
