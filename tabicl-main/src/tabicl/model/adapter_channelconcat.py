import torch
import torch.nn as nn


class ChannelWiseConcatAdapter(nn.Module):
    """Channel-wise adapter: project each channel embedding, concat, then project to TabICL dim.

    Input:
        x: (B, C, D_mantis)
    Output:
        z: (B, D_tabicl)

    Notes:
    - Supports variable C across datasets via pad/truncate to max_channels.
    - "逐个通道输入" 的实现方式是对每个通道共享同一套投影层（逐通道投影），
      再按通道维度拼接（concat/flatten），最后映射到 TabICL 输入维度。
    """

    def __init__(
        self,
        mantis_emb_dim: int,
        tabicl_input_dim: int,
        *,
        max_channels: int = 30,
        per_channel_dim: int = 16,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()

        if max_channels < 1:
            raise ValueError(f"max_channels must be >= 1, got {max_channels}")
        if per_channel_dim < 1:
            raise ValueError(f"per_channel_dim must be >= 1, got {per_channel_dim}")

        self.mantis_emb_dim = int(mantis_emb_dim)
        self.tabicl_input_dim = int(tabicl_input_dim)
        self.max_channels = int(max_channels)
        self.per_channel_dim = int(per_channel_dim)

        self.channel_proj = nn.Sequential(
            nn.LayerNorm(self.mantis_emb_dim) if use_layernorm else nn.Identity(),
            nn.Linear(self.mantis_emb_dim, self.per_channel_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        self.out_proj = nn.Linear(self.max_channels * self.per_channel_dim, self.tabicl_input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Input must be (B, C, D). Got shape {tuple(x.shape)}")

        B, C, D = x.shape
        if D != self.mantis_emb_dim:
            raise ValueError(
                f"Last dim mismatch: expected mantis_emb_dim={self.mantis_emb_dim}, got {D}."
            )

        if C < self.max_channels:
            pad = x.new_zeros((B, self.max_channels - C, D))
            x = torch.cat([x, pad], dim=1)
        elif C > self.max_channels:
            x = x[:, : self.max_channels, :]

        # Per-channel projection (shared weights across channels)
        z = self.channel_proj(x)  # (B, max_channels, per_channel_dim)
        z = self.dropout(z)

        # Concat over channels
        z = z.reshape(B, self.max_channels * self.per_channel_dim)

        # Final projection to TabICL input dim
        return self.out_proj(z)

    def extra_repr(self) -> str:
        return (
            f"mantis_emb_dim={self.mantis_emb_dim}, tabicl_input_dim={self.tabicl_input_dim}, "
            f"max_channels={self.max_channels}, per_channel_dim={self.per_channel_dim}"
        )
