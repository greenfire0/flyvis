# mc_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from flyvis.task.decoder import ActivityDecoder, Conv2dHexSpace


class DecoderTinyRes(ActivityDecoder):
    """
    Two lightweight residual blocks + 1×1 read-out (u,v).
    Keeps hex masking, adds GroupNorm + SiLU activations.
    """

    def __init__(
        self,
        connectome,
        base_channels: int = 8,      # after projection
        mid_channels:  int = 32,     # width inside residual blocks
        kernel_size:   int = 3,
        dropout:       float = 0.1,
        const_weight:  float = None,
    ):
        super().__init__(connectome)
        p = (kernel_size - 1) // 2                # same-padding

        in_ch = len(connectome.output_cell_types)

        # ── 1×1 projection to base_channels ────────────────────────────
        self.proj = Conv2dHexSpace(in_ch, base_channels, 1,
                                   const_weight=const_weight)

        # ── residual block helper ──────────────────────────────────────
        def res_block(ch: int):
            return nn.Sequential(
                Conv2dHexSpace(ch, mid_channels, kernel_size,
                               padding=p, const_weight=const_weight),
                nn.GroupNorm(4, mid_channels), nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                Conv2dHexSpace(mid_channels, ch, kernel_size,
                               padding=p, const_weight=const_weight),
                nn.GroupNorm(4, ch),
            )

        self.block1 = res_block(base_channels)
        self.block2 = res_block(base_channels)

        # ── final 1×1 read-out to 2 flow channels ─────────────────────
        self.head = Conv2dHexSpace(base_channels, 2, 1,
                                   const_weight=const_weight)

    # ------------------------------------------------------------------
    def forward(self, activity):
        """
        activity : (B, T, Ncells)  DMN activity
        returns  : (B, T, 2, Npix) cartesian flow
        """
        self.dvs_channels.update(activity)
        x = F.relu(self.dvs_channels.output)           # (B,T,C_hex,Nhex)

        B, T, C, Nhex = x.shape

        # put hexals onto regular grid
        grid = torch.zeros(B, T, C, self.H, self.W,
                           dtype=x.dtype, device=x.device)
        grid[:, :, :, self.u, self.v] = x
        grid = grid.view(B * T, C, self.H, self.W)     # merge time

        y = self.proj(grid)
        y = y + self.block1(y)
        y = y + self.block2(y)

        flow = self.head(y)                           # (B*T,2,H,W)
        flow = flow.view(B, T, 2, self.H, self.W)[:, :, :, self.u, self.v]
        return flow
