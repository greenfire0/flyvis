import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from datamate import Namespace

from flyvis import device
from flyvis.connectome import ConnectomeFromAvgFilters
from flyvis.utils.activity_utils import LayerActivity
from flyvis.utils.hex_utils import get_hex_coords
from flyvis.utils.nn_utils import n_params

__all__ = ["DecoderGAVP", "init_decoder"]


# ───────────────────────────────────────── helper layers ────────────────────────────────────────── #
class GlobalAvgPool(nn.Module):
    """Average over the last spatial dimension (hexals)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W) → (B, C)
        return x.mean(dim=-1)


class Conv2dConstWeight(nn.Conv2d):
    """Conv2d whose weights/bias can be initialised to a constant value."""
    def __init__(self, *args, const_weight: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if const_weight is not None:
            nn.init.constant_(self.weight, const_weight)
            if self.bias is not None:
                nn.init.constant_(self.bias, const_weight)


class Conv2dHexSpace(Conv2dConstWeight):
    """2‑D convolution where the kernel is masked to a regular hexagon."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *,
                 const_weight: Optional[float] = None, **kw):
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for a symmetric hex mask")

        pad = (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size,
                         padding=pad, const_weight=const_weight, **kw)

        # Pre‑compute a mask selecting the hexagonally arranged taps ----------------
        if kernel_size > 1:
            u, v = get_hex_coords(kernel_size // 2)
            u -= u.min(); v -= v.min()
            mask = torch.zeros_like(self.weight)
            mask[:, :, u, v] = 1.0
            self.register_buffer("_hex_mask", mask)
        else:
            self._hex_mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._hex_mask is not None:
            self.weight.data.mul_(self._hex_mask)
        return super().forward(x)


# ────────────────────────────────────────── decoder ─────────────────────────────────────────────── #
class MCDecoderGAVP(nn.Module):
    """A light fully‑convolutional decoder used in the original Fly‑DMN papers.

    The architecture is deliberately tiny:
        * **Base**  – one or more Conv→BN→ReLU→Dropout blocks
        * **Decoder** – a final linear Conv that maps the features to 3 flow channels

    Differences from the previous version
    --------------------------------------
    * **ReLU** instead of Softplus ⇒ outputs can be negative.
    * No extra normalisation channel; the last conv is linear.
    * Default dropout reduced to *0.2* (was 0.5) for faster convergence.
    """

    def __init__(self,
                 connectome: ConnectomeFromAvgFilters,
                 shape: List[int],
                 kernel_size: int,
                 p_dropout: float = 0.0,
                 batch_norm: bool = True,
                 n_out_features: Optional[int] = None,
                 const_weight: Optional[float] = None):
        super().__init__()
        self.dvs_channels = LayerActivity(None, connectome, use_central=False)

        # --- geometry ---------------------------------------------------------
        radius = connectome.config.extent
        u, v = get_hex_coords(radius)
        u -= u.min();  v -= v.min()

        self.register_buffer("u", torch.as_tensor(u, dtype=torch.long))
        self.register_buffer("v", torch.as_tensor(v, dtype=torch.long))
        self.H, self.W = int(u.max() + 1), int(v.max() + 1)

        # --- network ----------------------------------------------------------
        in_ch  = len(connectome.output_cell_types)
        out_ch = shape[-1]
        self._out_channels = out_ch
        self.out_channels  = out_ch * (n_out_features or 1)
        self.n_out_features = n_out_features

        blocks: List[nn.Module] = []
        for ch in shape[:-1]:
            if ch == 0:
                continue
            blocks.append(Conv2dHexSpace(in_ch, ch, kernel_size, const_weight=const_weight))
            if batch_norm:
                blocks.append(nn.BatchNorm2d(ch))
            blocks.append(nn.ReLU(inplace=True))
            if p_dropout > 0:
                blocks.append(nn.Dropout(p_dropout))
            in_ch = ch
        self.base = nn.Sequential(*blocks)

        # final mapping to flow (3 channels: dx, dy, confidence) ---------------
        self.decoder = Conv2dHexSpace(in_ch, self.out_channels, kernel_size,
                                      const_weight=const_weight)

        self.head = GlobalAvgPool() if n_out_features else nn.Identity()

        self.num_parameters = n_params(self)
        logging.info(f"Initialized decoder with {self.num_parameters} params")

    # ---------------------------------------------------------------------
    def forward(self, activity: torch.Tensor) -> torch.Tensor:
        """activity: (B, T, N_cells) → flow: (B, T, C=3, N_hex)"""
        self.dvs_channels.update(activity)
        x = F.relu(self.dvs_channels.output)  # (B, T, C_in, N_hex)
        B, T, C_in, N_hex = x.shape

        # map to Cartesian image grid ---------------------------------------
        x_map = x.new_zeros((B, T, C_in, self.H, self.W))
        x_map[..., self.u, self.v] = x
        x_map = x_map.view(-1, C_in, self.H, self.W)   # (B*T, C_in, H, W)

        y = self.decoder(self.base(x_map))             # (B*T, C_out, H, W)
        y = y.view(B, T, self.out_channels, self.H, self.W)[..., self.u, self.v]

        if self.n_out_features:
            y = self.head(y).view(B, T, self._out_channels, self.n_out_features)
        return y


# ───────────────────────────────────────── factory ──────────────────────────────────────────────── #

def init_decoder(decoder_config: Namespace, connectome: ConnectomeFromAvgFilters) -> nn.Module:
    """Instantiate a decoder from a *datamate* Namespace config."""
    cfg = decoder_config.deepcopy()
    cls = globals()[cfg.pop("type")]  # e.g. "DecoderGAVP"
    return cls(connectome=connectome, **cfg)
