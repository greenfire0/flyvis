import torch
import numpy as np
import cv2
from flyvis.utils.hex_utils import get_hex_coords



def hex_to_square(arr: torch.Tensor) -> np.ndarray:
    """(T,C,N_hex) → (T,H,W,C)   numpy"""
    T, C, N = arr.shape
    u, v = get_hex_coords(15);  u -= u.min();  v -= v.min()
    H, W = int(u.max()+1), int(v.max()+1)
    sq = np.zeros((T, H, W, C), dtype=arr.cpu().numpy().dtype)
    for i, (uu, vv) in enumerate(zip(u, v)):
        sq[:, uu, vv, :] = arr[:, :, i]
    return sq

def flow_to_rgb(flow_xy: np.ndarray) -> np.ndarray:
    """Visualise flow with classical HSV colour wheel."""
    T, H, W, _ = flow_xy.shape
    hsv = np.zeros((T, H, W, 3), np.uint8)
    mag, ang = cv2.cartToPolar(flow_xy[..., 0], flow_xy[..., 1])
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / (np.percentile(mag, 95) + 1e-6) * 255,
                          0, 255).astype(np.uint8)
    return np.stack([cv2.cvtColor(hsv[i], cv2.COLOR_HSV2BGR) for i in range(T)], 0)

class MatchFlowCDF:
    """
    Remap FT3D flow magnitudes so their empirical CDF matches Sintel.
    Works entirely on the device of 'flow' (no CPU copies each step).
    """
    def __init__(self, q_src: torch.Tensor, q_tgt: torch.Tensor):
        self.q_src = q_src         # (Q,)
        self.q_tgt = q_tgt         # (Q,)

    @torch.no_grad()
    def __call__(self, sample: dict):
        flow = sample["flow"]                       # (H,W,2) or (2,H,W) – adapt as needed
        orig_shape = flow.shape

        f = flow.view(-1, 2)                        # (N,2)
        mag = torch.linalg.norm(f, dim=1)           # (N,)
        dir = f / (mag.unsqueeze(1) + 1e-8)

        # piece-wise linear interpolate using torch.searchsorted
        idx = torch.searchsorted(self.q_src.to(mag.device), mag.clamp(max=self.q_src[-1]))
        idx = idx.clamp(min=1, max=len(self.q_src) - 1)

        x0  = self.q_src[idx-1]
        x1  = self.q_src[idx]
        y0  = self.q_tgt[idx-1]
        y1  = self.q_tgt[idx]

        t   = (mag - x0) / (x1 - x0 + 1e-8)
        new_mag = y0 + t * (y1 - y0)                # (N,)

        flow_matched = (dir * new_mag.unsqueeze(1)).view(orig_shape)
        sample["flow"] = flow_matched
        return sample

class FT3DMatched(torch.utils.data.Dataset):
    def __init__(self, base_ds, flow_cdf_matcher):
        self.ds   = base_ds
        self.xfrm = flow_cdf_matcher
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        sample = self.ds[idx]
        return self.xfrm(sample)

