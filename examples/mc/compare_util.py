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
