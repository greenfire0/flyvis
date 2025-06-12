#!/usr/bin/env python3
"""
Dump a few FlyingThings3D sequences to video:
  rgb.(mp4|avi)  and  flow.(mp4|avi)
in  <OUT_DIR>/seq_000/, <OUT_DIR>/seq_001/, …

Plays at 5 fps  (1 frame = 0.2 s).  Falls back to MJPG/AVI if MP4
codec is unavailable, so every file is guaranteed to be playable.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from flyvis.datasets.FT3D import MultiTaskFlyingThings3D
from flyvis.utils.hex_utils import get_hex_coords

# ── CONFIG ─────────────────────────────────────────────────────────────
N_SEQUENCES = 3
ROOT_DIR    = "/mnt/s/datasets/FlyingThings3D"
OUT_DIR     = Path("debug_samples")
DATA_ARGS   = dict(ft3d_path=ROOT_DIR, tasks=["rgb", "flow"], augment=False)
FPS         = 5                    # 1 frame / 0.2 s
# ───────────────────────────────────────────────────────────────────────

def flow_to_color(flow: np.ndarray, clip: float = 4.0) -> np.ndarray:
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)           # 0–180
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / clip * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def radius_from_nhex(n: int) -> int:
    r = int((-3 + (9 - 12 * (1 - n)) ** 0.5) / 6 + 1e-6)
    if 1 + 3 * r * (r + 1) != n:
        raise ValueError(f"{n} cannot be written as 1+3r(r+1)")
    return r

def hex_to_square(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.cpu().numpy()
    if arr.ndim != 3:
        raise RuntimeError("Expected (T,*,*) tensor")
    T, A, B = arr.shape
    if A > 6 and B <= 6:           # (T, N_hex, C)
        N_hex, C = A, B
    elif B > 6 and A <= 6:         # (T, C, N_hex) → transpose
        arr = arr.transpose(0, 2, 1)
        N_hex, C = B, A
    else:
        raise RuntimeError("Cannot detect hex dimension")

    r = radius_from_nhex(N_hex)
    u, v = get_hex_coords(r)
    u -= u.min(); v -= v.min()
    H, W = u.max() + 1, v.max() + 1

    if C == 1:
        sq = np.zeros((T, H, W), arr.dtype)
        for i, (uu, vv) in enumerate(zip(u, v)):
            sq[:, uu, vv] = arr[:, i, 0]
    else:
        sq = np.zeros((T, H, W, C), arr.dtype)
        for i, (uu, vv) in enumerate(zip(u, v)):
            sq[:, uu, vv, :] = arr[:, i, :]
    return sq

# ── helper: safe VideoWriter ───────────────────────────────────────────
def open_writer(base: Path, size: tuple[int, int]):
    fourcc = cv2.VideoWriter_fourcc
    mp4_path = base.with_suffix(".mp4")
    vw = cv2.VideoWriter(str(mp4_path), fourcc(*"mp4v"), FPS, size)
    if vw.isOpened():
        return vw, mp4_path
    # fallback → MJPG/AVI
    avi_path = base.with_suffix(".avi")
    vw = cv2.VideoWriter(str(avi_path), fourcc(*"MJPG"), FPS, size)
    if vw.isOpened():
        return vw, avi_path
    raise RuntimeError("Could not open VideoWriter with mp4v or MJPG")

# ── main dump routine ─────────────────────────────────────────────────
def dump():
    ds = MultiTaskFlyingThings3D(**DATA_ARGS, _init_cache=True)
    OUT_DIR.mkdir(exist_ok=True)

    for seq_idx in range(min(N_SEQUENCES, len(ds))):
        sample   = ds[seq_idx]
        rgb_sq   = hex_to_square(sample["rgb"])
        flow_sq  = hex_to_square(sample["flow"])

        seq_dir  = OUT_DIR / f"seq_{seq_idx:03d}"
        seq_dir.mkdir(exist_ok=True)

        H, W = rgb_sq.shape[1:3]
        #rgb_writer , rgb_file  = open_writer(seq_dir / "rgb" , (W, H))
        flow_writer, flow_file = open_writer(seq_dir / "flow", (W, H))

        for frame_rgb, frame_flow in zip(rgb_sq, flow_sq):
         #   rgb_writer.write((np.clip(frame_rgb, 0, 1) * 255)
                            # .astype(np.uint8)[..., ::-1])
            flow_writer.write(flow_to_color(frame_flow))

        #rgb_writer.release()
        flow_writer.release()
        print(f"[+] seq {seq_idx}:{flow_file.name}")

if __name__ == "__main__":
    dump()
