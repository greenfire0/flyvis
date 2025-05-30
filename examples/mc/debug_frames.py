# examples/mc/train_ft3d_compare.py
from __future__ import annotations
import os, time, numpy as np, torch, cv2
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datamate import Namespace

from flyvis.network import Network
from flyvis.datasets.FT3D import MultiTaskFlyingThings3D
from flyvis.task.decoder import DecoderGAVP
from flyvis.task.objectives import epe
from flyvis.utils.hex_utils import get_hex_coords
from compare_util import hex_to_square, flow_to_rgb
# ─────────── tweak-here hyper-params ───────────────────────────
FT3D_ROOT      = Path(os.getenv("FT3D_ROOT", "/mnt/s/datasets/FlyingThings3D"))
FLOW_SCALE     = 1 / 169.0
EPOCHS         = 50
BATCH_SIZE     = 8
DT             = 1 / 50                  # dataset dt
LR_SHARED      = 1e-4
LR_CUSTOM      = 1e-4
NUM_WORKERS    = 0                       # per request
SAVE_DEBUG_EVERY = 25                   # dump PNGs every n epochs (0 → off)
OUT_DIR        = Path("debug_epoch_frames")
FRAME_INDICES  = None                    # None ⇒ first/mid/last frame
# ───────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

# ─────────────── debug-frame helpers ───────────────────────────

# ───────────── debug-frame helpers ───────────────────────────
# ────────────────── debug collage helper ──────────────────────
def save_debug_grid(epoch: int,
                    rgb:  torch.Tensor,   # (T,C,N_hex) on CPU
                    pred: torch.Tensor,   # (T,2,N_hex) on CPU
                    gt:   torch.Tensor):  # (T,2,N_hex) on CPU
    """
    Write one 2×2 PNG on epoch 1 only:
        TL = RGB frame (first 3 channels)
        TR = black
        BL = predicted flow
        BR = ground-truth flow
    """
    if (epoch %5)!=0:
        print(epoch%5)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rgb_sq   = hex_to_square(rgb)               # (T,H,W,C)
    pred_rgb = flow_to_rgb(hex_to_square(pred))
    gt_rgb   = flow_to_rgb(hex_to_square(gt))

    f = 0                                       # first frame is enough
    # keep only first 3 channels to avoid 8-ch mismatch
    vis_rgb  = rgb_sq[f, ..., :3]
    rgb_img  = (np.clip(vis_rgb, 0, 1) * 255).astype(np.uint8)

    pred_img = pred_rgb[f]
    gt_img   = gt_rgb[f]
    H, W, _  = rgb_img.shape
    black    = np.zeros_like(rgb_img)

    top    = np.concatenate([rgb_img, black],  axis=1)
    bottom = np.concatenate([pred_img, gt_img], axis=1)
    grid   = np.concatenate([top, bottom], axis=0)

    cv2.imwrite(str(OUT_DIR / "epoch01_grid.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def save_debug_flow(epoch: int,
                      pred: torch.Tensor,
                      gt: torch.Tensor):
    """PNG side-by-side of (pred | GT) for a handful of frames."""
    if SAVE_DEBUG_EVERY == 0 or epoch % SAVE_DEBUG_EVERY:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_np = pred[0].cpu()                 # (T,2,N_hex)
    gt_np   = gt[0].cpu()
    pred_rgb = flow_to_rgb(hex_to_square(pred_np))
    gt_rgb   = flow_to_rgb(hex_to_square(gt_np))

    idx = FRAME_INDICES or [0,
                            pred_rgb.shape[0] // 2,
                            pred_rgb.shape[0] - 1]
    for i in idx:
        canvas = np.concatenate([pred_rgb[i], gt_rgb[i]], axis=1)
        cv2.imwrite(str(OUT_DIR / f"ep{epoch:03d}_f{i:02d}.png"), canvas)