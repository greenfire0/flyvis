from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import re
import os
import cv2
import numpy as np
from datamate import Directory, Namespace, root
from tqdm import tqdm
# Re‑use FlyVis hexagon utilities & augmentation modules
from flyvis import renderings_dir
TAG_FLOAT = 202021.25 

###############################################################################
#                            Data loading helpers                             #
###############################################################################

def download_flyingthings3d(*, flow: bool = True) -> Path:
    """Locate the FlyingThings3D root on disk.

    Priority order:
    1. Environment variable ``$FT3D_ROOT``
    2. ``~/datasets/FlyingThings3D``
    3. Raises *FileNotFoundError* with a helpful message.
    """
    guess = os.getenv("FT3D_ROOT") or Path.home() / "datasets" / "FlyingThings3D"
    path = Path(guess)
    if not path.is_dir():
        raise FileNotFoundError("FlyingThings3D root not found. Set $FT3D_ROOT or pass ft3d_path=")
    return path.resolve()


def load_ft3d_sequence(dir_path: Path, sample_fn, *, start: int = 0, end: Optional[int] = None):
    """
    Load sorted sequence of files with names like OpticalFlowIntoFuture_0011_L.pfm.
    Extracts frame number from filename via regex.
    """
    def frame_index(p: Path) -> int:
        match = re.search(r"(\d{4})", p.stem)
        if not match:
            raise ValueError(f"Could not extract frame number from {p.name}")
        return int(match.group(1))

    files = sorted(
        [p for p in dir_path.iterdir() if p.is_file()],
        key=frame_index
    )
    files = files[start:end]
    return np.stack([sample_fn(p) for p in files]) 


def sample_ft3d_rgb(path: Path) -> np.ndarray:
    """Load an FT3D PNG and return luminance (H, W) in [0, 1] float32."""
    # OpenCV reads as BGR uint8 by default
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    # Convert to luminance (ITU-R BT.709 weights, matching FlyVis formula)
    # Note: channels are B, G, R here
    return 0.0722 * bgr[..., 0] + 0.7152 * bgr[..., 1] + 0.2126 * bgr[..., 2]


def read_pfm(path):
    """
    Read a .pfm optical-flow file and return the first two channels (u, v).

    Parameters
    ----------
    path : str | Path
        Path to the .pfm file.

    Returns
    -------
    np.ndarray
        H × W × 2 array, dtype float32.
    """
    path = Path(path)
    with path.open('rb') as f:
        # ── header ──────────────────────────────────────────────
        header = f.readline().decode('ascii').rstrip()
        if header == 'PF':
            n_channels = 3
        elif header == 'Pf':
            n_channels = 1
        else:
            raise ValueError(f'{path} is not a valid PFM (got header {header!r}).')

        # skip optional comment lines
        dims_line = f.readline().decode('ascii')
        while dims_line.startswith('#'):
            dims_line = f.readline().decode('ascii')

        m = re.match(r'^(\d+)\s+(\d+)$', dims_line.strip())
        if m is None:
            raise ValueError(f'Malformed PFM header in {path}.')
        width, height = map(int, m.groups())

        scale = float(f.readline().decode('ascii').strip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)

        # ── data ────────────────────────────────────────────────
        data = np.fromfile(f, endian + 'f')
        expected = width * height * n_channels
        if data.size != expected:
            raise ValueError(
                f'Expected {expected} floats, found {data.size} in {path}.'
            )

    data = data.reshape((height, width, n_channels))
    data = np.flipud(data)             # PFM is stored from bottom up
    data *= scale                      # apply scale if present

    # use only u,v channels, cast to float32
    return data[..., :2].astype(np.float32)

