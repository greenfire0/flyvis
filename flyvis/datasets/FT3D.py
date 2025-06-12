import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

import numpy as np
import pandas as pd
import torch
from datamate import Directory, root
from tqdm import tqdm

# Re‑use FlyVis hexagon utilities & augmentation modules
from flyvis import renderings_dir

from .augmentation.hex import (
    ContrastBrightness,
    GammaCorrection,
    HexFlip,
    HexRotate,
    PixelNoise,
)
from .augmentation.temporal import (
    CropFrames,
    Interpolate,
)
from .datasets import MultiTaskDataset
from .rendering import BoxEye
from .rendering.utils import split
import re
from .FT3D_util import (
    download_flyingthings3d,
    load_ft3d_sequence,
    read_pfm,
    sample_ft3d_rgb,
)
logger = logging.getLogger(__name__)

__all__ = [
    "RenderedFlyingThings3D",
    "MultiTaskFlyingThings3D",
    "AugmentedFlyingThings3D",
]
TAG_FLOAT = 202021.25



###############################################################################
#                              Rendered dataset                               #
###############################################################################

@root(renderings_dir)
class RenderedFlyingThings3D(Directory):
    """Hex‑projects FlyingThings3D sequences once and stores NumPy blobs."""

    def __init__(
        self,
        tasks: List[str] = ["flow"],
        boxfilter: Dict[str, int] = dict(extent=15, kernel_size=13),
        vertical_splits: int = 3,
        n_frames: Optional[int] = 9,
        center_crop_fraction: float = 0.7,
        unittest: bool = False,
        ft3d_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__()

        ft3d_path = Path(ft3d_path) if ft3d_path else download_flyingthings3d()
        boxfilter = BoxEye(**boxfilter)

        seq_roots = (ft3d_path / "frames_cleanpass" / "TRAIN").rglob("left")

        for i, rgb_dir in enumerate(tqdm(sorted(seq_roots), desc="Rendering FT3D")):
            seq_id = rgb_dir.parent.name
            group = rgb_dir.parent.parent.name  # A/B/C
            flow_dir = ft3d_path / "optical_flow" / "TRAIN"  / group / seq_id /"into_future"/ "left"

            if not flow_dir.exists():
                logger.warning("skipped sequence",flow_dir)
                continue
            if n_frames is not None and (len(sorted(list(rgb_dir.iterdir()))) < n_frames or
                                          len(sorted(list(flow_dir.iterdir()))) < n_frames):
                logger.warning("skipped sequence",len(sorted(list(rgb_dir.iterdir()))) )
                continue  # skip short sequences


            # RGB ----------------------------------------------------------------
            rgb = load_ft3d_sequence(rgb_dir,
                                      sample_ft3d_rgb,
                                        start=0,
                                          end=n_frames
                                            if not unittest else 4)
            rgb_split = split(
                rgb,
                boxfilter.min_frame_size[1] + 2 * boxfilter.kernel_size,
                vertical_splits,
                center_crop_fraction,
            )
            rgb_hex = boxfilter(rgb_split).cpu().numpy()

            # FLOW ---------------------------------------------------------------
            if "flow" in tasks:
                flow = load_ft3d_sequence(flow_dir,
                                           read_pfm,
                                             start=0,
                                               end=n_frames
                                                 if not unittest else 3)
                #F, H, W, C = flow.shape        # unpack once
                #print(f"frames={F}, height={H}, width={W}, channels={C}")
                flow_split = split(
                    flow,
                    boxfilter.min_frame_size[1] + 2 * boxfilter.kernel_size,
                    vertical_splits,
                    center_crop_fraction,
                    -2 ## because w is in second to last col
                )
                flow_hex = torch.cat(
                    (
                        boxfilter(flow_split[..., 0], ftype="mean"),
                        boxfilter(flow_split[..., 1], ftype="mean"),
                    ),
                    dim=2,
                ).cpu().numpy()

            for j in range(rgb_hex.shape[0]):


                path = f"sequence_{i:05d}_{group}_{seq_id}_split_{j:02d}"
                self[f"{path}/lum"] = rgb_hex[j]
                if "flow" in tasks:
                    self[f"{path}/flow"] = flow_hex[j]
            if unittest:
                break

    def __call__(self, seq_id: int) -> Dict[str, np.ndarray]:
        blob = self[sorted(self)[seq_id]]
        return {k: blob[k][:] for k in sorted(blob)}

###############################################################################
#                            Multi‑task dataset                               #
###############################################################################

class MultiTaskFlyingThings3D(MultiTaskDataset):
    """Lightweight loader that streams the pre‑rendered blobs."""

    original_framerate: int = 15  # Hz
    dt: float = 1 / 50
    tasks: List[str] = []
    valid_tasks: List[str] = ["lum", "flow"]

    def __init__(
        self,
        tasks: List[str] = ["flow"],
        boxfilter: Dict[str, int] = dict(extent=15, kernel_size=13),
        vertical_splits: int = 3,
        n_frames: int = 9,
        center_crop_fraction: float = 0.7,
        dt: float = 1 / 50,
        augment: bool = True,
        random_temporal_crop: bool = True,
        all_frames: bool = False,
        resampling: bool = True,
        interpolate: bool = True,
        p_flip: float = 0.5,
        p_rot: float = 5 / 6,
        contrast_std: float = 0.2,
        brightness_std: float = 0.1,
        gaussian_white_noise: float = 0.08,
        gamma_std: Optional[float] = None,
        _init_cache: bool = True,
        unittest: bool = False,
        flip_axes: List[int] = [0, 1],
        ft3d_path: Optional[Union[str, Path]] = None,
    ):
        invalid = [t for t in tasks if t not in self.valid_tasks]
        if invalid:
            raise ValueError(f"invalid tasks {invalid}")
        self.tasks = [t for t in self.valid_tasks if t in tasks]
        self.data_keys = self.tasks

        self.n_frames = n_frames if not unittest else 3
        self.dt = dt
        self.all_frames = all_frames
        assert vertical_splits >= 1 , "vertical_splits must be greater than 1"
        self.vertical_splits = vertical_splits
        self.center_crop_fraction = center_crop_fraction

        self.p_flip = p_flip
        self.p_rot = p_rot
        self.contrast_std = contrast_std
        self.brightness_std = brightness_std
        self.gaussian_white_noise = gaussian_white_noise
        self.gamma_std = gamma_std
        self.random_temporal_crop = random_temporal_crop
        self.flip_axes = flip_axes
        self.fix_augmentation_params = False
        


        # augmentation params (HexFlip/Rotate etc.) – defer to parent helpers
        self.contrast_std = contrast_std
        self.brightness_std = brightness_std
        self.gaussian_white_noise = gaussian_white_noise
        self.gamma_std = gamma_std
        self.ft3d_path = Path(ft3d_path or os.getenv("FT3D_ROOT", ""))
        if not self.ft3d_path.is_dir():
            raise FileNotFoundError("ft3d_path not found. \
                                    Pass it explicitly or set $FT3D_ROOT")

        self.rendered = RenderedFlyingThings3D(
            tasks=tasks,
            boxfilter=boxfilter,
            vertical_splits=vertical_splits,
            n_frames=n_frames,
            center_crop_fraction=center_crop_fraction,
            unittest=unittest,
            ft3d_path=self.ft3d_path,
        )
        assert len(self.rendered) > 0, "RenderedFlyingThings3D is empty."
        self.init_augmentation()
        
        self.augment = augment
        self.piecewise_resample.augment = resampling
        self.linear_interpolate.augment = interpolate
        self._augmentations_are_initialized = True
        self._SEQ_RE = re.compile(
        r"sequence_(\d{5})_([A-C])_([A-Za-z0-9]+)_split_(\d{2})"
        )
        #self.arg_df = pd.DataFrame(dict(index=np.arange(len(self.rendered))))
        if _init_cache:
            self.init_cache()

    # ---------- dataset.py ----------
    def init_cache(self):

        self.cached_sequences: List[Dict[str, torch.Tensor]] = []
        arg_records: List[Dict[str, Any]] = []          # ← NEW

        key_list = sorted(self.rendered)                # fixed order once

        for i, key in enumerate(key_list):
            blob = self.rendered(i)

            # ---------- build the cache entry ----------
            entry = {
                k: torch.tensor(blob[k], dtype=torch.float32)
                for k in self.data_keys if k in blob
            }
            self.cached_sequences.append(entry)

            # ---------- gather metadata for arg_df ----------
            n_frames = next(iter(entry.values())).shape[0]          # any modality works
            m = re.match(r"sequence_(\d{5})_", key)                 # 00000 .. 99999
            original_idx = int(m.group(1)) if m else i

            arg_records.append(
                dict(
                    index=i,
                    original_index=original_idx,
                    name=key,
                    original_n_frames=n_frames,
                )
            )

        # ---------- finalise the dataframe ----------
        self.arg_df = pd.DataFrame(arg_records)


    def init_augmentation(self) -> None:
        """Create all temporal-and-spatial augmentation callables."""
        # temporal window crop
        self.temporal_crop = CropFrames(
            self.n_frames, all_frames=self.all_frames,
            random=self.random_temporal_crop
        )
        # photometric + geometric aug
        self.jitter  = ContrastBrightness(self.contrast_std, self.brightness_std)
        self.noise   = PixelNoise(self.gaussian_white_noise)
        extent = 15 # or boxfilter["extent"] if you want it parametric
        self.rotate = HexRotate(extent, p_rot=self.p_rot)
        self.flip = HexFlip (extent, p_flip=self.p_flip)

        # time-axis resamplers (same objects Sintel uses)
        self.piecewise_resample = Interpolate(
            self.original_framerate, 1 / self.dt, mode="nearest-exact"
        )
        self.linear_interpolate = Interpolate(
            self.original_framerate, 1 / self.dt, mode="linear"
        )
        # gamma
        self.gamma_correct = GammaCorrection(1.0, self.gamma_std)
    def apply_augmentation(
        self,
        data: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
        """Crop + optional resample / interpolate."""
        def temporal_ops(x):
            x = self.temporal_crop(x)
            if self.linear_interpolate.augment:
                x = self.linear_interpolate(x)
            elif self.resampling:
                x = self.piecewise_resample(x)
            return x

        out = {"lum": temporal_ops(data["lum"])}
        for k in self.tasks:
            if k in data and k != "lum":
                out[k] = temporal_ops(data[k])
        return out

    def get_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return one sequence as a dict of tensors.
        Guarantees that only the keys requested via self.data_keys are present
        and that every returned value is a torch.Tensor (on CPU).
        """
        entry = self.cached_sequences[idx]
        #print(idx)
        # keep only the keys the user asked for
        sample = {k: entry[k] for k in self.data_keys if k in entry}

        if not sample:
            # It is better to raise than to silently return None
            raise KeyError(
                f"Sequence {idx} is missing the requested keys {self.data_keys}."
            )
        return self.apply_augmentation(sample) if self.augment else sample
        #return sample this was the original but is now removed
    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        # Optionally remove problematic attributes
        if 'rendered' in state:
            del state['rendered']  # or replace with something minimal
        return state

    def __setstate__(self, state):
        """Restore state from the unpickled state."""
        self.__dict__.update(state)
    def _parse_name(self, idx: int) -> tuple[str, str]:
        """
        Internal helper – given an integer `idx`, return `(group, seq_id)` as
        strings so paths can be reconstructed on-demand.
        """
        name = self.arg_df.loc[idx, "name"]
        m = self._SEQ_RE.match(name)
        if m is None:
            raise ValueError(f"Cannot parse sequence name: {name}")
        _, group, seq_id, _ = m.groups()
        return group, seq_id

    def cartesian_sequence(
        self,
        idx: int,
        *,
        vertical_splits: int = 1,
        outwidth: int = 716,
        center_crop_fraction: float = 1.0,
        sampling: slice = slice(None),
    ) -> np.ndarray:
        """
        Return the RGB frames of the original FT3D sequence in Cartesian
        coordinates (no hex projection).

        Example
        -------
        >>> seq = ft3d.cartesian_sequence(5, vertical_splits=1)

        """
        group, seq_id = self._parse_name(idx)
        rgb_dir = (
            self.ft3d_path
            / "frames_cleanpass" / "TRAIN" / group / seq_id / "left"
        )
        frames = [
            sample_ft3d_rgb(p)                    # H×W×3 uint8
            for p in sorted(rgb_dir.iterdir())[sampling]
        ]
        rgb = np.stack(frames, axis=0)            # F×H×W×3
        return split(
            rgb,
            outwidth,
            vertical_splits,
            center_crop_fraction,
        )

    def cartesian_flow(
        self,
        idx: int,
        *,
        vertical_splits: int = 1,
        outwidth: int = 417,
        center_crop_fraction: float = 1.0,
        sampling: slice = slice(None),
    ) -> np.ndarray:
        """
        Return optical-flow fields (u, v) in Cartesian coordinates.

        Example
        -------
        >>> flow = ft3d.cartesian_flow(5, vertical_splits=1)
        """
        group, seq_id = self._parse_name(idx)
        flow_dir = (
            self.ft3d_path
            / "optical_flow" / "TRAIN"
            / group / seq_id / "into_future" / "left"
        )
        frames = [
            read_pfm(p)                           # H×W×2 float32
            for p in sorted(flow_dir.iterdir())[sampling]
        ]
        flow = np.stack(frames, axis=0)           # F×H×W×2
        # split() needs the u and v channels last; no extra axis shift needed
        return split(
            flow,
            outwidth,
            vertical_splits,
            center_crop_fraction,
            -2,                                   # width axis for FT3D flow
        )
    # ---------------------------------------------------------------------
    # Augmentation master-switch (getter + setter)
    # ---------------------------------------------------------------------
    @property
    def augment(self) -> bool:
        """Current augmentation state (True = ON)."""
        return self._augment


    @augment.setter
    def augment(self, value: bool) -> None:
        """
        Toggle all augmentation sub-modules in one go.

        Setting this BEFORE init_augmentation() runs is harmless because we
        guard with _augmentations_are_initialized.
        """
        self._augment = value

        # make sure sub-objects exist before we touch them
        if not getattr(self, "_augmentations_are_initialized", False):
            return

        # ── temporal crop randomness ───────────────────────────────────────
        self.temporal_crop.random = self.random_temporal_crop if value else False

        # ── photometric / geometric aug toggles ────────────────────────────
        self.jitter.augment        = value
        self.noise.augment         = value
        self.rotate.augment        = value
        self.flip.augment          = value
        self.gamma_correct.augment = value

        # ── time-axis resamplers (controlled by flags, not by `value`) ─────
        self.piecewise_resample.augment = self.resampling    # nearest-exact
        self.linear_interpolate.augment = self.interpolate   # linear interp

        
###############################################################################
#                          Deterministic augmentation                          #
###############################################################################

class AugmentedFlyingThings3D(MultiTaskFlyingThings3D):
    """Adds deterministic flips/rotations/temporal splits for evaluation."""

    valid_flip_axes = [0, 1, 2, 3]
    valid_rotations = [0, 1, 2, 3, 4, 5]

    def __init__(
        self,
        flip_axes: List[int] = [0, 1],
        n_rotations: List[int] = [0, 1, 2, 3, 4, 5],
        temporal_split: bool = False,
        **kwargs,
    ):
        super().__init__(augment=False, random_temporal_crop=False, **kwargs)

        if any(ax not in self.valid_flip_axes for ax in flip_axes):
            raise ValueError(f"invalid flip axes {flip_axes}")
        if any(r not in self.valid_rotations for r in n_rotations):
            raise ValueError(f"invalid rotations {n_rotations}")

        self.flip_axes = flip_axes
        self.n_rotations = n_rotations
        self.temporal_split = temporal_split

        # build deterministic variants on first demand
        self._built = False
        self._build()


    def _build(self):
        if self._built:
            return
