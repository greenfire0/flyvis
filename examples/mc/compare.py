# examples/mc/train_ft3d_compare.py
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from compare_util import FT3DMatched, MatchFlowCDF
from datamate import Namespace
from debug_frames import save_debug_grid
from torch.optim import Adam
from torch.utils.data import DataLoader

from flyvis.datasets.FT3D import MultiTaskFlyingThings3D
from flyvis.network import Network
from flyvis.task.decoder import DecoderGAVP
from flyvis.task.mc_decoder import MCDecoderGAVP as DecoderGAVP
from flyvis.task.objectives import epe

# ─────────── tweak-here hyper-params ───────────────────────────
FT3D_ROOT      = Path(os.getenv("FT3D_ROOT", "/mnt/s/datasets/FlyingThings3D"))
FLOW_SCALE     = 1
EPOCHS         = 50
BATCH_SIZE     = 8
DT             = 1 / 50                  # dataset dt
LR_SHARED      = 1e-4
LR_CUSTOM      = 1e-4
NUM_WORKERS    = 0                       # per request
SAVE_DEBUG_EVERY = 25                   # dump PNGs every n epochs (0 → off)
OUT_DIR        = Path("debug_epoch_frames")
FRAME_INDICES  = None                    # None ⇒ first/mid/last frame
LR_NET_SHARED   = 1e-4      # ↓ network (“fly”)   – slower
LR_DEC_SHARED   = 5e-4      # ↑ decoder           – faster
LR_NET_CUSTOM   = 1e-4      # custom-connectome run
LR_DEC_CUSTOM   = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ───────────── dataloader helper (no shuffle/pin-mem) ──────────
def build_loader(batch: int, augment: bool):
    ds = MultiTaskFlyingThings3D(
        ft3d_path=FT3D_ROOT,
        tasks=["rgb", "flow"],
        augment=augment,
        _init_cache=True,
    )
    data = dataset1[0]
    lum = ds[0]["rgb"]
    flow = data["flow"]

    ft3d_matched = torch.utils.data.Dataset()   # tiny wrapper

    percentiles      = np.linspace(0, 100, 1000)
    q_src            = np.percentile(mags_ft3d,   percentiles)   # 1 × 1000
    q_tgt            = np.percentile(mags_sintel, percentiles)   # 1 × 1000
    q_src = torch.as_tensor(q_src,  dtype=torch.float32)   # shape (Q,)
    q_tgt = torch.as_tensor(q_tgt,  dtype=torch.float32)

    flow_matcher   = MatchFlowCDF(q_src, q_tgt)
    dataset1_matched = FT3DMatched(dataset1, flow_matcher)

    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=True,
    )

# ───────────── custom network factory  ─────────────────────────
def build_custom_network():
    node_cfg = Namespace(
        bias=Namespace(
            type="RestingPotential",
            groupby=["type", "index"],
            initial_dist="Normal", mode="sample",
            requires_grad=True, mean=0.5, std=0.05,
            penalize=Namespace(activity=True), seed=0,
        ),
        time_const=Namespace(
            type="TimeConstant",
            groupby=["type", "index"],
            initial_dist="Value", value=0.05,
            requires_grad=True,
        ),
    )
    edge_cfg = Namespace(
        sign=Namespace(
            type="SynapseSign",
            groupby=[
                "source_type", "target_type",
                "source_index", "target_index"
            ],
            initial_dist="Value", requires_grad=False,
        ),
        syn_count=Namespace(
            type="SynapseCount",
            groupby=[
                "source_type", "target_type",
                "source_index", "target_index",
                "dv", "du",
            ],
            initial_dist="Lognormal", mode="mean", requires_grad=True, std=1.0,
        ),
        syn_strength=Namespace(
            type="SynapseCountScaling",
            groupby=[
                "source_type", "target_type",
                                        ],
            initial_dist="Value", requires_grad=False,
            scale=0.01, clamp="non_negative",
        ),
    )
    return Network(node_config=node_cfg, edge_config=edge_cfg).to(device)

# ───────────── generic epoch loop helper ───────────────────────
def current_lrs(opt):
    "Return list of LR values for each param group (usually len==1)."
    return [g['lr'] for g in opt.param_groups]

def tensor_stats(t: torch.Tensor):
    "Return mean & std detached to float."
    return float(t.mean()), float(t.std())

def print_hdr(epoch, label):
    print(f"\n── {label.upper()} epoch {epoch:03d} ─────────────────────────────────")

# ───────────── generic epoch loop helper ───────────────────────
# ───────────── generic epoch loop helper ───────────────────────
def run_epoch(net,
              dec,
              loader,
              optimizer=None,
              *,
              epoch: int = 0,
              label: str = ""):
    """
    Pass once over `loader`.

    Returns
    -------
    mean_loss : float
    pred      : torch.Tensor (T,2,N_hex) – CPU
    gt        : torch.Tensor (T,2,N_hex) – CPU
    rgb_samp  : torch.Tensor (T,C,N_hex) – CPU   (first mini-batch only)
    """
    train = optimizer is not None
    net.train() if train else net.eval()

    losses, rgb_samp = [], None
    first_conv = getattr(dec, "base", [None])[0]

    with torch.set_grad_enabled(train):
        for batch in loader:
            rgb  = batch["rgb"].float()               # (B,T,C,N_hex)
            flow = batch["flow"].float() * FLOW_SCALE

            if rgb.shape[2] == 1:                     # mono → multi-channel
                rgb = rgb.repeat_interleave(
                    len(net.stimulus.input_index), dim=2
                )                                     # (B,T,8,N_hex) e.g.

            rgb, flow = rgb.to(device), flow.to(device)
            init_state = net.steady_state(0.5, DT, rgb.size(0))
            pred = dec(net.simulate(rgb, DT, initial_state=init_state))
            loss = epe(pred, flow)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            if rgb_samp is None:                      # keep first mini-batch
                rgb_samp = rgb.detach().cpu()[0]      # drop batch dim → (T,C,N)

            losses.append(loss.item())

        # quick layer-0 stats (optional)
        if first_conv is not None:
            w_mean = first_conv.weight.abs().mean().item()
            g_mean = (first_conv.weight.grad.abs().mean().item()
                      if first_conv.weight.grad is not None else float('nan'))
            #print(f"[layer-0] ⟨|w|⟩={w_mean:.3e}  ⟨|∇w|⟩={g_mean:.3e}")

    if train:
        pass
        #print(f"LR={ [g['lr'] for g in optimizer.param_groups] }")

    # detach to CPU & squeeze batch dim ( [0] )
    return (float(np.mean(losses)),
            pred.detach().cpu()[0],          # (T,2,N_hex)
            flow.detach().cpu()[0],          # (T,2,N_hex)
            rgb_samp)                        # (T,C,N_hex)



# ───────────── full training / comparison ──────────────────────
def train_and_compare():
    train_loader = build_loader(BATCH_SIZE, augment=True)
    val_loader   = build_loader(BATCH_SIZE, augment=False)

    experiments = {
        "shared": (Network,             (LR_NET_SHARED,  LR_DEC_SHARED)),
        "custom": (build_custom_network,(LR_NET_CUSTOM, LR_DEC_CUSTOM)),
}

    for label, (builder, (lr_net,lr_dec)) in experiments.items():
        print(f"\n[ training {label.upper()} network ]")
        net  = builder().to(device)
        dec  = DecoderGAVP(net.connectome, shape=[8, 2], kernel_size=5).to(device)
        opt = Adam([
            {"params": net.parameters(), "lr": lr_net},
            {"params": dec.parameters(), "lr": lr_dec},
        ], eps=1e-8)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", 0.5, patience=3)

        t0 = time.perf_counter()
        for epoch in range(1, EPOCHS + 1):
            train_loss, _, _, _ = run_epoch(
                net, dec, train_loader, opt, epoch=epoch, label=label
            )
            val_loss, pred, gt, rgb = run_epoch(
                net, dec, val_loader, optimizer=None, epoch=epoch, label=label
            )

            save_debug_grid(epoch, rgb, pred, gt)  # writes only on epoch 1
            sched.step(val_loss)

            print(f"{label:<6}  ep {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        mins = (time.perf_counter() - t0) / 60
        print(f"[ {label.upper()} done ]  {mins:.1f} min total")

# ————————————————————————————————————————————————————————
if __name__ == "__main__":
    os.environ["FT3D_ROOT"] = str(FT3D_ROOT)
    torch.cuda.empty_cache()
    train_and_compare()
