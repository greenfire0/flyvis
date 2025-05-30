import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from flyvis.network import Network
from flyvis.task.decoder import DecoderGAVP
from flyvis.datasets.sintel import MultiTaskSintel
from flyvis.task.objectives import epe  # end-point error

# ── 0. Select device ───────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("Using GPU:", torch.cuda.get_device_name())

# ── 1. Build vanilla network + decoder ─────────────────────────────────────────
network = Network().to(device)
decoder = DecoderGAVP(network.connectome, shape=[8, 2], kernel_size=5).to(device)

# ── 2. Training dataset (with augmentation) ────────────────────────────────────
train_dataset = MultiTaskSintel(
    tasks=["flow"],
    boxfilter=dict(extent=15, kernel_size=13),
    vertical_splits=3,
    n_frames=19,
    dt=1/50,
    augment=True,
    random_temporal_crop=True,
    p_flip=0.5, p_rot=5/6,
    contrast_std=0.2, brightness_std=0.1,
    gaussian_white_noise=0.08,
    resampling=True, interpolate=True,
)
gen = torch.Generator(device=device)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, generator=gen)

# ── 3. Validation dataset (no augmentation) ────────────────────────────────────
val_dataset = MultiTaskSintel(
    tasks=["flow"],
    boxfilter=dict(extent=15, kernel_size=13),
    vertical_splits=3,
    n_frames=19,
    dt=1/50,
    augment=False,
    resampling=True, interpolate=True,
)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# ── 4. Optimizer & loss ────────────────────────────────────────────────────────
optimizer = Adam((*network.parameters(), *decoder.parameters()), lr=1e-5)
loss_fn = epe
t_pre = 0.5
dt = train_dataset.dt
initial_state = network.steady_state(t_pre, dt, batch_size=4)

# ── 5. Training loop ───────────────────────────────────────────────────────────
errors = []
for epoch in range(10_000):
    lum, flow = next(iter(train_loader)).values()
    lum, flow = lum.to(device), flow.to(device)

    optimizer.zero_grad()
    network.stimulus.zero()
    network.stimulus.add_input(lum)

    activity = network(network.stimulus(), dt=dt, state=initial_state)
    y_pred = decoder(activity)
    loss = loss_fn(y_pred, flow)
    loss.backward()
    optimizer.step()

    errors.append(loss.item())

    # Every 100 epochs: print validation loss and smoothed training loss
    if epoch % 100 == 0 and epoch > 0:
        with torch.no_grad():
            val_batch = next(iter(val_loader))
            val_lum = val_batch["lum"].to(device)
            val_flow = val_batch["flow"].to(device)
            network.stimulus.zero()
            network.stimulus.add_input(val_lum)
            val_state = network.fade_in_state(1.0, dt, val_lum[:, 0])
            val_resp = network.simulate(val_lum, dt, initial_state=val_state)
            val_pred = decoder(val_resp)
            val_loss = loss_fn(val_pred, val_flow).item()

        avg_loss = sum(errors[-100:]) / 100
        print(f"── Epoch {epoch:05d} | avg train loss (last 100) = {avg_loss:.4f} | val loss = {val_loss:.4f}")
