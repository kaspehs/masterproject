import os as _os
# Limit CPU threads for stability and to avoid BLAS oversubscription
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from legacy_code.helper_functions import *
from ODE_pinn_helper import *
from architectures import BasicMLP, SirenMLP
from legacy_code.ODE import build_ode_setup, print_ode_summary

# Prefer Apple GPU (MPS) if available; use float32 for stability/speed
#device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = torch.device("cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Dataset parameters
t_points = 200  # number of time samples for plotting/reference
input_size = 1
output_size = 2

# Plot limits (normalized displacement y/D and wake variable q)
Y_PLOT_LIMIT = 2.0
Q_PLOT_LIMIT = 2.0

# Load shared ODE configuration
ode_setup = build_ode_setup()
cfg = ode_setup["config"]
ODE_params = TrainableODEParams(
    ode_setup["ode_params"],
    dtype=dtype,
    device=device,
)
u_ic = ode_setup["u_ic"]
T_MIN_CONST = ode_setup["T_MIN_CONST"]
T_MAX_CONST = ode_setup["T_MAX_CONST"]
D = cfg.D
print_ode_summary(ode_setup)

# Architecture parameters for the baseline PINN
num_hidden_layers = 4
hidden_size = 128
USE_RFF = True
FOURIER_FEATURES = 64
FOURIER_SIGMA = 30.0
USE_SIREN = True
SIREN_W0 = 30.0
SIREN_W0_HIDDEN = 1.0
num_chunks = 64
n_per_chunk = 32
chunk_dirichlet_alpha = 5.0  # Dirichlet concentration for random chunk lengths
chunk_length_min_frac = 0.5  # Minimum chunk length relative to uniform average
chunk_length_max_frac = 1.5  # Maximum chunk length relative to uniform average

# Basic fully-connected network (no Fourier features / RWF)
# Optimisation / training parameters
total_steps = int(5e4)
grad_clip_max_norm = 1e5
causal_weight = 5.0
lambda_freq = 1000
grad_norm_alpha = 0.9

base_lr = 1e-3
decay_rate = 0.9
decay_steps = 2000
warmup_steps = 3000

steps_per_log_plot = 1000
log_every_n_steps = 100
dirac_val_points = 1024  # Number of deterministic Dirac validation samples

def main():
    # Prepare plotting grid (only used for visualisation)
    T_range = float(T_MAX_CONST - T_MIN_CONST)
    t = np.linspace(T_MIN_CONST, T_MAX_CONST, t_points)
    t_std = 2.0 * (t - T_MIN_CONST) / max(T_range, 1e-12) - 1.0
    _ = t_std  # currently unused but kept for consistency

    # Normalised time bounds
    t_min = torch.tensor(-1.0, dtype=dtype, device=device)
    t_max = torch.tensor(1.0, dtype=dtype, device=device)
    t_scale = torch.tensor(T_range / 2.0, dtype=dtype, device=device)

    # Instantiate baseline PINN (Basic MLP or SIREN)
    if USE_SIREN:
        model = SirenMLP(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_hidden_layers=num_hidden_layers,
            w0=SIREN_W0,
            w0_hidden=SIREN_W0_HIDDEN,
        ).to(device=device, dtype=dtype)
    else:
        model = BasicMLP(
            input_dim=input_size,
            hidden_dim=hidden_size,
            output_dim=output_size,
            depth=num_hidden_layers,
            fourier_features=FOURIER_FEATURES if USE_RFF else 0,
            sigma=FOURIER_SIGMA,
            dtype=dtype,
        ).to(device=device, dtype=dtype)

    optimizer = optim.Adam(
        list(model.parameters()) + list(ODE_params.parameters()),
        lr=base_lr,
    )
    lr_scheduler = LrSchedule(base_lr, decay_rate, warmup_steps, decay_steps)

    import time as _time
    run_dir = _os.path.join("runs", f"pinn_ode_{_time.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir=run_dir)

    stepper = ODETrainingStepper(
        model,
        t_min,
        t_max,
        u_ic,
        ODE_params,
        t_scale,
        optimizer,
        grad_clip_max_norm,
        n_per_chunk,
        num_chunks,
        causal_weight,
        lambda_freq,
        log_every_n_steps,
        writer,
        lr_scheduler,
        grad_norm_alpha,
        chunk_alpha=chunk_dirichlet_alpha,
        chunk_min_frac=chunk_length_min_frac,
        chunk_max_frac=chunk_length_max_frac,
        dirac_val_points=dirac_val_points,
        diameter=D,
        y_plot_limit=Y_PLOT_LIMIT,
        q_plot_limit=Q_PLOT_LIMIT,
        vPINN = True
    )

    # Initial diagnostic plot
    stepper.log_yq_curves(title="y/q prediction at step 0")

    for _ in range(total_steps):
        model.train()
        stepper.step()
        if (stepper.steps % steps_per_log_plot) == 0:
            stepper.log_yq_curves()

    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()
