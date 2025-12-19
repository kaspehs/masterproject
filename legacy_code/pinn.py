# Limit CPU threads for stability and to avoid BLAS oversubscription
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from legacy_code.helper_functions import *
from pinn_helper import *
from architectures import *

# Prefer Apple GPU (MPS) if available; use float32 for stability/speed
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
dtype = torch.float32
torch.set_default_dtype(dtype)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

#Dataset parameters
x_points, t_points = 100, 400 #Dimentions of dataset
input_size = 2   # number of features in your data
output_size = 1 #Output size

# Hardcoded domain constants (raw standardized coordinates)
X0_CONST, L_CONST, T_MIN_CONST, T_MAX_CONST = 0.0, 10.0, 0.0, 10.0

# TensorBoard run naming (set to a string to override timestamp folder)
LOG_RUN_NAME = None  # e.g., "pinn_exp1"; None uses timestamped default

#Architechture parameters
num_hidden_layers = 4 #Depth of network
hidden_size = 256  # number of hidden units
num_fourier_features = 32 #Embedding space
fourier_sigma = 4.0
n_r = 64 #Batchsize is n_r*num_chunks
num_chunks = 32 #Time chunks for causal training
n_bc = 64 #Number of colocation points for enforcing BCs
n_ic = 64 #Number of colocation points for enforcing BCs
use_rwf=True
rwf_mu = 1.0; rwf_sigma = 0.1
factorize_output=False

#Optimization parameters
total_steps = int(1e5)
grad_clip_max_norm = 200  # gradient clipping threshold (L2 norm)
causal_weight = 1.0
lambda_freq = 1000
grad_norm_alpha = 0.9

#Learning rate parameters
base_lr = 3e-4
decay_rate = 0.9
decay_steps = 3000
warmup_steps = 3000

#Logging parameters
steps_per_heatmap = 1000  # log heatmap figures to TensorBoard every N epochs (0 to disable)
log_every_n_steps = 100

def main():
    #Loading the data
    data = np.load('data_generation/data/data_kdv.npz')
    g_u = data['g_u']
    u_init = data['u'] #[x0->xn]
    xt = data['xt'] #[number of points][x, t]

    #Define the MLP model
    # SimpleMLP moved to pinn_helper.SimpleMLP

  
    # Physics-driven normalization:
    # Map x -> [-1, 1], t -> [0, 1], keep u in physical units (no scaling)
    L_range = float(L_CONST - X0_CONST)
    T_range = float(T_MAX_CONST - T_MIN_CONST)
    x_std = 2.0 * (xt[:, 0] - X0_CONST) / max(L_range, 1e-12) - 1.0
    t_std = (xt[:, 1] - T_MIN_CONST) / max(T_range, 1e-12)
    X_val = torch.as_tensor(np.stack([x_std, t_std], axis=1), dtype=dtype)
    y_val = torch.as_tensor(g_u[0].reshape(-1, 1), dtype=dtype)

    # Domain in standardized coordinates and IC in physical units
    x0 = torch.tensor(-1.0, dtype=dtype, device=device)
    L  = torch.tensor( 1.0, dtype=dtype, device=device)
    t_min = torch.tensor(0.0, dtype=dtype, device=device)
    t_max = torch.tensor(1.0, dtype=dtype, device=device)
    u0 = torch.as_tensor(u_init[0], dtype=dtype, device=device)

    # Chain-rule scales from std->physical: dx_phys/dx_std = L_range/2, dt_phys/dt_std = T_range, du scaling = 1
    x_scale = torch.tensor(L_range/2.0, dtype=dtype, device=device)
    t_scale = torch.tensor(T_range,      dtype=dtype, device=device)
    u_scale = torch.tensor(1.0,          dtype=dtype, device=device)

    model = SimpleMLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                      num_hidden_layers=num_hidden_layers, fourier_features=num_fourier_features, sigma=fourier_sigma,
                      use_rwf=use_rwf, rwf_mu = rwf_mu, rwf_sigma = rwf_sigma, factorize_output=factorize_output)
    # Ensure model is on selected device/dtype
    model.to(device=device, dtype=dtype)
   
    # Initialize with base LR; it will be updated per-epoch by lr_schedule
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    """
⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠛⢉⢉⠉⠉⠻⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⠟⠠⡰⣕⣗⣷⣧⣀⣅⠘⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⠃⣠⣳⣟⣿⣿⣷⣿⡿⣜⠄⣿⣿⣿⣿⣿
⣿⣿⣿⣿⡿⠁⠄⣳⢷⣿⣿⣿⣿⡿⣝⠖⠄⣿⣿⣿⣿⣿
⣿⣿⣿⣿⠃⠄⢢⡹⣿⢷⣯⢿⢷⡫⣗⠍⢰⣿⣿⣿⣿⣿
⣿⣿⣿⡏⢀⢄⠤⣁⠋⠿⣗⣟⡯⡏⢎⠁⢸⣿⣿⣿⣿⣿
⣿⣿⣿⠄⢔⢕⣯⣿⣿⡲⡤⡄⡤⠄⡀⢠⣿⣿⣿⣿⣿⣿
⣿⣿⠇⠠⡳⣯⣿⣿⣾⢵⣫⢎⢎⠆⢀⣿⣿⣿⣿⣿⣿⣿
⣿⣿⠄⢨⣫⣿⣿⡿⣿⣻⢎⡗⡕⡅⢸⣿⣿⣿⣿⣿⣿⣿
⣿⣿⠄⢜⢾⣾⣿⣿⣟⣗⢯⡪⡳⡀⢸⣿⣿⣿⣿⣿⣿⣿
⣿⣿⠄⢸⢽⣿⣷⣿⣻⡮⡧⡳⡱⡁⢸⣿⣿⣿⣿⣿⣿⣿
⣿⣿⡄⢨⣻⣽⣿⣟⣿⣞⣗⡽⡸⡐⢸⣿⣿⣿⣿⣿⣿⣿
⣿⣿⡇⢀⢗⣿⣿⣿⣿⡿⣞⡵⡣⣊⢸⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⡀⡣⣗⣿⣿⣿⣿⣯⡯⡺⣼⠎⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣧⠐⡵⣻⣟⣯⣿⣷⣟⣝⢞⡿⢹⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⡆⢘⡺⣽⢿⣻⣿⣗⡷⣹⢩⢃⢿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣷⠄⠪⣯⣟⣿⢯⣿⣻⣜⢎⢆⠜⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⡆⠄⢣⣻⣽⣿⣿⣟⣾⡮⡺⡸⠸⣿⣿⣿⣿
⣿⣿⡿⠛⠉⠁⠄⢕⡳⣽⡾⣿⢽⣯⡿⣮⢚⣅⠹⣿⣿⣿
⡿⠋⠄⠄⠄⠄⢀⠒⠝⣞⢿⡿⣿⣽⢿⡽⣧⣳⡅⠌⠻⣿
⠁⠄⠄⠄⠄⠄⠐⡐⠱⡱⣻⡻⣝⣮⣟⣿⣻⣟⣻⡺⣊
"""
    # TensorBoard writer
    import os as _os, time as _time
    run_dir = _os.path.join(
        "runs",
        LOG_RUN_NAME if LOG_RUN_NAME else f"pinn_kdv_{_time.strftime('%Y%m%d-%H%M%S')}"
    )
    writer = SummaryWriter(log_dir=run_dir)

    lr_scheduler = LrSchedule(base_lr, decay_rate, warmup_steps, decay_steps)

    stepper = TrainingStepper(model, x0, L, t_min, t_max, u0, x_scale, t_scale, u_scale,
                              optimizer, grad_clip_max_norm, n_r, n_bc, n_ic, num_chunks, causal_weight,
                              lambda_freq, log_every_n_steps, writer, x_points, lr_scheduler, grad_norm_alpha)

    stepper.define_validation_data(X_val, y_val)

    for steps in range(total_steps):

        model.train()

        stepper.step()

        if (stepper.steps % steps_per_heatmap) == 0:
            stepper.log_heatmap()
                
    # Close TensorBoard writer
    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()
