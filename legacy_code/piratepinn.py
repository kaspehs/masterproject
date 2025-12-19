import os as _os
# Limit CPU threads (useful if running on CPU)
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
x_points, t_points = 100, 200 #Dimentions of dataset
input_size = 2   # number of features in your data
output_size = 1 #Output size

# Hardcoded domain constants (raw standardized coordinates)
X0_CONST, L_CONST, T_MIN_CONST, T_MAX_CONST = 0.0, 10.0, 0.0, 10.0

# TensorBoard run naming (set to a string to override timestamp folder)
LOG_RUN_NAME = None  # e.g., "pinn_exp1"; None uses timestamped default

#Architechture parameters
num_blocks = 1 #Depth of networks
hidden_size = 128  # number of hidden units
fourier_features = 128
sigma = 2.0
num_fourier_features_x = 16  # x features
num_fourier_features_t = 16  # t features
fourier_sigma_x = 4.0
fourier_sigma_t = 2.0
# Sine embedding for PirateNet's U-branch
USE_SINE_EMBED = False
W0_EMBED = 5.0
n_r = 128 #Batchsize is n_r*num_chunks
num_chunks = 16 #Time chunks for causal training
n_bc = 1 #Number of colocation points for enforcing BCs
n_ic = 512 #Number of colocation points for enforcing IC
use_rwf=True
rwf_mu = 1.0; rwf_sigma = 0.1
factorize_output=False

#Optimization parameters
total_steps = int(1e5)
grad_clip_max_norm = 1e5  # gradient clipping threshold (L2 norm)
causal_weight = 1.0
lambda_freq = 1000
grad_norm_alpha = 0.9
# GradNorm clamp limits (min/max lambda); configurable
LAMBDA_MIN = 0.3
LAMBDA_MAX = 3.0

#Learning rate parameters
base_lr = 1e-3
decay_rate = 0.9
decay_steps = 2000
warmup_steps = 3000

#Logging parameters
steps_per_heatmap = 1000  # log heatmap figures to TensorBoard every N epochs (0 to disable)
log_every_n_steps = 100

def main():
    #Loading the data
    data = np.load('data_generation/data/data_kdv.npz')
    g_u = data['g_u']
    u_init = data['u'] #[x0->xn]
    xt = data['xt'] #[number of points][x, t] [[x0->xn, t0], [x0->xn, t1]]
    print(xt)
    print(u_init)
    # Physics-driven normalization for inputs: x in [-1,1], t in [-1,1]; keep u unscaled
    L_range = float(L_CONST - X0_CONST)
    T_range = float(T_MAX_CONST - T_MIN_CONST)
    x_std = 2.0 * (xt[:, 0] - X0_CONST) / max(L_range, 1e-12) - 1.0
    t_std = 2.0 * (xt[:, 1] - T_MIN_CONST) / max(T_range, 1e-12) - 1.0
    X_val = np.stack([x_std, t_std], axis=1)
    y_val = g_u[0].reshape(-1, 1)

    # Domain in standardized coordinates and IC in physical units
    x0 = torch.tensor(-1.0, dtype=dtype, device=device)
    L  = torch.tensor( 1.0, dtype=dtype, device=device)
    t_min = torch.tensor(-1.0, dtype=dtype, device=device)
    t_max = torch.tensor(1.0, dtype=dtype, device=device)
    u0 = torch.as_tensor(u_init[0], dtype=dtype, device=device)

    # Instantiate Simple MLP on [x, t]
    model = PirateNet(input_size=input_size,
                      output_size=output_size,
                      depth=num_blocks,
                      fourier_features=fourier_features,
                      sigma = sigma,  
                      x_features=num_fourier_features_x,
                      t_features=num_fourier_features_t,
                      sigma_x=fourier_sigma_x,
                      sigma_t=fourier_sigma_t,
                      periodic_x=True,
                      x_period_L=1.0,
                      dtype=dtype,
                      use_rwf=use_rwf, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma,
                      factorize_output=factorize_output,
                      use_sine_embed=USE_SINE_EMBED,
                      w0_embed=W0_EMBED).to(device=device, dtype=dtype)

    # Prepare u_mean (we keep u in physical units here) and chain-rule scales
    x_scale = torch.tensor(L_range/2.0, dtype=dtype, device=device)
    t_scale = torch.tensor(T_range/2.0,  dtype=dtype, device=device)
    u_scale = torch.tensor(1.0,          dtype=dtype, device=device)

    # TensorBoard writer
    import os as _os, time as _time
    run_dir = _os.path.join(
        "runs",
        LOG_RUN_NAME if LOG_RUN_NAME else f"pinn_kdv_{_time.strftime('%Y%m%d-%H%M%S')}"
    )
    writer = SummaryWriter(log_dir=run_dir)


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
    # Initialize the last layer to map features to IC across all times
    stats = model.physics_init(X_val, u0 = u0, add_bias=True, return_diagnostics=True)

    print('PirateNet Physics Initialization:')
    print("RMSE:", stats["rmse"].cpu().numpy())
    print("rel L2:", stats["rel_l2"].cpu().numpy())
    print("max |err|:", stats["max_abs"].cpu().numpy())
    print("||theta||_2:", stats["theta_l2"].cpu().numpy())

    # Stepper-based training like pinn.py
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    lr_scheduler = LrSchedule(base_lr, decay_rate, warmup_steps, decay_steps)
    stepper = TrainingStepper(model, x0, L, t_min, t_max, u0,
                              x_scale, t_scale, u_scale,
                              optimizer, grad_clip_max_norm,
                              n_r, n_bc, n_ic, num_chunks, causal_weight,
                              lambda_freq, log_every_n_steps, writer, x_points,
                              lr_scheduler, grad_norm_alpha)
    
    # Apply GradNorm clamp limits from config
    try:
        stepper.balancer.w_min = float(LAMBDA_MIN)
        stepper.balancer.w_max = float(LAMBDA_MAX)
    except Exception:
        pass
    
    stepper.define_validation_data(torch.as_tensor(X_val, dtype=dtype), torch.as_tensor(y_val, dtype=dtype))

    # Initial heatmap
    stepper.log_heatmap()

    for _ in range(total_steps):
        model.train()
        stepper.step()
        if (stepper.steps % steps_per_heatmap) == 0:
            stepper.log_heatmap()

    # Close TensorBoard writer
    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()
