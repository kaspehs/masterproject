import torch
import numpy as np

def _lhs_1d(n: int, low: torch.Tensor, high: torch.Tensor, device: torch.device, dtype: torch.dtype):
    """Latin Hypercube Sampling in 1D over [low, high]. Returns shape (n,)."""
    if n <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    # Stratified centers: (i + u) / n for i=0..n-1
    u = torch.rand(n, device=device, dtype=dtype)
    strata = (torch.arange(n, device=device, dtype=dtype) + u) / float(n)
    return low + strata * (high - low)

def sample_collocation(n_r: int,
                       n_bc: int,
                       n_ic: int,
                       x0: torch.Tensor,
                       L: torch.Tensor,
                       t_min: torch.Tensor,
                       t_max: torch.Tensor,
                       device: torch.device,
                       use_lhs: bool = False,
                       normalize_terms: bool = True):
    """
    Sample interior collocation points xb_r in [x0,L] x [t_min,t_max] and boundary-condition times t_bc.
    Returns:
      xb_r: (n_r, 2) tensor with columns [x, t]
      t_bc: (n_bc,) tensor of times for BC enforcement at x=0 and x=L
    """
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
    if use_lhs:
        # LHS for interior: pair a permuted t_r with x_r to avoid grid
        x_r = _lhs_1d(n_r, x0, L, device, x0.dtype)
        t_r = _lhs_1d(n_r, t_min, t_max, device, t_min.dtype)
        # Randomly permute one dimension to decorrelate
        if n_r > 0:
            perm = torch.randperm(n_r, device=device)
            t_r = t_r[perm]
        xb_r = torch.stack([x_r, t_r], dim=1) if n_r > 0 else torch.empty(0, 2, device=device, dtype=x0.dtype)
        # LHS for BC times and IC x-samples
        t_bc = _lhs_1d(n_bc, t_min, t_max, device, t_min.dtype)
        x_ic = _lhs_1d(n_ic, x0, L, device, x0.dtype)
    else:
        x_r = torch.rand(n_r, device=device, dtype=x0.dtype) * (L - x0) + x0
        t_r = torch.rand(n_r, device=device, dtype=t_min.dtype) * (t_max - t_min) + t_min
        xb_r = torch.stack([x_r, t_r], dim=1)
        t_bc = torch.rand(n_bc, device=device, dtype=t_min.dtype) * (t_max - t_min) + t_min
        x_ic = torch.rand(n_ic, device=device, dtype=x0.dtype) * (L - x0) + x0
    return xb_r, t_bc, x_ic