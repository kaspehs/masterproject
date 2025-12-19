import torch
import numpy as np
from torch import nn, autograd
from legacy_code.helper_functions import figure_compare_data

def d(outputs, inputs, retain_graph=True, create_graph=True):
    """First derivative helper that is safe for higher-order calls.

    If `outputs` does not require grad (can happen after vectorization or when
    a branch is constant), return a zeros-like tensor instead of calling
    autograd.grad, which would error.
    """
    if not isinstance(outputs, torch.Tensor) or not outputs.requires_grad:
        return torch.zeros_like(inputs)
    grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )[0]
    if grad is None:
        return torch.zeros_like(inputs)
    return grad

def d_t(u, t):   return d(u, t)
def d_x(u, x):   return d(u, x)
def d_xx(u, x):  return d(d_x(u, x), x)
def d_xxx(u, x): return d(d_xx(u, x), x)

def residual_kdv_loss(model,
                  # Time range for causal chunking in standardized coords
                  t_min,
                  t_max,
                  #A list of spatial domain collocation points for use in each chunk
                  x_r,
                  # scales for inputs and outputs for the PDE
                  x_scale, t_scale, u_scale,
                  #Number of time domain chunks
                  n_chunks: int = 16,
                  #How much the model has to learn the early time stages before the latter
                  causal_weight: float = 1.0,
                ):
    """
    Returns total_loss, dict_of_terms. All inputs are 1D tensors of same length per group.
    L is domain length in x for periodic BCs.
    """
    
    # Vectorized evaluation over all chunks in one forward/backward pass
    x_r_det = x_r.clone().detach()  # (n_chunks, n_r)
    device = x_r_det.device
    dtype = x_r_det.dtype

    # Build per-chunk time samples uniformly within each interval [t0, t1] over [t_min, t_max]
    idx = torch.arange(int(n_chunks), device=device, dtype=dtype)
    Lt = (t_max - t_min) if torch.is_tensor(t_max) else torch.tensor(float(t_max - t_min), device=device, dtype=dtype)
    t0 = t_min + (idx / float(n_chunks)) * Lt
    t1 = t_min + ((idx + 1) / float(n_chunks)) * Lt
    t_r = torch.rand_like(x_r_det) * (t1[:, None] - t0[:, None]) + t0[:, None]  # (n_chunks, n_r)

    # Stack inputs and run model once
    xb_all = torch.stack([x_r_det, t_r], dim=2).reshape(-1, 2).clone().detach().requires_grad_(True)
    pred_all = model(xb_all).squeeze(-1)

    # Derivatives (vectorized)
    grads = d(pred_all, xb_all)
    ux, ut = grads[:, 0], grads[:, 1]
    uxx = d(ux, xb_all)[:, 0]
    uxxx = d(uxx, xb_all)[:, 0]

    # Coefficients (scalars)
    c2s = (6.0 * u_scale * t_scale / x_scale)
    c3 = (t_scale / (x_scale**3))

    # Residual per sample, then mean per chunk
    res_raw = ut + c2s * (pred_all * ux) + c3 * uxxx
    res2 = res_raw.pow(2).view(int(n_chunks), -1).mean(dim=1)  # (n_chunks,)

    # Causal weighting applied sequentially but without extra model calls
    chunk_losses: list[torch.Tensor] = []
    residual_loss = xb_all.new_zeros(())
    wi_log = [0.0] * int(n_chunks)
    processed = 0
    for M in range(int(n_chunks)):
        prev = (torch.stack(chunk_losses).sum().detach() if chunk_losses else xb_all.new_zeros(())).detach()
        wi = torch.exp(-float(causal_weight) * prev).detach()
        if (M >= 4) and (float(wi) < 1e-3):
            break
        wi_log[M] = float(wi)
        loss_m = res2[M]
        residual_loss = residual_loss + wi * loss_m
        chunk_losses.append(loss_m.detach())
        processed += 1

    denom = float(processed if processed > 0 else 1)
    residual_loss = residual_loss / denom

    return residual_loss, {
        "residual": residual_loss.item(),
        "wi": wi_log,
        "kept": processed,
    }

def init_condition_loss(model,
        # IC ground truth, IC collocatin points at t=O
                  u0, x_ic, t0, x0, L
            ):
    
    #Ensures gradients are possible
    x_ic = x_ic.clone().detach().requires_grad_(True)

    # ---- Initial condition at t=t0 with interpolation from u0(x) ----
    # Build evaluation inputs at IC time
    t0_val = float(t0.item())
    t0v = torch.full_like(x_ic, fill_value=t0_val)
    xbIC = torch.stack([x_ic, t0v], dim=1).clone().detach().requires_grad_(True)

    # Interpolate ground-truth u0 at x_ic along standardized x-grid [x0, L]
    x_grid = torch.linspace(x0, L, steps=u0.numel(), device=xbIC.device, dtype=xbIC.dtype)
    def _interp1d_sorted(xg, yg, xq, eps: float = 1e-12):
        idx = torch.searchsorted(xg, xq, right=False)
        idx = idx.clamp(min=1, max=xg.numel()-1)
        x0i = xg[idx-1]; x1i = xg[idx]
        y0i = yg[idx-1]; y1i = yg[idx]
        w = (xq - x0i) / (x1i - x0i + eps)
        return y0i + w * (y1i - y0i)
    u0_interp = _interp1d_sorted(x_grid, u0.view(-1), x_ic)

    uIC = model(xbIC).squeeze(-1)
    ic_loss = torch.mean((uIC - u0_interp)**2)

    return ic_loss, {
        "ic": ic_loss.item()
    }

def periodic_bc_loss(model,
        t_bc, x0, L,
):
    
    t_bc = t_bc.clone().detach().requires_grad_(True)

    # ---- Periodic boundary conditions at x=0 and x=L ----
    # u(t,0) == u(t,L), ux(t,0) == ux(t,L), uxx(t,0) == uxx(t,L) (smooth periodicity)
    x0_val = float(x0.item())
    L_val  = float(L.item())
    x0v = torch.full_like(t_bc, fill_value=x0_val)
    xLv = torch.full_like(t_bc, fill_value=L_val)

    xb0 = torch.stack([x0v, t_bc], dim=1).clone().detach().requires_grad_(True)
    xbL = torch.stack([xLv, t_bc], dim=1).clone().detach().requires_grad_(True)

    u_0 = model(xb0).squeeze(-1)
    u_L = model(xbL).squeeze(-1)

    g0 = d(u_0, xb0)
    gL = d(u_L, xbL)
    ux_0 = g0[:, 0]
    ux_L = gL[:, 0]
    g0_2 = d(ux_0, xb0)
    gL_2 = d(ux_L, xbL)
    uxx_0 = g0_2[:, 0]
    uxx_L = gL_2[:, 0]

    bc_loss = torch.mean((u_0 - u_L)**2) + \
              torch.mean((ux_0 - ux_L)**2) + \
              torch.mean((uxx_0 - uxx_L)**2)
    
    return bc_loss, {
        "bc": bc_loss.item()
    }

def mse_data_loss(model, xb, yb):
    """MSE on provided data. Accepts numpy arrays and moves to model's device/dtype."""
    dev = next(model.parameters()).device
    dtp = next(model.parameters()).dtype
    if not isinstance(xb, torch.Tensor):
        xb = torch.as_tensor(xb, dtype=dtp, device=dev)
    else:
        xb = xb.to(device=dev, dtype=dtp)
    if not isinstance(yb, torch.Tensor):
        yb = torch.as_tensor(yb, dtype=dtp, device=dev)
    else:
        yb = yb.to(device=dev, dtype=dtp)

    pred = model(xb).squeeze(-1)
    target = yb.squeeze(-1)
    data_loss = torch.mean((pred - target)**2)

    return data_loss, {
        "data": data_loss.item(),
    }

class LrSchedule():
    def __init__(self,max_lr, decay_rate, warmup_steps, decay_steps):
        self.max_lr = max_lr; self.decay_rate = decay_rate; self.warmup_steps = warmup_steps; self.decay_steps = decay_steps

    def get_lr(self, step):
        if step <= self.warmup_steps:
            return self.max_lr * step/self.warmup_steps
        return self.max_lr * (self.decay_rate)**((step-self.warmup_steps)/self.decay_steps) 

def sample_collocation(n_r: int,
                       n_bc: int,
                       n_ic: int,
                       num_chunks: int,
                       x0: torch.Tensor,
                       L: torch.Tensor,
                       t_min: torch.Tensor,
                       t_max: torch.Tensor,
                       device: torch.device,
                       ):
    """
    Sample interior collocation points xb_r in [x0,L] x [t_min,t_max] and boundary-condition times t_bc.
    Returns:
      x_r: (n_chunks, n_r) tensor with random x values
      t_bc: (n_bc,) tensor of times for BC enforcement at x=0 and x=L
    """
    #Build (num_chunks, n_r) by stacking generated tensors; avoid torch.Tensor(list_of_tensors)
    x_r_list = [torch.rand(n_r, device=device, dtype=x0.dtype) * (L - x0) + x0 for _ in range(num_chunks)]
    x_r = torch.stack(x_r_list, dim=0)
    t_bc = torch.rand(n_bc, device=device, dtype=t_min.dtype) * (t_max - t_min) + t_min
    x_ic = torch.rand(n_ic, device=device, dtype=x0.dtype) * (L - x0) + x0
    return x_r, t_bc, x_ic

class TrainingStepper:
    def __init__(self, model, #The NN model
                 x0, L, t_min, t_max, u0, #Dataset spesific parameters
                 x_scale, t_scale, u_scale, #Scaler spesific parameters
                 optimizer, grad_clip_max_norm, #Optimizer spesific parameters
                 n_r, n_bc, n_ic, num_chunks, causal_weight, lambda_freq, #Training spesific parameters
                 log_every_n_steps, writer, x_points, lr_scheduler, alpha): #Logging spesific parameters
        self.model = model
        self.x0 = x0; self.L = L; self.t_min = t_min; self.t_max = t_max; self.u0 = u0
        self.x_scale = x_scale; self.t_scale = t_scale; self.u_scale = u_scale
        self.optimizer = optimizer; self.grad_clip_max_norm = grad_clip_max_norm
        self.n_r = n_r; self.n_bc = n_bc; self.n_ic = n_ic; self.num_chunks = num_chunks
        self.causal_weight = causal_weight; self.lamda_freq = lambda_freq
        self.log_every_n_steps = log_every_n_steps; self.writer = writer; self.x_points = x_points
        self.lr_scheduler = lr_scheduler; self.alpha = alpha

        self.device = next(model.parameters()).device
        self.steps = 0
        #self.lam = {'ic': 1.0, 'bc': 1.0, 'res': 1.0}
        self.lam = {'ic': 1.0, 'res': 1.0}
        self.balancer = GradNormBalancer(self.model, self.lam.keys(), alpha = self.alpha)

        self.lr = 0.0

        self.xb = None; self.yb = None
        self.y_min = None; self.y_max = None

    def define_validation_data(self, xb, yb):
        self.xb = xb; self.yb = yb
        self.y_min = torch.min(self.yb); self.y_max = torch.max(self.yb)

    @torch.no_grad()
    def _validate(self):
        data_loss, terms = mse_data_loss(self.model, self.xb, self.yb)
        return data_loss

    def _log(self,
                      train: dict,
                      val_data_loss: float,
                      grad_norm_mean: float | None = None):
        self.writer.add_scalar('lr', self.lr, self.steps)
        self.writer.add_scalar('loss/train/total', train.get('total', float('nan')), self.steps)
        self.writer.add_scalar('loss/train/res', train.get('res', float('nan')), self.steps)
        self.writer.add_scalar('loss/train/ic', train.get('ic', float('nan')), self.steps)
        self.writer.add_scalar('loss/train/bc', train.get('bc', float('nan')), self.steps)
        self.writer.add_scalar('loss/val/data', val_data_loss, self.steps)

        if grad_norm_mean is not None:
            self.writer.add_scalar('grad/total_norm_mean', grad_norm_mean, self.steps)

        # Log current lambda weights for losses (res, bc, ic)
        try:
            for k, v in self.lam.items():
                lam_f = float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
                self.writer.add_scalar(f'lambda/{k}', lam_f, self.steps)
        except Exception:
            pass

        # Log per-block alpha parameters if present (e.g., PirateNet residual blocks)
        try:
            blks = getattr(self.model, 'blocks', None)
            if blks is not None:
                for i, b in enumerate(blks, start=1):
                    a = getattr(b, 'alpha', None)
                    if a is not None:
                        try:
                            self.writer.add_scalar(f'blocks/alpha_{i}', float(a.detach().cpu()), self.steps)
                        except Exception:
                            pass
        except Exception:
            pass

        print(
        f"Epoch {self.steps+1:02d} "
        f"train(total={train.get('total', float('nan')):.3e}, res={train.get('res', float('nan')):.3e}, bc={train.get('bc', float('nan')):.3e}), ic={train.get('ic', float('nan')):.3e} "
        f"val(data={val_data_loss:.3e}) "
        f"[lr={self.lr:.3e}]"
            )

    def step(self):
        self.steps += 1

        #Updates learningrate
        for g in self.optimizer.param_groups:
            self.lr = self.lr_scheduler.get_lr(self.steps)
            g["lr"] = self.lr

        # Sample interior collocation points and IC, BC points
        x_r, t_bc, x_ic = sample_collocation(
            n_r=self.n_r,
            n_bc=self.n_bc,
            n_ic=self.n_ic,
            num_chunks=self.num_chunks,
            x0=self.x0, L=self.L, t_min=self.t_min, t_max=self.t_max,
            device=self.device,
        )

        #Calculate losses
        residual_loss, terms1 = residual_kdv_loss(self.model, self.t_min, self.t_max, x_r, self.x_scale, self.t_scale, self.u_scale, self.num_chunks, self.causal_weight)
        #bc_loss, terms2 = periodic_bc_loss(self.model, t_bc, self.x0, self.L)
        bc_loss, terms2 = torch.tensor(0.0), None
        ic_loss, terms3 = init_condition_loss(self.model, self.u0, x_ic, self.t_min, self.x0, self.L)
        # Freeze λ weights wrt θ
        lam_res = self.lam['res'].detach() if torch.is_tensor(self.lam['res']) else torch.tensor(float(self.lam['res']), device=self.device)
        #lam_bc  = self.lam['bc' ].detach() if torch.is_tensor(self.lam['bc' ]) else torch.tensor(float(self.lam['bc' ]), device=self.device)
        lam_ic  = self.lam['ic' ].detach() if torch.is_tensor(self.lam['ic' ]) else torch.tensor(float(self.lam['ic' ]), device=self.device)
        #loss = residual_loss * lam_res + bc_loss * lam_bc + ic_loss * lam_ic
        loss = residual_loss * lam_res + ic_loss * lam_ic
        
        #Updates loss weighting every lambda_freq steps
        if (self.steps % self.lamda_freq) == 0:
            #self.lam = self.balancer.update({'ic':ic_loss, 'bc': bc_loss, 'res': residual_loss})
            self.lam = self.balancer.update({'ic':ic_loss, 'res': residual_loss})

        #Backpropagation
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
        self.optimizer.step()

        #Logs if it is time for it
        if (self.steps % self.log_every_n_steps) == 0:
            # Log causal weights per chunk (detached floats)
            wi = terms1.get('wi') if isinstance(terms1, dict) else None
            if wi is not None:
                try:
                    kept = int(terms1.get('kept', 0))
                except Exception:
                    kept = 0
                for m, w in enumerate(wi):
                    try:
                        self.writer.add_scalar(f'causal/w_chunk_{m:02d}', float(w), self.steps)
                    except Exception:
                        pass
                # Compact bar figure
                try:
                    import numpy as _np, matplotlib.pyplot as _plt
                    _wi = _np.asarray(wi, dtype=_np.float32)
                    fig, ax = _plt.subplots(figsize=(6, 2))
                    ax.bar(_np.arange(_wi.size), _wi, width=0.9)
                    ax.set_ylim(0.0, 1.0)
                    ax.set_xlabel('chunk')
                    ax.set_ylabel('w')
                    self.writer.add_figure('causal/weights_bar', fig, self.steps)
                    _plt.close(fig)
                except Exception:
                    pass
                # Summary counters
                try:
                    self.writer.add_scalar('causal/kept_count', kept, self.steps)
                    self.writer.add_scalar('causal/sum_w', float(sum(float(x) for x in wi)), self.steps)
                except Exception:
                    pass

            self._log({'total': loss.item(), 'res': residual_loss.item(), 'ic': ic_loss.item(), 'bc': bc_loss.item()},
                      self._validate().item(), total_grad_norm.item())

    def log_heatmap(self):
        self.model.eval()
        # Visualize predictions vs truth over the ENTIRE dataset (train+val+test)
        xb = self.xb.to(self.device)
        with torch.no_grad():
            preds = self.model(xb).detach().cpu().numpy()
        truth = self.yb.detach().cpu().numpy()
        fig = figure_compare_data(preds, truth, self.x_points, self.y_min, self.y_max, title_prefix="All[train+val+test]")
        self.writer.add_figure('qual/full_pred_truth', fig, self.steps)
        import matplotlib.pyplot as _plt
        _plt.close(fig)

class GradNormBalancer:
    """
    Maintains global weights for multiple loss terms based on gradient norms.
    Updates every `freq` steps using EMA with coefficient `alpha`.
    """
    def __init__(self, model: nn.Module, names=("ic","bc","res"), alpha=0.9, eps=1e-8, device=None):
        self.model  = model
        self.names  = tuple(names)
        self.alpha  = alpha
        self.eps    = eps
        self.w_min  = 0.1
        self.w_max  = 10
        self.device = device or next(model.parameters()).device

        # EMA of grad norms; init to 1 for neutrality
        self.g_ema  = {n: torch.tensor(1.0, device=self.device) for n in self.names}
        # Public weights (mean ~ 1 at start)
        self.w      = {n: torch.tensor(1.0, device=self.device) for n in self.names}
        self.step   = 0
    
    def _global_grad_norm(self, loss):
        params = [p for p in self.model.parameters() if p.requires_grad]
        grads  = torch.autograd.grad(loss, params, retain_graph=True,
                                     create_graph=False, allow_unused=True)
        sq = None
        for g in grads:
            if g is not None:
                # ignore non-finite entries (rare but safer)
                g = g.detach()
                if not torch.isfinite(g).all():
                    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                v = (g*g).sum()
                sq = v if sq is None else (sq + v)
        if sq is None:
            return torch.tensor(0.0, device=self.device)
        return torch.sqrt(sq + self.eps)

    @torch.no_grad()
    def update(self, losses: dict):
        # 1) measure pre-clip grad norms for each term
        g = {n: self._global_grad_norm(losses[n]) for n in self.names}

        # 2) EMA on norms
        for n in self.names:
            self.g_ema[n] = self.alpha*self.g_ema[n] + (1 - self.alpha)*torch.clamp(g[n], min=self.eps)

        # 3) inverse-proportional proposal
        inv = {n: 1.0 / torch.clamp(self.g_ema[n], min=self.eps) for n in self.names}

        # 4) normalize so mean weight = 1
        s = sum(inv.values())
        k = float(len(self.names))
        w_hat = {n: (inv[n] * (k / s)) for n in self.names}

        # 5) clamp to [w_min, w_max]
        for n in self.names:
            self.w[n] = torch.clamp(w_hat[n], self.w_min, self.w_max)
        
        return {n: self.w[n].detach() for n in self.names}

def physics_init(model, X, u0, ridge=1e-6, add_bias=True, return_diagnostics=False):
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype
    X = torch.as_tensor(X, dtype=dtype, device=device)
    u0 = torch.as_tensor(u0, dtype=dtype, device=device)
    X = X.to(device=device, dtype=dtype)

    # Build y from u0 by repeating across time-major blocks,
    # or by 1D interpolation on x if shapes don't divide evenly.
    if u0 is None:
        raise ValueError("physics_init: either y or u0 must be provided")
    u0 = u0.to(device=device, dtype=dtype).view(-1)
    N = X.size(0)
    M = u0.numel()
    if M > 0 and (N % M) == 0:
        y = u0.repeat(N // M).unsqueeze(1)
    else:
        x_samples = X[:, 0]
        x_grid = torch.linspace(x_samples.min(), x_samples.max(), steps=M, device=device, dtype=dtype)
        idx = torch.searchsorted(x_grid, x_samples, right=False).clamp_(1, M - 1)
        xL = x_grid[idx - 1]; xR = x_grid[idx]
        yL = u0[idx - 1];     yR = u0[idx]
        w = (x_samples - xL) / (xR - xL + torch.finfo(dtype).eps)
        y = (yL + w * (yR - yL)).unsqueeze(1)

    H = model.features(X)  # (N, hidden) — should be linear features at t=0
    Phi = torch.cat([H, torch.ones(H.size(0), 1, device=device, dtype=dtype)], dim=1) if add_bias else H

    # Use normal equations with ridge (supported on MPS):
    A = Phi.T @ Phi
    if ridge and float(ridge) > 0.0:
        A = A + ridge * torch.eye(A.size(0), device=device, dtype=dtype)
    b = Phi.T @ y
    theta = torch.linalg.solve(A, b)  # (hidden[+1], out)

    # Load into head
    if add_bias:
        model.out.weight.data.copy_(theta[:-1, :].mT)
        model.out.bias.data.copy_(theta[-1, :])
    else:
        model.out.weight.data.copy_(theta.mT)
        torch.nn.init.zeros_(model.out.bias)

    if not return_diagnostics:
        return theta

    # Diagnostics
    y_hat = Phi @ theta                      # (N, out)
    err   = y_hat - y
    mse   = (err.pow(2).mean(dim=0))         # per-output
    rmse  = mse.sqrt()
    rel_l2 = (err.norm(dim=0) / (y.norm(dim=0).clamp_min(1e-12)))
    max_abs = err.abs().max(dim=0).values
    # Coeff magnitudes
    theta_l2 = theta.norm(dim=0)             # per-output
    theta_linf = theta.abs().max(dim=0).values
    # Conditioning
    # Approximate conditioning via eigenvalues of A = Phi^T Phi
    try:
        evals = torch.linalg.eigvalsh(A)
        cond = (evals.max() / evals.min().clamp_min(1e-15)).item()
    except Exception:
        cond = float('nan')

    return {
        "theta": theta,
        "rmse": rmse,                        # tensor of size (out,)
        "rel_l2": rel_l2,                    # tensor of size (out,)
        "max_abs": max_abs,                  # tensor of size (out,)
        "theta_l2": theta_l2,
        "theta_linf": theta_linf,
        "cond_Phi": cond,
        "y_hat_sample": y_hat[:10].detach(), # small peek
    }

def pirate_log_epoch_scalars(writer,
                      epoch: int,
                      train: dict,
                      val: dict,
                      weights: tuple,
                      lr: float,
                      blocks: int,
                      alphas, 
                      grad_norm_mean: float | None = None):
    w_data, w_pde, w_bc = weights
    writer.add_scalar('lr', lr, epoch)
    writer.add_scalar('weights/w_data', w_data, epoch)
    writer.add_scalar('weights/w_pde', w_pde, epoch)
    writer.add_scalar('weights/w_bc', w_bc, epoch)
    writer.add_scalar('loss/train/total', train.get('total', float('nan')), epoch)
    writer.add_scalar('loss/train/data', train.get('data', float('nan')), epoch)
    writer.add_scalar('loss/train/pde', train.get('pde', float('nan')), epoch)
    writer.add_scalar('loss/train/bc', train.get('bc', float('nan')), epoch)
    writer.add_scalar('loss/val/total', val.get('total', float('nan')), epoch)
    writer.add_scalar('loss/val/data', val.get('data', float('nan')), epoch)
    writer.add_scalar('loss/val/pde', val.get('pde', float('nan')), epoch)
    writer.add_scalar('loss/val/bc', val.get('bc', float('nan')), epoch)
    # Log per-block alpha values if provided
    if alphas is not None:
        for i in range(blocks):
            try:
                a = float(alphas[i])
            except Exception:
                # fall back if alphas is a tensor/list with different indexing
                a = float(alphas[i].detach().cpu()) if hasattr(alphas[i], 'detach') else None
            if a is not None:
                writer.add_scalar(f'blocks/alpha_{i+1}', a, epoch)
    if grad_norm_mean is not None:
        writer.add_scalar('grad/total_norm_mean', grad_norm_mean, epoch)
