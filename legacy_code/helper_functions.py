import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.colors as colors

def plot_data(g_u, x_points, t_points = None):
    data = np.reshape(g_u, (t_points, x_points))
    data = data.transpose() #Makes
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.colorbar(label="Value")  # adds a colorbar
    plt.title("Heatmap from 2D array")
    plt.show()

def plot_compare_data(y_pred, y_truth, x_points, t_points = None):

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True, constrained_layout = True)

    y_pred = np.reshape(y_pred, (-1, x_points))
    y_pred = y_pred.transpose() #Makes
    y_truth = np.reshape(y_truth, (-1, x_points))
    y_truth = y_truth.transpose() #Makes
    y_diff = abs(y_pred-y_truth)

    # shared scale for A & B
    vmin_ab = min(y_truth.min(), y_pred.min())
    vmax_ab = max(y_truth.max(), y_pred.max())
    norm = colors.Normalize(vmin=vmin_ab, vmax=vmax_ab)

    # independent scale for C
    norm_diff = colors.Normalize(vmin=0.0, vmax=y_diff.max())

    im0 = axes[0].imshow(y_pred, cmap='viridis', norm=norm, aspect='auto', origin='lower')
    axes[0].set_title("Predicted")
    im1 = axes[1].imshow(y_truth, cmap='viridis', norm=norm, aspect='auto', origin='lower')
    axes[1].set_title("Truth")
    im2 = axes[2].imshow(y_diff, cmap='RdBu_r', norm=norm_diff, aspect='auto', origin='lower')
    axes[2].set_title("Difference")

    axes[-1].set_xlabel("t index")
    for ax in axes: ax.set_ylabel("x index")

    # one colorbar for A & B (shared)
    fig.colorbar(im0, ax=axes[:2], orientation='vertical', fraction=0.025, pad=0.06, label="Predicted/Truth value")

    # separate colorbar for C
    fig.colorbar(im2, ax=axes[2],  orientation='vertical', fraction=0.025, pad=0.08, label="Difference value")

    #plt.tight_layout()
    plt.show()

def visualize_test(model, loader, x_points):
    y_pred, y_truth = preds_and_truth(model, loader)
    plot_compare_data(y_pred, y_truth, x_points)

def visualize_test_whole_timeseries(model, loader, x_points):
    y_pred, y_truth = preds_and_truth(model, loader)
    plot_compare_data(y_pred, y_truth, x_points)

@torch.no_grad()
def preds_and_truth(model, loader, device=None, as_numpy=True):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    yps, yts = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yp = model(xb)
        yps.append(yp.detach().cpu().view(-1))
        yts.append(yb.detach().cpu().view(-1))
    y_pred = torch.cat(yps); y_true = torch.cat(yts)
    if as_numpy:
        return y_pred.numpy(), y_true.numpy()
    return y_pred, y_true

def log_epoch_scalars(writer,
                      epoch: int,
                      train: dict,
                      val: dict,
                      weights: tuple,
                      lr: float,
                      grad_norm_mean: float | None = None,
                      alphas: list[float] | None = None):
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
    if grad_norm_mean is not None:
        writer.add_scalar('grad/total_norm_mean', grad_norm_mean, epoch)
    # Optional: per-block residual blend coefficients (alpha)
    if alphas is not None:
        for i, a in enumerate(alphas, start=1):
            try:
                writer.add_scalar(f'blocks/alpha_{i}', float(a), epoch)
            except Exception:
                # Be robust to non-float types
                pass

def figure_compare_data(y_pred, y_truth, x_points, y_min, y_max, title_prefix: str = "Val"):
    import numpy as _np
    import matplotlib.pyplot as _plt
    import matplotlib.colors as _colors

    y_pred = _np.reshape(y_pred, (-1, x_points)).T
    y_truth = _np.reshape(y_truth, (-1, x_points)).T
    y_diff = _np.abs(y_pred - y_truth)

    norm = _colors.Normalize(vmin=y_min, vmax=y_max)
    norm_diff = _colors.Normalize(vmin=0.0, vmax=(y_max - y_min))

    fig, axes = _plt.subplots(3, 1, figsize=(8, 10), sharex=True, constrained_layout=True)
    im0 = axes[0].imshow(y_pred, cmap='viridis', norm=norm, aspect='auto', origin='lower')
    axes[0].set_title(f"{title_prefix}: Predicted")
    im1 = axes[1].imshow(y_truth, cmap='viridis', norm=norm, aspect='auto', origin='lower')
    axes[1].set_title(f"{title_prefix}: Truth")
    im2 = axes[2].imshow(y_diff, cmap='RdBu_r', norm=norm_diff, aspect='auto', origin='lower')
    axes[2].set_title(f"{title_prefix}: Difference")
    axes[-1].set_xlabel("t index")
    for ax in axes:
        ax.set_ylabel("x index")
    fig.colorbar(im0, ax=axes[:2], orientation='vertical', fraction=0.025, pad=0.06, label="Pred/Truth value")
    fig.colorbar(im2, ax=axes[2], orientation='vertical', fraction=0.025, pad=0.08, label="Difference value")
    return fig

def train_test_val(X, y, train, test):
        # 60% train, 20% val, 20% test
    temp_size = 1-train
    test_size = test/temp_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=42, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size, random_state=42, shuffle=False
    )

    return X_train, y_train, X_val, y_val, X_test, y_test

def scaling(X_train, y_train, X_val, y_val, X_test, y_test, device, return_scales=True,
            u0=None, augment_ic: bool=False, dtype=torch.float32):
    """
    Standardize X and y using train statistics. Optionally, augment X with the full
    initial condition vector u0 (scaled with y's scaler) concatenated to each row.

    Returns:
      If return_scales and augment_ic:
        X_train, y_train, X_val, y_val, X_test, y_test,
        x_scaler, y_scaler, x_scale, t_scale, u_scale,
        u0_feat_t (torch tensor of scaled u0 on device)
      If return_scales and not augment_ic:
        X_train, y_train, X_val, y_val, X_test, y_test,
        x_scaler, y_scaler, x_scale, t_scale, u_scale
      If not return_scales:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    X_train = x_scaler.transform(X_train)
    X_val   = x_scaler.transform(X_val)
    X_test  = x_scaler.transform(X_test)
    y_train = y_scaler.transform(y_train)
    y_test  = y_scaler.transform(y_test)
    y_val   = y_scaler.transform(y_val)

    # Optional IC augmentation
    u0_feat_t = None
    if augment_ic:
        if u0 is None:
            raise ValueError("augment_ic=True requires u0 (initial condition vector)")
        u0 = np.asarray(u0).reshape(-1)
        # Scale IC with y scaler for consistent units
        u0_scaled = (u0 - y_scaler.mean_[0]) / y_scaler.scale_[0]
        feat_train = np.tile(u0_scaled, (X_train.shape[0], 1))
        feat_val   = np.tile(u0_scaled, (X_val.shape[0],   1))
        feat_test  = np.tile(u0_scaled, (X_test.shape[0],  1))
        X_train = np.concatenate([X_train, feat_train], axis=1)
        X_val   = np.concatenate([X_val,   feat_val],   axis=1)
        X_test  = np.concatenate([X_test,  feat_test],  axis=1)
        u0_feat_t = torch.as_tensor(u0_scaled, dtype=dtype, device=device)

    x_scale = torch.tensor(x_scaler.scale_[0], dtype=dtype, device=device)
    t_scale = torch.tensor(x_scaler.scale_[1], dtype=dtype, device=device)
    u_scale = torch.tensor(y_scaler.scale_[0], dtype=dtype, device=device)

    if not return_scales:
        return X_train, y_train, X_val, y_val, X_test, y_test

    if augment_ic:
        return (X_train, y_train, X_val, y_val, X_test, y_test,
                x_scaler, y_scaler, x_scale, t_scale, u_scale, u0_feat_t)

    return X_train, y_train, X_val, y_val, X_test, y_test, x_scaler, y_scaler, x_scale, t_scale, u_scale

def to_loader(X, y, batch_size=128, shuffle=False,
              num_workers: int = 0,
              persistent_workers: bool = False,
              pin_memory: bool = False,
              dtype=torch.float32):
    X_t = torch.as_tensor(X, dtype=dtype)
    y_t = torch.as_tensor(y, dtype=dtype)
    ds = TensorDataset(X_t, y_t)
    # persistent_workers only valid when num_workers > 0
    pw = persistent_workers if num_workers > 0 else False
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, persistent_workers=pw,
                      pin_memory=pin_memory)

def fit_scalers_multi_ic(X_list, y_list):
    """
    Fit StandardScaler for X and y over concatenated lists of arrays.
    X_list, y_list: lists of arrays for all training realizations.
    Returns x_scaler, y_scaler.
    """
    X_concat = np.concatenate(X_list, axis=0)
    y_concat = np.concatenate(y_list, axis=0)
    x_scaler = StandardScaler().fit(X_concat)
    y_scaler = StandardScaler().fit(y_concat)
    return x_scaler, y_scaler

def transform_with_scalers(X, y, device, x_scaler, y_scaler, u0=None, augment_ic=False, dtype=torch.float32):
    """
    Transform X and y with provided scalers. Optionally append scaled u0 features and
    return u0_feat_t (torch tensor on device) for use in the PINN loss.
    """
    X_t = x_scaler.transform(X)
    y_t = y_scaler.transform(y)
    u0_feat_t = None
    if augment_ic:
        if u0 is None:
            raise ValueError("augment_ic=True requires u0")
        u0 = np.asarray(u0).reshape(-1)
        u0_scaled = (u0 - y_scaler.mean_[0]) / y_scaler.scale_[0]
        feat = np.tile(u0_scaled, (X_t.shape[0], 1))
        X_t = np.concatenate([X_t, feat], axis=1)
        u0_feat_t = torch.as_tensor(u0_scaled, dtype=dtype, device=device)
    return X_t, y_t, u0_feat_t

def compute_scaled_domain_and_u0(x0_raw: float,
                                 L_raw: float,
                                 t_min_raw: float,
                                 t_max_raw: float,
                                 u0_raw,
                                 x_scaler: StandardScaler,
                                 y_scaler: StandardScaler,
                                 device: torch.device,
                                 dtype: torch.dtype,
                                 ):
    """
    Convert raw domain bounds (x0, L, t_min, t_max) into standardized coordinates
    using `x_scaler`, and scale the initial condition vector `u0_raw` using `y_scaler`.

    Returns torch tensors on the requested device/dtype:
      x0_s, L_s, t_min_s, t_max_s, u0_scaled_t
    """
    # Standardize bounds with the fitted scalers (same transform applied to inputs X)
    x0_s = (float(x0_raw)   - float(x_scaler.mean_[0])) / float(x_scaler.scale_[0])
    L_s  = (float(L_raw)    - float(x_scaler.mean_[0])) / float(x_scaler.scale_[0])
    t0_s = (float(t_min_raw) - float(x_scaler.mean_[1])) / float(x_scaler.scale_[1])
    tM_s = (float(t_max_raw) - float(x_scaler.mean_[1])) / float(x_scaler.scale_[1])

    x0_t   = torch.tensor(x0_s, dtype=dtype, device=device)
    L_t    = torch.tensor(L_s,  dtype=dtype, device=device)
    t_min_t = torch.tensor(t0_s, dtype=dtype, device=device)
    t_max_t = torch.tensor(tM_s, dtype=dtype, device=device)

    # Scale u0 with y_scaler to match standardized targets
    u0_arr = np.asarray(u0_raw).reshape(-1, 1)
    u0_scaled = y_scaler.transform(u0_arr).reshape(-1)
    u0_t = torch.as_tensor(u0_scaled, dtype=dtype, device=device)

    return x0_t, L_t, t_min_t, t_max_t, u0_t

def build_loaders_per_ic(u, g_u, xt, train_frac, test_frac, device,
                         batch_size_train=128, batch_size_eval=256,
                         augment_ic=True, random_seed=42,
                         dtype=torch.float32,
                         num_workers_train: int = 4,
                         num_workers_eval: int = 2):
    """
    Build per-IC DataLoaders for train/val/test splits by realization.
    Returns:
      train_loaders: list of (loader, u0_feat_t)
      val_loaders:   list of (loader, u0_feat_t)
      test_loaders:  list of (loader, u0_feat_t)
      x_scaler, y_scaler, x_scale, t_scale, u_scale
    Notes:
      - xt is shared across realizations and assumed shape (N, 2)
      - g_u[r] is flattened (N,) per realization
    """
    R = u.shape[0]
    idx = np.arange(R)
    rng = np.random.RandomState(random_seed)
    rng.shuffle(idx)
    n_train = max(1, int(R * train_frac))
    n_test  = max(1, int(R * test_frac))
    n_val   = max(0, R - n_train - n_test)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val: n_train+n_val+n_test]

    # Fit scalers on all training samples concatenated
    X_train_list = [xt for _ in train_idx]
    y_train_list = [g_u[r].reshape(-1, 1) for r in train_idx]
    x_scaler, y_scaler = fit_scalers_multi_ic(X_train_list, y_train_list)

    # Helper to transform and make loader list
    def make_split(idxs, is_train: bool):
        out = []
        is_cuda = torch.cuda.is_available()
        for r in idxs:
            X_r, y_r, u0_feat_t = transform_with_scalers(
                xt, g_u[r].reshape(-1, 1), device,
                x_scaler, y_scaler,
                u0=u[r], augment_ic=augment_ic,
                dtype=dtype,
            )
            loader = to_loader(
                X_r, y_r,
                batch_size=(batch_size_train if is_train else batch_size_eval),
                shuffle=is_train,
                num_workers=(num_workers_train if is_train else num_workers_eval),
                persistent_workers=False,
                pin_memory=is_cuda,
                dtype=dtype,
            )
            out.append((loader, u0_feat_t))
        return out

    train_loaders = make_split(train_idx, True)
    val_loaders   = make_split(val_idx, False)
    test_loaders  = make_split(test_idx, False)

    x_scale = torch.tensor(x_scaler.scale_[0], dtype=dtype, device=device)
    t_scale = torch.tensor(x_scaler.scale_[1], dtype=dtype, device=device)
    u_scale = torch.tensor(y_scaler.scale_[0], dtype=dtype, device=device)

    return train_loaders, val_loaders, test_loaders, x_scaler, y_scaler, x_scale, t_scale, u_scale

def build_mixed_loaders_per_ic(u, g_u, xt, train_frac, test_frac, device,
                               batch_size_train=128, batch_size_eval=256,
                               augment_ic=True, random_seed=42,
                               dtype=torch.float32,
                               num_workers_train: int = 0,
                               num_workers_eval: int = 0):
    """
    Build single train/val/test DataLoaders that mix samples from all ICs per split.
    Keeps IC features appended to X rows (when augment_ic=True).
    Returns:
      train_loader, val_loader, test_loader, x_scaler, y_scaler, x_scale, t_scale, u_scale
    """
    R = u.shape[0]
    idx = np.arange(R)
    rng = np.random.RandomState(random_seed)
    rng.shuffle(idx)
    n_train = max(1, int(R * train_frac))
    n_test  = max(1, int(R * test_frac))
    n_val   = max(0, R - n_train - n_test)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val: n_train+n_val+n_test]

    # Fit scalers on all training samples concatenated
    X_train_list = [xt for _ in train_idx]
    y_train_list = [g_u[r].reshape(-1, 1) for r in train_idx]
    x_scaler, y_scaler = fit_scalers_multi_ic(X_train_list, y_train_list)

    def make_split(idxs, is_train: bool):
        Xs, Ys = [], []
        for r in idxs:
            X_r, y_r, _u0_feat = transform_with_scalers(
                xt, g_u[r].reshape(-1, 1), device,
                x_scaler, y_scaler,
                u0=u[r], augment_ic=augment_ic,
                dtype=dtype,
            )
            Xs.append(X_r); Ys.append(y_r)
        X_all = np.concatenate(Xs, axis=0) if Xs else np.empty((0, xt.shape[1] + (u.shape[1] if augment_ic else 0)))
        Y_all = np.concatenate(Ys, axis=0) if Ys else np.empty((0, 1))
        return to_loader(
            X_all, Y_all,
            batch_size=(batch_size_train if is_train else batch_size_eval),
            shuffle=is_train,
            num_workers=(num_workers_train if is_train else num_workers_eval),
            persistent_workers=False,
            pin_memory=False,
            dtype=dtype,
        )

    train_loader = make_split(train_idx, True)
    val_loader   = make_split(val_idx, False)
    test_loader  = make_split(test_idx, False)

    x_scale = torch.tensor(x_scaler.scale_[0], dtype=dtype, device=device)
    t_scale = torch.tensor(x_scaler.scale_[1], dtype=dtype, device=device)
    u_scale = torch.tensor(y_scaler.scale_[0], dtype=dtype, device=device)

    return train_loader, val_loader, test_loader, x_scaler, y_scaler, x_scale, t_scale, u_scale

def build_loaders_single_ic(u0, g_u0, xt, train_frac, test_frac, device,
                            batch_size_train=128, batch_size_eval=256,
                            random_seed=42,
                            dtype=torch.float32,
                            num_workers_train: int = 0,
                            num_workers_eval: int = 0):
    """
    Build train/val/test loaders by splitting samples within a single IC realization.
    - Fits scalers on train samples only.
    - Does not append IC features (only [x, t] -> y).

    Args:
      u0: initial condition vector shape (X,) [unused here; kept for API symmetry]
      g_u0: flattened target array shape (N,) for this IC
      xt: shared coordinates array shape (N, 2)
    Returns:
      train_loader, val_loader, test_loader, x_scaler, y_scaler, x_scale, t_scale, u_scale
    """
    N = xt.shape[0]
    # Forecast-style split: do NOT shuffle. Split by time order so that
    # train uses earliest t, val uses mid t, test uses latest t.
    t = xt[:, 1]
    unique_t = np.unique(t)
    T = unique_t.shape[0]
    # Compute counts in time steps
    n_train_t = int(round(T * train_frac))
    n_test_t  = int(round(T * test_frac))
    n_val_t   = T - n_train_t - n_test_t
    if T >= 3:
        if n_train_t < 1: n_train_t = 1
        if n_test_t  < 1: n_test_t  = 1
        if n_val_t   < 1:
            if n_train_t >= n_test_t and n_train_t > 1:
                n_train_t -= 1; n_val_t += 1
            elif n_test_t > 1:
                n_test_t  -= 1; n_val_t += 1
    # Final adjustment to sum exactly to T
    total_t = n_train_t + n_val_t + n_test_t
    if total_t != T:
        n_train_t += (T - total_t)

    # Determine cutoff times
    train_t_max = unique_t[n_train_t - 1]
    val_t_max = unique_t[n_train_t + n_val_t - 1] if n_val_t > 0 else train_t_max

    # Build index masks by time intervals
    train_mask = (t <= train_t_max)
    val_mask = (t > train_t_max) & (t <= val_t_max) if n_val_t > 0 else np.zeros_like(train_mask, dtype=bool)
    test_mask = (t > val_t_max) if n_test_t > 0 else np.zeros_like(train_mask, dtype=bool)

    train_idx = np.nonzero(train_mask)[0]
    val_idx   = np.nonzero(val_mask)[0]
    test_idx  = np.nonzero(test_mask)[0]

    X_train = xt[train_idx]
    y_train = g_u0.reshape(-1, 1)[train_idx]
    X_val   = xt[val_idx] if len(val_idx) > 0 else np.empty((0, xt.shape[1]))
    y_val   = g_u0.reshape(-1, 1)[val_idx] if len(val_idx) > 0 else np.empty((0, 1))
    X_test  = xt[test_idx] if len(test_idx) > 0 else np.empty((0, xt.shape[1]))
    y_test  = g_u0.reshape(-1, 1)[test_idx] if len(test_idx) > 0 else np.empty((0, 1))

    # Fit scalers on train
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    X_train = x_scaler.transform(X_train)
    y_train = y_scaler.transform(y_train)
    X_val   = x_scaler.transform(X_val) if len(val_idx) > 0 else X_val
    y_val   = y_scaler.transform(y_val) if len(val_idx) > 0 else y_val
    X_test  = x_scaler.transform(X_test) if len(test_idx) > 0 else X_test
    y_test  = y_scaler.transform(y_test) if len(test_idx) > 0 else y_test

    # Build loaders
    train_loader = to_loader(
        X_train, y_train,
        batch_size=batch_size_train, shuffle=False,
        num_workers=num_workers_train, persistent_workers=False,
        pin_memory=False, dtype=dtype,
    )
    val_loader = to_loader(
        X_val, y_val,
        batch_size=batch_size_eval, shuffle=False,
        num_workers=num_workers_eval, persistent_workers=False,
        pin_memory=False, dtype=dtype,
    ) if len(val_idx) > 0 else to_loader(
        np.empty((0, xt.shape[1])), np.empty((0, 1)),
        batch_size=batch_size_eval, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=False, dtype=dtype,
    )
    test_loader = to_loader(
        X_test, y_test,
        batch_size=batch_size_eval, shuffle=False,
        num_workers=num_workers_eval, persistent_workers=False,
        pin_memory=False, dtype=dtype,
    ) if len(test_idx) > 0 else to_loader(
        np.empty((0, xt.shape[1])), np.empty((0, 1)),
        batch_size=batch_size_eval, shuffle=False,
        num_workers=0, persistent_workers=False, pin_memory=False, dtype=dtype,
    )

    x_scale = torch.tensor(x_scaler.scale_[0], dtype=dtype, device=device)
    t_scale = torch.tensor(x_scaler.scale_[1], dtype=dtype, device=device)
    u_scale = torch.tensor(y_scaler.scale_[0], dtype=dtype, device=device)

    y_min = np.min(y_train)
    y_max = np.max(y_train)

    return train_loader, val_loader, test_loader, x_scaler, y_scaler, x_scale, t_scale, u_scale, y_min, y_max

def scale_dataset(u0, g_u0, xt, device, dtype):
    x_scaler = StandardScaler().fit(xt)
    y_scaler = StandardScaler().fit(u0.transpose())
    X_val = torch.as_tensor(x_scaler.transform(xt), dtype=dtype, device = device)
    y_val = torch.as_tensor(y_scaler.transform(g_u0.transpose()), dtype=dtype, device = device)
    u0_scaled = torch.as_tensor(y_scaler.transform(u0.transpose()), dtype=dtype, device = device)
    x_scale = torch.tensor(x_scaler.scale_[0], dtype=dtype, device=device)
    t_scale = torch.tensor(x_scaler.scale_[1], dtype=dtype, device=device)
    u_scale = torch.tensor(y_scaler.scale_[0], dtype=dtype, device=device)
    u_mean = torch.tensor(y_scaler.mean_[0], dtype=dtype, device=device)

    return X_val, y_val, u0_scaled, x_scale, t_scale, u_scale, u_mean, x_scaler, y_scaler

def make_single_ic_loader_from_mixed(mixed_loader: DataLoader, ic_feat_vec: torch.Tensor,
                                     batch_size: int = 256, tol: float = 1e-8) -> DataLoader:
    """
    From a mixed-IC DataLoader (TensorDataset of X,y with IC features appended to X[:,2:]),
    build a new DataLoader that only contains samples whose IC feature vector matches ic_feat_vec.
    """
    ds = mixed_loader.dataset
    if not isinstance(ds, TensorDataset):
        raise ValueError("Expected TensorDataset in mixed_loader.dataset")
    X_all, y_all = ds.tensors
    if X_all.size(1) <= 2:
        raise ValueError("X does not contain appended IC features (need columns > 2)")
    feats = X_all[:, 2:]
    fv = ic_feat_vec.to(dtype=feats.dtype, device=feats.device)
    if fv.ndim > 1:
        fv = fv.view(-1)
    # Match by max-abs-diff within tolerance
    mask = (feats - fv).abs().max(dim=1).values < tol
    X_sel = X_all[mask]
    y_sel = y_all[mask]
    ds_sel = TensorDataset(X_sel, y_sel)
    return DataLoader(ds_sel, batch_size=batch_size, shuffle=False, num_workers=0, persistent_workers=False)

@torch.no_grad()
def evaluate_regression(model, loader, y_scaler, device=None, y_train_mean=None, eps=1e-12, dtype = torch.float64):
    """
    Returns:
      dict with: mse, rel_L2, rel_MSE_vs_mean, r2
    Notes:
      - If you pass y_train_mean, rel_MSE_vs_mean and r2 use that constant baseline.
        Otherwise they use the test set mean (standard R^2 definition).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    y_true_chunks, y_pred_chunks = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        yp = model(xb)


        # squeeze trailing singleton if present
        if yp.ndim > 1 and yp.size(-1) == 1: yp = yp.squeeze(-1)
        if yb.ndim > 1 and yb.size(-1) == 1: yb = yb.squeeze(-1)

        # flatten per-sample dimensions and collect
        y_true_chunks.append(yb.reshape(yb.size(0), -1))
        y_pred_chunks.append(yp.reshape(yp.size(0), -1))

    yt = torch.cat(y_true_chunks, dim=0)  # (N, D)
    yp = torch.cat(y_pred_chunks, dim=0)  # (N, D)

    yt = torch.tensor(y_scaler.inverse_transform(yt), dtype=dtype, device=device)
    yp = torch.tensor(y_scaler.inverse_transform(yp), dtype=dtype, device=device)

    # MSE over all samples/elements
    mse = ((yp - yt) ** 2).mean().item()

    # Relative L2 (energy-normalized): sum of squares ratio
    rel_L2 = ((yp - yt).pow(2).sum() / (yt.pow(2).sum().clamp_min(eps))).item()

    # Baseline = constant mean (train mean preferred; else test mean)
    if y_train_mean is None:
        ybar = yt.mean()  # test mean (standard R^2)
    else:
        ybar = torch.as_tensor(y_train_mean, device=yt.device, dtype=yt.dtype)
    mse_baseline = ((yt - ybar) ** 2).mean().clamp_min(eps).item()

    rel_MSE_vs_mean = mse / mse_baseline
    # R^2 = 1 - SSE/SST (consistent with chosen baseline mean)
    r2 = 1.0 - (
        ( (yt - yp).pow(2).sum() ) /
        ( (yt - ybar).pow(2).sum().clamp_min(eps) )
    ).item()

    return {
        "mse": mse,
        "rel_L2": rel_L2,
        "rel_MSE_vs_mean": rel_MSE_vs_mean,
        "r2": r2,
    }
