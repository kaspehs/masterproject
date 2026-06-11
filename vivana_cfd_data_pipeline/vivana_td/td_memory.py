from __future__ import annotations

from typing import Any

import numpy as np


def resolve_td_memory_config(raw_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(raw_cfg or {})
    mode_value = cfg.get("td_memory_mode", cfg.get("mode", "fixed_n_memory"))
    mode = str(mode_value).strip().lower()
    aliases = {
        "fixed": "fixed_n_memory",
        "fixed_n": "fixed_n_memory",
        "n_memory": "fixed_n_memory",
        "fixed_tau_s": "fixed_tau",
        "tau": "fixed_tau",
        "tau_over_tref": "tau_over_tref",
        "tau_over_t_ref": "tau_over_tref",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"fixed_n_memory", "fixed_tau", "tau_over_tref"}:
        raise ValueError("vivana_td.td_memory_mode must be one of: fixed_n_memory, fixed_tau, tau_over_tref.")
    tau_s_raw = cfg.get("td_memory_tau_s", cfg.get("tau_s", None))
    tau_s = None if tau_s_raw is None else float(tau_s_raw)
    tau_over_tref = float(cfg.get("td_tau_over_tref", cfg.get("tau_over_tref", 4.0)))
    if tau_s is not None and (not np.isfinite(tau_s) or tau_s <= 0.0):
        raise ValueError("vivana_td.td_memory_tau_s must be positive and finite when provided.")
    if not np.isfinite(tau_over_tref) or tau_over_tref <= 0.0:
        raise ValueError("vivana_td.td_tau_over_tref must be positive and finite.")
    if mode == "fixed_tau" and tau_s is None:
        raise ValueError("vivana_td.td_memory_mode='fixed_tau' requires vivana_td.td_memory_tau_s.")
    return {
        "mode": mode,
        "tau_s": tau_s,
        "tau_over_tref": tau_over_tref,
    }


def resolve_td_n_memory(
    params: dict[str, float],
    *,
    dt: float,
    flow_speed: float,
    diameter: float,
    memory_cfg: dict[str, Any] | None,
) -> float:
    cfg = resolve_td_memory_config(memory_cfg)
    mode = str(cfg["mode"])
    if mode == "fixed_n_memory":
        return max(1.0, float(round(float(params["n_memory"]))))
    dt_value = float(dt)
    flow_speed_value = float(flow_speed)
    diameter_value = float(diameter)
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError("dt must be positive and finite when resolving TD memory.")
    if not np.isfinite(diameter_value) or diameter_value <= 0.0:
        raise ValueError("diameter must be positive and finite when resolving TD memory.")
    if not np.isfinite(flow_speed_value) or abs(flow_speed_value) <= 0.0:
        raise ValueError("flow_speed must be finite and non-zero when resolving TD memory.")
    if mode == "fixed_tau":
        tau_value = float(cfg["tau_s"])
    else:
        fhat0 = float(params["fhat0"])
        if not np.isfinite(fhat0) or fhat0 <= 0.0:
            raise ValueError("params['fhat0'] must be positive and finite when resolving TD memory.")
        tau_value = float(cfg["tau_over_tref"]) * float(diameter_value) / (fhat0 * abs(flow_speed_value))
    return max(1.0, float(round(tau_value / dt_value)))
