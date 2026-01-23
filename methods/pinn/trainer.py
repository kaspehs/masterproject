from __future__ import annotations

from HNN_helper import Config


def train(cfg: Config, config_name: str) -> None:
    raise NotImplementedError(
        "PINN method is not implemented in the refactor yet. "
        "Legacy prototypes exist under `legacy_code/` (e.g. `legacy_code/pinn_ODE.py`). "
        "If you tell me which PINN you want (ODE-PINN for the VIV oscillator vs KdV), "
        "I can port it into `methods/pinn/` with the new config structure."
    )

