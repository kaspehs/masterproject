from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path


def _load_impl():
    try:
        return importlib.import_module("CFD_Data.prepare_cfd_npz_for_td_training")
    except ModuleNotFoundError:
        module_path = Path(__file__).resolve().parents[1] / "CFD_Data" / "prepare_cfd_npz_for_td_training.py"
        spec = importlib.util.spec_from_file_location("CFD_Data.prepare_cfd_npz_for_td_training", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load CFD prep module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


_IMPL = _load_impl()

for _name, _value in vars(_IMPL).items():
    if not _name.startswith("_"):
        globals()[_name] = _value

__all__ = getattr(_IMPL, "__all__", [name for name in globals() if not name.startswith("_")])


if __name__ == "__main__":
    main()
