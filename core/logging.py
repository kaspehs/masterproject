from __future__ import annotations

import os
import time
from typing import Optional, Tuple

from torch.utils.tensorboard import SummaryWriter


def setup_writer(
    run_dir_root: str,
    config_name: str,
    run_name_override: Optional[str] = None,
    append_timestamp: bool = True,
) -> Tuple[SummaryWriter, str]:
    timestamp = time.strftime("%m%d-%H%M%S")
    base_name = (run_name_override or "").strip() or config_name
    run_name = f"{base_name}_{timestamp}" if append_timestamp else base_name
    submit_dir = os.getenv("SLURM_SUBMIT_DIR")
    if submit_dir:
        run_dir = os.path.join(submit_dir, "logs", run_name)
    else:
        run_dir = os.path.join(run_dir_root, run_name)
    return SummaryWriter(log_dir=run_dir), run_name
