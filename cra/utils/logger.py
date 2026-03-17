"""Logging utilities with TensorBoard, WandB, and console output."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class Logger:
    """Simple logger that writes to console, file, TensorBoard, and WandB."""

    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "cradex",
        wandb_entity: str = "",
        wandb_config: dict | None = None,
    ) -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Console + file log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = open(
            os.path.join(log_dir, f"train_{timestamp}.log"), "w"
        )

        # TensorBoard
        self.tb_writer = None
        if use_tensorboard and HAS_TB:
            tb_dir = os.path.join(log_dir, "tb")
            self.tb_writer = SummaryWriter(log_dir=tb_dir)

        # WandB
        self.wandb_run = None
        if use_wandb and HAS_WANDB:
            self.wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity or None,
                name=experiment_name,
                dir=log_dir,
                config=wandb_config or {},
                reinit=True,
            )
        elif use_wandb and not HAS_WANDB:
            print("Warning: wandb requested but not installed (pip install wandb)")

        self.log_text(f"Experiment: {experiment_name}")
        self.log_text(f"Log dir: {log_dir}")
        self.start_time = time.time()

    def log_text(self, msg: str) -> None:
        """Log a text message to console and file."""
        elapsed = time.time() - self.start_time if hasattr(self, 'start_time') else 0
        timestamp = f"[{elapsed:8.1f}s]"
        line = f"{timestamp} {msg}"
        print(line, flush=True)
        if self.log_file and not self.log_file.closed:
            self.log_file.write(line + "\n")
            self.log_file.flush()

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log scalar metrics to TensorBoard and WandB."""
        if self.tb_writer is not None:
            for key, value in metrics.items():
                tag = f"{prefix}/{key}" if prefix else key
                self.tb_writer.add_scalar(tag, value, step)
            self.tb_writer.flush()

        if self.wandb_run is not None:
            log_dict = {}
            for key, value in metrics.items():
                tag = f"{prefix}/{key}" if prefix else key
                log_dict[tag] = value
            log_dict["global_step"] = step
            wandb.log(log_dict, step=step)

    def close(self) -> None:
        if self.log_file and not self.log_file.closed:
            self.log_file.close()
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb_run is not None:
            wandb.finish()
