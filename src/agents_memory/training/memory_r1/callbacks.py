"""Structured metrics logging callback for Memory-R1 GRPO training.

Hooks into TRL GRPOTrainer lifecycle to emit JSON-lines metrics:
- run_start: config snapshot at training begin
- step_metrics: per-logging-step RL metrics (reward, loss, KL, etc.)
- validation: periodic val EM/F1 evaluation
- run_end: training summary
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class MemoryR1MetricsCallback(TrainerCallback):
    """TrainerCallback for structured JSON-lines metrics logging."""

    def __init__(
        self,
        metrics_path: str | Path,
        phase: str,
        eval_fn: Any | None = None,
        eval_kwargs: dict | None = None,
        eval_every: int = 50,
        checkpoint_every: int = 100,
        checkpoint_dir: str | Path | None = None,
    ):
        super().__init__()
        self.metrics_path = Path(metrics_path)
        self.phase = phase
        self.eval_fn = eval_fn
        self.eval_kwargs = eval_kwargs or {}
        self.eval_every = eval_every
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._file = None
        self._start_time = None
        self._best_val_em = 0.0
        self._last_eval_step = -1
        self._last_ckpt_step = -1
        self._tokenizer = None  # Captured from trainer at train_begin

    def _write_record(self, record: dict) -> None:
        if self._file is None:
            return
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def on_train_begin(self, args, state, control, model=None, processing_class=None, **kwargs):
        # Capture tokenizer from trainer kwargs
        self._tokenizer = processing_class or kwargs.get("tokenizer")

        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.metrics_path, "a")
        self._start_time = time.time()

        config_snapshot = {
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_steps": args.max_steps,
            "num_train_epochs": args.num_train_epochs,
            "warmup_steps": args.warmup_steps,
            "logging_steps": args.logging_steps,
            "save_steps": args.save_steps,
            "bf16": args.bf16,
            "fp16": args.fp16,
        }

        self._write_record({
            "event": "run_start",
            "phase": self.phase,
            "config": config_snapshot,
            "eval_every": self.eval_every,
            "checkpoint_every": self.checkpoint_every,
            "timestamp": time.time(),
        })

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        record = {
            "event": "step_metrics",
            "phase": self.phase,
            "global_step": state.global_step,
            "epoch": state.epoch,
            "timestamp": time.time(),
            "wall_time_s": time.time() - self._start_time if self._start_time else 0,
        }

        # Capture all numeric metrics from TRL
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                record[key.replace("/", "_")] = value

        self._write_record(record)

    def on_step_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        step = state.global_step
        tokenizer = processing_class or self._tokenizer

        # Periodic validation
        if (
            self.eval_fn is not None
            and self.eval_every > 0
            and step > 0
            and step % self.eval_every == 0
            and step != self._last_eval_step
        ):
            self._last_eval_step = step
            try:
                eval_result = self.eval_fn(
                    model=model, tokenizer=tokenizer, **self.eval_kwargs
                )
                record = {
                    "event": "validation",
                    "phase": self.phase,
                    "global_step": step,
                    "timestamp": time.time(),
                    **eval_result,
                }
                self._write_record(record)

                val_em = eval_result.get("val_em", 0.0)
                if val_em > self._best_val_em and self.checkpoint_dir:
                    self._best_val_em = val_em
                    best_dir = self.checkpoint_dir / "best"
                    best_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(best_dir))
                    if tokenizer is not None:
                        tokenizer.save_pretrained(str(best_dir))
                    print(f"  [metrics] New best val_em={val_em:.4f} at step {step}")

            except Exception as e:
                self._write_record({
                    "event": "validation_error",
                    "phase": self.phase,
                    "global_step": step,
                    "error": str(e),
                    "timestamp": time.time(),
                })
                print(f"  [metrics] Validation error at step {step}: {e}")

        # Periodic checkpointing
        if (
            self.checkpoint_dir
            and self.checkpoint_every > 0
            and step > 0
            and step % self.checkpoint_every == 0
            and step != self._last_ckpt_step
        ):
            self._last_ckpt_step = step
            ckpt_dir = self.checkpoint_dir / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            if tokenizer is not None:
                tokenizer.save_pretrained(str(ckpt_dir))
            print(f"  [metrics] Checkpoint saved at step {step}")

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self._start_time if self._start_time else 0
        self._write_record({
            "event": "run_end",
            "phase": self.phase,
            "total_steps": state.global_step,
            "total_wall_time_s": elapsed,
            "best_val_em": self._best_val_em,
            "timestamp": time.time(),
        })
        print(
            f"\n[metrics] Training complete. {state.global_step} steps in {elapsed:.0f}s. "
            f"Best val_em={self._best_val_em:.4f}"
        )
        print(f"[metrics] Metrics saved to {self.metrics_path}")

        if self._file is not None:
            self._file.close()
            self._file = None
