#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _git_commit(cwd: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


class TrainingRunLogger:
    def __init__(self, cfg, progress_interval: int = 10000):
        self.cfg = cfg
        self.progress_interval = max(1, int(progress_interval))
        self.started_at = _utc_now_iso()
        self.start_time = time.time()
        self.summary_path = os.path.join(cfg.out_dir, "run_summary.json")
        self.progress_path = os.path.join(cfg.out_dir, "progress.jsonl")
        self.git_commit = _git_commit(os.getcwd())
        self._last_progress_epoch: Optional[int] = None

    def _feature_flags(self) -> Dict[str, bool]:
        return {
            "use_albedo_features": bool(self.cfg.use_albedo_features),
            "use_spec_features": bool(self.cfg.use_spec_features),
            "use_normal_features": bool(self.cfg.use_normal_features),
            "use_roughness_feature": bool(self.cfg.use_roughness_feature),
            "use_pdf_feature": bool(self.cfg.use_pdf_feature),
        }

    def _sparse_progress_entry(
        self,
        epoch: int,
        metrics: Dict[str, float],
        phase: str,
    ) -> Dict[str, object]:
        return {
            "epoch": int(epoch),
            "phase": str(phase),
            "train_loss": _safe_float(metrics.get("loss")),
            "val_loss": _safe_float(metrics.get("val_loss")),
            "yhat_mean": _safe_float(metrics.get("yhat_mean")),
            "elapsed_seconds": time.time() - self.start_time,
        }

    def should_log_progress(
        self,
        epoch: int,
        phase_changed: bool,
        is_final: bool,
    ) -> bool:
        if epoch == 0:
            return True
        if is_final:
            return True
        if phase_changed:
            return True
        return ((epoch + 1) % self.progress_interval) == 0

    def append_progress(
        self,
        epoch: int,
        metrics: Dict[str, float],
        phase: str,
    ) -> None:
        if self._last_progress_epoch == epoch:
            return
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        entry = self._sparse_progress_entry(epoch, metrics, phase)
        with open(self.progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self._last_progress_epoch = epoch

    def write_summary(
        self,
        status: str,
        best_epoch: Optional[int],
        best_metrics: Optional[Dict[str, float]],
        last_epoch: Optional[int],
        last_metrics: Optional[Dict[str, float]],
    ) -> None:
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        finished_at = _utc_now_iso()
        summary = {
            "status": status,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "duration_seconds": time.time() - self.start_time,
            "out_dir": self.cfg.out_dir,
            "git_commit": self.git_commit,
            "training": {
                "max_epochs": int(self.cfg.max_epochs),
                "encoder_bootstrap_epochs": int(self.cfg.encoder_bootstrap_epochs),
                "tex_w": int(self.cfg.tex_w),
                "tex_h": int(self.cfg.tex_h),
                "train_latent_texture": bool(self.cfg.train_latent_texture),
                "train_decoder": bool(self.cfg.train_decoder),
                "features": self._feature_flags(),
            },
            "best_epoch": _safe_int(best_epoch),
            "best_val_loss": _safe_float(None if best_metrics is None else best_metrics.get("val_loss")),
            "best_train_loss": _safe_float(None if best_metrics is None else best_metrics.get("loss")),
            "best_yhat_mean": _safe_float(None if best_metrics is None else best_metrics.get("yhat_mean")),
            "exported_epoch": _safe_int(best_epoch),
            "last_epoch": _safe_int(last_epoch),
            "last_train_loss": _safe_float(None if last_metrics is None else last_metrics.get("loss")),
            "last_val_loss": _safe_float(None if last_metrics is None else last_metrics.get("val_loss")),
            "last_yhat_mean": _safe_float(None if last_metrics is None else last_metrics.get("yhat_mean")),
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
