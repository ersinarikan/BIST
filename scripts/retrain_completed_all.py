#!/usr/bin/env python3
"""
Retrain all completed symbol–horizon pairs using the current training/evaluation logic.

This script:
- Loads environment similar to the systemd service (EnvironmentFile + DATABASE_URL from secret)
- Uses ContinuousHPOPipeline.run_training with stored best_params for each completed task
- Updates the pipeline state with refreshed training DirHit metrics
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


def _load_env_like_service() -> None:
    """Load environment similar to systemd service."""
    # Load EnvironmentFile if present
    env_file = Path("/etc/default/bist-pattern")
    if env_file.exists():
        try:
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :]
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
        except Exception as exc:
            print(f"Warn: cannot parse env file: {exc}")

    # DATABASE_URL from secret (like ExecStartPre)
    try:
        pw = Path("/opt/bist-pattern/.secrets/db_password").read_text().strip()
        os.environ[
            "DATABASE_URL"
        ] = f"postgresql://bist_user:{pw}@127.0.0.1:6432/bist_pattern_db"
    except Exception as exc:
        print(f"Warn: cannot set DATABASE_URL: {exc}")

    # Ensure project on path
    if "/opt/bist-pattern" not in os.sys.path:
        os.sys.path.insert(0, "/opt/bist-pattern")


def _load_best_params(path: Path) -> Tuple[Optional[Dict], Optional[Dict]]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        best_params = (data.get("best_params") or {}).copy()
        # Carry HPO metadata needed for parity
        for k in ("features_enabled", "feature_params", "feature_flags", "hyperparameters"):
            if k in data:
                best_params[k] = data[k]
        # ✅ FIX: Add best_trial_number to best_params and hpo_result for seed alignment
        if "best_trial_number" in data:
            best_params["best_trial_number"] = data["best_trial_number"]
        elif "best_trial" in data and isinstance(data["best_trial"], dict):
            best_trial_number = data["best_trial"].get("number")
            if best_trial_number is not None:
                best_params["best_trial_number"] = best_trial_number
        # Ensure hpo_result has best_trial_number
        if "best_trial_number" not in data and "best_trial_number" in best_params:
            data["best_trial_number"] = best_params["best_trial_number"]
        return best_params, data
    except Exception as exc:
        print("Failed to load best_params:", path, exc)
        return None, None


def main() -> int:
    _load_env_like_service()

    # Import after env is ready
    from scripts.continuous_hpo_training_pipeline import ContinuousHPOPipeline  # noqa: WPS433

    pipe = ContinuousHPOPipeline()
    items = list(pipe.state.items())
    completed = [
        (key, t)
        for key, t in items
        if getattr(t, "status", "") == "completed" and getattr(t, "best_params_file", None)
    ]
    # ✅ FIX: Sort alphabetically by key (symbol_horizon) for consistent processing order
    completed = sorted(completed, key=lambda x: x[0])
    print(f"Completed tasks found: {len(completed)}")

    updated = 0
    failed = 0
    skipped = 0

    for key, t in completed:
        # ✅ FIX: Type guard - best_params_file is guaranteed to be non-None by filter above
        best_params_file = getattr(t, "best_params_file", None)
        if not best_params_file:
            print("Skip (missing best_params_file):", key)
            skipped += 1
            continue
        bp = Path(best_params_file)
        if not bp.exists():
            print("Skip (missing best_params):", key)
            skipped += 1
            continue

        best_params, hpo_result = _load_best_params(bp)
        if not best_params:
            print("Skip (cannot load best_params):", key)
            skipped += 1
            continue

        print("Retraining", key, "...")
        try:
            res = pipe.run_training(t.symbol, t.horizon, best_params, hpo_result=hpo_result)
        except Exception as exc:  # noqa: BLE001
            print("Retrain error:", key, exc)
            res = None

        if not isinstance(res, dict):
            print("Retrain failed:", key)
            failed += 1
            continue

        # Update metrics and persist incrementally
        now = datetime.now().isoformat()
        t.training_completed_at = now
        t.training_dirhit = res.get("adaptive_dirhit")
        t.training_dirhit_online = res.get("adaptive_dirhit")
        t.training_dirhit_wfv = res.get("wfv_dirhit")
        t.status = "completed"
        pipe.state[key] = t
        pipe.save_state()
        updated += 1
        print("Updated", key, "WFV=", res.get("wfv_dirhit"), "Online=", res.get("adaptive_dirhit"))

    print("Summary: updated=", updated, "failed=", failed, "skipped=", skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
