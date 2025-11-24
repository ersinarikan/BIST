#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def main() -> int:
    results_dir = Path("/opt/bist-pattern/results")
    allowed: set[str] = {"Poisson", "Bayesian", "Bernoulli", "MVS", "No"}

    files_scanned = 0
    with_bt = 0
    invalid_bt: List[Tuple[str, Any]] = []
    bayesian_with_subsample: List[Tuple[str, Any]] = []
    examples_invalid: List[Tuple[str, Any]] = []
    examples_bayes: List[Tuple[str, Any]] = []

    for p in results_dir.rglob("*.json"):
        try:
            data: Dict[str, Any] = json.loads(p.read_text())
        except Exception:
            continue
        best_params: Dict[str, Any] = data.get("best_params") or {}
        if not isinstance(best_params, dict):
            continue
        files_scanned += 1
        bt = best_params.get("cat_bootstrap_type")
        sub = best_params.get("cat_subsample")

        if bt is not None:
            with_bt += 1
            if isinstance(bt, bool):
                invalid_bt.append((str(p), bt))
                if len(examples_invalid) < 5:
                    examples_invalid.append((str(p), bt))
            elif isinstance(bt, str):
                if bt not in allowed:
                    invalid_bt.append((str(p), bt))
                    if len(examples_invalid) < 5:
                        examples_invalid.append((str(p), bt))
            else:
                invalid_bt.append((str(p), bt))
                if len(examples_invalid) < 5:
                    examples_invalid.append((str(p), bt))

        if isinstance(bt, str) and bt == "Bayesian" and sub is not None:
            bayesian_with_subsample.append((str(p), sub))
            if len(examples_bayes) < 5:
                examples_bayes.append((str(p), sub))

    print("files_scanned:", files_scanned)
    print("best_params_with_cat_bt:", with_bt)
    print("invalid_bootstrap_type_count:", len(invalid_bt))
    print("bayesian_with_subsample_count:", len(bayesian_with_subsample))
    print("examples_invalid:")
    for e in examples_invalid:
        print("  ", e)
    print("examples_bayesian_with_subsample:")
    for e in examples_bayes:
        print("  ", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


