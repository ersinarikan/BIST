#!/usr/bin/env python3
"""
Check training log for key ML capabilities and emit a concise report.

Usage:
  python scripts/check_training_log_capabilities.py --log /opt/bist-pattern/logs/train_....log

Outputs JSON with found/absent/inconclusive per checklist item.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, Any, List


def compile_patterns() -> Dict[str, List[re.Pattern]]:
    # Each capability has a list of regexes; any match => found
    return {
        'purged_embargo_horizon_aware': [
            re.compile(r"Using Purged Time-Series CV .*purge=\d+.*, embargo=\d+.*HORIZON-AWARE")
        ],
        'feature_importance_selection': [
            re.compile(r"Training temporary model for feature importance selection", re.I),
            re.compile(r"Feature importance selection: \d+ .* → \d+ features")
        ],
        'training_caps_empirical': [
            re.compile(r"Empirical cap .* P\d+\(\|ret\|\)=\d+\.\d+")
        ],
        'meta_stacking_ridge': [
            re.compile(r"Meta-learner trained .* \(OOF-based Ridge \+ StandardScaler\)")
        ],
        'directional_loss_metric': [
            re.compile(r"dir_acc", re.I),  # xgboost verbose eval with custom metric name
        ],
        'early_stopping_logged': [
            re.compile(r"best_iteration=\d+"),  # sklearn wrapper path
        ],
        'adaptive_learning_phase2_started': [
            re.compile(r"Phase 2: GERÇEK Adaptive Learning", re.I)
        ],
        'adaptive_learning_incremental_done': [
            re.compile(r"GERÇEK incremental learning: .*\+\d+ .* rounds")
        ],
        'regime_detection_training': [
            re.compile(r"Market regime score: \d+\.\d+")
        ],
        # Predict-time only signals (may be absent in training logs): keep as inconclusive by default
        'smart_ensemble_predict': [
            re.compile(r"Smart ensemble", re.I),
            re.compile(r"falling back to weighted average", re.I),
        ],
        'inference_caps_applied': [
            re.compile(r"delta clipped .* \(cap=\d+\.\d+\)")
        ],
        # Sample weights are not explicitly logged; mark inconclusive unless we add logging later
        'cv_sample_weights': []
    }


def evaluate_log(log_text: str) -> Dict[str, Any]:
    pats = compile_patterns()
    result: Dict[str, Any] = {}
    for key, regexes in pats.items():
        if not regexes:
            result[key] = {'status': 'inconclusive', 'evidence': None}
            continue
        found_any = False
        evidence: List[str] = []
        for rgx in regexes:
            m = rgx.search(log_text)
            if m:
                found_any = True
                # capture a nearby context line if possible
                evidence.append(m.group(0))
        result[key] = {
            'status': 'found' if found_any else 'absent',
            'sample': evidence[:3] if evidence else None,
        }

    # Adaptive learning success heuristic
    if result['adaptive_learning_phase2_started']['status'] == 'found':
        if result['adaptive_learning_incremental_done']['status'] == 'found':
            result['adaptive_learning'] = {'status': 'success'}
        else:
            # Look for explicit warning
            if re.search(r"Adaptive learning: Hiçbir model güncellenmedi", log_text):
                result['adaptive_learning'] = {'status': 'started_but_no_update'}
            else:
                result['adaptive_learning'] = {'status': 'started_unknown'}
    else:
        result['adaptive_learning'] = {'status': 'not_started'}

    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True, help='Path to training log file')
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(json.dumps({'error': 'log_not_found', 'path': args.log}))
        return 1

    with open(args.log, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    report = evaluate_log(text)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
