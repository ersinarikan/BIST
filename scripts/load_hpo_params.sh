#!/usr/bin/env bash
# Load HPO parameters into environment variables for training scripts

CONFIG_FILE="/opt/bist-pattern/config/best_hpo_params.json"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "⚠️  No HPO config found at $CONFIG_FILE, using defaults"
    exit 0
fi

# Ensure path is available to subprocesses
export CONFIG_FILE

# Use Python to parse JSON and emit export statements, then eval to apply
eval "$(python3 <<'PYEOF'
import json
import os
import sys

try:
    config_path = os.environ.get('CONFIG_FILE', '/opt/bist-pattern/config/best_hpo_params.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # For each horizon, export the best_params
    for horizon_key, data in sorted(config.items(), key=lambda x: int(x[0][:-1])):
        horizon_num = horizon_key[:-1]  # '1d' -> '1'
        params = data.get('best_params', {})
        
        # Export XGBoost params (with 'xgb_' prefix removed)
        for key, val in sorted(params.items()):
            if key.startswith('xgb_'):
                env_key = f"OPTUNA_XGB_{key.replace('xgb_', '').upper()}"
                print(f"export {env_key}='{val}'")
        
        # Export LightGBM params (with 'lgb_' prefix removed)
        for key, val in sorted(params.items()):
            if key.startswith('lgb_'):
                env_key = f"OPTUNA_LGB_{key.replace('lgb_', '').upper()}"
                print(f"export {env_key}='{val}'")
        
        # Export CatBoost params (with 'cat_' prefix removed)
        for key, val in sorted(params.items()):
            if key.startswith('cat_'):
                env_key = f"OPTUNA_CAT_{key.replace('cat_', '').upper()}"
                print(f"export {env_key}='{val}'")
        
        # Also check separate best_params_xgb, best_params_lgb, best_params_cat if available
        params_xgb = data.get('best_params_xgb', {})
        for key, val in sorted(params_xgb.items()):
            env_key = f"OPTUNA_XGB_{key.upper()}"
            print(f"export {env_key}='{val}'")
        
        params_lgb = data.get('best_params_lgb', {})
        for key, val in sorted(params_lgb.items()):
            env_key = f"OPTUNA_LGB_{key.upper()}"
            print(f"export {env_key}='{val}'")
        
        params_cat = data.get('best_params_cat', {})
        for key, val in sorted(params_cat.items()):
            env_key = f"OPTUNA_CAT_{key.upper()}"
            print(f"export {env_key}='{val}'")
        
        # Export adaptive_k if present
        if data.get('adaptive_k') is not None:
            print(f"export ML_ADAPTIVE_K_{horizon_num}D='{data['adaptive_k']:.4f}'")
        
        # Export pattern_weight if present
        if data.get('pattern_weight') is not None:
            print(f"export ML_PATTERN_WEIGHT_SCALE_{horizon_num}D='{data['pattern_weight']:.4f}'")
    
except Exception as e:
    print(f"⚠️  Failed to load HPO config: {e}", file=sys.stderr)
    sys.exit(0)  # Don't fail, just skip
    
PYEOF
)"
