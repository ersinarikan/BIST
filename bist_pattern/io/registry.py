import os
import json
from datetime import datetime


def write_manifest(model_dir: str, symbol: str, horizons: list[int], feature_columns: list[str], horizon_features_map: dict) -> str:
    manifest = {
        'symbol': symbol,
        'trained_at': datetime.now().isoformat(),
        'horizons': [f"{h}d" for h in horizons],
        'feature_count': int(len(feature_columns or [])),
        'horizon_features': {f"{h}d": list(horizon_features_map.get(f"{symbol}_{h}d_features", [])) for h in horizons},
    }
    path = os.path.join(model_dir, f"{symbol}_manifest.json")
    with open(path, 'w') as wf:
        json.dump(manifest, wf)
    return path
