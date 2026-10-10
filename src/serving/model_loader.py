import json
import os
from typing import Any

import joblib
from xgboost import XGBClassifier

PRODUCTION_DIR = "models/production"


class ModelArtifacts:
    def __init__(self, model: Any, preprocessor: Any, version: str, threshold: float):
        self.model = model
        self.preprocessor = preprocessor
        self.version = version
        self.threshold = threshold


def load_production_artifacts(prod_dir: str = PRODUCTION_DIR) -> ModelArtifacts:
    """
    Load the production model bundle from disk.
    OS-agnostic, portable, and ultra-fast (no runtime database dependencies).
    """
    meta_path = os.path.join(prod_dir, "metadata.json")
    model_path = os.path.join(prod_dir, "model.json")
    prep_path = os.path.join(prod_dir, "preprocessor.joblib")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Production bundle not found in '{prod_dir}'. Run promotion/export first.")

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    # 1. Load XGBoost model using native universal format
    model = XGBClassifier()
    model.load_model(model_path)

    # 2. Load preprocessor
    preprocessor = joblib.load(prep_path)

    version_str = str(metadata.get("version", "unknown"))
    threshold = float(metadata.get("threshold", 0.35))

    return ModelArtifacts(
        model=model,
        preprocessor=preprocessor,
        version=version_str,
        threshold=threshold,
    )
