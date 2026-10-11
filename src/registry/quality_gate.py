import argparse
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from mlflow.tracking import MlflowClient

from src.registry.model_registry import (
    MODEL_NAME,
    set_model_stage,
)
from src.training.evaluate import evaluate_predictions


def evaluate_version_on_test_data(
    model_name: str, alias_or_version: str, test_df: pd.DataFrame
) -> dict[str, Any]:
    """Load a specific model version and its preprocessor, evaluating on test data."""
    client = MlflowClient()

    if alias_or_version.isdigit():
        mv = client.get_model_version(name=model_name, version=alias_or_version)
        model_uri = f"models:/{model_name}/{alias_or_version}"
    else:
        mv = client.get_model_version_by_alias(name=model_name, alias=alias_or_version)
        model_uri = f"models:/{model_name}@{alias_or_version}"

    run_id = mv.run_id
    run = client.get_run(run_id)
    threshold = float(run.data.params.get("threshold", 0.35))

    model = mlflow.xgboost.load_model(model_uri)
    preprocessor = mlflow.sklearn.load_model(f"runs:/{run_id}/preprocessor")

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"].astype(int)

    X_test_p = preprocessor.transform(X_test)
    y_proba = model.predict_proba(X_test_p)[:, 1]

    metrics = evaluate_predictions(y_test.to_numpy(), y_proba, threshold=threshold)
    metrics["version"] = str(mv.version)
    return metrics


def run_quality_gate(
    model_name: str = MODEL_NAME,
    test_data_path: str = "data/processed/test.csv",
    min_improvement: float = 0.0,
) -> dict[str, Any]:
    """
    Champion vs Challenger Quality Gate.
    Compares @candidate against @production on identical held-out test data.
    Promotes candidate if it beats production; otherwise rejects it.
    """
    client = MlflowClient()
    test_df = pd.read_csv(test_data_path)

    # 1. Fetch current production champion
    try:
        prod_metrics = evaluate_version_on_test_data(model_name, "production", test_df)
    except Exception as e:  # noqa: BLE001
        print(f"No active production model found: {e}")
        return {"decision": "ERROR", "reason": "No production model"}

    # 2. Fetch candidate challenger
    try:
        cand_metrics = evaluate_version_on_test_data(model_name, "candidate", test_df)
    except Exception as e:  # noqa: BLE001
        print(f"No active candidate model found: {e}")
        return {
            "decision": "NO_OP",
            "production_version": prod_metrics["version"],
            "reason": f"No active candidate model found: {e}",
        }

    if prod_metrics["version"] == cand_metrics["version"]:
        print(f"\n[Quality Gate] No active challenger: Version {prod_metrics['version']} is already the active Production model.")
        return {
            "decision": "NO_OP",
            "production_version": prod_metrics["version"],
            "reason": "Candidate and production are the same version",
        }

    prod_pr_auc = prod_metrics["pr_auc"]
    cand_pr_auc = cand_metrics["pr_auc"]
    diff = cand_pr_auc - prod_pr_auc

    print(f"\n--- Quality Gate: Champion (v{prod_metrics['version']}) vs Challenger (v{cand_metrics['version']}) ---")
    print(f"Production (v{prod_metrics['version']}) Test PR-AUC: {prod_pr_auc:.4f} | Recall: {prod_metrics['recall']:.4f}")
    print(f"Candidate  (v{cand_metrics['version']}) Test PR-AUC: {cand_pr_auc:.4f} | Recall: {cand_metrics['recall']:.4f}")
    print(f"PR-AUC Difference: {diff:+.4f} (Required minimum improvement: {min_improvement:+.4f})")

    # 3. Decision rule
    if diff > min_improvement:
        print(f"\n>>> DECISION: PROMOTE! Candidate v{cand_metrics['version']} beats Production.")
        set_model_stage(model_name=model_name, version=cand_metrics["version"], stage="production")
        decision = "PROMOTED"
    else:
        print(f"\n>>> DECISION: REJECT! Candidate v{cand_metrics['version']} did not beat Production.")
        client.set_model_version_tag(
            name=model_name,
            version=cand_metrics["version"],
            key="quality_gate",
            value="rejected",
        )
        decision = "REJECTED"

    return {
        "decision": decision,
        "production_version": prod_metrics["version"],
        "candidate_version": cand_metrics["version"],
        "production_pr_auc": prod_pr_auc,
        "candidate_pr_auc": cand_pr_auc,
        "diff": round(diff, 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Candidate vs Production Model Quality Gate.")
    parser.add_argument("--test-data", default="data/processed/test.csv", help="Path to held-out test data.")
    parser.add_argument("--min-gain", type=float, default=0.0, help="Minimum PR-AUC gain required to promote.")
    args = parser.parse_args()

    run_quality_gate(test_data_path=args.test_data, min_improvement=args.min_gain)
