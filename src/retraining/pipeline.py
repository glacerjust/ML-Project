import argparse
from typing import Any
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from xgboost import XGBClassifier

from src.data.preprocessing import build_preprocessor, split_data
from src.registry.model_registry import MODEL_NAME, register_model, set_model_stage
from src.training.evaluate import evaluate_predictions

RETRAINING_PARAMS = {
    "n_estimators": 80,
    "learning_rate": 0.06,
    "max_depth": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "scale_pos_weight": 2.0,
    "random_state": 42,
    "eval_metric": "logloss",
}


def run_retraining_pipeline(
    data_path: str = "data/raw/Telco-Customer-Churn.csv",
    params: dict[str, Any] = RETRAINING_PARAMS,
    threshold: float = 0.35,
    run_name: str = "retrained-candidate",
) -> dict[str, Any]:
    """
    Executes an automated retraining cycle on fresh/updated data.
    Trains, evaluates, logs to MLflow, and registers strictly as @candidate.
    """
    df = pd.read_csv(data_path)
    X_train, X_val, _X_test, y_train, y_val, _y_test = split_data(df)

    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)

    mlflow.set_experiment("telco-churn")

    with mlflow.start_run(run_name=run_name) as run:
        # 1. Log training metadata
        mlflow.log_params(params)
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("retrained_trigger", "manual_or_drift")

        # 2. Train model
        model = XGBClassifier(**params)
        model.fit(X_train_p, y_train)

        # 3. Evaluate candidate model
        y_val_proba = model.predict_proba(X_val_p)[:, 1]
        metrics = evaluate_predictions(y_val.to_numpy(), y_val_proba, threshold=threshold)
        mlflow.log_metrics(metrics)

        # 4. Log artifacts
        mlflow.xgboost.log_model(model, name="model")
        mlflow.sklearn.log_model(
            preprocessor,
            name="preprocessor",
            serialization_format="cloudpickle",
        )

        run_id = run.info.run_id

    # 5. Register newly minted model strictly as @candidate (Version incremented)
    candidate_version = register_model(run_id=run_id, model_name=MODEL_NAME)
    set_model_stage(model_name=MODEL_NAME, version=candidate_version, stage="candidate")

    print(f"\n[RETRAINING COMPLETE] Candidate Version {candidate_version} produced.")
    print(f"Validation PR-AUC: {metrics['pr_auc']:.4f} | Recall: {metrics['recall']:.4f}")

    return {
        "run_id": run_id,
        "candidate_version": candidate_version,
        "metrics": metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger model retraining pipeline.")
    parser.add_argument("--data", default="data/raw/Telco-Customer-Churn.csv", help="Path to new labeled dataset.")
    args = parser.parse_args()

    run_retraining_pipeline(data_path=args.data)
