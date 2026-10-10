import argparse
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from xgboost import XGBClassifier

from src.data.preprocessing import build_preprocessor, load_processed_data
from src.training.evaluate import evaluate_predictions

BASELINE_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 1.0,
    "scale_pos_weight": 1.0,
    "random_state": 42,
    "eval_metric": "logloss",
}

TUNED_PARAMS = {
    "n_estimators": 75,
    "learning_rate": 0.07,
    "max_depth": 3,
    "subsample": 0.82,
    "colsample_bytree": 0.75,
    "scale_pos_weight": 2.0,
    "random_state": 42,
    "eval_metric": "logloss",
}


def train_model(
    params: dict[str, Any] | None = None,
    experiment_name: str = "telco-churn",
    threshold: float = 0.5,
    run_name: str | None = None,
):
    if params is None:
        params = BASELINE_PARAMS

    X_train, X_val, _X_test, y_train, y_val, _y_test = load_processed_data()

    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # 1. Log parameters
        mlflow.log_params(params)
        mlflow.log_param("threshold", threshold)

        # 2. Train model
        model = XGBClassifier(**params)
        model.fit(X_train_p, y_train)

        # 3. Evaluate on validation set
        y_val_proba = model.predict_proba(X_val_p)[:, 1]
        metrics = evaluate_predictions(y_val.to_numpy(), y_val_proba, threshold=threshold)

        # 4. Log metrics
        mlflow.log_metrics(metrics)

        # 5. Log artifacts (model & preprocessor)
        mlflow.xgboost.log_model(model, name="model")
        mlflow.sklearn.log_model(
            preprocessor,
            name="preprocessor",
            serialization_format="cloudpickle",
        )

        print(f"Run completed [{run_name or 'unnamed'}]. Val PR-AUC: {metrics['pr_auc']:.4f}, Recall: {metrics['recall']:.4f}")
        return model, preprocessor, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train customer churn model (1 run per execution).")
    parser.add_argument("--tuned", action="store_true", help="Use tuned hyperparameters instead of baseline.")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (default: 0.5 baseline, 0.35 tuned).")
    args = parser.parse_args()

    if args.tuned:
        threshold = args.threshold if args.threshold is not None else 0.35
        train_model(TUNED_PARAMS, threshold=threshold, run_name="xgboost-tuned")
    else:
        threshold = args.threshold if args.threshold is not None else 0.5
        train_model(BASELINE_PARAMS, threshold=threshold, run_name="xgboost-baseline")
