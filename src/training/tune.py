import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from src.data.preprocessing import build_preprocessor, split_data

# Suppress Optuna's verbose log output per trial
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial, X_train_p, y_train, X_val_p, y_val) -> float:
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=25),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 3.0),
        "random_state": 42,
        "eval_metric": "logloss",
    }

    model = XGBClassifier(**params)
    model.fit(X_train_p, y_train)

    y_val_proba = model.predict_proba(X_val_p)[:, 1]
    val_pr_auc = average_precision_score(y_val, y_val_proba)

    return float(val_pr_auc)


def run_tuning(n_trials: int = 20):
    df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
    X_train, X_val, _X_test, y_train, y_val, _y_test = split_data(df)

    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train_p, y_train, X_val_p, y_val),
        n_trials=n_trials,
    )

    print(f"Completed {n_trials} trials.")
    print(f"Best Validation PR-AUC: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")

    return study


if __name__ == "__main__":
    run_tuning(n_trials=20)
