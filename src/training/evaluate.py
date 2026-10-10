from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """
    Compute comprehensive classification metrics for churn evaluation.
    Allows threshold adjustment to balance Precision and Recall.
    """
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    return metrics


if __name__ == "__main__":
    import pandas as pd

    from src.data.preprocessing import split_data
    from src.training.train import train_baseline

    # Train model and get predictions
    model, preprocessor = train_baseline()

    df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    X_val_p = preprocessor.transform(X_val)
    # Get continuous probabilities for class 1 (Churn = Yes)
    y_val_proba = model.predict_proba(X_val_p)[:, 1]

    # Evaluate at default threshold 0.5
    print("\n--- Evaluation at Threshold 0.5 ---")
    results_05 = evaluate_predictions(y_val.to_numpy(), y_val_proba, threshold=0.5)
    for k, v in results_05.items():
        print(f"{k}: {v}")

    # Evaluate at lower threshold 0.35 (Prioritize Recall)
    print("\n--- Evaluation at Threshold 0.35 (Catch more churners) ---")
    results_035 = evaluate_predictions(y_val.to_numpy(), y_val_proba, threshold=0.35)
    for k, v in results_035.items():
        print(f"{k}: {v}")
