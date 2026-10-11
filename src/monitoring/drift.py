import argparse
from typing import Any

import pandas as pd
from scipy.stats import ks_2samp

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


def detect_numerical_drift(
    reference: pd.Series, current: pd.Series, alpha: float = 0.05
) -> dict[str, Any]:
    """
    Two-sample Kolmogorov-Smirnov (KS) test for numerical distribution drift.
    Null Hypothesis (H0): Both samples come from the same distribution.
    If p-value < alpha: Reject H0 -> Statistically significant drift detected.
    """
    ref_clean = reference.dropna()
    curr_clean = current.dropna()

    res = ks_2samp(ref_clean, curr_clean)
    p_value = float(res.pvalue)
    statistic = float(res.statistic)

    return {
        "drift_detected": bool(p_value < alpha),
        "p_value": round(p_value, 5),
        "ks_statistic": round(statistic, 4),
        "ref_mean": round(float(ref_clean.mean()), 2),
        "curr_mean": round(float(curr_clean.mean()), 2),
    }


def analyze_dataset_drift(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, alpha: float = 0.05
) -> dict[str, Any]:
    """
    Analyze numerical features for data drift against the baseline reference dataset.
    """
    feature_reports: dict[str, Any] = {}
    drifted_features: list[str] = []

    for col in NUMERICAL_FEATURES:
        if col in reference_df.columns and col in current_df.columns:
            report = detect_numerical_drift(reference_df[col], current_df[col], alpha=alpha)
            feature_reports[col] = report
            if report["drift_detected"]:
                drifted_features.append(col)

    overall_drift = len(drifted_features) > 0

    return {
        "overall_drift_detected": overall_drift,
        "drifted_features": drifted_features,
        "features": feature_reports,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate data drift against reference baseline.")
    parser.add_argument("--reference", default="data/processed/train.csv", help="Path to reference baseline dataset.")
    parser.add_argument("--current", required=True, help="Path to current/production batch dataset to analyze.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for KS test.")
    args = parser.parse_args()

    ref_df = pd.read_csv(args.reference)
    curr_df = pd.read_csv(args.current)

    report = analyze_dataset_drift(ref_df, curr_df, alpha=args.alpha)
    print(f"\n--- Drift Analysis Report (alpha={args.alpha}) ---")
    print(f"Overall Drift Detected: {report['overall_drift_detected']}")
    if report["drifted_features"]:
        print(f"Drifted Features: {report['drifted_features']}")
    else:
        print("All numerical features align with baseline reference distribution.")

    for feat, rep in report["features"].items():
        status = "DRIFT" if rep["drift_detected"] else "OK"
        print(f"  [{status}] {feat}: p-value={rep['p_value']}, ref_mean={rep['ref_mean']}, curr_mean={rep['curr_mean']}")
