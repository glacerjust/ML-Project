import pandas as pd
from src.monitoring.drift import analyze_dataset_drift


def test_drift_detector_detects_no_drift_on_identical_distributions():
    df = pd.read_csv("data/processed/train.csv")
    sample_a = df.sample(n=150, random_state=42)
    sample_b = df.sample(n=150, random_state=99)

    report = analyze_dataset_drift(sample_a, sample_b, alpha=0.01)
    assert not report["overall_drift_detected"]
    assert len(report["drifted_features"]) == 0


def test_drift_detector_detects_significant_shift():
    df = pd.read_csv("data/processed/train.csv")
    sample_a = df.sample(n=150, random_state=42)
    sample_b = df.sample(n=150, random_state=99).copy()

    # Artificially inflate MonthlyCharges by $50
    sample_b["MonthlyCharges"] = sample_b["MonthlyCharges"] + 50.0

    report = analyze_dataset_drift(sample_a, sample_b, alpha=0.01)
    assert report["overall_drift_detected"]
    assert "MonthlyCharges" in report["drifted_features"]
