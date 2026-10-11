import os
from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types and drop identifiers without leaking test statistics."""
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    """Split dataframe into stratified Train, Val, and Test sets."""
    df = clean_data(df)
    # Safely map 'Yes'/'No' to 1/0 regardless of pandas string vs object dtype
    if df["Churn"].isin(["Yes", "No"]).any():
        y = (df["Churn"] == "Yes").astype(int)
    else:
        y = df["Churn"].astype(int)
    X = df.drop(columns=["Churn"])

    temp_size = test_size + val_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=random_state, stratify=y
    )

    relative_test_size = test_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_and_save_data(
    raw_path: str = "data/raw/Telco-Customer-Churn.csv",
    processed_dir: str = "data/processed",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean, split, and persist datasets to data/processed/ for reproducible lineage."""
    df = pd.read_csv(raw_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, test_size=test_size, val_size=val_size, random_state=random_state
    )

    train_df = X_train.copy()
    train_df["Churn"] = y_train

    val_df = X_val.copy()
    val_df["Churn"] = y_val

    test_df = X_test.copy()
    test_df["Churn"] = y_test

    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    return train_df, val_df, test_df


def load_processed_data(processed_dir: str = "data/processed"):
    """Load train, val, and test partitions from data/processed/."""
    train_path = os.path.join(processed_dir, "train.csv")
    val_path = os.path.join(processed_dir, "val.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        prepare_and_save_data(processed_dir=processed_dir)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]

    X_val = val_df.drop(columns=["Churn"])
    y_val = val_df["Churn"]

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor() -> ColumnTransformer:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ]
    )


if __name__ == "__main__":
    train_df, val_df, test_df = prepare_and_save_data()
    print("Saved processed partitions to data/processed/:")
    print(f"  train.csv: {train_df.shape}")
    print(f"  val.csv:   {val_df.shape}")
    print(f"  test.csv:  {test_df.shape}")
