import joblib
import pandas as pd

from src.data.preprocessing import clean_data

PREPROCESSOR_PATH = "models/production/preprocessor.joblib"


def test_preprocessing_handles_blank_total_charges():
    """Verify that blank strings in TotalCharges are coerced and do not crash."""
    raw_sample = pd.DataFrame([{
        "customerID": "9999-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 20.0,
        "TotalCharges": " ",  # Blank string from raw database
    }])

    cleaned = clean_data(raw_sample)
    assert "customerID" not in cleaned.columns
    assert pd.isna(cleaned["TotalCharges"].iloc[0])

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    transformed = preprocessor.transform(cleaned)
    # Total features: 3 numerical + 43 one-hot columns = 46 features
    assert transformed.shape == (1, 46)


def test_preprocessing_handles_unknown_categories():
    """Verify that unseen categories in production do not crash the encoder."""
    unseen_sample = pd.DataFrame([{
        "gender": "Non-binary",               # Unseen category
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Starlink",        # Unseen category
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Bi-Annual",             # Unseen category
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Crypto",            # Unseen category
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0,
    }])

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    transformed = preprocessor.transform(unseen_sample)

    assert transformed.shape == (1, 46)
