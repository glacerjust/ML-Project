import numpy as np
from xgboost import XGBClassifier

MODEL_PATH = "models/production/model.json"


def test_model_prediction_output_contract():
    """Verify that model output probabilities are valid and bounded [0.0, 1.0]."""
    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    # 46 features matching the actual ColumnTransformer output
    dummy_features = np.zeros((2, 46))

    probas = model.predict_proba(dummy_features)

    assert probas.shape == (2, 2)
    assert not np.isnan(probas).any()
    assert np.all((probas >= 0.0) & (probas <= 1.0))
    assert np.allclose(probas.sum(axis=1), 1.0)
