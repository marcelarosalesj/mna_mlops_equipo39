import pytest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.stages.evaluate import evaluate_model


def test_evaluate():
    config_params = {}

    data = {
        "feature1": [5, 2, 8, 9, 1],
        "feature2": [3, 7, 1, 5, 2],
        "62": [1, 0, 1, 1, 0],
    }
    df_test = pd.DataFrame(data)

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1, 0, 1, 0, 1])

    metrics, disp = evaluate_model(config_params, df_test, mock_model)

    expected_accuracy = 0.6
    expected_mse = 0.4
    expected_rmse = np.sqrt(expected_mse)
    expected_cm = confusion_matrix(
        df_test["62"], mock_model.predict(df_test.iloc[:, :-1])
    )

    assert metrics["accuracy"] == pytest.approx(expected_accuracy, 0.01)
    assert metrics["mse"] == pytest.approx(expected_mse, 0.01)
    assert metrics["rmse"] == pytest.approx(expected_rmse, 0.01)

    assert isinstance(disp, ConfusionMatrixDisplay)
    np.testing.assert_array_equal(disp.confusion_matrix, expected_cm)
