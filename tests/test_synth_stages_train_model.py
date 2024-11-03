from unittest.mock import patch, MagicMock
import pandas as pd
from src.stages.train_model import train_model


@patch("src.stages.train_model.available_models")
def test_train_model(mock_available_models):
    config_params = {
        "train": {
            "algo": "mock_algorithm",
            "model_name": "test_model",
            "model_path": "/path/to/model",
            "param1": 10,
            "param2": 0.1,
        }
    }
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "target": [0, 1, 0, 1, 0],
    }
    df_train = pd.DataFrame(data)
    mock_model_instance = MagicMock()
    mock_available_models.get.return_value = MagicMock(return_value=mock_model_instance)

    model, model_path = train_model(config_params, df_train)

    mock_available_models.get.assert_called_once_with("mock_algorithm")
    mock_model_instance.fit.assert_called_once()  # Verifica que `fit` fue llamado

    assert model_path == "/path/to/model"
