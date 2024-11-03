import pandas as pd
from unittest.mock import patch
from src.stages.features_transform import features_transform


@patch("src.stages.features_transform._encode_features")
@patch("src.stages.features_transform._encode_target")
def test_features_transform(mock_encode_target, mock_encode_features):
    data = {"feature1": [1, 2, 3], "feature2": [10, 20, 30], "OUTPUT Grade": [0, 1, 0]}
    df_train = pd.DataFrame(data)
    df_test = pd.DataFrame(data)
    df_val = pd.DataFrame(data)

    mock_encode_features.return_value = (
        df_train.drop("OUTPUT Grade", axis=1),
        df_test.drop("OUTPUT Grade", axis=1),
        df_val.drop("OUTPUT Grade", axis=1),
    )
    mock_encode_target.return_value = (
        df_train["OUTPUT Grade"],
        df_test["OUTPUT Grade"],
        df_val["OUTPUT Grade"],
    )

    df_train_transformed, df_test_transformed, df_val_transformed = features_transform(
        df_train, df_test, df_val
    )

    assert df_train_transformed.shape == (3, 3)
    assert df_test_transformed.shape == (3, 3)
    assert df_val_transformed.shape == (3, 3)

    mock_encode_features.assert_called_once()
    mock_encode_target.assert_called_once()
