import pandas as pd

from src.stages.data_split import split_data


def test_split_data():

    input_data = pd.DataFrame(
        {
            "Columna1": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            "Columna2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "OUTPUT Grade": [
                True,
                False,
                True,
                False,
                True,
                True,
                True,
                False,
                False,
                True,
            ],
            "Columna4": [10.5, 20.5, 30.5, 40.5, 50.5, 60.5, 70.5, 80.5, 90.5, 100.5],
        }
    )
    config = {
        "split_data": {
            "train_proportion": 0.3,
            "random_state": 42,
            "val_proportion_wrt_test": 0.5,
        }
    }

    expected_dt_train, expected_dt_test, expected_dt_val = split_data(
        config, input_data
    )

    assert isinstance(expected_dt_train, pd.core.frame.DataFrame), "Wrong data type"
    assert isinstance(expected_dt_test, pd.core.frame.DataFrame), "Wrong data type"
    assert isinstance(expected_dt_val, pd.core.frame.DataFrame), "Wrong data type"
