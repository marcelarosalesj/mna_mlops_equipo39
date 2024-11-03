from src.stages.data_load import load_data
import pandas as pd


def test_load_data():
    df = load_data()
    assert isinstance(df, pd.core.frame.DataFrame), "Wrong data type"
