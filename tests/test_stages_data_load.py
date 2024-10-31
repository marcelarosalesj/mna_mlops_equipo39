from src.stages.data_load import load_data
import pandas


def test_load_data():
    df = load_data()
    assert isinstance(df, pandas.core.frame.DataFrame), "Wrong data type"
