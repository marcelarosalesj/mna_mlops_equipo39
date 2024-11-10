import argparse
import numpy as np
import pandas as pd

from src.utils import read_config_params
from src.utils_features_transform import create_pipeline

from sklearn.preprocessing import LabelEncoder


def _encode_target(y_train, y_test, y_val):
    """
    Run encoder for target
    """

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)

    return y_train, y_test, y_val


def _encode_features(X_train, X_test, X_val):
    """
    Run encodeing pipeline for features
    """
    pipeline = create_pipeline()

    X_train = pipeline.fit_transform(X_train)

    import pickle

    with open("artifacts/model_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    X_test = pipeline.transform(X_test)
    X_val = pipeline.transform(X_val)

    return X_train, X_test, X_val


def features_transform(df_train, df_test, df_val):
    """
    Aplicando el pipeline al conjunto de train, val y test
    """

    target_column = "OUTPUT Grade"

    X_train = df_train.drop(target_column, axis=1)
    X_test = df_test.drop(target_column, axis=1)
    X_val = df_val.drop(target_column, axis=1)

    y_train = df_train[target_column]
    y_test = df_test[target_column]
    y_val = df_val[target_column]

    X_train, X_test, X_val = _encode_features(X_train, X_test, X_val)

    y_train, y_test, y_val = _encode_target(y_train, y_test, y_val)

    train_dataset = np.column_stack([X_train, y_train])
    test_dataset = np.column_stack([X_test, y_test])
    val_dataset = np.column_stack([X_val, y_val])

    print("finish transformations")

    df_train = pd.DataFrame(train_dataset)
    df_test = pd.DataFrame(test_dataset)
    df_val = pd.DataFrame(val_dataset)

    print(f"shapes for train -> {train_dataset.shape}")
    print(f"shapes for test -> {test_dataset.shape}")
    print(f"shapes for val -> {val_dataset.shape}")

    return df_train, df_test, df_val


def features_transform_dvc(config_params):
    df_train = pd.read_csv(config_params["split_data"]["train_dataset_path"])
    df_test = pd.read_csv(config_params["split_data"]["test_dataset_path"])
    df_val = pd.read_csv(config_params["split_data"]["val_dataset_path"])

    df_train, df_test, df_val = features_transform(df_train, df_test, df_val)

    df_train.columns = df_train.columns.astype(str)
    df_test.columns = df_test.columns.astype(str)
    df_val.columns = df_val.columns.astype(str)

    df_train.to_csv(config_params["features"]["features_train_dataset"], index=False)
    df_test.to_csv(config_params["features"]["features_test_dataset"], index=False)
    df_val.to_csv(config_params["features"]["features_val_dataset"], index=False)
    print("Done saving artifacts")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    params = read_config_params(args.config)

    features_transform_dvc(params)
