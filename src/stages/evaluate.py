import json
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    mean_squared_error,
)
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

from src.utils import read_config_params


def get_model(config_params):
    with open(config_params["train"]["model_path"], "rb") as ff:
        model = pickle.load(ff)
    return model


def evaluate_model(config_params, dvc_enabled=False):
    """
    evaluate model
    """

    df_test = pd.read_csv(config_params["features"]["features_test_dataset"])
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1:]
    y_test = y_test["62"].to_numpy()  # Convert to numpy array

    # df_val = pd.read_csv(config_params["features"]["features_val_dataset"])
    # X_val = df_val.iloc[:, :-1]
    # y_val = df_val.iloc[:, -1:]

    model = get_model(config_params)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy = {accuracy}")
    print(f"MSE = {mse}")
    print(f"RMSE = {rmse}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
    disp.plot().figure_.savefig(config_params["evaluate"]["cm_file"])

    metrics = {
        "accuracy": accuracy,
        "mse": mse,
        "rmse": rmse,
    }
    if dvc_enabled:
        with open(config_params["evaluate"]["metrics_file"], "w") as ff:
            json.dump(metrics, ff)
    else:
        return metrics


def cross_validate_model(config_params):
    df_train = pd.read_csv(config_params["features"]["features_train_dataset"])
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1:]
    y_train = y_train["62"].to_numpy()  # Convert to numpy array

    model = get_model(config_params)

    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(
        f"Accuracy con validacion cruzada del conjunto de entrenamiento: {np.mean(scores):.4f}"
    )
    return scores


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument("--dvc", dest="dvc", required=True, action="store_true")
    args = args_parser.parse_args()

    params = read_config_params(args.config)

    evaluate_model(params, args.dvc)
