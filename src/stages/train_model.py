import pickle
import argparse
import pandas as pd
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from src.utils import read_config_params

available_models = {
    "knn": KNeighborsClassifier,
    "dt": DecisionTreeClassifier,
    "rf": RandomForestClassifier,
    "xgb": XGBClassifier,
}


def train_model(config_params, dvc_enabled=False):
    """
    train model
    """
    df_train = pd.read_csv(config_params["features"]["features_train_dataset"])

    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1:]
    y_train = y_train.values.ravel()

    train_params = config_params["train"].copy()

    selected_algorithm = train_params.pop("algo")
    model_name = train_params.pop("model_name")
    model_path = train_params.pop("model_path")

    model = available_models.get(selected_algorithm)

    print(f"Training model {model_name} with {train_params}")

    model = model(**train_params)

    model.fit(X_train, y_train)

    if dvc_enabled:
        with open(model_path, "wb") as ff:
            pickle.dump(model, ff)
    else:
        return model


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument("--dvc", dest="dvc", required=True, action="store_true")
    args = args_parser.parse_args()

    params = read_config_params(args.config)

    train_model(params, args.dvc)
