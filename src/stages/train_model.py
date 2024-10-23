import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

available_models = {
    "knn": KNeighborsClassifier,
}


def train_model(config_params):
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
    with open(model_path, "wb") as ff:
        pickle.dump(model, ff)
