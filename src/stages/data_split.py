import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import read_config_params


def split_data(config_params, dvc_enabled=False):
    data = pd.read_csv(config_params["load_data"]["dataset_csv"])

    X = data.drop("OUTPUT Grade", axis=1)
    y = data["OUTPUT Grade"]

    print(
        f"Splitting dataset into train and test-val dataset - {config_params['split_data']['train_proportion']} train proportion"
    )
    X_train, X_testval, y_train, y_testval = train_test_split(
        X,
        y,
        train_size=config_params["split_data"]["train_proportion"],
        stratify=y,
        random_state=config_params["split_data"]["random_state"],
    )

    print(
        f"Splitting test-val dataset into test and validation dataset - {config_params["split_data"]["val_proportion_wrt_test"]} val proportion with respect to test"
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_testval,
        y_testval,
        train_size=config_params["split_data"]["val_proportion_wrt_test"],
        stratify=y_testval,
        random_state=config_params["split_data"]["random_state"],
    )

    df_train = X_train.merge(y_train, left_index=True, right_index=True)
    df_test = X_test.merge(y_test, left_index=True, right_index=True)
    df_val = X_val.merge(y_val, left_index=True, right_index=True)

    if dvc_enabled:
        df_train.to_csv(config_params["split_data"]["train_dataset_path"], index=False)
        df_test.to_csv(config_params["split_data"]["test_dataset_path"], index=False)
        df_val.to_csv(config_params["split_data"]["val_dataset_path"], index=False)
        print("Done saving artifacts")
    else:
        return df_train, df_test, df_val


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument("--dvc", dest="dvc", required=True, action="store_true")
    args = args_parser.parse_args()

    params = read_config_params(args.config)

    split_data(params, args.dvc)
