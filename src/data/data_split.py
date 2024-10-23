import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(config_params):
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

    df_train.to_csv(config_params["split_data"]["train_dataset_path"])
    df_test.to_csv(config_params["split_data"]["test_dataset_path"])
    df_val.to_csv(config_params["split_data"]["val_dataset_path"])
    print("Done saving artifacts")
