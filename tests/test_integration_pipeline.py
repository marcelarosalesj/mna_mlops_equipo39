from src.stages.data_load import load_data
from src.stages.data_split import split_data
from src.stages.features_transform import features_transform
from src.stages.train_model import train_model
from src.stages.evaluate import evaluate_model


def test_integration_pipeline():

    config_params = {
        "load_data": {"dataset_csv": "artifacts/downloaded_dataset.csv"},
        "split_data": {
            "train_proportion": 0.7,
            "val_proportion_wrt_test": 0.5,
            "random_state": 12,
            "train_dataset_path": "artifacts/split_train_dataset.csv",
            "test_dataset_path": "artifacts/split_test_dataset.csv",
            "val_dataset_path": "artifacts/split_val_dataset.csv",
        },
        "features": {
            "features_train_dataset": "artifacts/features_train_dataset.csv",
            "features_test_dataset": "artifacts/features_test_dataset.csv",
            "features_val_dataset": "artifacts/features_val_dataset.csv",
        },
        "train": {
            "model_name": "SGP",
            "algo": "rf",
            "model_path": "artifacts/model_test1.pkl",
            "n_estimators": 100,
            "max_depth": 6,
            "random_state": 12,
        },
        "evaluate": {
            "metrics_file": "artifacts/metrics.json",
            "cm_file": "artifacts/confusion_matrix.png",
        },
    }

    data = load_data()
    df_train, df_test, df_val = split_data(config_params, data)
    df_train, df_test, df_val = features_transform(df_train, df_test, df_val)
    model, _ = train_model(config_params, df_train)
    metrics, _ = evaluate_model(config_params, df_test, model)

    assert metrics.get("accuracy")
    assert metrics.get("mse")
    assert metrics.get("rmse")
