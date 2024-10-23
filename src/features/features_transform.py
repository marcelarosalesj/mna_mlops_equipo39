import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder


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
    catNOM_pipeline = Pipeline(
        steps=[
            (
                "OneHot",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
            )
        ]
    )
    catORD_pipeline = Pipeline(steps=[("Ordinal", OrdinalEncoder())])

    catNOM_pipeline_nombres = [
        "Sex",
        "Graduated High-school Type",
        "Scholarship Type",
        "Additional Work",
        "Regular Artistic/Sports Activity",
        "Do you have a Partner",
        "Transportation",
        "Accommodation in Cyprus",
        "Mothers Education",
        "Fathers Education",
        "Parental Status",
        "Mothers Occupation",
        "Fathers Occupation",
        "Attendance to Seminars",
        "Impact on Success",
        "Attendance to Classes",
        "Preparation to Midterm 1",
        "Preparation to Midterm 2",
        "Taking Notes in Classes",
        "Listening in Classes",
        "Discussion Improves Success",
        "Flip-Classroom",
    ]
    catORD_pipeline_nombres = [
        "Student Age",
        "Number of Siblings",
        "Total Salary",
        "Weekly Study Hours",
        "Reading Frequency (Non-Scientific)",
        "Reading Frequency (Scientific)",
        "Cumulative GPA Last Semester",
        "Expected GPA at Graduation",
        "Student Age",
        "Number of Siblings",
    ]

    pipeline = ColumnTransformer(
        transformers=[
            ("OHE", catNOM_pipeline, catNOM_pipeline_nombres),
            ("Ordinal", catORD_pipeline, catORD_pipeline_nombres),
        ]
    )

    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)
    X_val = pipeline.transform(X_val)

    return X_train, X_test, X_val


def features_transform(config_params):
    """
    Aplicando el pipeline al conjunto de train, val y test
    """
    df_train = pd.read_csv(config_params["split_data"]["train_dataset_path"])
    df_test = pd.read_csv(config_params["split_data"]["test_dataset_path"])
    df_val = pd.read_csv(config_params["split_data"]["val_dataset_path"])

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

    print(f"finish transformations")

    df_train = pd.DataFrame(train_dataset)
    df_test = pd.DataFrame(test_dataset)
    df_val = pd.DataFrame(val_dataset)

    print(f"shapes for train -> {train_dataset.shape}")
    print(f"shapes for test -> {test_dataset.shape}")
    print(f"shapes for val -> {val_dataset.shape}")

    df_train.to_csv(config_params["features"]["features_train_dataset"])
    df_test.to_csv(config_params["features"]["features_test_dataset"])
    df_val.to_csv(config_params["features"]["features_val_dataset"])
    print("Done saving artifacts")
