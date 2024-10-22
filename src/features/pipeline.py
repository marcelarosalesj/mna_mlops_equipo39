import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder


def _encode_target(config_params):
    df_train = pd.read_csv(config_params["split_data"]["train_dataset_path"])
    df_test = pd.read_csv(config_params["split_data"]["test_dataset_path"])
    df_val = pd.read_csv(config_params["split_data"]["val_dataset_path"])

    X_train = df_train.drop("OUTPUT Grade", axis=1)
    X_test = df_test.drop("OUTPUT Grade", axis=1)
    X_val = df_val.drop("OUTPUT Grade", axis=1)
    y_train = df_train["OUTPUT Grade"]
    y_test = df_test["OUTPUT Grade"]
    y_val = df_val["OUTPUT Grade"]

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)

    return X_train, y_train, X_test, y_test, X_val, y_val


def _build_pipeline():

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

    # Combinar las transformaciones en un ColumnTransformer
    pipeline = ColumnTransformer(
        transformers=[
            ("OHE", catNOM_pipeline, catNOM_pipeline_nombres),
            ("Ordinal", catORD_pipeline, catORD_pipeline_nombres),
        ]
    )
    return pipeline


def feature_engineering(config_params):
    """
    Aplicando el pipeline al conjunto de train, val y test
    """
    pipeline = _build_pipeline()
    X_train, _, X_test, _, X_val, _ = _encode_target(config_params)

    X_train = pipeline.fit_transform(X_train)
    X_val = pipeline.transform(X_val)
    X_test = pipeline.transform(X_test)

    return pipeline
