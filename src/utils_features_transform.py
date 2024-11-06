from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def create_pipeline():
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

    return pipeline


def pipeline_transform(df):
    pipeline = create_pipeline()
    transformed_df = pipeline.transform(df)
    return transformed_df
