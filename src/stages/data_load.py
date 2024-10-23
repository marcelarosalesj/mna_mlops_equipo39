import argparse
import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.utils import read_config_params


def load_data(config_params):
    """
    read dataset for student grade prediction and save as CSV file
    """
    fetched_data = fetch_ucirepo(id=856)
    data = pd.concat([fetched_data.data.features, fetched_data.data.targets], axis=1)

    column_names = [
        "Student Age",
        "Sex",
        "Graduated High-school Type",
        "Scholarship Type",
        "Additional Work",
        "Regular Artistic/Sports Activity",
        "Do you have a Partner",
        "Total Salary",
        "Transportation",
        "Accommodation in Cyprus",
        "Mothers Education",
        "Fathers Education",
        "Number of Siblings",
        "Parental Status",
        "Mothers Occupation",
        "Fathers Occupation",
        "Weekly Study Hours",
        "Reading Frequency (Non-Scientific)",
        "Reading Frequency (Scientific)",
        "Attendance to Seminars",
        "Impact on Success",
        "Attendance to Classes",
        "Preparation to Midterm 1",
        "Preparation to Midterm 2",
        "Taking Notes in Classes",
        "Listening in Classes",
        "Discussion Improves Success",
        "Flip-Classroom",
        "Cumulative GPA Last Semester",
        "Expected GPA at Graduation",
        "COURSE ID",
        "OUTPUT Grade",
    ]
    data.columns = column_names
    print(f"Number of features in dataset: {len(column_names)}")
    print(f"Shape of initial dataset {data.shape}")

    data.to_csv(config_params["load_data"]["dataset_csv"], index=False)
    print("Done saving artifacts")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    params = read_config_params(args.config)

    load_data(params)
