import students_grades_predictions_class

model = students_grades_predictions_class.StudentPerformanceModel(dataset_id=856)
model.load_data()
model.preprocess_data()

# EDA
#students_grades_predictions_class.DataExplorer.explore_data(model.data) # def explore_data(data):
#students_grades_predictions_class.DataExplorer.plot_correlation_matrix(model.X_train, model.y_train) # def plot_correlation_matrix(X_data, y_data):
#students_grades_predictions_class.DataExplorer.plot_categorical_distributions(model.X_train, model.data.columns[:-1], model.y_train) # def plot_categorical_distributions(data, columns, target)

# Modelling and processing
model.encode_target().build_pipeline().apply_pipeline().train_model_dtree().evaluate_model("test").cross_validate_model()