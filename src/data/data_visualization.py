import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class DataExplorer:
    def __init__(self, config_params):
        self.data = pd.read_csv(config_params["load_data"]["dataset_csv"])
        self.train_data = pd.read_csv(config_params["split_data"]["train_dataset_path"])
        self.test_data = pd.read_csv(config_params["split_data"]["test_dataset_path"])
        self.val_data = pd.read_csv(config_params["split_data"]["val_dataset_path"])

    def explore_data(self):
        print("\nHEADER\n")
        print(self.data.head().T)
        print("\n\nDATA DESCRIPTION\n")
        print(self.data.describe().T)
        print("\n\nDATA INFORMATION\n")
        print(self.data.info())
        print("\n\nDATA FREQUENCY\n")
        print(self.data.nunique())

    @staticmethod
    def _plot_correlation_matrix(data):
        corr_matrix = data.corr(method="pearson", numeric_only=True)
        plt.figure(figsize=(17, 10))
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1
        )
        plt.title("Pearson correlation matrix")
        plt.show()

    def plot_correlation_matrix_for_train_data(self):
        self._plot_correlation_matrix(self.train_data)

    def plot_categorical_distributions_for_train_data(self):
        _, axes = plt.subplots(7, 5, figsize=(20, 28))
        axes = axes.ravel()

        for col, ax in zip(self.data.columns[:-1], axes):
            sns.countplot(
                x=self.train_data[col],
                hue=self.train_data["OUTPUT Grade"],
                ax=ax,
                palette="bright",
            )
            ax.set(title=f"{col}", xlabel=None)
