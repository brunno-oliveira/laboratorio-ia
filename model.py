from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    plot_confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


class Model:
    @staticmethod
    def fit_and_predict(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        list_model: List = None,
    ) -> List:
        models_base_predict = []
        for result in list_model:
            name, model = result
            model.fit(X_train, y_train)
            predict = model.predict(X_test)
            accuracy = round(accuracy_score(y_test, predict), 4)
            f1 = round(f1_score(y_test, predict, average="macro"), 4)
            precision = round(precision_score(y_test, predict, average="macro"), 4)
            recall = round(recall_score(y_test, predict, average="macro"), 4)
            models_base_predict.append(
                {
                    "name": name,
                    "model": model,
                    "predict": predict,
                    "accuracy": accuracy,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                }
            )

        return models_base_predict

    @staticmethod
    def plot_results(list_predict, X_test, y_test, export_files=True):
        for result in list_predict:
            print(f"Model: {result['name']}")
            metrics = {
                "Accuracy": [result["accuracy"]],
                "F1": [result["f1"]],
                "Precision": [result["precision"]],
                "Recall": [result["recall"]],
            }

            metrics_df = pd.DataFrame.from_dict(
                metrics, orient="index", columns=["Valor"],
            )
            print(metrics_df)
            print()
            if result["name"] == "MLP":
                print(f"Alpha: {result['model'].alpha}")
                print(f"Hidden Layers Sizes: {result['model'].hidden_layer_sizes}")
            elif result["name"] == "SVM":
                print(f"C: {result['model'].C}")
                # Sigma e gamma são a mesma coisa, but
                # if gamma='scale' (default) is passed then it uses
                # 1 / (n_features * X.var()) as value of gamma,
                # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
                print(f"Sigma: {result['model'].gamma}")

            print()
            report = classification_report(y_test, result["predict"], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            plot_confusion_matrix(result["model"], X_test, y_test)
            print()
            if export_files:
                plt.savefig(f"outputs/img/matrix_{result['name'].lower()}.png")
            plt.show()
            print("--------------------------------------------")
