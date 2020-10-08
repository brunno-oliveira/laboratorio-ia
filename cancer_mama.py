from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import pandas as pd
import numpy as np
import random

from model import Model

random.seed(47)
RANDOM_NUM = 47
np.random.seed(47)


class CancerMama:
    def __init__(self):
        self.model = Model()
        self.df = self.get_data()
        self.df_novo = self.get_new_data()

    @staticmethod
    def get_data():
        return pd.read_csv("data_source/Material 02 - 2 - Cancer de Mama - Dados.csv")

    @staticmethod
    def get_new_data():
        return pd.read_csv(
            "data_source/Material 02 - 2 - Cancer de Mama - Dados - Novos Casos.csv"
        )

    @staticmethod
    def get_models():
        return [("SVM", SVC(),), ("MLP", MLPClassifier(),)]

    def run_hold_out(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.df.iloc[:, 1:-1],  # 1 coluna = ID
            self.df.iloc[:, -1],  # Remove target
            test_size=0.2,
            random_state=RANDOM_NUM,
        )
        result = self.model.fit_and_predict(
            X_train, X_test, y_train, y_test, self.get_models()
        )

        self.model.plot_results(result, X_test, y_test)

        return result

    def cross_val(self):
        for model in self.get_models():
            scores = cross_val_score(
                model[1], self.df.iloc[:, 1:-1], self.df.iloc[:, -1], cv=10, n_jobs=-1
            )
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        return scores


if __name__ == "__main__":
    CancerMama().cross_val()
