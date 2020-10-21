from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

ConvergenceWarning("ignore")

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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df.iloc[:, 1:-1],  # 1 coluna = ID
            self.df.iloc[:, -1],  # Remove target
            test_size=0.2,
            random_state=RANDOM_NUM,
        )

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
        return [("SVM", SVC(random_state=RANDOM_NUM),), ("MLP", MLPClassifier(random_state=RANDOM_NUM),)]

    def run_hold_out(self, verbose=True):
        result = self.model.fit_and_predict(
            self.X_train, self.X_test, self.y_train, self.y_test, self.get_models()
        )

        if verbose:
            self.model.plot_results(result, self.X_test, self.y_test)

        return result

    def cross_val(self, verbose=True):
        cross_val_results = []
        for model in self.get_models():
            score = cross_val_score(
                model[1], self.X_train, self.y_train, cv=10, n_jobs=-1, verbose=False
            )
            predict = cross_val_predict(
                model[1], self.X_train, self.y_train, cv=10, n_jobs=-1, verbose=False
            )

            cross_val_results.append((model[0], score, predict))

            if verbose:
                print(f"Model: {model[0]}")
                print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
                print(confusion_matrix(self.y_train, predict))

        return cross_val_results


if __name__ == "__main__":
    CancerMama().run_hold_out()
    CancerMama().cross_val()
