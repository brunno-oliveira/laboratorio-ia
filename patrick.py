import random

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from data import Diabetes, ImpostoRenda
from techniques import best_model, cv, hold_out

ConvergenceWarning("ignore")

random.seed(47)
RANDOM_NUM = 47
np.random.seed(47)

DATABASES = [
    {"name": "Imposto de Renda", "db": ImpostoRenda()},
    {"name": "Diabetes", "db": Diabetes()},
]

MODELS = [
    {
        "name": "MLP",
        "model": MLPClassifier(max_iter=2000, random_state=RANDOM_NUM),
        "params": [
            {"name": "decay", "attr": "alpha", "options": (0.1, 0.4, 0.7)},
            {
                "name": "size",
                "attr": "hidden_layer_sizes",
                "options": (1, 10, 20, 30, 40),
            },
        ],
    },
    # {
    #     "name": "SVM",
    #     "model": SVC(),
    #     "params": [
    #         {"name": "C", "attr": "C", "options": ()},
    #         {"name": "Sigma", "attr": "gamma", "options": ()},
    #     ],
    # },
    # {"name": "KNN", "model": KNN},
    # {"name": "RF", "model": RF},
]

TECHNIQUES = [
    {"name": "Hold-out", "method": hold_out},
    {"name": "CV", "method": cv},
    {"name": "Melhor Modelo", "method": best_model},
]


if __name__ == "__main__":
    for db in DATABASES:
        X_train, X_test, y_train, y_test = db["db"].get_splited_data(RANDOM_NUM)
        for model in MODELS:
            for technique in TECHNIQUES:
                result = technique["method"](
                    X_train, X_test, y_train, y_test, model["model"], model["params"],
                )
                print("----------------------------------------------")
                print(f"Base de Dados: {db['name']}")
                print(f"Modelo: {model['name']}")
                print(f"Técnica: {technique['name']}")
                params = {p["name"]: p["value"] for p in result.params}
                print(f"Parâmetros: {params}")
                print(f"Acurácia: {result.accuracy}")
                print(f"Matriz de Confusão: \n{result.confusion_matrix}")
