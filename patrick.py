import random

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from data import Diabetes, ImpostoRenda, CancerMama, Veiculo, PrevisaoTempo
from techniques import best_model, cv, hold_out

ConvergenceWarning("ignore")

random.seed(47)
RANDOM_NUM = 47
np.random.seed(47)

DATABASES = [
    {"name": "Câncer de Mama", "db": CancerMama()},
    # {"name": "Veículo", "db": Veiculo()},
    # {"name": "Previsão do Tempo", "db": PrevisaoTempo()},
    # {"name": "Imposto de Renda", "db": ImpostoRenda()},
    # {"name": "Diabetes", "db": Diabetes()},
]

MODELS = [
    {
        "name": "MLP",
        "model": MLPClassifier,
        "params": [
            {"name": "decay", "attr": "alpha", "options": (0.1, 0.4, 0.7)},
            {
                "name": "size",
                "attr": "hidden_layer_sizes",
                "options": (1, 10, 20, 30, 40),
            },
        ],
        "init_params": {
            "max_iter": 2000,
            "random_state": RANDOM_NUM
        },
        "default_params": {
            "alpha": [0.0001],
            "hidden_layer_sizes": [100]
        }
    },
    {
        "name": "SVM",
        "model": SVC,
        "params": [
            {"name": "C", "attr": "C", "options": (1, 10, 100, 1000)},
            {"name": "Sigma", "attr": "gamma", "options": (1, 0.1, 0.001, 0.0001)},
        ],
        "init_params": {
            "random_state": RANDOM_NUM
        },
        "default_params": {
            "C": [1],
            "gamma": [1]
        }
    },
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
                    X_train, X_test, y_train, y_test, model["model"], model["init_params"], model["params"], model["default_params"]
                )
                print("----------------------------------------------")
                print(f"Base de Dados: {db['name']}")
                print(f"Modelo: {model['name']}")
                print(f"Técnica: {technique['name']}")
                params = {p["name"]: p["value"] for p in result.params}
                print(f"Parâmetros: {params}")
                print(f"Acurácia: {result.accuracy}")
                print(f"Matriz de Confusão: \n{result.confusion_matrix}")
