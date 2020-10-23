from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV


@dataclass
class Result:
    params: List[dict]
    accuracy: float
    confusion_matrix: str


def hold_out(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    classifier,
    params: List[dict],
) -> Result:
    classifier.fit(x_train, y_train)
    predict = classifier.predict(x_test)

    for param in params:
        param.update({"value": classifier.get_params().get(param["attr"])})

    return Result(
        params=params,
        accuracy=round(accuracy_score(y_test, predict), 4),
        confusion_matrix=confusion_matrix(y_test, predict),
    )


def cv(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    classifier,
    params: List[dict],
) -> Result:
    classifier = GridSearchCV(classifier, {}, n_jobs=-1, cv=10)
    classifier.fit(x_train, y_train)
    predict = classifier.predict(x_test)

    for param in params:
        param.update(
            {"value": classifier.get_params().get(f"estimator__{param['attr']}")}
        )

    return Result(
        params=params,
        accuracy=round(accuracy_score(y_test, predict), 4),
        confusion_matrix=confusion_matrix(y_test, predict),
    )


def best_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    classifier,
    params: List[dict],
) -> Result:
    cv_params = {p["attr"]: p["options"] for p in params}
    classifier = GridSearchCV(classifier, cv_params, n_jobs=-1, cv=10)
    classifier.fit(x_train, y_train)
    predict = classifier.predict(x_test)

    for param in params:
        param.update(
            {"value": classifier.get_params().get(f"estimator__{param['attr']}")}
        )

    return Result(
        params=params,
        accuracy=round(accuracy_score(y_test, predict), 4),
        confusion_matrix=confusion_matrix(y_test, predict),
    )
