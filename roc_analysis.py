import math

import numpy as np


def get_x_y_4x4(data: tuple) -> tuple:
    x = 1 - data[3] / (data[3] + data[2]) if data[3] + data[2] != 0 else 0
    y = data[0] / (data[0] + data[1]) if data[0] + data[1] != 0 else 0
    return x, y


def get_x_y(data: tuple) -> tuple:
    if len(data) == 4:
        return get_x_y_4x4(data)
    sqrt = int(math.sqrt(len(data)))
    data = np.array(data)
    shape = (sqrt, sqrt)
    data = data.reshape(shape)
    x_list = []
    y_list = []
    for i, num in enumerate(np.diag(data)):
        temp_x, temp_y = get_x_y_4x4(
            (
                num,
                data[i].sum() - num,
                data[:, i].sum() - num,
                np.diag(data).sum() - num,
            )
        )
        x_list.append(temp_x)
        y_list.append(temp_y)
    return sum(x_list) / len(x_list), sum(y_list) / len(y_list)


def calculate_distance(x: float, y: float) -> float:
    return math.sqrt(((0 - x) ** 2) + ((1 - y) ** 2))


def get_distance(data: tuple) -> float:
    x, y = get_x_y(data)
    return calculate_distance(x, y)


def get_shorter_distance(data: tuple) -> tuple:
    data = [(item[0], get_distance(item[1])) for item in data]
    return min(data, key=lambda x: x[1])


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    databases = {
        "CANCER_MAMA": (
            ("RNA_HO", (89, 1, 2, 47,)),
            ("RNA_CV", (88, 3, 3, 45,)),
            ("RNA_BM", (88, 1, 3, 47,)),
            ("KNN", (89, 3, 2, 45,)),
            ("SVM_HO", (87, 0, 4, 48,)),
            ("SVM_CV", (89, 0, 2, 48,)),
            ("SVM_BM", (89, 0, 2, 48,)),
            ("RF_HO", (88, 1, 3, 47,)),
            ("RF_CV", (88, 1, 3, 47,)),
            ("RF_BM", (88, 1, 3, 47,)),
        ),
        "VEICULO": (
            ("RNA_HO", (28, 1, 3, 5, 1, 17, 15, 0, 13, 22, 24, 1, 1, 2, 1, 33,),),
            ("RNA_CV", (43, 34, 39, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 6, 4, 37,),),
            ("RNA_BM", (43, 0, 0, 0, 0, 29, 10, 1, 0, 12, 32, 0, 0, 1, 1, 38,),),
            ("KNN", (38, 1, 1, 2, 0, 19, 20, 0, 4, 20, 22, 2, 1, 2, 0, 35,),),
            ("SVM_HO", (42, 0, 0, 0, 0, 18, 14, 0, 0, 21, 27, 0, 1, 3, 2, 39,),),
            ("SVM_CV", (42, 0, 0, 0, 0, 18, 14, 0, 0, 21, 27, 0, 1, 3, 2, 39,),),
            ("SVM_BM", (43, 0, 0, 0, 0, 27, 16, 1, 0, 13, 27, 0, 0, 2, 0, 38,),),
            ("RF_HO", (43, 0, 0, 0, 0, 21, 18, 0, 0, 18, 21, 0, 0, 3, 4, 39,),),
            ("RF_CV", (42, 1, 2, 0, 0, 20, 15, 0, 0, 19, 21, 1, 1, 2, 5, 38,),),
            ("RF_BM", (42, 0, 1, 0, 0, 23, 16, 0, 0, 17, 21, 1, 1, 2, 5, 38,),),
        ),
        "PREVISAO_TEMPO": (
            ("RNA_HO", (4, 0, 0, 3,)),
            ("RNA_CV", (4, 0, 0, 3,)),
            ("RNA_BM", (4, 0, 0, 3,)),
            ("KNN", (4, 0, 0, 3,)),
            ("SVM_HO", (4, 0, 0, 3,)),
            ("SVM_CV", (4, 0, 0, 3,)),
            ("SVM_BM", (4, 0, 0, 3,)),
            ("RF_HO", (4, 0, 0, 3,)),
            ("RF_CV", (4, 0, 0, 3,)),
            ("RF_BM", (4, 0, 0, 3,)),
        ),
        "IR": (
            ("RNA_HO", (4, 3, 3, 0, 0, 0, 0, 0, 0,)),
            ("RNA_CV", (4, 3, 3, 0, 0, 0, 0, 0, 0,)),
            ("RNA_BM", (4, 1, 0, 0, 2, 0, 0, 0, 3,)),
            ("KNN", (4, 0, 0, 0, 3, 0, 0, 0, 3,)),
            ("SVM_HO", (4, 0, 0, 0, 3, 0, 0, 0, 3,)),
            ("SVM_CV", (4, 0, 0, 0, 3, 0, 0, 0, 3,)),
            ("SVM_BM", (4, 0, 0, 0, 3, 0, 0, 0, 3,)),
            ("RF_HO", (4, 0, 0, 0, 3, 0, 0, 0, 3,)),
            ("RF_CV", (4, 0, 0, 0, 3, 0, 0, 0, 3,)),
            ("RF_BM", (4, 0, 0, 0, 3, 0, 0, 0, 3,)),
        ),
        "DIABETES": (
            ("RNA_HO", (70, 31, 30, 22,)),
            ("RNA_CV", (92, 40, 8, 13,)),
            ("RNA_BM", (81, 26, 19, 27,)),
            ("KNN", (89, 26, 11, 27,)),
            ("SVM_HO", (92, 19, 8, 34,)),
            ("SVM_CV", (92, 19, 8, 34,)),
            ("SVM_BM", (92, 19, 8, 34,)),
            ("RF_HO", (90, 18, 10, 35,)),
            ("RF_CV", (89, 17, 11, 36,)),
            ("RF_BM", (89, 18, 11, 35,)),
        ),
    }
    for name, data in databases.items():
        print(name, get_shorter_distance(data))
