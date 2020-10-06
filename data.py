import pandas as pd


class CancerMama:
    @staticmethod
    def get_data():
        return pd.read_csv("data_source/Material 02 - 2 - Cancer de Mama - Dados.csv")

    @staticmethod
    def get_new_data():
        return pd.read_csv(
            "data_source/Material 02 - 2 - Cancer de Mama - Dados - Novos Casos.csv"
        )


class EstimativaVolume:
    @staticmethod
    def get_data():
        return pd.read_csv(
            "data_source/Material 02 - 3 – Estimativa de Volume - Dados.csv"
        )

    @staticmethod
    def get_new_data():
        return pd.read_csv(
            "data_source/Material 02 - 3 – Estimativa de Volume - Dados - Novos Casos.csv"
        )


class Biomassa:
    @staticmethod
    def get_data():
        return pd.read_csv("data_source/Material 02 - 4 - R - Biomassa - Dados.csv")

    @staticmethod
    def get_new_data():
        return pd.read_csv(
            "data_source/Material 02 - 3 – Estimativa de Volume - Dados - Novos Casos.csv"
        )


class Veiculos:
    @staticmethod
    def get_data():
        return pd.read_csv("data_source/Material 02 - 5 - C - Veiculos - Dadoscsv")


class PrevisaoTempo:
    @staticmethod
    def get_data():
        return pd.read_csv(
            "data_source/Material 02 - 6 - C - Previsao do Tempo - Dados.csv"
        )


class ImpostoRenda:
    @staticmethod
    def get_data():
        return pd.read_csv("data_source/Material 02 - 7 – C - IR - Dados.csv")

    @staticmethod
    def get_new_data():
        return pd.read_csv(
            "data_source/Material 02 - 7 – C - IR - Dados - Novos Casos.csv"
        )


class ImpostoRenda:
    @staticmethod
    def get_data():
        return pd.read_csv("data_source/Material 02 - 8 – R - Admissao - Dados.csv")

    @staticmethod
    def get_new_data():
        return pd.read_csv(
            "data_source/Material 02 - 8 – R - Admissao - Novos Casos.csv"
        )


class Diabetes:
    @staticmethod
    def get_data():
        return pd.read_csv("data_source/Material 02 - 9 – C - Diabetes - Dados.csv")
