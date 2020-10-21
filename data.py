import abc
import pandas as pd


class Data(abc.ABC):
    """
    Abstract class to set the methods subclasses must to implement
    """
    @staticmethod
    @abc.abstractmethod
    def get_data() -> pd.DataFrame:
        raise NotImplemented

    @staticmethod
    @abc.abstractmethod
    def get_new_data() -> pd.DataFrame:
        raise NotImplemented


class CancerMama(Data):
    @staticmethod
    def get_data() -> pd.DataFrame:
        return pd.read_csv(
            "data_source/Material 02 - 2 - Cancer de Mama - Dados.csv"
        )

    @staticmethod
    def get_new_data() -> pd.DataFrame:
        return pd.read_csv(
            "data_source/Material 02 - 2 - Cancer de Mama - Dados - Novos Casos.csv"
        )


class EstimativaVolume(Data):
    @staticmethod
    def get_data() -> pd.DataFrame:
        return pd.read_csv(
            "data_source/Material 02 - 3 – Estimativa de Volume - Dados.csv"
        )

    @staticmethod
    def get_new_data() -> pd.DataFrame:
        return pd.read_csv(
            "data_source/Material 02 - 3 – Estimativa de Volume - Dados - Novos Casos.csv"
        )


class Biomassa(Data):
    @staticmethod
    def get_data() -> pd.DataFrame:
        return pd.read_csv("data_source/Material 02 - 4 - R - Biomassa - Dados.csv")

    @staticmethod
    def get_new_data() -> pd.DataFrame:
        return pd.read_csv(
            "data_source/Material 02 - 4 - R - Biomassa - Dados - Novos Casos.csv"
        )


class Veiculos(Data):
    @staticmethod
    def get_data() -> pd.DataFrame:
        return pd.read_csv("data_source/Material 02 - 5 - C - Veiculos - Dados.csv")

    @staticmethod
    def get_new_data() -> pd.DataFrame:
        raise NotImplemented


class PrevisaoTempo(Data):
    @staticmethod
    def get_data() -> pd.DataFrame:
        return pd.read_csv(
            "data_source/Material 02 - 6 - C - Previsao do Tempo - Dados.csv"
        )

    @staticmethod
    def get_new_data() -> pd.DataFrame:
        raise NotImplemented


class ImpostoRenda(Data):
    @staticmethod
    def get_data() -> pd.DataFrame:
        return pd.read_csv("data_source/Material 02 - 7 – C - IR - Dados.csv")

    @staticmethod
    def get_new_data() -> pd.DataFrame:
        return pd.read_csv(
            "data_source/Material 02 - 7 – C - IR - Dados - Novos Casos.csv"
        )


class Admissao(Data):
    @staticmethod
    def get_data() -> pd.DataFrame:
        return pd.read_csv("data_source/Material 02 - 8 – R - Admissao - Dados.csv")

    @staticmethod
    def get_new_data() -> pd.DataFrame:
        return pd.read_csv("data_source/Material 02 - 8 – R - Admissao - Novos Casos.csv")


class Diabetes(Data):
    @staticmethod
    def get_data() -> pd.DataFrame:
        return pd.read_csv("data_source/Material 02 - 9 – C - Diabetes - Dados.csv")

    @staticmethod
    def get_new_data() -> pd.DataFrame:
        raise NotImplemented
