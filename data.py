import abc

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class Data(abc.ABC):
    """
    Abstract class to set the methods subclasses must to implement
    """

    first_column: int = 0

    @staticmethod
    @abc.abstractmethod
    def get_data() -> pd.DataFrame:
        raise NotImplemented

    @staticmethod
    @abc.abstractmethod
    def get_new_data() -> pd.DataFrame:
        raise NotImplemented

    @staticmethod
    def encode_object_columns(df: pd.DataFrame) -> pd.DataFrame:
        le = preprocessing.LabelEncoder()
        for col_name in df.columns:
            if df[col_name].dtype == object:
                df[col_name] = le.fit_transform(df[col_name])
        return df

    def get_splited_data(self, random_state, test_size=0.2) -> tuple:
        df = self.get_data()
        df = self.encode_object_columns(df)

        return train_test_split(
            df.iloc[:, self.first_column : -1],
            df.iloc[:, -1],  # Remove target
            test_size=test_size,
            random_state=random_state,
        )


class CancerMama(Data):
    @staticmethod
    def get_data() -> pd.DataFrame:
        return pd.read_csv("data_source/Material 02 - 2 - Cancer de Mama - Dados.csv")

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


class Veiculo(Data):
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
    first_column = 0

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
        return pd.read_csv(
            "data_source/Material 02 - 8 – R - Admissao - Novos Casos.csv"
        )


class Diabetes(Data):
    first_column = 1

    @staticmethod
    def get_data() -> pd.DataFrame:
        return pd.read_csv("data_source/Material 02 - 9 – C - Diabetes - Dados.csv")

    @staticmethod
    def get_new_data() -> pd.DataFrame:
        raise NotImplemented
