from src.stages.features_transform import _encode_target, _encode_features, features_transform, features_transform_dvc
from src.utils import read_config_params
import unittest
import pandas as pd
import numpy

class TestTransform(unittest.TestCase):

    # Verificar que el Label encoder sea aplicado correctamente
    def test_label_encoder(self):
        df = []
        config = read_config_params('params.yaml')
        split = ['train_dataset_path', 'test_dataset_path', 'val_dataset_path']
        for set in split:
            df.append(pd.read_csv(config['split_data'][set]))
        y_encoded_splits = _encode_target(df[0]["OUTPUT Grade"], df[1]["OUTPUT Grade"], df[2]["OUTPUT Grade"])
        for split in y_encoded_splits:
            for array in split:
                self.assertIsInstance(array, numpy.int64, 'La función "_encode_target()" no está realizando la codificación')

    # Verificar que la funcion '_encode_target' solo devuelva la variable de salida
    def test_y_output(self):
        df = []
        config = read_config_params('params.yaml')
        split = ['train_dataset_path', 'test_dataset_path', 'val_dataset_path']
        for set in split:
            df.append(pd.read_csv(config['split_data'][set]))
        y_encoded_splits = _encode_target(df[0]["OUTPUT Grade"], df[1]["OUTPUT Grade"], df[2]["OUTPUT Grade"])
        for y in y_encoded_splits:
            self.assertEqual(y.ndim, 1, 'La función "_encode_target()" no está devolviendo solo la variable de salida')

    # Verificar que el OneHot y Ordinal encoder sea aplicado correctamente
    def test_onehot_and_ordinal_encoder(self):
        df = []
        config = read_config_params('params.yaml')
        split = ['train_dataset_path', 'test_dataset_path', 'val_dataset_path']
        for set in split:
            df.append(pd.read_csv(config['split_data'][set]))
        x_encoded_splits = _encode_features(df[0].drop("OUTPUT Grade", axis=1), df[1].drop("OUTPUT Grade", axis=1), df[2].drop("OUTPUT Grade", axis=1))
        for split in x_encoded_splits:
            for col in split:
                for array in col:
                    self.assertIsInstance(array, numpy.float64, 'La función "_encode_features()" no está realizando la codificación')

    # Verificar que la función 'features_transform' devuelva 3 Dataframes
    def test_df_transform(self):
        df = []
        config = read_config_params('params.yaml')
        split = ['train_dataset_path', 'test_dataset_path', 'val_dataset_path']
        for set in split:
            df.append(pd.read_csv(config['split_data'][set]))
        df_transformations = features_transform(df[0], df[1], df[2])
        self.assertEqual(len(df_transformations), 3, 'La función "features_transform()" no está devolviendo 3 conjuntos de datos')
        for trans_df in df_transformations:
            self.assertIsInstance(trans_df, pd.core.frame.DataFrame, 'La función "features_transform()" no está a los conjuntos de datos como DataFrames')

    # Verificar que las columnas de los conjuntos de datos sean strings
    def test_column_strings(self):
        config = read_config_params('params.yaml')
        features_transform_dvc(config)
        transformed_df = ['train_dataset_path', 'test_dataset_path', 'val_dataset_path']
        for set in transformed_df:
            df = pd.read_csv(config['split_data'][set])
            for col in df:
                self.assertIsInstance(col, str, 'La función "features_transform_dvc()" no transformó el nombre de las columnas del conjunto de datos en stings')

if __name__ == '__main__':
    unittest.main()