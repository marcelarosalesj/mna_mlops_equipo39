from src.stages.data_split import split_data, split_data_dvc
from src.utils import read_config_params
import unittest
import pandas as pd
import os

class TestSplit(unittest.TestCase):

    # Verificar que se entreguen 3 conjuntos de datos
    def test_num_splits(self):
        config = read_config_params('params.yaml')
        results = split_data(config, pd.read_csv(config['load_data']['dataset_csv']))
        self.assertEqual(len(results), 3)

    # Verificar que se entregue un DataFrame como resultado de la division del conjunto de datos
    def test_df_split(self):
        config = read_config_params('params.yaml')
        df1, df2, df3 = split_data(config, pd.read_csv(config['load_data']['dataset_csv']))
        self.assertIsInstance(df1, pd.core.frame.DataFrame)
        self.assertIsInstance(df2, pd.core.frame.DataFrame)
        self.assertIsInstance(df3, pd.core.frame.DataFrame)

    # Verificar que el conjunto de datos contenga la variable de salida
    def test_df_complete(self):
        config = read_config_params('params.yaml')
        df1, df2, df3 = split_data(config, pd.read_csv(config['load_data']['dataset_csv']))
        self.assertIn("OUTPUT Grade", df1.columns)
        self.assertIn("OUTPUT Grade", df2.columns)
        self.assertIn("OUTPUT Grade", df3.columns)

    # Verificar que se guarden correctamente los artefactos en el path correspondiente
    def test_save_artifacts_and_verify_strings(self):
        config = read_config_params('params.yaml')
        paths = ['train_dataset_path', 'test_dataset_path', 'val_dataset_path']
        for path in paths:
            if os.path.exists(config['split_data'][path]):
                os.remove(config['split_data'][path])
        split_data_dvc(config)
        for path in paths:
            self.assertTrue(os.path.exists(config['split_data'][path]))
    
    # Verificar que las columnas de los conjuntos de datos sean strings
    def test_column_strings(self):
        config = read_config_params('params.yaml')
        paths = ['train_dataset_path', 'test_dataset_path', 'val_dataset_path']
        split_data_dvc(config)
        for path in paths:
            df = pd.read_csv(config['split_data'][path])
            for col in df.columns:
                self.assertIsInstance(col, str)

if __name__ == '__main__':
    unittest.main()