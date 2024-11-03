from src.stages.data_load import load_data, load_data_dvc
from src.utils import read_config_params
import unittest
import pandas
import os

class TestLoad(unittest.TestCase):

    # Verificar que se carge la base de datos correctamente como un DataFrame
    def test_df_load(self):
        self.assertIsInstance(load_data(), pandas.core.frame.DataFrame, 'La función "load_data()" no está entregando un DataFrame')

    # Verificar que el DataFrame contenga la variable de salida
    def test_df_complete(self):
        self.assertIn("OUTPUT Grade", load_data().columns, 'La función "load_data() no entrega un DataFrame con la variable de salida')

    # Verificar que se guarde correctamente la base de datos en el path correspondiente
    def test_save_artifact(self):
        config = read_config_params('params.yaml')
        if os.path.exists(config["load_data"]["dataset_csv"]):
            os.remove(config["load_data"]["dataset_csv"])
        load_data_dvc(config)
        self.assertTrue(os.path.exists(config["load_data"]["dataset_csv"]), 'La función "load_data_dvc()" no está guardando la base de datos en el lugar especificado')

if __name__ == '__main__':
    unittest.main()