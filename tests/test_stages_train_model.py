from src.stages.train_model import train_model, train_model_dvc
from src.utils import read_config_params
import unittest
import pandas as pd
import os

class TestTrain(unittest.TestCase):

    # Verificar que el modelo se ha ajustado (se han generado parámetros)
    def test_fit_applied(self):
        config = read_config_params('params.yaml')
        dft = pd.read_csv(config["features"]["features_train_dataset"])
        fit_model, _= train_model(config, dft)
        if config['train']['algo'] == 'knn':
            self.assertTrue(hasattr(fit_model, "n_neighbors"))
        elif config['train']['algo'] in ('dt', 'rf', 'xgb'):
            self.assertTrue(hasattr(fit_model, "max_depth"))

    # Verificar que se guarde correctamente el modelo
    def test_save_fit_model(self):
        config = read_config_params('params.yaml')
        if os.path.exists(config['train']['model_path']):
            os.remove(config['train']['model_path'])
        train_model_dvc(config)
        self.assertTrue(os.path.exists(config['train']['model_path']))

if __name__ == '__main__':
    unittest.main()