from src.stages.evaluate import get_model, cross_validate_model, evaluate_model,  evaluate_model_dvc
from sklearn.metrics import ConfusionMatrixDisplay
from src.utils import read_config_params
import unittest
import pandas as pd
import numpy as np
import os

class TestEvaluate(unittest.TestCase):

    # Verificar que la función de 'cross_validate_model' entregue el tipo de dato correcto y la cantidad de puntuaciones correspondientes al numero de cv
    def test_cross_validation_output(self):
        config = read_config_params('params.yaml')
        scores = cross_validate_model(config)
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), 5)

    # Verificar que la función 'evaluate_model' entregue correctamente las métricas
    def test_metrics(self):
        config = read_config_params('params.yaml')
        test = pd.read_csv(config["features"]["features_test_dataset"])
        fit_model = get_model(config)
        metrics, matrix = evaluate_model(config, test, fit_model)
        self.assertIsInstance(metrics, dict)
        for metric in metrics:
            self.assertIsInstance(metric, str)
        self.assertIsInstance(matrix, ConfusionMatrixDisplay)

    # Verificar que se guarde correctamente la imagen de la matriz de confusión
    def test_save_matrix(self):
        config = read_config_params('params.yaml')
        if os.path.exists(config["evaluate"]["cm_file"]):
            os.remove(config["evaluate"]["cm_file"])
        evaluate_model_dvc(config)
        self.assertTrue(os.path.exists(config["evaluate"]["cm_file"]))

    # Verificar que se guarden correctamente las métricas
    def test_save_metrics(self):
        config = read_config_params('params.yaml')
        if os.path.exists(config["evaluate"]["metrics_file"]):
            os.remove(config["evaluate"]["metrics_file"])
        evaluate_model_dvc(config)
        self.assertTrue(os.path.exists(config["evaluate"]["metrics_file"]))

if __name__ == '__main__':
    unittest.main()