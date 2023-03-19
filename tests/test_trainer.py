import unittest
import os
import sys

current_dir = os.getcwd()
scripts_dir = os.path.join(current_dir, "train")
sys.path.append(scripts_dir)

import pandas as pd
from darts import TimeSeries
from trainer import Trainer
from config import PATH_TO_SAVE_MODELS, DATA_PATH
from preprocess import DataPreprocessor


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.trainer = Trainer()
        self.trainer.load_preprocess_data(DATA_PATH)

    def test_get_best_hyperparameters_file(self):
        hyper_data_path = os.path.join(
            scripts_dir, "hyperparameters_results.csv"
        )
        # Check if the file exists and is not empty
        self.assertTrue(os.path.exists(hyper_data_path))
        self.assertTrue(os.path.getsize(hyper_data_path) > 0)
        # Check if the file has the correct columns
        hyper_data = pd.read_csv(hyper_data_path)
        self.assertIn("mae_score", hyper_data.columns)
        self.assertIn("n_epochs", hyper_data.columns)
        # Check if the file has 1 or more rows
        self.assertGreaterEqual(hyper_data.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
