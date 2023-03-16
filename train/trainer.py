import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from preprocess import DataPreprocessor
from config import PATH_TO_SAVE_MODELS, DATA_PATH
from darts import TimeSeries
from darts.metrics import mape, mae, rmse
from darts.models import TCNModel
from datetime import datetime


class Trainer:
    def __init__(self):

            
        self.df = None
        self.train = None
        
        self.hyperparameters_path = "hyperparameters_results.csv"
        self.hp_df = None
        self.hyperparameters = None
        
        self.model = None
        
        self.current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        
    def load_preprocess_data(self, data_path):
        """
        Preprocesses data using DataPreprocessor class and returns a TimeSeries object for training.
        """
        self.df = DataPreprocessor(data_path).preprocess()
        self.train = TimeSeries.from_dataframe(self.df, 
                                                time_col="ds", 
                                                value_cols=self.df.columns.tolist()[1:], 
                                                fill_missing_dates=True,
                                                freq = "10T", #10 minutes
                                                fillna_value = 0,
                                                )
        return self.train
    
    def get_best_hyperparameters(self):
        """
        Reads the hyperparameters_results.csv file, deletes rows with rmse_score == inf, 
        and gets the best hyperparameters based on the rmse_score.
        """
        self.hp_df = pd.read_csv(self.hyperparameters_path)
        self.hp_df = self.hp_df[self.hp_df["rmse_score"] != np.inf]
        self.hyperparameters = self.hp_df.sort_values(by="rmse_score", ascending=True).iloc[0].to_dict()
        
    def fit_model(self):
        """
        Trains a TCNModel using the best hyperparameters and fits it on the training set.
        """
        self.get_best_hyperparameters()
        self.model = TCNModel(input_chunk_length=int(self.hyperparameters["input_chunk_length"]),
                              output_chunk_length=1, 
                              n_epochs=int(self.hyperparameters["n_epochs"]),
                              num_layers=int(self.hyperparameters["num_layers"]),
                              num_filters=int(self.hyperparameters["num_filters"]),
                              dropout=self.hyperparameters["dropout"],
                              random_state=13,
                              optimizer_cls=torch.optim.Adam,
                              pl_trainer_kwargs={"accelerator": "gpu", "devices": [0]},
                              batch_size=1024*3,
                              )
        self.model.fit(self.train)
        
    def save_model(self, PATH_TO_SAVE_MODELS):
        """
        Saves the trained model with the current time in the filename and saves the info 
        about the hyperparameters and the model in models_info.csv.
        """
        self.PATH_TO_SAVE_MODELS = PATH_TO_SAVE_MODELS        
        self.model.save(os.path.join(self.PATH_TO_SAVE_MODELS, f"tcn_model_{self.current_time}.pt"))
        print(f"Model saved at {os.path.join(self.PATH_TO_SAVE_MODELS, f'tcn_model_{self.current_time}.pt')}")
        
        model_info_dict = {
            "model_name": f"tcn_model_{self.current_time}.pt",
            "rmse_score": self.hyperparameters["rmse_score"],
            "mae_score": self.hyperparameters["mae_score"],
            "trained_date": self.current_time,
            "input_chunk_length": int(self.hyperparameters["input_chunk_length"]),
            "output_chunk_length": 1,
            "n_epochs": int(self.hyperparameters["n_epochs"]),
            "num_layers": int(self.hyperparameters["num_layers"]),
            "num_filters": int(self.hyperparameters["num_filters"]),
            "dropout": self.hyperparameters["dropout"],
            "random_state": 13,
            "optimizer_cls": "torch.optim.Adam",
            "pl_trainer_kwargs": {"accelerator": "gpu", "devices": [0]},
            "batch_size": 1024*3,
            "model_class": "TCNModel",
            "model_type": "Time Series",
            "last_date": self.df["ds"].max()
        }

        self.models_info_path = os.path.join(PATH_TO_SAVE_MODELS, 'models_info.csv')

        if not os.path.exists(self.models_info_path):
            models_info = pd.DataFrame(model_info_dict)
            models_info.to_csv(self.models_info_path, index=False)
        else:
            models_info = pd.DataFrame(model_info_dict)
            models_info.to_csv(self.models_info_path, mode='a', header=False, index=False)
        
if __name__ == "__main__":
    assistant = Trainer()
    assistant.load_preprocess_data(DATA_PATH)
    assistant.fit_model()
    assistant.save_model(PATH_TO_SAVE_MODELS)