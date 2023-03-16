import pandas as pd
from preprocess import df
from config import PATH_TO_SAVE

import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.metrics import mape, mae, rmse
import torch
from darts.models import TCNModel
from datetime import datetime
import os


train = TimeSeries.from_dataframe(df, 
                                   time_col="ds", 
                                   value_cols=df.columns.tolist()[1:], 
                                   fill_missing_dates=True,
                                   freq = "10T", #10 minutes
                                   fillna_value = 0,
                                   )

# Split the series into train and eval
#train, val = series.split_before(0.8)

hyperparameters_path = "hyperparameters_results.csv"
hp_df = pd.read_csv(hyperparameters_path)

# Delete rows with rmse_score == inf 
hp_df = hp_df[hp_df["rmse_score"] != np.inf]

# Get the best hyperparameters
hyperparameters = hp_df.sort_values(by="rmse_score", ascending=True).iloc[0].to_dict()

model = TCNModel(input_chunk_length=int(hyperparameters["input_chunk_length"]),
                 output_chunk_length=1, 
                 n_epochs=int(hyperparameters["n_epochs"]),
                 num_layers=int(hyperparameters["num_layers"]),
                 num_filters=int(hyperparameters["num_filters"]),
                 dropout=hyperparameters["dropout"],
                 random_state=13,
                 optimizer_cls = torch.optim.Adam,
                 pl_trainer_kwargs={"accelerator": "gpu", "devices": [0]},
                 batch_size=1024*3,
                 )
                 
# Fit model on training set
model.fit(train)


# Get current time in str 
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
# Save model tcn_model.pt
model.save(os.path.join(PATH_TO_SAVE, f"tcn_model_{current_time}.pt"))

print(f"Model saved at {os.path.join(PATH_TO_SAVE, f'tcn_model_{current_time}.pt')}")

# Save info about hyperparameters and model in .json file 
model_info_dict = {
    "input_chunk_length": int(hyperparameters["input_chunk_length"]),
    "output_chunk_length": 1,
    "n_epochs": int(hyperparameters["n_epochs"]),
    "num_layers": int(hyperparameters["num_layers"]),
    "num_filters": int(hyperparameters["num_filters"]),
    "dropout": hyperparameters["dropout"],
    "random_state": 13,
    "optimizer_cls": "torch.optim.Adam",
    "pl_trainer_kwargs": {"accelerator": "gpu", "devices": [0]},
    "batch_size": 1024*3,
    "rmse_score": hyperparameters["rmse_score"],
    "mae_score": hyperparameters["mae_score"],
    "model_name": f"tcn_model_{current_time}.pt",
    "train_date": current_time,
    "model_class": "TCNModel",
    "model_type": "Time Series",
}

# Save to os.path.join(PATH_TO_SAVE, 'models_info.csv)

# Create an empty dataframe if models_info.csv does not exist
models_info_path = os.path.join(PATH_TO_SAVE, 'models_info.csv')
if not os.path.exists(models_info_path):
    models_info = pd.DataFrame(columns=model_info_dict.keys())
    models_info.to_csv(models_info_path, index=False)

models_info = pd.DataFrame(model_info_dict)
models_info.to_csv(models_info_path, mode='a', header=False, index=False)


