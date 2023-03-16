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


# Create model

model = TCNModel(input_chunk_length=24, 
                 output_chunk_length=1, 
                 n_epochs=20, 
                 num_layers=10,
                 num_filters=256,
                 dropout=0.1,
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