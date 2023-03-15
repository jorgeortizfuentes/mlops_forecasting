import pandas as pd
from preprocess import df

import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.metrics import mape, mae, rmse

series = TimeSeries.from_dataframe(df, 
                                   time_col="ds", 
                                   value_cols=df.columns.tolist()[1:], 
                                   fill_missing_dates=True,
                                   freq = "10T", #10 minutes
                                   fillna_value = 0,
                                   )

# Split the series into train and eval
train, val = series.split_before(0.8)
