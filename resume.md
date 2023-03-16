[/home/jlortiz/awto_mle_challenge/api/main.py]
```python
from datetime import datetime
import uvicorn
from fastapi import FastAPI
import os
import sys

# Add scripts directory to path
if os.getcwd().endswith("api"):
    print("hola")
    current_dir = os.getcwd()
    scripts_dir = current_dir.replace("api", "train")
else:
    current_dir = os.getcwd()
    scripts_dir = os.path.join(current_dir, "train")

sys.path.append(scripts_dir)

from predict import PowerPredictor
from config import PATH_TO_SAVE_MODELS

app = FastAPI()

@app.get("/predict/{date}")
def predict(date: str):
    """
    Predicts the power output for a given datetime.

    Args:
        date (str): The datetime for which to make predictions (format: "YYYY-MM-DD HH:MM").

    Returns:
        dict: A dictionary with the predicted power output.
    """
    # Check if date is in %Y-%m-%d %H:%M format
    try:
        date = datetime.strptime(date, "%Y-%m-%d %H:%M")
    except:
        return {"error": "Date format is not correct. Please use %Y-%m-%d %H:%M format."}
    
    # Check if %M is a multiple of 10, if not, round it to the highest multiple of 10
    if date.minute % 10 != 0:
        date = date.replace(minute=(date.minute // 10 + 1) * 10)
    
    # Predict power output
    power_predictor = PowerPredictor(PATH_TO_SAVE_MODELS)
    predicted_power_output = power_predictor.predict(date)
    return predicted_power_output

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8282)
```

[/home/jlortiz/awto_mle_challenge/train/config.py]
```python
DATA_PATH = "/home/jlortiz/awto_mle_challenge/data/wind_power_generation.csv"
PATH_TO_SAVE_MODELS = "/home/jlortiz/awto_mle_challenge/models/"
```

[/home/jlortiz/awto_mle_challenge/train/predict.py]
```python
import os
import pandas as pd
from datetime import datetime
from darts.models import TCNModel
from config import PATH_TO_SAVE_MODELS

class PowerPredictor:
    """
    A class for predicting power output using a trained TCNModel.
    """
    def __init__(self, model_path: str):
        """
        Initializes the PowerPredictor object.

        Args:
            param model_path (str): The path to the trained TCNModel.
        """
        self.model_path = model_path
        self.models_info = pd.read_csv(os.path.join(model_path, "models_info.csv")).iloc[-1].to_dict()
        self.last_date = datetime.strptime(self.models_info["last_date"], "%Y-%m-%d %H:%M:%S")
        self.model_name = self.models_info["model_name"]
        self.model = TCNModel.load(os.path.join(model_path, self.model_name))

    def get_time_position(self, date: datetime):
        """
        Calculates the number of predictions to make based on the time between the input date and the last date
        of the model's training data.

        Args:
            date (datetime): The datetime for which to make predictions.
        
        Returns: 
            int: the number of predictions to make.
        """
        # Count the number of times between the last date and the date we want (by 10 minutes)
        return (date - self.last_date).days * 144 + (date - self.last_date).seconds // 600

    def predict(self, date: datetime):
        """        
        Predicts the power output for the given datetime.
        
        Args:
            date (datetime): last date to make predictions.

        Returns:
            dict: dict with the predicted power output.
        """
        n_predictions = self.get_time_position(date)
        output = self.model.predict(n_predictions)
        return output["ActivePower"].pd_dataframe().to_dict()


if __name__ == "__main__":
    # Example usage
    power_predictor = PowerPredictor(PATH_TO_SAVE_MODELS)
    date = datetime(2020, 4, 5, 18, 50)
    predicted_power_output = power_predictor.predict(date)
    print(predicted_power_output)
    

```

[/home/jlortiz/awto_mle_challenge/train/search_hyperparameters.py]
```python
import pandas as pd
from preprocess import DataPreprocessor
from config import PATH_TO_SAVE_MODELS, DATA_PATH
from darts.metrics import mae, rmse
import optuna
from darts import TimeSeries
from darts.models import TCNModel
import torch


preprocessor = DataPreprocessor(DATA_PATH)
df = preprocessor.preprocess()

series = TimeSeries.from_dataframe(df,
                                   time_col="ds", 
                                   value_cols=df.columns.tolist()[1:], 
                                   fill_missing_dates=True,
                                   freq = "10T", #10 minutes
                                   fillna_value = 0,
                                   )

# Split the series into train and eval
train, val = series.split_before(0.8)
results = []
# Define the objective function to optimize
def objective(trial):
    # Define the hyperparameters to search over
    input_chunk_length = trial.suggest_int('input_chunk_length', 10, 100)
    num_layers = trial.suggest_int('num_layers', 1, 10)
    num_filters = trial.suggest_int('num_filters', 8, 256)
    n_epochs = trial.suggest_int('n_epochs', 5, 20) # 5 y 20
    dropout = trial.suggest_uniform('dropout', 0.0, 0.5)

    # Create the TCN model with the current hyperparameters
    model = TCNModel(input_chunk_length=input_chunk_length,
                     output_chunk_length=1,
                     num_layers=num_layers,
                     num_filters=num_filters,
                     n_epochs=n_epochs,
                     dropout=dropout,
                     random_state=13,
                     optimizer_cls = torch.optim.Adam,
                     optimizer_kwargs={"lr": 1e-3},
                     pl_trainer_kwargs={"accelerator": "gpu", "devices": [1]},
                     batch_size=1024*3)

    # Fit the model on the training set
    model.fit(train)

    # Make predictions on the validation set
    pred_val = model.predict(len(val), verbose=True)

    # Evaluate the model's performance using root mean squared error
    rmse_score = rmse(val["ActivePower"], pred_val["ActivePower"])
    mae_score = mae(val["ActivePower"], pred_val["ActivePower"])
    
    results_d = {"input_chunk_length": input_chunk_length,
                    "num_layers": num_layers,
                    "num_filters": num_filters,
                    "n_epochs": n_epochs,
                    "dropout": dropout,
                    "rmse_score": rmse_score,
                    "mae_score": mae_score}
    results.append(results_d)
    pd.DataFrame(results).to_csv("hyperparameters_results.csv")
    return rmse_score

# Set up the optuna study
study = optuna.create_study(direction='minimize')

# Run the optimization
study.optimize(objective, n_trials=50, show_progress_bar=True)
```

[/home/jlortiz/awto_mle_challenge/train/trainer.py]
```python
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
```

[/home/jlortiz/awto_mle_challenge/train/preprocess.py]
```python

import pandas as pd
from config import DATA_PATH

class DataPreprocessor:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        
    def rename_columns(self):
        """Rename "Unnamed: 0" to "ds"."""
        self.df.rename(columns={"Unnamed: 0": "ds"}, inplace=True)
        
    def convert_datetime(self):
        """Convert ds to datetime and remove timezone."""
        self.df["ds"] = pd.to_datetime(self.df["ds"]).dt.tz_localize(None)
        
    def remove_nan(self):
        """Remove all NaN in ActivePower column."""
        self.df = self.df.dropna(subset=["ActivePower"])
        
    def fill_nan(self):
        """Convert NaN in another columns to 0."""
        self.df = self.df.fillna(0)
        
    def delete_duplicates_columns(self):
        """Delete WTG column and duplicates columns (Blade3PitchAngle, WindDirection)."""
        self.df = self.df.drop(columns=["WTG", "Blade3PitchAngle", "WindDirection"])
        
    def reset_index(self):
        """Reset index."""
        self.df.reset_index(drop=True, inplace=True)
        
    def preprocess(self):
        """Apply all preprocessing methods."""
        self.rename_columns()
        self.convert_datetime()
        self.remove_nan()
        self.fill_nan()
        self.delete_duplicates_columns()
        self.reset_index()
        return self.df

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(DATA_PATH)
    preprocessor.preprocess()

    # Access preprocessed data
    print(preprocessor.df.head())

```

[/home/jlortiz/awto_mle_challenge/train/run_pipeline.py]
```python

```

[/home/jlortiz/awto_mle_challenge/train/hyperparameters_results.csv]
```python
,input_chunk_length,num_layers,num_filters,n_epochs,dropout,rmse_score,mae_score
0,32,2,132,6,0.3573020203892282,599.7193224743062,401.40348090045524
1,51,4,50,13,0.1685108997804483,563.2919988587394,376.81431689963523
2,56,9,256,19,0.19290336802781033,576.3154204161744,385.33070613141564
3,46,9,139,15,0.2820215439171283,560.7664346261014,375.2878626947116
4,28,10,122,20,0.4080716329718478,552.0631500831722,370.2601237175553
5,14,8,108,7,0.4322901419266914,593.1210807367607,396.6899990990549
6,52,9,202,5,0.03903885143181829,600.965996262965,402.287354011261
7,21,6,23,13,0.13261824518050908,596.9806584831852,399.4345577526966
8,72,6,60,17,0.22957850475949443,582.6402554351079,389.4198003764312
9,33,7,20,18,0.321587517371701,inf,inf
10,97,1,181,10,0.4772979216373184,inf,inf
11,37,10,105,16,0.32422022319799115,584.1193086350808,390.50584658126206
12,73,10,178,20,0.39571721209768274,589.5358864277398,394.10706297414504
13,41,4,145,15,0.2864385168376412,591.0084860889343,395.34015750662235
14,21,8,100,10,0.4810646029119503,589.3830826464874,394.07144309166677
15,67,10,150,20,0.38616590122297617,560.1142873607952,374.8484263555829
16,70,10,215,20,0.39568316593897146,575.1462523532854,384.4274491978456
17,90,8,82,18,0.49457588780146605,589.6775421120528,394.29788981855717
18,64,4,156,11,0.41936748183642575,590.7440895639604,394.97676652532425
19,84,7,246,17,0.3429636281526173,461.6472534387056,337.17959874934314
20,82,5,256,15,0.3454659423930888,588.7721638199384,393.6370014250704
21,82,7,226,20,0.3750893754212301,1.0014521556233025e+34,1.613375208825243e+33
22,62,10,181,18,0.4382724076914072,564.2331589041639,377.56056949090356
23,82,7,161,17,0.3644709676228632,579.6345165050499,387.2230573749557
24,98,9,225,19,0.3038186209805563,540.8035586004462,363.42714649427546

```