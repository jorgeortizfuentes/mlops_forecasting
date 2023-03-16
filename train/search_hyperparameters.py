import pandas as pd
from preprocess import df
from darts.metrics import mae, rmse
import optuna
from darts import TimeSeries
from darts.models import TCNModel
import torch

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