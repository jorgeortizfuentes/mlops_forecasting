import os
from datetime import datetime

import pandas as pd
from config import PATH_TO_SAVE_MODELS
from darts.models import TCNModel


class PowerPredictor:
    """
    A class for predicting power output using a trained TCNModel.
    """

    def __init__(self, model_path: str):
        """Initializes the PowerPredictor object.

        Args:
            param model_path (str): The path to the trained TCNModel.
        """
        self.model_path = model_path
        self.models_info = (
            pd.read_csv(os.path.join(model_path, "models_info.csv"))
            .iloc[-1]
            .to_dict()
        )
        self.last_date = datetime.strptime(
            self.models_info["last_date"], "%Y-%m-%d %H:%M:%S"
        )
        self.model_name = self.models_info["model_name"]
        self.model = TCNModel.load(
            os.path.join(model_path, self.model_name),
            map_location="cpu",
        )
        self.model.to_cpu()

    def get_time_position(self, date: datetime):
        """Calculates the number of predictions to make based on the time
        between the input date and the last date of the model's training data.

        Args:
            date (datetime): The datetime for which to make predictions.

        Returns:
            int: the number of predictions to make.
        """
        return (date - self.last_date).days * 144 + (
            date - self.last_date
        ).seconds // 600

    def predict(self, date: datetime):
        """Predicts the power output for the given datetime.

        Args:
            date (datetime): last date to make predictions.

        Returns:
            dict: dict with the predicted power output.
        """
        n_predictions = self.get_time_position(date)
        output = self.model.predict(
            n_predictions,
            num_loader_workers=7,
            verbose=True,
            batch_size=1024 * 30,
        )
        output = output["ActivePower"].pd_dataframe().reset_index(drop=False)
        output["ds"] = output["ds"].dt.strftime("%Y-%m-%d %H:%M")
        data = output.to_dict(orient="records")
        return self.convert_nan_to_zero(data)

    def convert_nan_to_zero(self, data: dict):
        """Converts all NaN values to 0

        Args:
            data (dict): The data to convert.

        Returns:
            dict: The converted data.
        """
        data = [
            {
                "ds": d["ds"],
                "ActivePower": 0
                if pd.isna(d["ActivePower"])
                else d["ActivePower"],
            }
            for d in data
        ]
        return data


if __name__ == "__main__":
    # Example usage
    power_predictor = PowerPredictor(PATH_TO_SAVE_MODELS)
    date = datetime(year=2020, month=5, day=3, hour=18, minute=52)
    predicted_power_output = power_predictor.predict(date)
    print(predicted_power_output)
