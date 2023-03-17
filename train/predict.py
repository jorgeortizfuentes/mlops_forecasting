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
        self.model = TCNModel.load(os.path.join(model_path, self.model_name), map_location="cpu")
        self.model.to_cpu()


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
        output = output["ActivePower"].pd_dataframe().reset_index(drop=False)
        output["ds"] = output["ds"].dt.strftime("%Y-%m-%d %H:%M")
        return output.to_dict(orient="records")


if __name__ == "__main__":
    # Example usage
    power_predictor = PowerPredictor(PATH_TO_SAVE_MODELS)
    date = datetime(2020, 4, 5, 18, 50)
    predicted_power_output = power_predictor.predict(date)
    print(predicted_power_output)
    
