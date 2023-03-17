from datetime import datetime
import uvicorn
from fastapi import FastAPI
import os
import sys

# Add scripts directory to path
if os.getcwd().endswith("api"):
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
    Predict the wind power generation from the last date of the training data (2020-03-30 23:50:00) to the date indicated. 

    DISCLAIMER: The model can only predict 887 predictions at 10-minute intervals. Therefore, the model only allows inferences between 2020-03-30 23:50:00 (last date of the training dataset) and 2020-04-06 03:40.

    Args:
        date (str): Tte datetime for which to make predictions (format: "YYYY-MM-DD HH:MM").

    Returns:
        dict: the predicted wind power output.
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