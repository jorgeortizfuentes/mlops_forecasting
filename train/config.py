import os

pwd = os.getcwd()

# Get pwd before awto_mle_challenge
pwd = pwd.split("awto_mle_challenge")[0]
DATA_PATH = os.path.join(
    pwd, "awto_mle_challenge", "data", "wind_power_generation.csv"
)
PATH_TO_SAVE_MODELS = os.path.join(pwd, "awto_mle_challenge", "models")
