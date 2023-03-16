import pandas as pd
from config import DATA_PATH

# Read csv
df = pd.read_csv(DATA_PATH)

# Rename "Unnamed: 0" to "ds"
df.rename(columns={"Unnamed: 0": "ds"}, inplace=True)

# Convert ds to datetime and remove timezone
df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

# Remove all NaN in ActivePower column
df = df.dropna(subset=["ActivePower"])

# Convert NaN in another columns to 0
#df = df.fillna(0)

# Delete WTG column
df = df.drop(columns=["WTG"])

# Delete duplicates columns (Blade3PitchAngle, WindDirection)
df = df.drop(columns=["Blade3PitchAngle", "WindDirection"])

# Reset index
df.reset_index(drop=True, inplace=True)