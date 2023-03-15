import pandas as pd

df_path = "/home/jorge/data/awto_mle_challenge/data/wind_power_generation.csv"
df = pd.read_csv(df_path)

# Rename "Unnamed: 0" to "ds"
df.rename(columns={"Unnamed: 0": "ds"}, inplace=True)

# Convert ds to datetime
df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)


# Remove all NaN in ActivePower column
df = df.dropna(subset=["ActivePower"])

# Convert NaN in another columns to 0
#df = df.fillna(0)

# Delete WTG column
df = df.drop(columns=["WTG"])

df.reset_index(drop=True, inplace=True)