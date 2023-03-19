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
        """Delete WTG column and duplicates columns
        (Blade3PitchAngle, WindDirection).
        """
        self.df = self.df.drop(
            columns=["WTG", "Blade3PitchAngle", "WindDirection"]
        )

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
