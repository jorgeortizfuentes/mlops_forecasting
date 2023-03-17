import unittest
import os
import sys

current_dir = os.getcwd()
scripts_dir = os.path.join(current_dir, "train")
sys.path.append(scripts_dir)

from preprocess import DataPreprocessor
from config import DATA_PATH

class TestDataPreprocessor(unittest.TestCase):
    
    def setUp(self):
        self.data_path = DATA_PATH
        self.preprocessor = DataPreprocessor(self.data_path)
        
    def test_rename_columns(self):
        self.preprocessor.rename_columns()
        self.assertEqual(self.preprocessor.df.columns[0], "ds")
        
    def test_convert_datetime(self):
        self.preprocessor.rename_columns()
        self.preprocessor.convert_datetime()
        self.assertEqual(self.preprocessor.df["ds"].dtype, "datetime64[ns]")
        
    def test_remove_nan(self):
        self.preprocessor.remove_nan()
        self.assertFalse(self.preprocessor.df["ActivePower"].isnull().values.any())
        
    def test_fill_nan(self):
        self.preprocessor.fill_nan()
        self.assertFalse(self.preprocessor.df.isnull().values.any())
        
    def test_delete_duplicates_columns(self):
        self.preprocessor.delete_duplicates_columns()
        self.assertNotIn("WTG", self.preprocessor.df.columns)
        self.assertNotIn("Blade3PitchAngle", self.preprocessor.df.columns)
        self.assertNotIn("WindDirection", self.preprocessor.df.columns)
        
    def test_reset_index(self):
        self.preprocessor.reset_index()
        self.assertEqual(self.preprocessor.df.index[0], 0)
        
if __name__ == "__main__":
    unittest.main()