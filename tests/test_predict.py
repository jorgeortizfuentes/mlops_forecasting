import unittest
import os
import sys

current_dir = os.getcwd()
scripts_dir = os.path.join(current_dir, "train")
sys.path.append(scripts_dir)

from predict import PowerPredictor
from datetime import datetime


class TestPowerPredictor(unittest.TestCase):
    def setUp(self):
        self.power_predictor = PowerPredictor("./models/")
        self.date_str = "2020-04-01 00:00"
        self.date = datetime.strptime(self.date_str, "%Y-%m-%d %H:%M")

    def test_get_time_position(self):
        time_pos = self.power_predictor.get_time_position(self.date)
        self.assertEqual(time_pos, 145)

    def test_predict(self):
        output = self.power_predictor.predict(self.date)
        self.assertIsInstance(output, dict)

if __name__ == '__main__':
    unittest.main()