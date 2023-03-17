from datetime import datetime
import os
import sys
import unittest

current_dir = os.getcwd()
scripts_dir = os.path.join(current_dir, "train")
sys.path.append(scripts_dir)

from predict import PowerPredictor


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
        self.assertIsInstance(output, list)
        # output[0] is a dict with keys "ds" and "ActivePower"
        self.assertIsInstance(output[0], dict)
        self.assertIsInstance(output[0]["ds"], str)
        self.assertIsInstance(output[0]["ActivePower"], float)
        self.assertEqual(output[-1]["ds"], self.date_str)


if __name__ == "__main__":
    unittest.main()
