import unittest
import os
import sys

current_dir = os.getcwd()
scripts_dir = os.path.join(current_dir, "api")
sys.path.append(scripts_dir)

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestMain(unittest.TestCase):
    def test_predict_success(self):
        response = client.get("/predict/2020-04-01 00:00")
        self.assertEqual(response.status_code, 200)
        self.assertIn("ActivePower", response.json())

    def test_predict_incorrect_format(self):
        response = client.get("/predict/2021-04-01")
        self.assertEqual(response.status_code, 200)
        self.assertIn("error", response.json())

if __name__ == "__main__":
    unittest.main()
