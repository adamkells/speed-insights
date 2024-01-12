import unittest
import pandas as pd
from speed_insights.model_comparison import ModelComparison
from sklearn.linear_model import LinearRegression


class TestModelComparison(unittest.TestCase):
    def setUp(self):
        X = pd.DataFrame([1, 2, 3])
        y = pd.DataFrame([4, 5, 6])
        Model1 = LinearRegression().fit(X, y)
        self.models = {"model1": Model1}
        self.model_comparison = ModelComparison(X, y)

    def test_add_model(self):
        # Call the method
        self.model_comparison.add_model(self.models["model1"], "model1")

        # Check the results
        self.assertEqual(self.model_comparison.models["model1"], self.models["model1"])

    def test_compute_agg_metrics(self):
        # Call the method
        self.model_comparison.add_model(self.models["model1"], "model1")
        self.model_comparison.compute_agg_metrics()

        # Check the results
        self.assertEqual(self.model_comparison.metrics.loc["model1"]["mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
