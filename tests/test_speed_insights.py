import unittest
import pandas as pd
from sklearn.linear_model import LinearRegression
from speed_insights.speed_insights import SpeedInsights


class TestSpeedInsights(unittest.TestCase):
    def setUp(self):
        # Create test data
        X = pd.DataFrame([1, 2, 3])
        y = pd.DataFrame([4, 5, 6])
        Model1 = LinearRegression().fit(X, y)
        Model2 = LinearRegression().fit(X, y)
        models = {"model1": Model1, "model2": Model2}

        # Initialize SpeedInsights
        self.speed_insights = SpeedInsights(X, y, models)

    def test_generate_insights(self):
        insights = self.speed_insights.generate_metrics()

        # Assert that insights are generated correctly
        self.assertIsNotNone(insights)
        self.assertIsInstance(insights, pd.DataFrame)

    def test_generate_visualisations(self):
        output_folder = "test_folder/"
        self.speed_insights.generate_metrics()
        self.speed_insights.generate_prediction_visualisations(output_folder)
        self.speed_insights.generate_feature_visualisations(output_folder)

    def test_select_outlier_predictions(self):
        # Test when metrics is None
        self.speed_insights.model_comparison.preds = None
        with self.assertRaises(AttributeError):
            self.speed_insights.select_outlier_predictions()

        # Test with valid metrics
        self.speed_insights.model_comparison.preds = pd.DataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["preds1", "preds2", "preds3"]
        )

        outliers = self.speed_insights.select_outlier_predictions(z_threshold=0)

        # Assert that outliers are selected correctly
        self.assertIsNotNone(outliers)


if __name__ == "__main__":
    unittest.main()
