import unittest
import pandas as pd
from speed_insights.model_comparison import ModelComparison

class TestModelComparison(unittest.TestCase):
    def setUp(self):
        # Create test data
        columns = ['model1', 'model2', 'model3', 'y_true']
        data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        self.preds = pd.DataFrame(data, columns=columns)
        self.model_comparison = ModelComparison(self.preds)

    def test_find_rows_for_investigation_mae(self):
        # Set up test data
        percentile = 0.99
        expected_threshold = 7.92
        expected_filtered_df = pd.DataFrame([[8.0, 9.0, 10.0, 11.0]], columns=['model1', 'model2', 'model3', 'y_true'])

        # Call the method
        self.model_comparison.find_rows_for_investigation_mae(percentile)

        # Check the results
        self.assertEqual(self.model_comparison.threshold, expected_threshold)
        pd.testing.assert_frame_equal(self.model_comparison.filtered_df, expected_filtered_df)

if __name__ == '__main__':
    unittest.main()