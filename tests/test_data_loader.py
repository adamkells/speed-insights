import unittest
import pandas as pd
from speed_insights.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    def test_data_loader(self):
        # Create test data
        X = pd.DataFrame([1, 2, 3])
        y = pd.DataFrame([4, 5, 6])

        # Initialize DataLoader
        data_loader = DataLoader(X, y)

        # Check if data is stored correctly
        pd.testing.assert_frame_equal(data_loader.X, X)
        pd.testing.assert_frame_equal(data_loader.y, y)


if __name__ == "__main__":
    unittest.main()
