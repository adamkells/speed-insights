import unittest
import pandas as pd
import os
from speed_insights.visualiser import HistogramVisualiser, ScatterplotVisualiser

class TestHistogramVisualiser(unittest.TestCase):
    def setUp(self):
        self.dataframe = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        self.folder_name = 'test_folder'
        self.visualiser = HistogramVisualiser(self.dataframe, self.folder_name)

    def test_create_figures(self):
        self.visualiser.create_figures()
        self.assertEqual(len(self.visualiser.figures), 2)

    def test_save_figures(self):
        self.visualiser.create_figures()
        self.visualiser.save_figures()
        # Add assertions to check if the figures are saved correctly
        assert os.path.exists(os.path.join(self.folder_name, f"histogram_A.png"))

class TestScatterplotVisualiser(unittest.TestCase):
    def setUp(self):
        self.dataframe = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'y_true': [7, 8, 9]})
        self.folder_name = 'test_folder'
        self.visualiser = ScatterplotVisualiser(self.dataframe, self.folder_name)

    def test_create_figures(self):
        self.visualiser.create_figures()
        self.assertEqual(len(self.visualiser.figures), 2)

    def test_save_figures(self):
        self.visualiser.create_figures()
        self.visualiser.save_figures()
        # Add assertions to check if the figures are saved correctly
        assert os.path.exists(os.path.join(self.folder_name, f"scatterplot_A.png"))

if __name__ == '__main__':
    unittest.main()