import pandas as pd
import numpy as np

from speed_insights.model_comparison import ModelComparison
from speed_insights.data_loader import DataLoader
from speed_insights.visualiser import (
    HistogramVisualiser,
    ScatterplotVisualiser,
    BoxplotVisualiser,
)
from speed_insights.row_selecter import RowSelecter

from logging import getLogger

logger = getLogger(__name__)


class SpeedInsights:
    def __init__(self, X, y, models):
        # TODO: Add docstrings and type hints
        # TODO: Add checks on form of input data that can be handled

        self.X = X
        self.y = self._check_data(y)
        self.models = models

        data_loader = DataLoader(self.X, self.y)
        self.model_comparison = ModelComparison(data_loader)
        for name, model in self.models.items():
            logger.info(f"Adding model {name}")
            self.model_comparison.add_model(model, name)

        logger.info("Computing predictions")
        self.model_comparison._compute_predictions()

    def _check_data(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a DataFrame")
        if not data.empty and not np.issubdtype(data.dtypes[0], np.number):
            raise ValueError("data must contain numeric values")
        return data

    def generate_metrics(self):
        logger.info("Computing aggregate metrics")
        self.model_comparison.compute_agg_metrics()

        return self.model_comparison.metrics

    def generate_feature_visualisations(self, output_folder):
        logger.info(f"Generating visualisations of features in {output_folder}")
        HistogramVisualiser(self.X, output_folder).create_figures()
        for x in self.X.columns:
            ScatterplotVisualiser(self.X, output_folder, x).create_figures()
            BoxplotVisualiser(self.X, output_folder, x).create_figures()

    def generate_prediction_visualisations(self, output_folder):
        logger.info(
            f"Generating visualisations of predictions vs truth in {output_folder}"
        )
        HistogramVisualiser(self.model_comparison.preds, output_folder).create_figures()
        ScatterplotVisualiser(
            self.model_comparison.preds, output_folder, x="y_true"
        ).create_figures()
        BoxplotVisualiser(
            self.model_comparison.preds, output_folder, x="y_true"
        ).create_figures()
        # Add Q-Q plot

    def select_outlier_predictions(self, z_threshold=2):
        if self.model_comparison.metrics is None:
            raise RuntimeError(
                "Please run generate_metrics before calling select_outlier_predictions"
            )
        row_selecter = RowSelecter(self.model_comparison.metrics)
        outliers_rows, outliers_columns = row_selecter.find_outlier_rows(z_threshold)
        data = self.X.iloc[outliers_rows, :]
        data["reason"] = outliers_columns

        return data
