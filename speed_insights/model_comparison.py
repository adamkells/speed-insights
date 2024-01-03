import numpy as np
import pandas as pd
from typing import Optional
from abc import ABC, abstractmethod

from speed_insights.metrics_calculator import METRIC_CALCULATORS

import logging

logger = logging.getLogger(__name__)


class ModelComparison:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.models = {}

        self.preds = pd.DataFrame()
        self.preds["y_true"] = data_loader.y

        self.metrics = pd.DataFrame(columns=list(METRIC_CALCULATORS.keys()))

        # self.investigation_rows = pd.DataFrame()

    def add_model(self, model, name):
        # Add a model to the comparison
        logging.info(f"Adding model {name}")
        self.models[name] = model
        self._compute_predictions()

    # def add_benchmark_model(self):
    #     # Add a benchmark model to the comparison
    #     y_pred = np.ones_like(self.data_loader.y) * np.mean(self.data_loader.y)
    #     self.add_model(y_pred, "mean_benchmark")

    def _compute_predictions(self):
        # Compare predictions of different models and return evaluation metrics
        # for each model, make predictions the compute metrics
        for name, model in self.models.items():
            if name not in self.preds.columns:
                # Make predictions
                logging.info(f"Computing predictions for {name}")
                self.preds[name] = model.predict(self.data_loader.X)

    def compute_agg_metrics(self):
        columns = [x for x in self.preds.columns if x != "y_true"]
        for column in columns:
            y_pred = self.preds[column]
            # Compute metrics
            metrics = {}
            for metric_name, metric_calculator_class in METRIC_CALCULATORS.items():
                metric_calculator = metric_calculator_class(self.data_loader.y, y_pred)
                metric_value = metric_calculator.compute_metric()
                metrics[metric_name] = metric_value

            print(metrics)
            # Put metrics in a pandas dataframe structure
            self.metrics.loc[column] = metrics.values()

    def compute_row_wise_metrics(self):
        pass

    # def find_rows_for_investigation(self, func, func_name, percentile=0.99):
    #     # Apply the given function to calculate the differences
    #     diff_df = self.preds.iloc[:, 2:].apply(
    #         lambda x: func(x, self.preds["y_true"]), axis=1
    #     )

    #     # Calculate the threshold for the top percentile values
    #     threshold = diff_df.stack().quantile(percentile)

    #     # Filter rows containing top percentile values
    #     filtered_df = diff_df[diff_df > threshold].dropna()
    #     filtered_df["filter_reason"] = f"{func_name} variance high"
    #     self.investigation_rows.append(filtered_df)


# class RowWiseModelComparisonFunctions(ABC):
#     def __init__(self):
#         self.name = 'custom_function'
#     @abstractmethod
#     def compute_metric(self, x, y):
#         pass

# class RowWiseMAE(ABC):
#     def __init__(self):
#         self.name = 'mae'
#     # Example of a custom function
#     def compute_metric(x, y):
#         # Replace this with your actual logic
#         return np.abs(x - y)
