import numpy as np
import pandas as pd
from typing import Optional
from abc import ABC, abstractmethod

from speed_insights.metrics_calculator import METRIC_CALCULATORS

import logging

logger = logging.getLogger(__name__)


class ModelComparison:
    def __init__(self, X, y):
        self.models = {}
        self.X = X
        self.y = y
        self.preds = pd.DataFrame()
        self.preds["y_true"] = self.y

        self.metrics = pd.DataFrame(columns=list(METRIC_CALCULATORS.keys()))

        # self.investigation_rows = pd.DataFrame()

    def add_model(self, model, name):
        # Add a model to the comparison
        logging.info(f"Adding model {name}")
        self.models[name] = model
        self._compute_predictions()

    def _compute_predictions(self):
        # Compare predictions of different models and return evaluation metrics
        # for each model, make predictions the compute metrics
        for name, model in self.models.items():
            if name not in self.preds.columns:
                # Make predictions
                logging.info(f"Computing predictions for {name}")
                self.preds[name] = model.predict(self.X)

    def compute_agg_metrics(self):
        columns = [x for x in self.preds.columns if x != "y_true"]
        for column in columns:
            y_pred = self.preds[column]
            # Compute metrics
            metrics = {}
            for metric_name, metric_calculator_class in METRIC_CALCULATORS.items():
                metric_calculator = metric_calculator_class(self.y, y_pred)
                metric_value = metric_calculator.compute_metric()
                metrics[metric_name] = metric_value

            # Put metrics in a pandas dataframe structure
            self.metrics.loc[column] = metrics.values()

    def compute_row_wise_metrics(self):
        pass
