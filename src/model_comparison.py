import numpy as np
import pandas as pd


class ModelComparison:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.models = {}

        self.preds = pd.DataFrame()
        self.preds['y_true'] = data_loader.y

        self.metrics = pd.DataFrame(columns=['mae', 'mse', 'r2'])

    def add_model(self, model, name):
        # Add a model to the comparison
        self.models[name] = model
        self._compute_predictions()

    def add_benchmark_model(self):
        # Add a benchmark model to the comparison
        y_pred = np.ones_like(self.data_loader.y) * np.mean(self.data_loader.y)
        self.add_model(y_pred, "mean_benchmark")

    def _compute_predictions(self):
        # Compare predictions of different models and return evaluation metrics
        # for each model, make predictions the compute metrics
        for name, model in self.models.items():
            if name not in self.preds.columns:
                # Make predictions
                self.preds[name] = model.predict(self.data_loader.X)

    def compute_agg_metrics(self):
        columns = [x for x in self.preds.columns if x != 'y_true']
        for column in columns:
            y_pred = self.preds[column]
            # Compute metrics
            metric_calculator = MetricCalculator(self.data_loader.y, y_pred)
            mae = metric_calculator.compute_mae()
            mse = metric_calculator.compute_mse()
            r2 = metric_calculator.compute_r2()
            # Put metrics in a pandas dataframe structure
            self.metrics.loc[column] = [mae, mse, r2]
            

    def compute_row_wise_metrics(self):
        pass
