import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from abc import ABC, abstractmethod


class MetricCalculatorInterface(ABC):
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    @abstractmethod
    def compute_metric(self):
        pass


class MAECalculator(MetricCalculatorInterface):
    def compute_metric(self):
        return mean_absolute_error(self.y_true, self.y_pred)


class MSECalculator(MetricCalculatorInterface):
    def compute_metric(self):
        return mean_squared_error(self.y_true, self.y_pred)
    

class RMSECalculator(MetricCalculatorInterface):
    def compute_metric(self):
        return np.sqrt(mean_squared_error(self.y_true, self.y_pred, squared=False))


class R2Calculator(MetricCalculatorInterface):
    def compute_metric(self):
        return r2_score(self.y_true, self.y_pred)
    

class MAPECalculator(MetricCalculatorInterface):
    def compute_metric(self):
        return np.mean(np.abs((self.y_true - self.y_pred) / self.y_true)) * 100
    

class SMAPECalculator(MetricCalculatorInterface):
    def compute_metric(self):
        return np.mean(np.abs((self.y_pred - self.y_true) / ((self.y_true + self.y_pred) / 2))) * 100


class MaxAbsoluteErrorCalculator(MetricCalculatorInterface):
    def compute_metric(self):
        return np.max(np.abs(self.y_true - self.y_pred))
    

class StdErrorCalculator(MetricCalculatorInterface):
    def compute_metric(self):
        return np.std(self.y_true - self.y_pred)
    
METRIC_CALCULATORS = {
    "mae": MAECalculator,
    "mse": MSECalculator,
    "r2": R2Calculator,
    "rmse": RMSECalculator
}