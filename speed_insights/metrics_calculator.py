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
    def __init__(self, y_true, y_pred):
        super().__init__(y_true, y_pred)

    def compute_metric(self):
        return mean_absolute_error(self.y_true, self.y_pred)


class MSECalculator(MetricCalculatorInterface):
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def compute_metric(self):
        return mean_squared_error(self.y_true, self.y_pred)


class R2Calculator(MetricCalculatorInterface):
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def compute_metric(self):
        return r2_score(self.y_true, self.y_pred)
