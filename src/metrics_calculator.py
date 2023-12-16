from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Class to compute metrics on an aggregated level
class MetricCalculator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def compute_mae(self):
        return mean_absolute_error(self.y_true, self.y_pred)

    def compute_mse(self):
        return mean_squared_error(self.y_true, self.y_pred)

    def compute_r2(self):
        return r2_score(self.y_true, self.y_pred)
