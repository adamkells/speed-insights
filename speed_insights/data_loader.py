import pandas as pd


class DataLoader:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        # Store data
        self.X = X
        self.y = y
