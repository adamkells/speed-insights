import pandas as pd


class RowSelecter:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def find_outlier_rows(self, z_threshold):
        # Calculate the z-score for each column in the dataframe
        z_scores = (self.df - self.df.mean()) / self.df.std()

        row_indices_list = []
        above_zero_columns = []
        for index, row in z_scores.iterrows():
            # Get indices where values are above
            if row.max() > z_threshold:
                row_indices_list.append(index)
                above_zero_columns.append(row.index[row > z_threshold].tolist())

        return row_indices_list, above_zero_columns

