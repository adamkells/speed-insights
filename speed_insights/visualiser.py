import seaborn as sns
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod
from typing import Optional

# Add logging
import logging

logger = logging.getLogger(__name__)


def _check_col_in_dataframe(column, dataframe):
    return column in dataframe.columns


class VisualiserInterface(ABC):
    def __init__(self, dataframe, folder_name):
        self.dataframe = dataframe
        self.folder_name = folder_name
        self.figures = {}

    @abstractmethod
    def create_figures(self):
        pass

    @abstractmethod
    def save_figures(self):
        pass


class HistogramVisualiser(VisualiserInterface):
    def __init__(self, dataframe, folder_name):
        super().__init__(dataframe, folder_name)

    def create_figures(self):
        logger.info("Creating histograms")
        for column in self.dataframe.columns:
            logger.info(f"Creating histogram for {column}")
            plt.figure()
            fig = sns.histplot(self.dataframe[column])
            self.figures[column] = fig

    def save_figures(self):
        os.makedirs(self.folder_name, exist_ok=True)
        for column, fig in self.figures.items():
            logger.info(f"Saving histogram for {column}")
            fig.figure.savefig(
                os.path.join(self.folder_name, f"histogram_{column}.png")
            )


class ScatterplotVisualiser(VisualiserInterface):
    def __init__(self, dataframe, folder_name, x: Optional[str] = "y_true"):
        super().__init__(dataframe, folder_name)
        if _check_col_in_dataframe(x, dataframe):
            self.x = x
        else:
            raise ValueError(f"Column {x} not in dataframe")

    def create_figures(self):
        logger.info("Creating scatterplots")
        for column in self.dataframe.columns:
            # check that column is not the default x
            if column == self.x:
                continue
            logger.info(f"Creating scatterplot for {column}")
            plt.figure()
            fig = sns.scatterplot(data=self.dataframe, x=self.x, y=column)
            self.figures[column] = fig

    def save_figures(self):
        os.makedirs(self.folder_name, exist_ok=True)
        for column, fig in self.figures.items():
            logger.info(f"Saving scatterplot for {column}")
            fig.figure.savefig(
                os.path.join(self.folder_name, f"scatterplot_{column}_{self.x}.png")
            )


class BoxplotVisualiser(VisualiserInterface):
    def __init__(self, dataframe, folder_name, x: Optional[str] = "y_true"):
        super().__init__(dataframe, folder_name)
        if _check_col_in_dataframe(x, dataframe):
            self.x = x
        else:
            raise ValueError(f"Column {x} not in dataframe")

    def create_figures(self):
        logger.info("Creating boxplots")
        for column in self.dataframe.columns:
            # check that column is not y_true
            if column == self.x:
                continue
            # check that column is not categorical
            if self.dataframe[column].dtype != "object":
                continue
            logger.info(f"Creating boxplot for {column}")
            plt.figure()
            fig = sns.boxplot(data=self.dataframe, y=self.x, x=column)
            self.figures[column] = fig

    def save_figures(self):
        os.makedirs(self.folder_name, exist_ok=True)
        for column, fig in self.figures.items():
            logger.info(f"Saving boxplot for {column}")
            fig.figure.savefig(
                os.path.join(self.folder_name, f"boxplot_{column}_{self.x}.png")
            )
