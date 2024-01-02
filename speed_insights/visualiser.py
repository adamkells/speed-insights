import seaborn as sns
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod

# Add logging
import logging

logger = logging.getLogger(__name__)


class VisualiserInterface(ABC):
    def __init__(self, dataframe, folder_name):
        self.dataframe = dataframe
        self.folder_name = folder_name
        self.figures = []

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
            self.figures.append(fig)

    def save_figures(self):
        os.makedirs(self.folder_name, exist_ok=True)
        for column, fig in zip(self.dataframe.columns, self.figures):
            logger.info(f"Saving histogram for {column}")
            fig.figure.savefig(
                os.path.join(self.folder_name, f"histogram_{column}.png")
            )


class ScatterplotVisualiser(VisualiserInterface):
    def __init__(self, dataframe, folder_name):
        super().__init__(dataframe, folder_name)

    def create_figures(self):
        logger.info("Creating scatterplots")
        for column in self.dataframe.columns:
            # check that column is not y_true
            if column == "y_true":
                continue
            logger.info(f"Creating scatterplot for {column}")
            plt.figure()
            fig = sns.scatterplot(data=self.dataframe, x="y_true", y=column)
            self.figures.append(fig)

    def save_figures(self):
        os.makedirs(self.folder_name, exist_ok=True)
        for column, fig in zip(self.dataframe.columns, self.figures):
            logger.info(f"Saving scatterplot for {column}")
            fig.figure.savefig(
                os.path.join(self.folder_name, f"scatterplot_{column}.png")
            )
