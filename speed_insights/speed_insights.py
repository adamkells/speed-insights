from speed_insights.model_comparison import ModelComparison
from speed_insights.data_loader import DataLoader
from speed_insights.visualiser import (
    HistogramVisualiser,
    ScatterplotVisualiser,
    BoxplotVisualiser,
)

from logging import getLogger

logger = getLogger(__name__)


class SpeedInsights:
    def __init__(self, X, y, models):
        # TODO: Add docstrings and type hints
        self.X = X
        self.y = y
        self.models = models

        data_loader = DataLoader(self.X, self.y)
        self.model_comparison = ModelComparison(data_loader)
        for name, model in self.models.items():
            logger.info(f"Adding model {name}")
            self.model_comparison.add_model(model, name)

        logger.info("Computing predictions")
        self.model_comparison._compute_predictions()

    def generate_metrics(self):
        logger.info("Computing aggregate metrics")
        self.model_comparison.compute_agg_metrics()

        return self.model_comparison.metrics

    def generate_feature_visualisations(self, output_folder):
        logger.info(f"Generating visualisations of features in {output_folder}")
        HistogramVisualiser(self.X, output_folder).create_figures()
        for x in self.X.columns:
            ScatterplotVisualiser(self.X, output_folder, x).create_figures()
            BoxplotVisualiser(self.X, output_folder, x).create_figures()

    def generate_prediction_visualisations(self, output_folder):
        if self.model_comparison is None:
            raise RuntimeError(
                "Please run generate_metrics before calling generate_prediction_visualisations"
            )

        logger.info(
            f"Generating visualisations of predictions vs truth in {output_folder}"
        )
        HistogramVisualiser(self.model_comparison.preds, output_folder).create_figures()
        ScatterplotVisualiser(
            self.model_comparison.preds, output_folder, x="y_true"
        ).create_figures()
        BoxplotVisualiser(
            self.model_comparison.preds, output_folder, x="y_true"
        ).create_figures()
        # Add Q-Q plot

    def select_outlier_predictions(self, threshold=0.1):
        pass
