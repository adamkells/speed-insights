from speed_insights.model_comparison import ModelComparison
from speed_insights.data_loader import DataLoader
from speed_insights.visualiser import HistogramVisualiser, ScatterplotVisualiser

from logging import getLogger

logger = getLogger(__name__)


class SpeedInsights:
    def __init__(self, X, y, models):
        # TODO: Add docstrings and type hints
        self.X = X
        self.y = y
        self.models = models

    def generate_metrics(self):
        logger.info("Generating metrics")
        data_loader = DataLoader(self.X, self.y)
        model_comparison = ModelComparison(data_loader)
        for name, model in self.models.items():
            logger.info(f"Adding model {name}")
            model_comparison.add_model(model, name)

        logger.info("Adding benchmark model")
        # model_comparison.add_benchmark_model()

        logger.info("Computing predictions")
        model_comparison._compute_predictions()

        logger.info("Computing aggregate metrics")
        model_comparison.compute_agg_metrics()
        self.model_comparison = model_comparison

        return model_comparison.metrics

    def generate_visualisations(self, output_folder):
        logger.info(f"Generating visualisations in {output_folder}")
        HistogramVisualiser(self.model_comparison.preds, output_folder).create_figures()
        ScatterplotVisualiser(
            self.model_comparison.preds, output_folder
        ).create_figures()
