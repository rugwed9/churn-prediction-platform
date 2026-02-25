"""End-to-end ML pipeline orchestrator.

Ties together data generation, feature engineering, model training,
evaluation, and artifact persistence. Provides both full pipeline
execution and individual stage entry points.
"""

import logging
import time

from src.config import Settings, load_config, setup_logging
from src.data.generator import ChurnDataGenerator
from src.features.store import FeatureStore
from src.models.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class ChurnPipeline:
    """End-to-end churn prediction pipeline."""

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or load_config()
        setup_logging(self.config.logging)

        self.generator = ChurnDataGenerator(self.config.data)
        self.feature_store = FeatureStore(self.config.features)
        self.trainer = ModelTrainer(self.config)

    def run(self) -> dict:
        """Execute the full pipeline: generate -> feature engineer -> train -> evaluate.

        Returns:
            Dict with dataset stats, model metrics, and feature importance.
        """
        start = time.perf_counter()
        logger.info("Starting churn prediction pipeline")

        # Stage 1: Generate data
        logger.info("Stage 1: Data generation")
        df = self.generator.generate()
        self.generator.save(df)

        # Stage 2: Feature engineering
        logger.info("Stage 2: Feature engineering")
        X, y, feature_names = self.feature_store.fit_transform(df)
        self.feature_store.save()

        logger.info(
            "Dataset: %d samples, %d features, %.1f%% positive",
            X.shape[0],
            X.shape[1],
            y.mean() * 100,
        )

        # Stage 3: Model training
        logger.info("Stage 3: Model training")
        models = self.trainer.train_all(X, y, feature_names)
        self.trainer.save_models()

        # Stage 4: Feature importance (SHAP)
        logger.info("Stage 4: Feature importance")
        from src.models.explainer import ChurnExplainer

        xgb_model = models["xgboost"]
        explainer = ChurnExplainer(xgb_model.model, feature_names)
        importance = explainer.compute_global_importance(X)

        elapsed = time.perf_counter() - start

        results = {
            "dataset": {
                "n_samples": len(df),
                "n_features": X.shape[1],
                "churn_rate": float(y.mean()),
            },
            "models": {name: tm.metrics.to_dict() for name, tm in models.items()},
            "feature_importance": dict(list(importance.items())[:10]),
            "pipeline_duration_seconds": round(elapsed, 1),
        }

        logger.info("Pipeline complete in %.1fs", elapsed)
        return results


def create_pipeline(config_path: str | None = None) -> ChurnPipeline:
    """Factory function."""
    config = load_config(config_path)
    return ChurnPipeline(config)
