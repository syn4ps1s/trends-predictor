"""Unified prediction engine for demand forecasting."""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    Unified inference engine for demand predictions.

    Orchestrates:
    - DeepAR+ neural network forecasts
    - MDP-based inventory optimization
    - Store-level aggregation and desegregation
    - Confidence intervals and scenario analysis
    """

    def __init__(
        self,
        deepar_pipeline=None,
        mdp_optimizer=None,
        store_aggregator=None,
        seasonality_processor=None,
        macro_processor=None,
    ):
        """
        Initialize prediction engine.

        Args:
            deepar_pipeline: Trained DeepAR+ pipeline
            mdp_optimizer: Trained MDP optimizer
            store_aggregator: Store aggregation module
            seasonality_processor: Seasonality feature processor
            macro_processor: Macro features processor
        """
        self.deepar = deepar_pipeline
        self.mdp = mdp_optimizer
        self.store_aggregator = store_aggregator
        self.seasonality_processor = seasonality_processor
        self.macro_processor = macro_processor

        logger.info("Initialized PredictionEngine")

    def forecast(
        self,
        X_context: np.ndarray,
        metadata: pd.DataFrame,
        forecast_horizon: int = 30,
        include_intervals: bool = True,
        include_optimization: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive demand forecasts.

        Args:
            X_context: Context sequences for forecasting
            metadata: DataFrame with temporal/store/product metadata
            forecast_horizon: Days to forecast ahead
            include_intervals: Whether to include confidence intervals
            include_optimization: Whether to include MDP optimization

        Returns:
            Dictionary with forecast dataframes
        """
        logger.info(
            f"Starting forecasting for {len(metadata)} records, "
            f"horizon={forecast_horizon} days"
        )

        # DeepAR+ Forecasting
        deepar_predictions = self.deepar.predict(
            X_context, return_intervals=include_intervals
        )

        # Prepare forecast dataframe
        forecasts = []
        for i, (_, row) in enumerate(metadata.iterrows()):
            forecast_record = {
                "store_id": row["store_id"],
                "product_id": row["product_id"],
                "forecast_timestamp": datetime.now(),
                "forecast_date": row["timestamp"] + timedelta(days=1),
                "point_forecast": deepar_predictions["point_forecast"][i, 0],
            }

            if include_intervals:
                forecast_record["lower_bound"] = deepar_predictions["lower_bound"][i, 0]
                forecast_record["upper_bound"] = deepar_predictions["upper_bound"][i, 0]

            forecasts.append(forecast_record)

        forecast_df = pd.DataFrame(forecasts)

        results = {"point_forecasts": forecast_df}

        # MDP Optimization
        if include_optimization and self.mdp is not None:
            optimized_df = self._apply_mdp_optimization(forecast_df)
            results["optimized_inventory"] = optimized_df

        # Aggregate to store level
        if self.store_aggregator is not None:
            store_aggregates = self.store_aggregator.aggregate(forecast_df)
            results["store_aggregates"] = store_aggregates

        logger.info(f"Forecast completed with {len(forecast_df)} predictions")

        return results

    def scenario_forecast(
        self,
        X_context: np.ndarray,
        scenarios: Dict[str, Dict[str, float]],
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts under different macro scenarios.

        Args:
            X_context: Context sequences
            scenarios: Dictionary of scenario names and parameter overrides

        Returns:
            Dictionary of scenario-specific forecasts
        """
        scenario_forecasts = {}

        for scenario_name, scenario_params in scenarios.items():
            logger.info(f"Generating forecast for scenario: {scenario_name}")

            # Apply scenario adjustments
            X_adjusted = self._apply_scenario_adjustments(X_context, scenario_params)

            # Forecast
            predictions = self.deepar.predict(X_adjusted, return_intervals=True)

            # Store results
            scenario_forecasts[scenario_name] = predictions

        return scenario_forecasts

    def ensemble_forecast(
        self,
        X_context: np.ndarray,
        weights: Dict[str, float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate ensemble forecasts combining multiple models/scenarios.

        Args:
            X_context: Context sequences
            weights: Dictionary of model weights

        Returns:
            Ensemble predictions
        """
        if weights is None:
            weights = {"deepar": 0.7, "trend": 0.2, "seasonal": 0.1}

        logger.info(f"Generating ensemble forecast with weights: {weights}")

        ensemble_pred = np.zeros_like(self.deepar.predict(X_context)["point_forecast"])

        # Weighted combination of different methods
        if weights.get("deepar", 0) > 0:
            deepar_pred = self.deepar.predict(X_context)["point_forecast"]
            ensemble_pred += weights["deepar"] * deepar_pred

        if weights.get("trend", 0) > 0:
            trend_pred = self._trend_extrapolation(X_context)
            ensemble_pred += weights["trend"] * trend_pred

        if weights.get("seasonal", 0) > 0:
            seasonal_pred = self._seasonal_forecast(X_context)
            ensemble_pred += weights["seasonal"] * seasonal_pred

        return {
            "ensemble_forecast": ensemble_pred,
            "component_forecasts": {
                "deepar": deepar_pred if weights.get("deepar", 0) > 0 else None,
                "trend": trend_pred if weights.get("trend", 0) > 0 else None,
                "seasonal": seasonal_pred if weights.get("seasonal", 0) > 0 else None,
            },
        }

    def explain_prediction(
        self,
        X_context: np.ndarray,
        prediction_idx: int,
    ) -> Dict[str, float]:
        """
        Explain individual prediction using SHAP or similar.

        Args:
            X_context: Context sequences
            prediction_idx: Index of prediction to explain

        Returns:
            Dictionary with feature importance
        """
        # Placeholder for SHAP integration
        importances = {
            "seasonality": 0.35,
            "macro_factors": 0.25,
            "store_baseline": 0.20,
            "recent_trend": 0.15,
            "collection_impact": 0.05,
        }

        logger.info(
            f"Explained prediction {prediction_idx}: {importances}"
        )

        return importances

    # ========== Helper Methods ==========

    def _apply_mdp_optimization(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """Apply MDP-based inventory optimization to forecasts."""
        optimized = forecast_df.copy()
        optimized["optimal_order"] = 0

        for _, row in optimized.iterrows():
            # Get optimal order for store
            order = self.mdp.get_optimal_order(
                current_inventory=int(row["point_forecast"]),
                predicted_demand_state="medium",
            )
            optimized.loc[_, "optimal_order"] = order

        return optimized

    def _apply_scenario_adjustments(
        self, X_context: np.ndarray, scenario_params: Dict[str, float]
    ) -> np.ndarray:
        """Apply scenario parameter adjustments to context."""
        X_adjusted = X_context.copy()

        # Apply multipliers from scenario
        for param, multiplier in scenario_params.items():
            # Find feature index (simplified)
            X_adjusted *= multiplier

        return X_adjusted

    def _trend_extrapolation(self, X_context: np.ndarray) -> np.ndarray:
        """Simple trend extrapolation."""
        # Extract trend from most recent context
        recent_trend = X_context[:, -30:, 0]  # Last 30 days
        trend = np.polyfit(np.arange(30), recent_trend, 1)[0]

        # Extrapolate
        future_steps = self.deepar.config.forecast_horizon
        trend_forecast = np.tile(trend, (X_context.shape[0], future_steps, 1))

        return trend_forecast.squeeze(-1)

    def _seasonal_forecast(self, X_context: np.ndarray) -> np.ndarray:
        """Seasonal forecast based on historical patterns."""
        # Extract seasonal component
        seasonal = X_context[:, -365::7, 0]  # Weekly seasonality

        # Repeat seasonal pattern
        future_steps = self.deepar.config.forecast_horizon
        seasonal_forecast = np.tile(seasonal, (1, future_steps // 52 + 1))[:, :future_steps]

        return seasonal_forecast
