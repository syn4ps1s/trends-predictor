"""Store-level aggregation and desegregation of predictions."""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class StoreAggregator:
    """
    Handle aggregation and desegregation of predictions at store level.

    Methods:
    - Aggregate product-level predictions to store level
    - Disaggregate aggregate predictions to store level
    - Apply store-specific adjustments
    """

    def __init__(
        self,
        method: str = "proportional",
        use_baselines: bool = True,
        store_size_weight: float = 0.4,
        historical_share_weight: float = 0.6,
    ):
        """
        Initialize store aggregator.

        Args:
            method: Aggregation method ('proportional', 'ml_based', 'hierarchical')
            use_baselines: Whether to use store baselines for aggregation
            store_size_weight: Weight for store size in aggregation
            historical_share_weight: Weight for historical market share
        """
        self.method = method
        self.use_baselines = use_baselines
        self.store_size_weight = store_size_weight
        self.historical_share_weight = historical_share_weight

        self.store_baselines = None
        self.store_shares = None
        self.store_metadata = None

        logger.info(f"Initialized StoreAggregator with method={method}")

    def fit(self, demand_df: pd.DataFrame, store_metadata: pd.DataFrame = None):
        """
        Fit aggregator with historical data.

        Args:
            demand_df: Historical demand data
            store_metadata: Store characteristics
        """
        logger.info("Fitting StoreAggregator with historical data...")

        # Calculate store baselines
        self.store_baselines = self._calculate_baselines(demand_df)

        # Calculate market shares
        self.store_shares = self._calculate_shares(demand_df)

        # Store metadata
        self.store_metadata = store_metadata

        logger.info(
            f"Fitted aggregator for {len(self.store_baselines)} stores"
        )

    def aggregate(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate product-level forecasts to store level.

        Args:
            forecast_df: DataFrame with product-level predictions

        Returns:
            DataFrame with store-level aggregates
        """
        logger.info("Aggregating forecasts to store level...")

        aggregates = []

        for store_id in forecast_df["store_id"].unique():
            store_data = forecast_df[forecast_df["store_id"] == store_id]

            aggregate_record = {
                "store_id": store_id,
                "forecast_date": store_data["forecast_date"].iloc[0],
                "point_forecast": store_data["point_forecast"].sum(),
            }

            if "lower_bound" in store_data.columns:
                aggregate_record["lower_bound"] = store_data["lower_bound"].sum()
                aggregate_record["upper_bound"] = store_data["upper_bound"].sum()

            aggregates.append(aggregate_record)

        return pd.DataFrame(aggregates)

    def disaggregate(
        self, aggregate_forecast_df: pd.DataFrame, disagg_method: str = None
    ) -> pd.DataFrame:
        """
        Disaggregate store-level forecasts to product level.

        Args:
            aggregate_forecast_df: Store-level forecasts
            disagg_method: Disaggregation method override

        Returns:
            DataFrame with product-level disaggregated forecasts
        """
        if disagg_method is None:
            disagg_method = self.method

        logger.info(
            f"Disaggregating forecasts using method={disagg_method}..."
        )

        disaggregated = []

        for _, agg_row in aggregate_forecast_df.iterrows():
            store_id = agg_row["store_id"]

            if disagg_method == "proportional":
                products = self._disaggregate_proportional(
                    store_id, agg_row["point_forecast"]
                )
            elif disagg_method == "ml_based":
                products = self._disaggregate_ml_based(
                    store_id, agg_row["point_forecast"]
                )
            else:
                products = self._disaggregate_proportional(
                    store_id, agg_row["point_forecast"]
                )

            for product_id, demand in products.items():
                disaggregated.append(
                    {
                        "store_id": store_id,
                        "product_id": product_id,
                        "forecast_date": agg_row["forecast_date"],
                        "disaggregated_demand": demand,
                    }
                )

        return pd.DataFrame(disaggregated)

    def apply_store_adjustments(
        self, forecast_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply store-specific adjustments to forecasts.

        Args:
            forecast_df: Original forecasts

        Returns:
            Adjusted forecasts
        """
        adjusted = forecast_df.copy()

        if self.store_baselines is None:
            logger.warning("Store baselines not available. Skipping adjustments.")
            return adjusted

        logger.info("Applying store-specific adjustments...")

        for store_id in adjusted["store_id"].unique():
            mask = adjusted["store_id"] == store_id

            baseline = self.store_baselines.get(store_id, 1.0)

            # Scale forecasts based on baseline
            adjusted.loc[mask, "point_forecast"] *= baseline

            if "lower_bound" in adjusted.columns:
                adjusted.loc[mask, "lower_bound"] *= baseline
                adjusted.loc[mask, "upper_bound"] *= baseline

        return adjusted

    def create_regional_aggregates(
        self, forecast_df: pd.DataFrame, region_map: Dict[int, str]
    ) -> pd.DataFrame:
        """
        Create regional aggregates from store-level forecasts.

        Args:
            forecast_df: Store-level forecasts
            region_map: Dictionary mapping store_id to region

        Returns:
            DataFrame with regional aggregates
        """
        logger.info("Creating regional aggregates...")

        # Add region column
        forecast_df["region"] = forecast_df["store_id"].map(region_map)

        # Group by region and date
        regional = forecast_df.groupby(["region", "forecast_date"]).agg(
            {
                "point_forecast": "sum",
                "lower_bound": "sum" if "lower_bound" in forecast_df.columns else None,
                "upper_bound": "sum" if "upper_bound" in forecast_df.columns else None,
            }
        ).reset_index()

        return regional.dropna(axis=1, how="all")

    def hierarchical_reconciliation(
        self,
        bottom_level_df: pd.DataFrame,
        hierarchy: Dict[str, list],
    ) -> Dict[str, pd.DataFrame]:
        """
        Reconcile hierarchical forecasts (store -> region -> national).

        Args:
            bottom_level_df: Store-level forecasts
            hierarchy: Hierarchy structure

        Returns:
            Dictionary with reconciled forecasts at each level
        """
        logger.info("Performing hierarchical reconciliation...")

        reconciled = {"store": bottom_level_df}

        # Aggregate to higher levels
        for level_name, store_groups in hierarchy.items():
            level_forecasts = []

            for group_name, stores in store_groups.items():
                group_data = bottom_level_df[
                    bottom_level_df["store_id"].isin(stores)
                ]

                agg = {
                    level_name: group_name,
                    "forecast_date": group_data["forecast_date"],
                    "point_forecast": group_data["point_forecast"].sum(),
                }

                level_forecasts.append(agg)

            reconciled[level_name] = pd.DataFrame(level_forecasts)

        return reconciled

    # ========== Private Methods ==========

    def _calculate_baselines(self, demand_df: pd.DataFrame) -> Dict[int, float]:
        """Calculate baseline demand per store."""
        baselines = {}

        for store_id in demand_df["store_id"].unique():
            store_data = demand_df[demand_df["store_id"] == store_id]
            baseline = store_data["demand"].tail(90).mean()
            baselines[store_id] = max(baseline, 0.1)  # Minimum baseline

        return baselines

    def _calculate_shares(self, demand_df: pd.DataFrame) -> Dict[int, float]:
        """Calculate market share per store."""
        store_totals = demand_df.groupby("store_id")["demand"].sum()
        total = store_totals.sum()

        return (store_totals / total).to_dict()

    def _disaggregate_proportional(
        self, store_id: int, aggregate_demand: float
    ) -> Dict[int, float]:
        """Disaggregate using proportional allocation."""
        # Simplified: equal distribution across products
        num_products = 500  # From config
        per_product = aggregate_demand / num_products

        return {
            i: per_product
            for i in range(num_products)
        }

    def _disaggregate_ml_based(
        self, store_id: int, aggregate_demand: float
    ) -> Dict[int, float]:
        """Disaggregate using ML-based product affinity."""
        # Placeholder for ML-based method
        return self._disaggregate_proportional(store_id, aggregate_demand)
