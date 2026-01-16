"""Store-specific feature extraction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class StoreFeatureProcessor:
    """Extract and process store-specific features."""

    def __init__(self):
        """Initialize store feature processor."""
        self.store_metadata = None

    def load_store_metadata(self, metadata_df: pd.DataFrame) -> None:
        """
        Load store metadata.

        Args:
            metadata_df: DataFrame with store information
        """
        self.store_metadata = metadata_df
        logger.info(f"Loaded metadata for {len(metadata_df)} stores")

    def create_store_features(self, num_days: int) -> pd.DataFrame:
        """
        Create comprehensive store features for all stores and days.

        Args:
            num_days: Number of days

        Returns:
            DataFrame with store features
        """
        if self.store_metadata is None:
            raise ValueError("Store metadata not loaded. Call load_store_metadata first.")

        features_list = []

        for _, store_row in self.store_metadata.iterrows():
            store_id = store_row["store_id"]

            # Static features
            store_size_norm = self._normalize_value(
                store_row["store_size_sqm"], 500, 5000
            )
            store_age_norm = self._normalize_value(
                store_row["store_age_years"], 0.5, 25
            )
            competitor_density_norm = store_row["competitor_density"]

            # Type embedding
            store_type_features = self._encode_store_type(store_row["store_type"])

            # Region embedding
            region_features = self._encode_region(store_row["region"])

            # Replicate features for each day
            for day in range(num_days):
                day_features = {
                    "store_id": store_id,
                    "day": day,
                    "store_size_norm": store_size_norm,
                    "store_age_norm": store_age_norm,
                    "competitor_density": competitor_density_norm,
                }

                # Add type features
                for key, val in store_type_features.items():
                    day_features[f"store_type_{key}"] = val

                # Add region features
                for key, val in region_features.items():
                    day_features[f"region_{key}"] = val

                features_list.append(day_features)

        return pd.DataFrame(features_list)

    def calculate_store_baselines(
        self, demand_df: pd.DataFrame, window_days: int = 90
    ) -> Dict[int, float]:
        """
        Calculate baseline demand for each store.

        Args:
            demand_df: DataFrame with demand data
            window_days: Window for calculating baseline

        Returns:
            Dictionary of store_id -> baseline demand
        """
        baselines = {}

        for store_id in demand_df["store_id"].unique():
            store_data = demand_df[demand_df["store_id"] == store_id]
            baseline = store_data["demand"].tail(window_days).mean()
            baselines[store_id] = baseline

        return baselines

    def calculate_store_shares(
        self, demand_df: pd.DataFrame
    ) -> Dict[int, float]:
        """
        Calculate market share (proportion of total demand) for each store.

        Args:
            demand_df: DataFrame with demand data

        Returns:
            Dictionary of store_id -> share
        """
        store_totals = demand_df.groupby("store_id")["demand"].sum()
        total_demand = store_totals.sum()
        shares = (store_totals / total_demand).to_dict()
        return shares

    def create_store_similarity_matrix(self) -> np.ndarray:
        """
        Create similarity matrix between stores based on characteristics.

        Returns:
            Store similarity matrix (num_stores x num_stores)
        """
        if self.store_metadata is None:
            raise ValueError("Store metadata not loaded")

        n_stores = len(self.store_metadata)
        similarity = np.zeros((n_stores, n_stores))

        for i in range(n_stores):
            for j in range(n_stores):
                if i == j:
                    similarity[i, j] = 1.0
                else:
                    # Type similarity
                    type_sim = (
                        1.0
                        if self.store_metadata.iloc[i]["store_type"]
                        == self.store_metadata.iloc[j]["store_type"]
                        else 0.5
                    )

                    # Region similarity
                    region_sim = (
                        1.0
                        if self.store_metadata.iloc[i]["region"]
                        == self.store_metadata.iloc[j]["region"]
                        else 0.5
                    )

                    # Size similarity
                    size_i = self.store_metadata.iloc[i]["store_size_sqm"]
                    size_j = self.store_metadata.iloc[j]["store_size_sqm"]
                    size_sim = 1.0 - abs(size_i - size_j) / max(size_i, size_j)

                    # Combined similarity
                    similarity[i, j] = (type_sim + region_sim + 2 * size_sim) / 4

        return similarity

    def extract_store_patterns(
        self, demand_df: pd.DataFrame, pattern_window: int = 30
    ) -> Dict[int, np.ndarray]:
        """
        Extract typical demand patterns for each store.

        Args:
            demand_df: DataFrame with demand data
            pattern_window: Window size for pattern extraction

        Returns:
            Dictionary of store_id -> pattern array
        """
        patterns = {}

        for store_id in demand_df["store_id"].unique():
            store_data = demand_df[demand_df["store_id"] == store_id].sort_values(
                "timestamp"
            )

            demand_values = store_data["demand"].values

            # Extract repeating pattern using correlation
            if len(demand_values) > pattern_window:
                recent_pattern = demand_values[-pattern_window:]
                patterns[store_id] = recent_pattern
            else:
                patterns[store_id] = demand_values

        return patterns

    def identify_store_segments(self, num_segments: int = 5) -> Dict[int, int]:
        """
        Segment stores using clustering based on characteristics.

        Args:
            num_segments: Number of segments

        Returns:
            Dictionary of store_id -> segment
        """
        if self.store_metadata is None:
            raise ValueError("Store metadata not loaded")

        from sklearn.cluster import KMeans

        # Features for clustering
        features = self.store_metadata[["store_size_sqm", "competitor_density"]].values

        kmeans = KMeans(n_clusters=num_segments, random_state=42)
        segments = kmeans.fit_predict(features)

        store_segments = {}
        for idx, store_id in enumerate(self.store_metadata["store_id"]):
            store_segments[store_id] = int(segments[idx])

        return store_segments

    @staticmethod
    def _normalize_value(value: float, min_val: float, max_val: float) -> float:
        """Normalize value to [0, 1]."""
        return (value - min_val) / (max_val - min_val)

    @staticmethod
    def _encode_store_type(store_type: str) -> Dict[str, int]:
        """Encode store type as one-hot."""
        types = ["flagship", "outlet", "regular", "online"]
        encoding = {t: (1 if t == store_type else 0) for t in types}
        return encoding

    @staticmethod
    def _encode_region(region: str) -> Dict[str, int]:
        """Encode region as one-hot."""
        regions = ["North", "South", "East", "West", "Central"]
        encoding = {r: (1 if r == region else 0) for r in regions}
        return encoding
