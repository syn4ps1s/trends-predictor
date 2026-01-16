"""Collection impact simulator for fashion demand data."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


class CollectionSimulator:
    """Simulate the impact of new collections on demand."""

    def __init__(
        self,
        collection_impact_duration: int = 14,
        peak_impact_day: int = 3,
        base_lift: float = 2.0,
    ):
        """
        Initialize collection simulator.

        Args:
            collection_impact_duration: Days collection impacts demand
            peak_impact_day: Day when peak impact occurs
            base_lift: Base demand multiplier during collection launch
        """
        self.collection_impact_duration = collection_impact_duration
        self.peak_impact_day = peak_impact_day
        self.base_lift = base_lift

    def simulate_collection_impact(
        self,
        demand_series: np.ndarray,
        collection_dates: List[datetime],
    ) -> np.ndarray:
        """
        Add collection impact to demand series.

        Args:
            demand_series: Base demand values
            collection_dates: Dates of new collections

        Returns:
            Adjusted demand with collection impacts
        """
        adjusted_demand = demand_series.copy()

        for collection_date in collection_dates:
            # Find indices affected by this collection
            for days_since in range(self.collection_impact_duration):
                # Decay impact over time
                decay_factor = np.exp(-days_since / (self.collection_impact_duration / 2))

                # Peak impact on specified day
                if days_since == self.peak_impact_day:
                    impact = self.base_lift * decay_factor
                else:
                    impact = (self.base_lift - 1) * decay_factor + 1

                # Apply to corresponding index
                idx = collection_date + timedelta(days=days_since)
                if hasattr(demand_series, 'index'):
                    matching_indices = demand_series.index[demand_series.index == idx]
                    if len(matching_indices) > 0:
                        adjusted_demand.loc[matching_indices] *= impact

        return adjusted_demand

    def get_collection_calendar(
        self,
        start_date: datetime,
        end_date: datetime,
        collections_per_year: int = 4,
    ) -> List[datetime]:
        """Get collection launch dates."""
        collections = []
        days_between = (end_date - start_date).days / collections_per_year

        for i in range(collections_per_year):
            collection_date = start_date + timedelta(days=i * days_between)
            collections.append(collection_date)

        return collections
