"""
Synthetic Fashion Demand Data Generator.

Generates synthetic time series data with:
- Strong seasonality (annual + weekly)
- Collection cycles
- Macro factors (GDP, inflation, consumer confidence)
- Store-specific characteristics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic fashion demand data."""

    def __init__(
        self,
        num_stores: int = 100,
        num_products: int = 500,
        num_days: int = 730,
        start_date: str = "2022-01-01",
        seed: int = 42,
    ):
        """
        Initialize the synthetic data generator.

        Args:
            num_stores: Number of retail stores
            num_products: Number of products
            num_days: Number of days to generate
            start_date: Start date for time series
            seed: Random seed for reproducibility
        """
        self.num_stores = num_stores
        self.num_products = num_products
        self.num_days = num_days
        self.start_date = pd.to_datetime(start_date)
        self.seed = seed

        np.random.seed(seed)
        logger.info(
            f"Initialized SyntheticDataGenerator: {num_stores} stores, "
            f"{num_products} products, {num_days} days"
        )

    def generate(self) -> pd.DataFrame:
        """
        Generate complete synthetic demand dataset.

        Returns:
            DataFrame with columns: [timestamp, store_id, product_id, demand,
                                     seasonality_factor, collection_impact,
                                     macro_factor, store_baseline]
        """
        logger.info("Starting synthetic data generation...")

        # Crear índice temporal
        dates = pd.date_range(self.start_date, periods=self.num_days, freq="D")

        # Generar datos combinatorios
        data_records = []

        for store_id in range(self.num_stores):
            store_baseline = self._generate_store_baseline(store_id)

            for product_id in range(self.num_products):
                # Baseline de producto
                product_baseline = np.random.uniform(50, 500)

                # Estacionalidad
                seasonal_pattern = self._generate_seasonality(dates)

                # Impacto de colecciones
                collection_impact = self._generate_collection_impact(dates)

                # Factor macro
                macro_factor = self._generate_macro_factor(dates)

                # Ruido
                noise = np.random.normal(0, 0.1, len(dates))

                # Demanda final = baseline * seasonalidad * colecciones * macro + ruido
                demand = (
                    (store_baseline * product_baseline)
                    * (1 + seasonal_pattern)
                    * (1 + collection_impact)
                    * (1 + macro_factor)
                    + noise
                )
                demand = np.maximum(demand, 0)  # No negativos

                for date, demand_val, seas, coll, macro in zip(
                    dates, demand, seasonal_pattern, collection_impact, macro_factor
                ):
                    data_records.append(
                        {
                            "timestamp": date,
                            "store_id": store_id,
                            "product_id": product_id,
                            "demand": max(0, demand_val),
                            "seasonality_factor": seas,
                            "collection_impact": coll,
                            "macro_factor": macro,
                            "store_baseline": store_baseline,
                        }
                    )

        df = pd.DataFrame(data_records)
        logger.info(f"Generated {len(df)} demand records")

        return df

    def _generate_store_baseline(self, store_id: int) -> float:
        """Generate baseline demand multiplier for a store."""
        # Log-normal distribution para variabilidad realista
        return np.exp(np.random.normal(0, 0.5))

    def _generate_seasonality(
        self, dates: pd.DatetimeIndex, primary_period: int = 365, secondary_period: int = 7
    ) -> np.ndarray:
        """Generate seasonality pattern (annual + weekly)."""
        t = np.arange(len(dates))

        # Componente anual (ciclo de moda)
        annual = 0.3 * np.sin(2 * np.pi * t / primary_period)

        # Componente semanal (patrones de compra)
        weekly = 0.15 * np.sin(2 * np.pi * t / secondary_period)

        return annual + weekly

    def _generate_collection_impact(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Generate impact of new collection launches."""
        impact = np.zeros(len(dates))

        # Nueva colección cada 3 meses (90 días)
        collection_frequency = 90
        ramp_up_days = 14
        peak_multiplier = 2.5
        decay_rate = 0.15

        for launch_day in range(0, len(dates), collection_frequency):
            if launch_day < len(dates):
                # Ramp-up phase
                for i in range(ramp_up_days):
                    if launch_day + i < len(dates):
                        impact[launch_day + i] = (i / ramp_up_days) * peak_multiplier

                # Decay phase
                for i in range(ramp_up_days, ramp_up_days + 60):
                    if launch_day + i < len(dates):
                        decay = np.exp(-decay_rate * (i - ramp_up_days) / 7)
                        impact[launch_day + i] = peak_multiplier * decay

        return impact

    def _generate_macro_factor(
        self,
        dates: pd.DatetimeIndex,
        gdp_growth: float = 0.03,
        inflation: float = 0.02,
    ) -> np.ndarray:
        """Generate macroeconomic factors."""
        t = np.arange(len(dates)) / 365  # Años

        # Tendencia de crecimiento GDP
        gdp_trend = gdp_growth * t

        # Inflación
        inflation_trend = inflation * t

        # Consumer confidence (variación estacional)
        confidence = 0.1 * np.sin(2 * np.pi * t)

        # Eventos estacionales
        seasonal_events = np.zeros(len(dates))
        for i, date in enumerate(dates):
            # Black Friday (noviembre) - sólo si existe
            if date.month == 11 and 15 <= date.day <= 30:
                seasonal_events[i] = 0.8
            # Holiday season (diciembre)
            elif date.month == 12:
                seasonal_events[i] = 0.6
            # Summer sales (julio-agosto)
            elif date.month in [7, 8]:
                seasonal_events[i] = 0.4

        macro_factor = gdp_trend - inflation_trend + confidence + seasonal_events / 10

        return np.clip(macro_factor, -0.5, 0.5)

    def generate_store_metadata(self) -> pd.DataFrame:
        """Generate store characteristics."""
        stores = []
        for store_id in range(self.num_stores):
            stores.append(
                {
                    "store_id": store_id,
                    "store_type": np.random.choice(
                        ["flagship", "outlet", "regular", "online"], p=[0.1, 0.2, 0.6, 0.1]
                    ),
                    "region": np.random.choice(["North", "South", "East", "West", "Central"]),
                    "store_size_sqm": np.random.uniform(500, 5000),
                    "store_age_years": np.random.uniform(0.5, 25),
                    "competitor_density": np.random.uniform(0.1, 1.0),
                }
            )
        return pd.DataFrame(stores)

    def generate_product_metadata(self) -> pd.DataFrame:
        """Generate product characteristics."""
        products = []
        for product_id in range(self.num_products):
            products.append(
                {
                    "product_id": product_id,
                    "category": np.random.choice(
                        ["Dresses", "Pants", "Tops", "Outerwear", "Accessories"]
                    ),
                    "price_segment": np.random.choice(["Economy", "Standard", "Premium"]),
                    "seasonality_sensitivity": np.random.uniform(0.5, 2.0),
                    "collection_affinity": np.random.uniform(0.1, 1.0),
                }
            )
        return pd.DataFrame(products)
