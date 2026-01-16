"""Macroeconomic features extraction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class MacroFeaturesProcessor:
    """Extract and process macroeconomic features."""

    def __init__(
        self,
        gdp_growth_annual: float = 0.03,
        inflation_rate: float = 0.02,
        confidence_volatility: float = 0.1,
    ):
        """
        Initialize macro features processor.

        Args:
            gdp_growth_annual: Annual GDP growth rate
            inflation_rate: Annual inflation rate
            confidence_volatility: Consumer confidence volatility
        """
        self.gdp_growth = gdp_growth_annual
        self.inflation_rate = inflation_rate
        self.confidence_volatility = confidence_volatility

    def generate_gdp_trend(
        self, num_periods: int, start_value: float = 100.0
    ) -> np.ndarray:
        """
        Generate GDP trend with growth.

        Args:
            num_periods: Number of periods
            start_value: Initial GDP value

        Returns:
            Array of GDP values
        """
        daily_growth = self.gdp_growth / 365
        gdp = start_value * np.exp(daily_growth * np.arange(num_periods))
        return gdp

    def generate_inflation_index(
        self, num_periods: int, start_value: float = 100.0
    ) -> np.ndarray:
        """
        Generate inflation index.

        Args:
            num_periods: Number of periods
            start_value: Initial value (typically 100)

        Returns:
            Array of inflation index values
        """
        daily_inflation = self.inflation_rate / 365
        inflation = start_value * np.exp(daily_inflation * np.arange(num_periods))
        return inflation

    def generate_consumer_confidence(self, num_periods: int) -> np.ndarray:
        """
        Generate consumer confidence index with volatility.

        Args:
            num_periods: Number of periods

        Returns:
            Array of confidence values (centered at 100)
        """
        # Base trend
        trend = 100 + 5 * np.sin(2 * np.pi * np.arange(num_periods) / 365)

        # Volatility
        volatility = np.random.normal(0, self.confidence_volatility * 10, num_periods)

        confidence = trend + volatility
        return np.clip(confidence, 50, 150)  # Bounded between 50-150

    def generate_unemployment_rate(self, num_periods: int) -> np.ndarray:
        """
        Generate unemployment rate with trend and seasonality.

        Args:
            num_periods: Number of periods

        Returns:
            Array of unemployment rates
        """
        base_rate = 5.0
        trend = -0.01 * np.arange(num_periods) / 365  # Declining trend
        seasonality = 0.3 * np.sin(2 * np.pi * np.arange(num_periods) / 365)
        noise = np.random.normal(0, 0.1, num_periods)

        unemployment = base_rate + trend + seasonality + noise
        return np.clip(unemployment, 2, 12)  # Realistic bounds

    def generate_exchange_rates(
        self, num_periods: int, base_currency: str = "USD", num_currencies: int = 3
    ) -> Dict[str, np.ndarray]:
        """
        Generate exchange rates (relevant for international fashion retail).

        Args:
            num_periods: Number of periods
            base_currency: Base currency
            num_currencies: Number of foreign currencies

        Returns:
            Dictionary of exchange rate arrays
        """
        rates = {}
        currencies = ["EUR", "GBP", "JPY", "CHF", "CAD"][:num_currencies]

        for currency in currencies:
            # Random walk for exchange rates
            returns = np.random.normal(0.0001, 0.01, num_periods)
            price = 100 * np.exp(np.cumsum(returns))
            rates[f"{base_currency}_{currency}"] = price

        return rates

    def create_macro_features_df(
        self, dates: pd.DatetimeIndex, include_vars: List[str] = None
    ) -> pd.DataFrame:
        """
        Create complete macro features dataframe.

        Args:
            dates: DatetimeIndex of dates
            include_vars: List of variables to include

        Returns:
            DataFrame with macro features
        """
        if include_vars is None:
            include_vars = ["gdp", "inflation", "confidence", "unemployment"]

        num_periods = len(dates)
        df = pd.DataFrame({"date": dates})

        if "gdp" in include_vars:
            df["gdp_index"] = self.generate_gdp_trend(num_periods)

        if "inflation" in include_vars:
            df["inflation_index"] = self.generate_inflation_index(num_periods)

        if "confidence" in include_vars:
            df["consumer_confidence"] = self.generate_consumer_confidence(num_periods)

        if "unemployment" in include_vars:
            df["unemployment_rate"] = self.generate_unemployment_rate(num_periods)

        if "exchange_rates" in include_vars:
            rates = self.generate_exchange_rates(num_periods)
            for rate_name, values in rates.items():
                df[rate_name] = values

        # Normalize features
        for col in df.columns:
            if col != "date":
                df[col] = self._normalize_feature(df[col].values)

        return df

    def _normalize_feature(self, values: np.ndarray, method: str = "standard") -> np.ndarray:
        """Normalize a feature."""
        if method == "standard":
            return (values - np.mean(values)) / np.std(values)
        elif method == "minmax":
            return (values - np.min(values)) / (np.max(values) - np.min(values))
        return values

    def calculate_purchasing_power_index(
        self, gdp_index: np.ndarray, inflation_index: np.ndarray
    ) -> np.ndarray:
        """
        Calculate purchasing power index.

        Args:
            gdp_index: GDP values
            inflation_index: Inflation values

        Returns:
            Purchasing power index
        """
        return (gdp_index / 100) / (inflation_index / 100)

    def identify_economic_shocks(
        self, confidence_index: np.ndarray, threshold_std: float = 2.0
    ) -> np.ndarray:
        """
        Identify economic shocks based on confidence swings.

        Args:
            confidence_index: Consumer confidence values
            threshold_std: Standard deviation threshold for shock detection

        Returns:
            Binary array indicating shock periods
        """
        mean = np.mean(confidence_index)
        std = np.std(confidence_index)
        shocks = np.abs(confidence_index - mean) > threshold_std * std
        return shocks.astype(int)
