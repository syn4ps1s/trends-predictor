"""Seasonality feature extraction and processing."""

import numpy as np
import pandas as pd
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class SeasonalityProcessor:
    """Extract and process seasonality components from demand data."""

    def __init__(self, primary_period: int = 365, secondary_period: int = 7):
        """
        Initialize seasonality processor.

        Args:
            primary_period: Primary seasonality period (days)
            secondary_period: Secondary seasonality period (days)
        """
        self.primary_period = primary_period
        self.secondary_period = secondary_period

    def extract_fourier_features(
        self, time_series: np.ndarray, num_terms: int = 10
    ) -> np.ndarray:
        """
        Extract Fourier features for seasonality.

        Args:
            time_series: Input time series
            num_terms: Number of Fourier terms

        Returns:
            Array of shape (len(time_series), 2*num_terms) with sin/cos components
        """
        t = np.arange(len(time_series))
        features = []

        # Términos para período primario (anual)
        for k in range(1, num_terms + 1):
            features.append(np.sin(2 * np.pi * k * t / self.primary_period))
            features.append(np.cos(2 * np.pi * k * t / self.primary_period))

        # Términos para período secundario (semanal)
        for k in range(1, 3):  # Menos términos para período corto
            features.append(np.sin(2 * np.pi * k * t / self.secondary_period))
            features.append(np.cos(2 * np.pi * k * t / self.secondary_period))

        return np.column_stack(features)

    def decompose_seasonal(
        self, time_series: pd.Series, period: int = 365, model: str = "additive"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform seasonal decomposition using statsmodels.

        Args:
            time_series: Input time series (pandas Series)
            period: Seasonality period
            model: 'additive' or 'multiplicative'

        Returns:
            Tuple of (trend, seasonal, residual) components
        """
        if len(time_series) < 2 * period:
            logger.warning(
                f"Time series length ({len(time_series)}) < 2*period ({2*period}). "
                "Skipping decomposition."
            )
            return None, None, None

        try:
            result = seasonal_decompose(time_series, model=model, period=period, extrapolate="fill")
            return result.trend.values, result.seasonal.values, result.resid.values
        except Exception as e:
            logger.error(f"Error in seasonal decomposition: {e}")
            return None, None, None

    def extract_cyclical_features(
        self, dates: pd.DatetimeIndex, scale: bool = True
    ) -> pd.DataFrame:
        """
        Extract cyclical temporal features.

        Args:
            dates: DatetimeIndex of dates
            scale: Whether to scale features to [-1, 1]

        Returns:
            DataFrame with cyclical features
        """
        features = pd.DataFrame({"date": dates})

        # Day of year (cyclical)
        day_of_year = dates.dayofyear.values
        features["day_sin"] = np.sin(2 * np.pi * day_of_year / 365)
        features["day_cos"] = np.cos(2 * np.pi * day_of_year / 365)

        # Day of week (cyclical)
        day_of_week = dates.dayofweek.values
        features["week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        features["week_cos"] = np.cos(2 * np.pi * day_of_week / 7)

        # Month cyclical
        month = dates.month.values
        features["month_sin"] = np.sin(2 * np.pi * month / 12)
        features["month_cos"] = np.cos(2 * np.pi * month / 12)

        # Quarter
        quarter = dates.quarter.values
        features["quarter"] = quarter / 4

        # Is weekend
        features["is_weekend"] = (dates.dayofweek >= 5).astype(int)

        return features

    def extract_holiday_features(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Extract holiday and special event features.

        Args:
            dates: DatetimeIndex of dates

        Returns:
            DataFrame with holiday indicators
        """
        features = pd.DataFrame({"date": dates})

        # Black Friday (Friday after Thanksgiving - 4th Thursday of November)
        features["is_black_friday"] = 0
        features["is_holiday_season"] = 0
        features["is_summer_sales"] = 0

        for i, date in enumerate(dates):
            # Black Friday approximation
            if date.month == 11 and date.day >= 20:
                features.loc[i, "is_black_friday"] = 1

            # Holiday season (December)
            if date.month == 12:
                features.loc[i, "is_holiday_season"] = 1

            # Summer sales
            if date.month in [7, 8]:
                features.loc[i, "is_summer_sales"] = 1

        return features

    def create_seasonal_lags(
        self, time_series: pd.Series, lags: List[int] = None
    ) -> pd.DataFrame:
        """
        Create lagged features with seasonal lags.

        Args:
            time_series: Input time series
            lags: List of lag periods (e.g., [7, 30, 365])

        Returns:
            DataFrame with lagged features
        """
        if lags is None:
            lags = [7, 30, 365]

        df = pd.DataFrame({"value": time_series})

        for lag in lags:
            df[f"lag_{lag}"] = time_series.shift(lag)

        return df

    def create_rolling_features(
        self, time_series: pd.Series, windows: List[int] = None
    ) -> pd.DataFrame:
        """
        Create rolling statistics features.

        Args:
            time_series: Input time series
            windows: List of rolling window sizes

        Returns:
            DataFrame with rolling features
        """
        if windows is None:
            windows = [7, 30, 90]

        df = pd.DataFrame()

        for window in windows:
            df[f"rolling_mean_{window}"] = time_series.rolling(window=window).mean()
            df[f"rolling_std_{window}"] = time_series.rolling(window=window).std()
            df[f"rolling_min_{window}"] = time_series.rolling(window=window).min()
            df[f"rolling_max_{window}"] = time_series.rolling(window=window).max()

        return df

    def normalize_seasonality(
        self, data: np.ndarray, method: str = "minmax"
    ) -> Tuple[np.ndarray, dict]:
        """
        Normalize seasonality components.

        Args:
            data: Input data
            method: 'minmax' or 'standard'

        Returns:
            Normalized data and scaling parameters
        """
        params = {}

        if method == "minmax":
            data_min = np.nanmin(data, axis=0)
            data_max = np.nanmax(data, axis=0)
            data_range = data_max - data_min
            data_range[data_range == 0] = 1  # Avoid division by zero

            normalized = (data - data_min) / data_range
            params = {"min": data_min, "max": data_max, "range": data_range}

        elif method == "standard":
            data_mean = np.nanmean(data, axis=0)
            data_std = np.nanstd(data, axis=0)
            data_std[data_std == 0] = 1  # Avoid division by zero

            normalized = (data - data_mean) / data_std
            params = {"mean": data_mean, "std": data_std}

        return normalized, params
