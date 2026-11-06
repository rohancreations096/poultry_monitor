"""Cleaning + feature engineering shared by training and the dashboard."""
import numpy as np
import pandas as pd

# Plausible physical ranges; values outside are treated as sensor faults -> NaN.
VALID_RANGES = {
    "house_temp_c": (10, 45),
    "humidity_pct": (10, 100),
    "ammonia_ppm": (0, 80),
    "feed_g_per_bird": (5, 250),
    "water_ml_per_bird": (10, 600),
    "mortality_pct": (0, 20),
    "cough_sneeze_score": (0, 5),
    "abnormal_droppings_pct": (0, 100),
    "lethargy_score": (0, 5),
    "performance_index": (40, 130),
}

BASE_FEATURES = [
    "bird_age_days", "house_temp_c", "humidity_pct", "ammonia_ppm",
    "feed_g_per_bird", "water_ml_per_bird", "mortality_pct",
    "cough_sneeze_score", "abnormal_droppings_pct", "lethargy_score",
    "performance_index", "days_since_vaccination",
]
ENGINEERED = [
    "water_feed_ratio", "feed_change_3d", "mortality_3d_avg",
    "cough_3d_avg", "droppings_3d_avg",
]
FEATURES = BASE_FEATURES + ENGINEERED


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-flock rolling features. Needs flock_id and day columns, sorted by time."""
    df = clean(df).sort_values(["flock_id", "day"]).reset_index(drop=True)
    g = df.groupby("flock_id", group_keys=False)
    df["water_feed_ratio"] = df["water_ml_per_bird"] / df["feed_g_per_bird"]
    prev_feed = g["feed_g_per_bird"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["feed_change_3d"] = (df["feed_g_per_bird"] - prev_feed) / prev_feed * 100
    df["mortality_3d_avg"] = g["mortality_pct"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["cough_3d_avg"] = g["cough_sneeze_score"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["droppings_3d_avg"] = g["abnormal_droppings_pct"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    return df


def risk_band(p: float) -> str:
    if p >= 0.70:
        return "HIGH"
    if p >= 0.40:
        return "WATCH"
    return "NORMAL"
