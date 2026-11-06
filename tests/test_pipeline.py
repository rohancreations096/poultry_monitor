"""Unit and integration tests for Poultry Disease Risk Classification Pipeline."""
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from data_gen import simulate_flock
from features import BASE_FEATURES, ENGINEERED, FEATURES, VALID_RANGES, add_features, clean, risk_band


def test_data_simulation():
    """Verify simulate_flock generates valid synthetic flock days."""
    df = simulate_flock(1)
    assert len(df) == 70
    assert df["flock_id"].iloc[0] == "F001"
    for col in BASE_FEATURES:
        assert col in df.columns
    assert "early_disease" in df.columns
    assert set(df["early_disease"].unique()).issubset({0, 1})


def test_cleaning_valid_ranges():
    """Verify out-of-range sensor readings are mapped to NaN."""
    dirty_data = pd.DataFrame([{
        "house_temp_c": 999.0,      # Out of range (10-45)
        "humidity_pct": -10.0,      # Out of range (10-100)
        "ammonia_ppm": 20.0,        # In range (0-80)
        "feed_g_per_bird": 60.0,
        "water_ml_per_bird": 120.0,
        "mortality_pct": 0.05,
        "cough_sneeze_score": 0.5,
        "abnormal_droppings_pct": 5.0,
        "lethargy_score": 0.5,
        "performance_index": 100.0,
    }])
    cleaned = clean(dirty_data)
    assert np.isnan(cleaned["house_temp_c"].iloc[0])
    assert np.isnan(cleaned["humidity_pct"].iloc[0])
    assert cleaned["ammonia_ppm"].iloc[0] == 20.0


def test_feature_engineering_calculations():
    """Verify rolling feature computations and ratios."""
    sample = pd.DataFrame([
        {"flock_id": "F001", "day": 0, "feed_g_per_bird": 50.0, "water_ml_per_bird": 100.0,
         "mortality_pct": 0.02, "cough_sneeze_score": 0.0, "abnormal_droppings_pct": 4.0,
         "bird_age_days": 20, "house_temp_c": 25.0, "humidity_pct": 60.0, "ammonia_ppm": 10.0,
         "lethargy_score": 0.2, "performance_index": 100.0, "days_since_vaccination": 5},
        {"flock_id": "F001", "day": 1, "feed_g_per_bird": 50.0, "water_ml_per_bird": 105.0,
         "mortality_pct": 0.04, "cough_sneeze_score": 1.0, "abnormal_droppings_pct": 6.0,
         "bird_age_days": 21, "house_temp_c": 25.0, "humidity_pct": 60.0, "ammonia_ppm": 10.0,
         "lethargy_score": 0.2, "performance_index": 100.0, "days_since_vaccination": 6},
        {"flock_id": "F001", "day": 2, "feed_g_per_bird": 45.0, "water_ml_per_bird": 110.0,
         "mortality_pct": 0.06, "cough_sneeze_score": 2.0, "abnormal_droppings_pct": 8.0,
         "bird_age_days": 22, "house_temp_c": 25.0, "humidity_pct": 60.0, "ammonia_ppm": 10.0,
         "lethargy_score": 0.5, "performance_index": 98.0, "days_since_vaccination": 7},
    ])
    feat_df = add_features(sample)
    
    for feat in ENGINEERED:
        assert feat in feat_df.columns
        
    # Check water_feed_ratio on day 0: 100 / 50 = 2.0
    assert pytest.approx(feat_df["water_feed_ratio"].iloc[0], 0.01) == 2.0
    # Check 3-day average mortality on day 2: (0.02 + 0.04 + 0.06) / 3 = 0.04
    assert pytest.approx(feat_df["mortality_3d_avg"].iloc[2], 0.01) == 0.04
    # Check 3-day average cough on day 2: (0.0 + 1.0 + 2.0) / 3 = 1.0
    assert pytest.approx(feat_df["cough_3d_avg"].iloc[2], 0.01) == 1.0
    # Check 3-day average droppings on day 2: (4.0 + 6.0 + 8.0) / 3 = 6.0
    assert pytest.approx(feat_df["droppings_3d_avg"].iloc[2], 0.01) == 6.0


def test_risk_band_thresholds():
    """Verify clinical triage band categorization."""
    assert risk_band(0.95) == "HIGH"
    assert risk_band(0.70) == "HIGH"
    assert risk_band(0.69) == "WATCH"
    assert risk_band(0.40) == "WATCH"
    assert risk_band(0.39) == "NORMAL"
    assert risk_band(0.05) == "NORMAL"


def test_trained_model_inference():
    """Verify saved decision tree pipeline loads and generates valid probabilities."""
    model_path = root_dir / "outputs" / "decision_tree_model.joblib"
    assert model_path.exists(), "Trained model artifact missing in outputs/"
    
    model = joblib.load(model_path)
    
    # Create healthy dummy row
    healthy_row = pd.DataFrame([{
        "bird_age_days": 35, "house_temp_c": 26.0, "humidity_pct": 62.0, "ammonia_ppm": 12.0,
        "feed_g_per_bird": 55.0, "water_ml_per_bird": 110.0, "mortality_pct": 0.04,
        "cough_sneeze_score": 0.5, "abnormal_droppings_pct": 4.0, "lethargy_score": 0.6,
        "performance_index": 100.0, "days_since_vaccination": 10,
        "water_feed_ratio": 2.0, "feed_change_3d": 0.0,
        "mortality_3d_avg": 0.04, "cough_3d_avg": 0.5, "droppings_3d_avg": 4.0
    }])[FEATURES]
    
    probs = model.predict_proba(healthy_row)
    assert probs.shape == (1, 2)
    assert pytest.approx(probs[0].sum(), 0.001) == 1.0
    # Healthy flock should be normal risk (< 0.40)
    assert probs[0, 1] < 0.40
    
    # Create acute sick dummy row
    sick_row = pd.DataFrame([{
        "bird_age_days": 45, "house_temp_c": 28.5, "humidity_pct": 75.0, "ammonia_ppm": 25.0,
        "feed_g_per_bird": 42.0, "water_ml_per_bird": 140.0, "mortality_pct": 0.18,
        "cough_sneeze_score": 3.8, "abnormal_droppings_pct": 28.0, "lethargy_score": 3.5,
        "performance_index": 85.0, "days_since_vaccination": 20,
        "water_feed_ratio": 3.33, "feed_change_3d": -15.0,
        "mortality_3d_avg": 0.16, "cough_3d_avg": 3.5, "droppings_3d_avg": 25.0
    }])[FEATURES]
    
    sick_probs = model.predict_proba(sick_row)
    # Sick flock should be HIGH risk (>= 0.70)
    assert sick_probs[0, 1] >= 0.70


def test_test_predictions_consistency():
    """Verify test predictions integrity on held-out flocks."""
    pred_path = root_dir / "outputs" / "test_predictions.csv"
    assert pred_path.exists()
    
    df_pred = pd.read_csv(pred_path)
    assert len(df_pred) == 1050
    assert df_pred["flock_id"].nunique() == 15
    assert "risk_probability" in df_pred.columns
    assert (df_pred["risk_probability"] >= 0.0).all() and (df_pred["risk_probability"] <= 1.0).all()
