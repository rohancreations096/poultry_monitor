"""Command-line interface (CLI) for Poultry Disease Risk Scoring.

Usage examples:
    python predict.py --droppings 25.0 --feed 42.0 --water 140.0 --cough 3.5 --mortality 0.16
    python predict.py --preset sick
    python predict.py --preset healthy
    python predict.py --file path/to/sample.csv
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from features import FEATURES, add_features, clean, risk_band

MODEL_PATH = Path(__file__).resolve().parent / "outputs" / "decision_tree_model.joblib"


def get_default_readings():
    return {
        "bird_age_days": 35,
        "house_temp_c": 26.0,
        "humidity_pct": 62.0,
        "ammonia_ppm": 12.0,
        "feed_g_per_bird": 55.0,
        "water_ml_per_bird": 110.0,
        "mortality_pct": 0.04,
        "cough_sneeze_score": 0.5,
        "abnormal_droppings_pct": 4.0,
        "lethargy_score": 0.6,
        "performance_index": 100.0,
        "days_since_vaccination": 10,
        "feed_change_3d": 0.0,
    }


def predict_sample(data: dict):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run 'python train.py' first.")
    model = joblib.load(MODEL_PATH)

    row = dict(data)
    # Derive missing rolling or ratio fields if not provided
    feed = row.get("feed_g_per_bird", 55.0)
    water = row.get("water_ml_per_bird", 110.0)
    mort = row.get("mortality_pct", 0.04)
    cough = row.get("cough_sneeze_score", 0.5)
    dropp = row.get("abnormal_droppings_pct", 4.0)
    feed_chg = row.get("feed_change_3d", 0.0)

    if "water_feed_ratio" not in row:
        row["water_feed_ratio"] = water / feed if feed > 0 else 2.0
    if "feed_change_3d" not in row:
        row["feed_change_3d"] = feed_chg
    if "mortality_3d_avg" not in row:
        row["mortality_3d_avg"] = row.get("mortality_3d_avg", mort)
    if "cough_3d_avg" not in row:
        row["cough_3d_avg"] = row.get("cough_3d_avg", cough)
    if "droppings_3d_avg" not in row:
        row["droppings_3d_avg"] = row.get("droppings_3d_avg", dropp)

    df = pd.DataFrame([row])
    # Apply bounds cleaning
    df = clean(df)
    # Ensure all required features exist
    for f in FEATURES:
        if f not in df:
            df[f] = float("nan")

    prob = float(model.predict_proba(df[FEATURES])[0, 1])
    band = risk_band(prob)

    advice_map = {
        "HIGH": "[HIGH RISK] ISOLATE AND INSPECT: High early disease probability detected. "
                "Quarantine shed, alert attending farm veterinarian, verify drinker chlorination, "
                "and collect diagnostic swab samples immediately.",
        "WATCH": "[WATCH] INTENSIFY MONITORING: Elevated disease risk indicators present. "
                 "Perform targeted inspection within 24 hours, check ventilation fans and drinker nipple flow.",
        "NORMAL": "[NORMAL] BASELINE HEALTH: Normal parameters. Continue routine daily sensor and health monitoring."
    }

    return {
        "risk_probability": round(prob, 4),
        "risk_percentage": round(prob * 100, 1),
        "risk_band": band,
        "action_protocol": advice_map[band],
        "input_features": {f: (None if pd.isna(df[f].iloc[0]) else round(float(df[f].iloc[0]), 3)) for f in FEATURES}
    }


def main():
    parser = argparse.ArgumentParser(description="Poultry Early Disease Risk Assessment CLI")
    parser.add_argument("--preset", choices=["healthy", "sick"], help="Use sample preset readings")
    parser.add_argument("--file", type=str, help="Path to CSV containing flock readings")
    parser.add_argument("--age", type=int, help="Bird age in days")
    parser.add_argument("--temp", type=float, help="House temperature in °C")
    parser.add_argument("--humidity", type=float, help="Relative humidity %%")
    parser.add_argument("--ammonia", type=float, help="Ammonia in ppm")
    parser.add_argument("--feed", type=float, help="Feed intake in g/bird")
    parser.add_argument("--water", type=float, help="Water intake in ml/bird")
    parser.add_argument("--feed-change", type=float, help="Feed change vs 3d prior %%")
    parser.add_argument("--mortality", type=float, help="Daily mortality %%")
    parser.add_argument("--cough", type=float, help="Respiratory cough/sneeze score (0-5)")
    parser.add_argument("--droppings", type=float, help="Abnormal droppings %% (0-100)")
    parser.add_argument("--lethargy", type=float, help="Lethargy score (0-5)")
    parser.add_argument("--perf", type=float, help="Performance index %%")
    parser.add_argument("--vacc", type=int, help="Days since vaccination")
    parser.add_argument("--json", action="store_true", help="Output result as pure JSON")

    args = parser.parse_args()

    if args.file:
        raw = pd.read_csv(args.file)
        feat_df = add_features(raw)
        model = joblib.load(MODEL_PATH)
        feat_df["risk_prob"] = model.predict_proba(feat_df[FEATURES])[:, 1]
        feat_df["risk_band"] = feat_df["risk_prob"].apply(risk_band)
        out_cols = [c for c in ["flock_id", "day", "risk_prob", "risk_band"] if c in feat_df.columns]
        print(feat_df[out_cols].to_string(index=False))
        return

    readings = get_default_readings()
    if args.preset == "sick":
        readings.update({
            "bird_age_days": 48,
            "house_temp_c": 28.5,
            "humidity_pct": 74.0,
            "ammonia_ppm": 24.0,
            "feed_g_per_bird": 41.0,
            "water_ml_per_bird": 145.0,
            "mortality_pct": 0.17,
            "cough_sneeze_score": 3.6,
            "abnormal_droppings_pct": 28.0,
            "lethargy_score": 3.2,
            "feed_change_3d": -16.5,
            "droppings_3d_avg": 26.0,
            "mortality_3d_avg": 0.15,
            "cough_3d_avg": 3.2,
        })
    elif args.preset == "healthy":
        readings.update({
            "bird_age_days": 30,
            "house_temp_c": 25.5,
            "humidity_pct": 60.0,
            "ammonia_ppm": 11.0,
            "feed_g_per_bird": 58.0,
            "water_ml_per_bird": 115.0,
            "mortality_pct": 0.03,
            "cough_sneeze_score": 0.2,
            "abnormal_droppings_pct": 3.0,
            "lethargy_score": 0.4,
            "feed_change_3d": +2.1,
            "droppings_3d_avg": 3.2,
            "mortality_3d_avg": 0.03,
            "cough_3d_avg": 0.3,
        })

    # Override with explicit CLI flags if passed
    if args.age is not None: readings["bird_age_days"] = args.age
    if args.temp is not None: readings["house_temp_c"] = args.temp
    if args.humidity is not None: readings["humidity_pct"] = args.humidity
    if args.ammonia is not None: readings["ammonia_ppm"] = args.ammonia
    if args.feed is not None: readings["feed_g_per_bird"] = args.feed
    if args.water is not None: readings["water_ml_per_bird"] = args.water
    if args.feed_change is not None: readings["feed_change_3d"] = args.feed_change
    if args.mortality is not None: readings["mortality_pct"] = args.mortality
    if args.cough is not None: readings["cough_sneeze_score"] = args.cough
    if args.droppings is not None: readings["abnormal_droppings_pct"] = args.droppings
    if args.lethargy is not None: readings["lethargy_score"] = args.lethargy
    if args.perf is not None: readings["performance_index"] = args.perf
    if args.vacc is not None: readings["days_since_vaccination"] = args.vacc

    result = predict_sample(readings)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "="*60)
        print("POULTRY FLOCK EARLY-DISEASE RISK ASSESSMENT")
        print("="*60)
        print(f"Risk Probability : {result['risk_probability']} ({result['risk_percentage']}%)")
        print(f"Risk Tier        : {result['risk_band']}")
        print(f"Action Protocol  : {result['action_protocol']}")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
