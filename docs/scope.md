# Project Scope: Intelligent Poultry Farm Health Monitor & Early Disease Symptoms Detection

**Project Focus:** Machine Learning & Predictive Analytics  
**Topic:** Decision Tree Classifier to Flag Flocks Showing Early Disease Symptoms  
**System Type:** Decision Support & Early Warning System (Data-Driven Poultry Health Monitor)

---

## 1. Problem Statement & Motivation
Modern poultry farming operates at high flock densities where infectious diseases (e.g., Avian Influenza, Infectious Bronchitis, Newcastle Disease, Coccidiosis) can propagate rapidly within sheds, leading to severe economic losses and high mortality rates if not detected early. 

Conventional methods rely on manual inspections or reactive treatment after overt mortality spikes occur. The objective of this project is to build an intelligent, predictive early-warning system that analyzes daily farm environmental sensor readings and flock health-log records to flag flocks exhibiting early, sub-clinical disease symptoms before catastrophic mortality occurs.

---

## 2. Core Objectives
1. **Data Ingestion & Cleaning:** Ingest multivariate daily flock records comprising ambient shed sensors (temperature, humidity, ammonia concentration) and operational health-log entries (feed consumption, water intake, mortality, respiratory cough/sneeze scores, droppings consistency, lethargy, performance index).
2. **Feature Engineering:** Calculate physiological dynamic indicators, including water-to-feed ratios and 3-day rolling trends (e.g., `feed_change_3d`, `mortality_3d_avg`, `cough_3d_avg`, `droppings_3d_avg`) to capture disease ramp-up dynamics.
3. **Leakage-Free Validation Strategy:** Implement strict flock-level partitioning (`GroupShuffleSplit` for hold-out testing and `GroupKFold` for 5-fold cross-validation) ensuring that no flock's data appears simultaneously in both training and test partitions.
4. **Interpretable Decision Tree Modeling:** Train a cost-sensitive / balanced Decision Tree classifier optimized on F1-score with hyperparameter tuning (`max_depth`, `criterion`, `min_samples_leaf`) to extract transparent, auditable clinical rules.
5. **Interactive Farm Health Monitor:** Provide actionable insights to farm managers and veterinarians through:
   - Streamlit interactive dashboard (`app.py`)
   - Standalone web-based executive monitor (`Poultry Farm Health Monitor.html`)
   - Direct risk banding: `HIGH` (≥70%), `WATCH` (40%–69%), and `NORMAL` (<40%) with prescribed bio-security workflows.

---

## 3. Dataset Architecture
- **Flock-Day Granularity:** 60 flocks monitored over a 70-day cycle = 4,200 flock-day observations.
- **Sensor & Health Signals (12 Base + 5 Engineered):**
  1. `bird_age_days`: Age of birds in days.
  2. `house_temp_c`: Shed ambient temperature (°C).
  3. `humidity_pct`: Shed relative humidity (%).
  4. `ammonia_ppm`: Ammonia gas level in shed air (ppm).
  5. `feed_g_per_bird`: Daily feed consumption per bird (grams).
  6. `water_ml_per_bird`: Daily water intake per bird (milliliters).
  7. `mortality_pct`: Daily mortality percentage.
  8. `cough_sneeze_score`: Clinical respiratory sound index (scale 0–5).
  9. `abnormal_droppings_pct`: Proportion of abnormal/watery droppings (0–100%).
  10. `lethargy_score`: Flock behavioral lethargy observation (scale 0–5).
  11. `performance_index`: Normalized growth/egg performance index (% of target).
  12. `days_since_vaccination`: Days elapsed since previous vaccination.
  13. `water_feed_ratio`: Ratio of water intake to feed consumption.
  14. `feed_change_3d`: Percentage change in feed consumption relative to the prior 3-day average.
  15. `mortality_3d_avg`: 3-day rolling average mortality (%).
  16. `cough_3d_avg`: 3-day rolling average respiratory score.
  17. `droppings_3d_avg`: 3-day rolling average abnormal droppings (%).
- **Target Variable (`early_disease`):** Binary indicator (0 = Healthy/Normal, 1 = Early Disease Symptoms Present). Positives represent ~12.8% of the observations during symptomatic incubation/onset.

---

## 4. Architectural Boundaries
```
project/
├── data/
│   └── poultry_flock_data.csv        # 4,200 flock-day observations
├── ml/
│   ├── data_gen.py                   # Latent disease pressure simulator with realistic sensor noise
│   ├── features.py                   # Data cleaning, physiological bounds, and rolling features
│   ├── train.py                      # GroupKFold CV, Decision Tree training, and model artifacts
│   └── outputs/                      # Saved models, metrics JSON, and evaluation charts
├── app.py                            # Streamlit Poultry Farm Health Monitor application
├── Poultry Farm Health Monitor.html  # Standalone interactive dashboard for executive review
├── docs/
│   └── scope.md                      # Project scope & specifications
├── report/
│   └── project_report.md             # Comprehensive technical and analytical report
└── requirements.txt                  # Python dependencies
```

---

## 5. Decision Support & Triage Workflow
- **HIGH Risk (Risk ≥ 70%):** Immediate veterinary notification, shed isolation, feeder/drinker line inspection, diagnostic swab sampling.
- **WATCH (40% ≤ Risk < 70%):** Flagged for intensive 24-hour observation, environmental sensor verification (ammonia & ventilation checks), re-assessment in next feeding cycle.
- **NORMAL (Risk < 40%):** Baseline bio-security protocol, automated sensor logging.
