# Poultry Farm Health Monitor: Machine Learning & Predictive Analytics for Early Disease Detection

An intelligent, interpretable decision-support system utilizing a cost-sensitive Decision Tree classifier to detect early, sub-clinical disease symptoms in poultry flocks from farm sensor readings and daily health logs.

---

## 📌 Executive Overview
- **Domain Focus:** Machine Learning & Predictive Analytics in Precision Livestock Farming.
- **Core Challenge:** Detecting disease during early incubation before exponential mortality surges occur, balancing sensitivity with full veterinary interpretability.
- **Model:** Balanced Decision Tree classifier (`criterion='entropy'`, `max_depth=3`, `min_samples_leaf=40`).
- **Validation:** Strict flock-level partitioning (`GroupShuffleSplit` & `GroupKFold`, $k=5$) ensuring zero data leakage across 60 flocks (4,200 flock-days).
- **Interface:** Deployed as an interactive Streamlit farm monitor (`app.py`) and a portable zero-dependency HTML executive dashboard (`Poultry Farm Health Monitor.html`).

---

## 📊 Evaluation Results (Held-Out Flocks: 15 Flocks, 1,050 Days)

| Metric | Decision Tree (Tuned) | Logistic Regression Baseline | Majority Class Baseline |
|---|:---:|:---:|:---:|
| **Recall (Early Disease Sensitivity)** | **0.8284 (82.8%)** | 0.8284 | 0.0000 |
| **Precision** | **0.6307 (63.1%)** | 0.6307 | 0.0000 |
| **F1-Score** | **0.7161 (71.6%)** | 0.7161 | 0.0000 |
| **ROC-AUC** | **0.9128** | 0.9114 | 0.5000 |
| **Accuracy** | **0.9162 (91.6%)** | 0.9162 | 0.8724 |

> **Note on Imbalance:** Positive disease days represent ~12.8% of observations. A naive model that blindly predicts "Healthy" achieves 87.24% accuracy while catching zero sick flocks. Thus, **Recall (0.83)** and **F1 (0.72)** are the primary decision metrics. The Decision Tree matches regularized Logistic Regression while providing transparent, human-auditable logic rules for farm veterinarians.

---

## 📂 Project Directory Structure

```
project/
├── data/
│   └── poultry_flock_data.csv          # 60 flocks x 70 days = 4,200 flock-day records
├── docs/
│   └── scope.md                        # Project scope, specifications & governance
├── ml/
│   ├── data_gen.py                     # Synthetic data generator with sensor faults & noise
│   ├── features.py                     # Cleaning rules & rolling feature engineering
│   ├── train.py                        # GroupKFold CV training & artifact generator
│   └── outputs/                        # Saved model, metrics, and visual artifacts
├── report/
│   └── project_report.md               # Comprehensive technical and scientific report
├── outputs/
│   ├── decision_tree_model.joblib      # Serialized scikit-learn pipeline
│   ├── metrics.json                    # Performance metrics & baseline statistics
│   ├── test_predictions.csv           # Predictions on unseen test flocks
│   ├── confusion_matrix.png            # Confusion matrix visualization
│   ├── feature_importance.png          # Top predictive features
│   ├── roc_curve.png                   # Receiver Operating Characteristic plot
│   ├── precision_recall.png            # Precision-Recall curve
│   └── decision_tree.png               # Visual plot of the learned tree
├── app.py                              # Streamlit Poultry Farm Health Monitor dashboard
├── Poultry Farm Health Monitor.html    # Standalone interactive dashboard (Single-file HTML/JS)
├── data_gen.py                         # Root-level data generator script
├── features.py                         # Shared cleaning & feature engineering module
├── train.py                            # Root-level training script
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
Activate the virtual environment and ensure dependencies are installed:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Generate Dataset (Optional)
The dataset is already generated in `data/poultry_flock_data.csv`. To re-simulate:
```powershell
python data_gen.py
```

### 3. Train & Evaluate the Decision Tree
Trains the model with 5-fold `GroupKFold` cross-validation, runs holdout evaluation, and outputs diagnostic charts:
```powershell
python train.py
```

### 4. Launch the Interactive Monitor
#### Option A: Streamlit Dashboard
```powershell
streamlit run app.py
```
*Features:*
- **Farm Overview:** Real-time HIGH/WATCH/NORMAL summary cards and sortable flock status table.
- **Flock Detail:** Longitudinal temporal trajectory charts for mortality, feed intake, water intake, and respiratory distress.
- **Check a Flock:** Interactive what-if simulator with live probability inference and clinical triage recommendations.
- **Model Report:** Confusion matrix, feature importance ranking, ROC curve, and decision tree architecture.

#### Option B: Standalone Portable Dashboard
Double click or open `Poultry Farm Health Monitor.html` in any web browser. It runs entirely offline without requiring a Python server!

---

## 🔍 Learned Clinical Decision Logic

The Decision Tree extracts intuitive rules:
1. **Primary Gate (`droppings_3d_avg` > 7.87%):**
   - If droppings show elevated moisture/abnormality, the model examines feed intake depression (`feed_change_3d`) and slight mortality creep (`mortality_pct`).
   - If both are present $\rightarrow$ **HIGH Risk (96%–100%)**: Isolate shed, collect diagnostic swabs, notify veterinarian.
2. **Secondary Gate (Droppings Normal $\le$ 7.87%):**
   - If droppings are normal, the model evaluates shed ammonia levels (`ammonia_ppm`) and respiratory score (`cough_3d_avg`).
   - If respiratory scores are elevated $\rightarrow$ **WATCH (42%)**: Verify ventilation fans and inspect for air-sac distress.
   - Otherwise $\rightarrow$ **NORMAL (22%)**: Healthy flock baseline.

---

## 🛡️ Triage Bands & Biosecurity Protocols

- **🔴 HIGH (Risk $\ge$ 70%):** Immediate veterinary notification, shed quarantine, drinker sanitation, diagnostic PCR/ELISA swabbing.
- **🟡 WATCH (40% $\le$ Risk < 70%):** Intensified monitoring (check within 24h), ventilation inspection, drinker nipple flow rate testing.
- **🟢 NORMAL (Risk < 40%):** Routine biosecurity and environmental logging.

---

## 📄 Detailed Technical Report
For in-depth mathematical formulations, data generation physics, subgroup slice analysis across sheds, and clinical implementation roadmaps, refer to:
[Comprehensive Project Report](report/project_report.md)
