"""Train and evaluate the Decision Tree that flags flocks with early disease symptoms.

Run:  python train.py
Outputs go to outputs/ (model, metrics, plots, predictions).
"""
import json
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, accuracy_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from features import FEATURES, add_features

SEED = 42


def train_and_evaluate(data_path="data/poultry_flock_data.csv", output_dir="outputs"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = add_features(pd.read_csv(data_path))
    X, y, groups = df[FEATURES], df["early_disease"], df["flock_id"]

    # Split by flock so the same flock never appears in both train and test (no leakage).
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_tr, X_te, y_tr, y_te = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("tree", DecisionTreeClassifier(class_weight="balanced", random_state=SEED)),
    ])
    grid = {
        "tree__criterion": ["gini", "entropy"],
        "tree__max_depth": [3, 4, 5, 6, 8],
        # Larger leaves keep the rules stable and readable (tiny leaves memorise noise).
        "tree__min_samples_leaf": [20, 40, 80],
    }
    search = GridSearchCV(
        pipe, grid, scoring="f1", cv=GroupKFold(n_splits=5), n_jobs=-1, refit=True
    )
    search.fit(X_tr, y_tr, groups=groups.iloc[train_idx])
    model = search.best_estimator_

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "best_params": {k.replace("tree__", ""): v for k, v in search.best_params_.items()},
        "cv_f1_mean": round(float(search.best_score_), 4),
        "test_accuracy": round(accuracy_score(y_te, pred), 4),
        "test_precision": round(precision_score(y_te, pred), 4),
        "test_recall": round(recall_score(y_te, pred), 4),
        "test_f1": round(f1_score(y_te, pred), 4),
        "test_roc_auc": round(roc_auc_score(y_te, proba), 4),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "n_train_flocks": int(groups.iloc[train_idx].nunique()),
        "n_test_flocks": int(groups.iloc[test_idx].nunique()),
        "test_positive_rate": round(float(y_te.mean()), 4),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
    }
    print(classification_report(y_te, pred, target_names=["normal", "early disease"]))
    print(json.dumps(metrics, indent=2))

    # Baselines for context in the report.
    base = {}
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    base["majority_class_accuracy"] = round(accuracy_score(y_te, dummy.predict(X_te)), 4)
    lr = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                   ("m", LogisticRegression(max_iter=1000, class_weight="balanced"))]).fit(X_tr, y_tr)
    base["logreg_f1"] = round(f1_score(y_te, lr.predict(X_te)), 4)
    base["logreg_recall"] = round(recall_score(y_te, lr.predict(X_te)), 4)
    metrics["baselines"] = base
    print("Baselines:", base)

    # Feature importance
    tree = model.named_steps["tree"]
    imp = pd.Series(tree.feature_importances_, index=FEATURES).sort_values(ascending=False)
    metrics["feature_importance"] = {k: round(float(v), 4) for k, v in imp.items() if v > 0}

    # Save artifacts
    joblib.dump(model, out_dir / "decision_tree_model.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    out = df.iloc[test_idx][["flock_id", "day", "early_disease"]].copy()
    out["risk_probability"] = proba
    out.to_csv(out_dir / "test_predictions.csv", index=False)

    # Plots
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_te, pred, display_labels=["Normal", "Early disease"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (held-out flocks)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    top = imp[imp > 0].head(10)[::-1]
    ax.barh(top.index, top.values, color="#2a6f97")
    ax.set_xlabel("Feature importance (Gini/entropy gain)")
    ax.set_title("What the tree relies on")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4))
    PrecisionRecallDisplay.from_predictions(y_te, proba, ax=ax, curve_kwargs={"color": "#2a6f97"})
    ax.set_title("Precision-recall curve")
    fig.tight_layout()
    fig.savefig(out_dir / "precision_recall.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4))
    RocCurveDisplay.from_predictions(y_te, proba, ax=ax, curve_kwargs={"color": "#2a6f97"})
    ax.set_title("ROC curve (AUC = {:.3f})".format(metrics["test_roc_auc"]))
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(tree, feature_names=FEATURES, class_names=["normal", "early disease"],
              filled=True, rounded=True, max_depth=3, fontsize=8, ax=ax, impurity=False)
    ax.set_title("Learned decision tree")
    fig.tight_layout()
    fig.savefig(out_dir / "decision_tree.png", dpi=140)
    plt.close(fig)
    print(f"Saved model, metrics and plots to {out_dir}/")


if __name__ == "__main__":
    train_and_evaluate()
