"""Poultry Farm Health Monitor - Streamlit dashboard.

Run:  streamlit run app.py
Shows held-out flocks (never seen during training) with their early-disease risk.
"""
import json

import joblib
import pandas as pd
import streamlit as st

from features import FEATURES, add_features, risk_band

st.set_page_config(page_title="Poultry Farm Health Monitor", page_icon="🐔", layout="wide")

BAND_COLOR = {"HIGH": "#c0392b", "WATCH": "#e0a100", "NORMAL": "#2e8b57"}


@st.cache_resource
def load_model():
    return joblib.load("outputs/decision_tree_model.joblib")


@st.cache_data
def load_data():
    raw = pd.read_csv("data/poultry_flock_data.csv")
    df = add_features(raw)
    test_ids = set(pd.read_csv("outputs/test_predictions.csv")["flock_id"])
    df = df[df["flock_id"].isin(test_ids)].reset_index(drop=True)
    df["risk"] = load_model().predict_proba(df[FEATURES])[:, 1]
    df["status"] = df["risk"].apply(risk_band)
    return df


model, df = load_model(), load_data()
metrics = json.load(open("outputs/metrics.json"))

st.title("🐔 Poultry Farm Health Monitor")
st.caption("Decision Tree early-warning system for flocks showing early disease symptoms. "
           "Data shown is synthetic.")

max_day = int(df["day"].max())
# Open on the day with the most flocks at WATCH/HIGH so the demo shows the alerts.
default_day = int(df[df["status"] != "NORMAL"].groupby("day").size().idxmax())
day = st.sidebar.slider("Monitoring day", 0, max_day, default_day)
st.sidebar.markdown("**Risk bands**  \nHIGH ≥ 70%  \nWATCH 40-69%  \nNORMAL < 40%")

today = df[df["day"] == day].sort_values("risk", ascending=False)

tab1, tab2, tab3, tab4 = st.tabs(["Farm overview", "Flock detail", "Check a flock", "Model report"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flocks monitored", len(today))
    c2.metric("HIGH risk", int((today["status"] == "HIGH").sum()))
    c3.metric("WATCH", int((today["status"] == "WATCH").sum()))
    c4.metric("NORMAL", int((today["status"] == "NORMAL").sum()))

    view = today[["flock_id", "shed", "bird_age_days", "risk", "status", "feed_change_3d",
                  "mortality_3d_avg", "cough_3d_avg", "droppings_3d_avg"]].copy()
    view["risk"] = (view["risk"] * 100).round(0)
    view = view.rename(columns={
        "flock_id": "Flock", "shed": "Shed", "bird_age_days": "Age (d)", "risk": "Risk %",
        "status": "Status", "feed_change_3d": "Feed change vs 3d (%)",
        "mortality_3d_avg": "Mortality 3d avg (%)", "cough_3d_avg": "Cough score 3d",
        "droppings_3d_avg": "Abnormal droppings 3d (%)"})

    def colour(v):
        return f"background-color: {BAND_COLOR[v]}; color: white; font-weight: 600"

    st.dataframe(
        view.style.map(colour, subset=["Status"]).format(
            {"Feed change vs 3d (%)": "{:+.1f}", "Mortality 3d avg (%)": "{:.2f}",
             "Cough score 3d": "{:.1f}", "Abnormal droppings 3d (%)": "{:.1f}", "Risk %": "{:.0f}"}),
        hide_index=True, width="stretch")

with tab2:
    flock = st.selectbox("Flock", sorted(df["flock_id"].unique()),
                         index=list(sorted(df["flock_id"].unique())).index(today.iloc[0]["flock_id"]))
    fd = df[(df["flock_id"] == flock) & (df["day"] <= day)].set_index("day")
    cur = fd.iloc[-1]
    st.markdown(f"### {flock} · {cur['shed']} · age {int(cur['bird_age_days'])} days · "
                f"{int(cur['flock_size']):,} birds")
    st.markdown(f"Current status: **:{'red' if cur['status']=='HIGH' else 'orange' if cur['status']=='WATCH' else 'green'}"
                f"[{cur['status']}]** ({cur['risk']*100:.0f}% risk)")
    a, b = st.columns(2)
    a.caption("Early-disease risk probability")
    a.line_chart(fd["risk"])
    b.caption("Daily mortality (%)")
    b.line_chart(fd["mortality_pct"])
    a.caption("Feed intake (g/bird) and water intake (ml/bird ÷ 2)")
    a.line_chart(pd.DataFrame({"feed": fd["feed_g_per_bird"], "water/2": fd["water_ml_per_bird"] / 2}))
    b.caption("Cough/sneeze score (0-5) and lethargy score (0-5)")
    b.line_chart(fd[["cough_sneeze_score", "lethargy_score"]])

with tab3:
    st.write("Enter today's readings for a flock to get a risk assessment. "
             "Rolling values default to the same as today's reading.")
    with st.form("whatif"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Bird age (days)", 1, 120, 35)
        temp = c1.number_input("House temp (°C)", 10.0, 45.0, 26.0)
        hum = c1.number_input("Humidity (%)", 10.0, 100.0, 62.0)
        nh3 = c1.number_input("Ammonia (ppm)", 0.0, 80.0, 12.0)
        feed = c2.number_input("Feed (g/bird)", 5.0, 250.0, 55.0)
        water = c2.number_input("Water (ml/bird)", 10.0, 600.0, 110.0)
        feed_chg = c2.number_input("Feed change vs last 3 days (%)", -60.0, 60.0, 0.0)
        mort = c2.number_input("Mortality today (%)", 0.0, 20.0, 0.04, step=0.01)
        cough = c3.slider("Cough/sneeze score", 0.0, 5.0, 0.5, 0.1)
        dropp = c3.slider("Abnormal droppings (%)", 0.0, 100.0, 4.0, 1.0)
        leth = c3.slider("Lethargy score", 0.0, 5.0, 0.6, 0.1)
        perf = c3.number_input("Performance index (% of expected)", 40.0, 130.0, 100.0)
        vacc = c3.number_input("Days since vaccination", 0, 120, 10)
        go = st.form_submit_button("Assess flock")
    if go:
        row = pd.DataFrame([{
            "bird_age_days": age, "house_temp_c": temp, "humidity_pct": hum, "ammonia_ppm": nh3,
            "feed_g_per_bird": feed, "water_ml_per_bird": water, "mortality_pct": mort,
            "cough_sneeze_score": cough, "abnormal_droppings_pct": dropp, "lethargy_score": leth,
            "performance_index": perf, "days_since_vaccination": vacc,
            "water_feed_ratio": water / feed, "feed_change_3d": feed_chg,
            "mortality_3d_avg": mort, "cough_3d_avg": cough, "droppings_3d_avg": dropp}])[FEATURES]
        p = float(model.predict_proba(row)[0, 1])
        band = risk_band(p)
        st.markdown(f"## Risk: **{p*100:.0f}%** — "
                    f"<span style='color:{BAND_COLOR[band]}'>{band}</span>", unsafe_allow_html=True)
        if band == "HIGH":
            st.error("Isolate-and-inspect recommended: check birds in this house, review water/feed lines, "
                     "and consider a vet visit.")
        elif band == "WATCH":
            st.warning("Increase monitoring frequency and re-check within 24 hours.")
        else:
            st.success("No early-disease pattern detected.")

with tab4:
    m = metrics
    c = st.columns(5)
    c[0].metric("Recall", f"{m['test_recall']:.2f}")
    c[1].metric("Precision", f"{m['test_precision']:.2f}")
    c[2].metric("F1", f"{m['test_f1']:.2f}")
    c[3].metric("ROC-AUC", f"{m['test_roc_auc']:.2f}")
    c[4].metric("Accuracy", f"{m['test_accuracy']:.2f}")
    st.caption(f"Held-out test: {m['n_test_flocks']} flocks, {m['n_test']} flock-days. "
               f"Best tree: {m['best_params']}. Note: accuracy is close to the majority-class baseline "
               f"({m['baselines']['majority_class_accuracy']:.2f}) because most days are healthy, "
               "so recall and F1 are the primary metrics to evaluate.")
    
    st.markdown("#### Diagnostic Evaluation Plots")
    col1, col2 = st.columns(2)
    col1.image("outputs/confusion_matrix.png", caption="Confusion Matrix on Unseen Flocks")
    col2.image("outputs/feature_importance.png", caption="Feature Importance (Gain)")
    
    col3, col4 = st.columns(2)
    col3.image("outputs/roc_curve.png", caption="Receiver Operating Characteristic (ROC)")
    col4.image("outputs/precision_recall.png", caption="Precision-Recall Curve")
    
    st.markdown("#### Learned Decision Tree Architecture")
    st.image("outputs/decision_tree.png", caption="Top levels of the learned Decision Tree rules")
