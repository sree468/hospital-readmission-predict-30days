import os
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# Import your modular logic components
from src.data_processor import load_and_merge_data, split_features_target
from src.pipeline_builder import get_preprocessor
from src.model_evaluator import assign_risk_tier

# Page layout configuration
st.set_page_config(page_title="Readmission Triage Dashboard", page_icon="🏥", layout="wide")

# Use st.cache_resource so the model trains ONCE when starting up, making page refreshes instant
@st.cache_resource
def load_and_train_pipeline():
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    RANDOM_STATE = 42
    
    # 1. Load data
    data_path = os.path.join(BASE_DIR, "data", "")
    df = load_and_merge_data(data_dir=data_path)
    X, y = split_features_target(df)
    
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    
    # 2. Preprocess
    preprocessor, _ = get_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # 3. Model Calibration
    calibrated_model = CalibratedClassifierCV(
        estimator=XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=RANDOM_STATE),
        method="sigmoid", cv=5
    )
    calibrated_model.fit(X_train_proc, y_train)
    
    # 4. Generate prediction dataframe
    report_df = X_test.copy()
    report_df["Probability"] = calibrated_model.predict_proba(X_test_proc)[:, 1]
    report_df["Risk_Tier"] = report_df["Probability"].apply(assign_risk_tier)
    
    return report_df.sort_values(by="Probability", ascending=False)

# --- USER INTERFACE APP LAYOUT ---
st.title("🏥 Clinical Readmission Risk Stratification")
st.caption("Live machine learning triage metrics filtered for hospital care managers and discharge planners.")

# Side status marker indicator panel
st.sidebar.header("System Status")
st.sidebar.success("XGBoost Calibrated Engine Live")

# Spinner notification context while loading/training the dataset initially
with st.spinner("Processing medical data records and updating predictive queues..."):
    master_predictions = load_and_train_pipeline()

# 1. Metrics Cards Summary Layout Row
high_count = len(master_predictions[master_predictions["Risk_Tier"] == "High"])
med_count = len(master_predictions[master_predictions["Risk_Tier"] == "Medium"])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="🔴 High Risk Patients", value=high_count)
with col2:
    st.metric(label="🟡 Medium Risk Patients", value=med_count)
with col3:
    st.metric(label="📊 Total Evaluation Records", value=len(master_predictions))

st.write("---")

# 2. Dynamic Triage Filter
st.subheader("Patient Management Discharge Queue")
tier_filter = st.multiselect(
    "Filter Queue by Operational Status:", 
    options=["High", "Medium", "Low"], 
    default=["High", "Medium"]
)

# Render processed table output matching selections
filtered_df = master_predictions[master_predictions["Risk_Tier"].isin(tier_filter)].head(100)

# Build custom display table framework columns
display_df = pd.DataFrame({
    "Record Identifier": [f"Admission Reference #{idx}" for idx in filtered_df.index],
    "Readmission Probability": filtered_df["Probability"].map(lambda p: f"{p*100:.2f}%"),
    "Triage Status Priority": filtered_df["Risk_Tier"]
})

st.dataframe(display_df, use_container_width=True, hide_index=True)