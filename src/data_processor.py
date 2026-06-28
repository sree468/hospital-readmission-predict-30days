import pandas as pd
import numpy as np

def clean_id_column(series):
    """Safely converts ID columns to string, strips whitespace, and makes lowercase."""
    return series.astype(str).str.strip().str.lower()

def load_and_merge_data(data_dir="./data/"):
    # 1. Load Dataframes
    patients = pd.read_csv(f"{data_dir}patients.csv")
    admissions = pd.read_csv(f"{data_dir}admissions.csv")
    diagnoses = pd.read_csv(f"{data_dir}diagnoses.csv")
    labs = pd.read_csv(f"{data_dir}lab_results.csv")
    prescriptions = pd.read_csv(f"{data_dir}prescriptions.csv")
    transition = pd.read_csv(f"{data_dir}care_transition_plans.csv")
    vitals = pd.read_csv(f"{data_dir}post_discharge_vitals.csv")

    # 2. Strict ID Standardizing
    for df_obj in [admissions, diagnoses, labs, transition, vitals]:
        if "admission_id" in df_obj.columns:
            df_obj["admission_id"] = clean_id_column(df_obj["admission_id"])
        if "patient_id" in df_obj.columns:
            df_obj["patient_id"] = clean_id_column(df_obj["patient_id"])
            
    patients["patient_id"] = clean_id_column(patients["patient_id"])
    prescriptions["patient_id"] = clean_id_column(prescriptions["patient_id"])

    # 3. Feature Aggregations
    diag_features = diagnoses.groupby(["patient_id", "admission_id"]).agg(
        num_diagnoses=("icd_code", "count"),
        primary_diagnosis=("primary_flag", "sum")
    ).reset_index()

    lab_features = labs.groupby(["patient_id", "admission_id"]).agg(
        total_tests=("lab_id", "count"),
        abnormal_tests=("abnormal_flag", "sum"),
        avg_lab_value=("result_value", "mean")
    ).reset_index()

    rx_features = prescriptions.groupby("patient_id").agg(
        total_prescriptions=("prescription_id", "count"),
        avg_refills=("refills", "mean")
    ).reset_index()

    transition_features = transition[[
        'patient_id', 'admission_id', 'pcp_notified', 'specialist_referral_sent',
        'meds_at_discharge_count', 'high_risk_medication_flag', 'transportation_arranged',
        'social_work_consult', 'estimated_readmission_risk_score'
    ]]

    vital_features = vitals[[
        'patient_id', 'admission_id', 'heart_rate_bpm', 'systolic_bp', 'diastolic_bp',
        'oxygen_saturation_pct', 'pain_score', 'mobility_score', 'fall_risk_level',
        'medication_reconciliation_done', 'follow_up_scheduled', 'weight_kg'
    ]]

    # 4. Merge Datasets
    df = admissions.merge(patients, on="patient_id", how="left")
    df = df.merge(diag_features, on=["patient_id", "admission_id"], how="left")
    df = df.merge(lab_features, on=["patient_id", "admission_id"], how="left")
    df = df.merge(rx_features, on="patient_id", how="left")
    df = df.merge(transition_features, on=["patient_id", "admission_id"], how="left")
    df = df.merge(vital_features, on=["patient_id", "admission_id"], how="left")

    # 5. Feature Engineering
    df["high_utilization"] = np.where(df["length_of_stay"] > 7, 1, 0)
    df["polypharmacy"] = np.where(df["num_medications"] >= 5, 1, 0)
    df["comorbidity_score"] = df["num_chronic_conditions"].fillna(0) + df["num_diagnoses"].fillna(0)
    df["abnormal_lab_ratio"] = df["abnormal_tests"].fillna(0) / (df["total_tests"].fillna(0) + 1)

    return df

def split_features_target(df, target="readmitted_30d"):
    y = df[target]
    X = df.drop(
        columns=[target, "patient_id", "admission_id", "first_name", "last_name", "phone"],
        errors="ignore"
    )
    return X, y