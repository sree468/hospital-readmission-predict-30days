# 🏥 Clinical Hospital Readmission Risk Stratification Engine

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)
![XGBoost](https://img.shields.io/badge/ML--Engine-XGBoost-11FFF8.svg)

An enterprise-grade, modular Machine Learning classification pipeline designed to address care gaps, predict 30-day patient hospital readmissions, and provide actionable risk triage workflows for healthcare care coordination teams.

---

## 📋 Problem Statement

Hospital readmissions within 30 days of discharge serve as a critical proxy for care gaps and patient risk, often triggering financial penalties under value-based care programs. Clinical teams require a robust early stratification tool to prioritize complex discharge planning, medication reconciliation, and home monitoring schedules.

This project aggregates seven distinct relational Electronic Health Record (EHR) tables, engineers clinical indicator flags (such as comorbidity loads and polypharmacy metrics), and trains a **Calibrated XGBoost Classifier** to deliver stable operational risk queues (**High**, **Medium**, **Low**) rather than raw, uncalibrated probability scores.

---

## 📂 Project Architecture & File Flow

The workspace moves away from monolithic Jupyter notebooks to adhere to rigorous software engineering separation of concerns:

```text
hospital_readmission_project/
│
├── data/                             # Clinical Relational Databases (CSV)
│   ├── admissions.csv                # Encounter master records and target labels
│   ├── patients.csv                  # Baseline demographics and baseline comorbidities
│   ├── diagnoses.csv                 # Patient ICD code assignments 
│   ├── lab_results.csv               # Lab test tracking records
│   ├── prescriptions.csv             # Medication order history
│   ├── care_transition_plans.csv     # Post-discharge tasks and coordination notes
│   └── post_discharge_vitals.csv     # Remote monitoring vital tracking telemetry
│
├── src/                              # Core Engine Modules
│   ├── __init__.py                   # Package initialization markup
│   ├── data_processor.py             # ID alignment, aggregations, and feature engineering
│   ├── pipeline_builder.py           # Preprocessing pipelines (Imputation + Scaling + OHE)
│   └── model_evaluator.py            # Statistical valuation and triage scoring
│
├── app_streamlit.py                  # Production Streamlit UI Dashboard Server
├── main.py                           # Local model exploration and training diagnostic script
└── requirements.txt                  # Project environment dependencies
