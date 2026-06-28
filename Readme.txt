# 🏥 Clinical Hospital Readmission Risk Stratification Engine

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)
![XGBoost](https://img.shields.io/badge/ML--Engine-XGBoost-11FFF8.svg)

An enterprise-grade, modular Machine Learning classification pipeline designed to address care gaps, predict 30-day patient hospital readmissions, and provide actionable risk triage workflows for healthcare care coordination teams.

---
## 📋 Problem Statement
Hospital readmissions within 30 days of discharge serve as a critical proxy for care gaps and patient risk, often triggering financial penalties under value-based care programs. Clinical teams require a robust early stratification tool to prioritize complex discharge planning, medication reconciliation, and home monitoring schedules.

This project aggregates seven distinct relational Electronic Health Record (EHR) tables, engineers clinical indicator flags (such as comorbidity loads and polypharmacy metrics), and trains a **Calibrated XGBoost Classifier** to deliver stable operational risk queues (High, Medium, Low) rather than raw, uncalibrated probability scores.

---
## 📂 Project Architecture & File Flow
The workspace moves away from monolithic Jupyter notebooks to adhere to rigorous software engineering separation of concerns:

```text
hospital_readmission_project/
│
├── data/                               # Clinical Relational Databases (CSV)
│   ├── admissions.csv                  # Encounter master records and target labels
│   ├── patients.csv                    # Baseline demographics and baseline comorbidities
│   ├── diagnoses.csv                   # Patient ICD code assignments 
│   ├── lab_results.csv                 # Lab test tracking records
│   ├── prescriptions.csv               # Medication order history
│   ├── care_transition_plans.csv      # Post-discharge tasks and coordination notes
│   └── post_discharge_vitals.csv       # Remote monitoring vital tracking telemetry
│
├── src/                                # Core Engine Modules
│   ├── __init__.py                     # Package initialization markup
│   ├── data_processor.py               # ID alignment, aggregations, and feature engineering
│   ├── pipeline_builder.py             # Preprocessing pipelines (Imputation + Scaling + OHE)
│   └── model_evaluator.py              # Statistical valuation and triage scoring
│
├── app_streamlit.py                    # Production Streamlit UI Dashboard Server
├── main.py                             # Local model exploration and training diagnostic script
└── requirements.txt                    # Project environment dependencies
🛠️ Feature Engineering & Data PipelineKey Standarization: Strict regex strip and text casing normalization across patient_id and admission_id to prevent missing-value merge generation bugs.Clinical Aggregations: Longitudinal records (Labs, Diagnoses) are compressed into single encounter-level descriptive attributes.Domain Risk Transformations:Polypharmacy: Evaluated as a binary indicator if a patient has $\ge 5$ active drug orders.High Utilization: Identifies extended lengths of stay greater than 7 days.Comorbidity Score: Tracks aggregate chronic conditions and acute diagnosis codes.📈 Evaluation MetricsModels are evaluated using metrics tailored for clinical utility rather than basic classification accuracy:AUROC: To evaluate the model's overall capability in distinguishing risk profiles across the entire population.AUPRC: To ensure strong model precision on the positive target class (Readmitted), accounting for potential data imbalances.Calibrated Probabilities: Utilizes CalibratedClassifierCV (Sigmoid method) to transform raw outputs into true event likelihoods, allowing for safe operational threshold tiering.🚀 Installation & Local Deployment1. Clone the WorkspaceBashgit clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
2. Configure the Isolated EnvironmentBash# Create environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\activate.ps1

# Activate environment (Linux / macOS / Git Bash)
source .venv/Scripts/activate
3. Install DependenciesBashpip install -r requirements.txt
4. Provide Raw DatasetsEnsure your 7 raw medical files (admissions.csv, patients.csv, etc.) are placed inside the ./data/ folder directory.
5. Launch the Dashboard AppBashstreamlit run app_streamlit.py
Your browser will open up http://localhost:8501 automatically, displaying the interactive clinical triage system dashboard.
🏥 Operational Interventions by Risk Queue
🔴 High Risk Queue (Probability $\ge$ 70%): Triggers immediate, mandatory clinical pharmacist medication reconciliation, home health aide dispatching, and a primary care follow-up visit scheduled within 48 hours.
🟡 Medium Risk Queue (Probability 40% - 69%): Generates automated primary care provider (PCP) discharge notifications and places the patient onto a 5-day telehealth call checking priority list.