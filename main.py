from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from src.data_processor import load_and_merge_data, split_features_target
from src.pipeline_builder import get_preprocessor
from src.model_evaluator import evaluate_model, generate_daily_report

def main():
    RANDOM_STATE = 42

    # 1. Processing data pipeline
    print("[1/5] Loading and merging data...")
    df = load_and_merge_data(data_dir="./data/")
    # Add this temporarily in main.py right after: df = load_and_merge_data()
    print("--- Row Count Diagnostics ---")
    print(f"Total rows in admissions: {len(df)}")
    print(f"Rows with valid labs: {df['total_tests'].notna().sum()}")
    print(f"Rows with valid vitals: {df['heart_rate_bpm'].notna().sum()}")
    X, y = split_features_target(df)

    # 2. Train Test Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Pipeline generation
    preprocessor, numerical_cols = get_preprocessor(X_train)

    # 4. Modeling & Training
    print("[2/5] Training models (Logistic Regression, Random Forest, XGBoost)...")
    
    # Logistic Regression
    lr_pipeline = Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1000))])
    lr_pipeline.fit(X_train, y_train)
    
    # Random Forest
    rf_pipeline = Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE))])
    rf_pipeline.fit(X_train, y_train)

    # XGBoost
    xgb_pipeline = Pipeline([("preprocessor", preprocessor), ("model", XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=RANDOM_STATE, eval_metric='logloss'))])
    xgb_pipeline.fit(X_train, y_train)

    # 5. Evaluate Pipelines
    print("[3/5] Evaluating performance on test set...")
    evaluate_model("Logistic Regression", y_test, lr_pipeline.predict(X_test), lr_pipeline.predict_proba(X_test)[:, 1])
    evaluate_model("Random Forest", y_test, rf_pipeline.predict(X_test), rf_pipeline.predict_proba(X_test)[:, 1])
    evaluate_model("XGBoost", y_test, xgb_pipeline.predict(X_test), xgb_pipeline.predict_proba(X_test)[:, 1])

    # 6. Probability Calibration & Operational Risk Reporting
    print("[4/5] Training calibrated classifier for stable risk thresholds...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    calibrated_model = CalibratedClassifierCV(
        estimator=XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=RANDOM_STATE),
        method="sigmoid", cv=5
    )
    calibrated_model.fit(X_train_proc, y_train)
    calibrated_probs = calibrated_model.predict_proba(X_test_proc)[:, 1]

    # 7. Generate Deliverable
    print("[5/5] Compiling operational outputs...")
    generate_daily_report(X_test, calibrated_probs)

if __name__ == "__main__":
    main()