import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, classification_report

def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"Model Evaluation: {name}")
    print("AUROC:", roc_auc_score(y_true, y_prob))
    print("AUPRC:", average_precision_score(y_true, y_prob))
    print("F1-Score:", f1_score(y_true, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

def assign_risk_tier(p):
    if p >= 0.70:
        return "High"
    elif p >= 0.40:
        return "Medium"
    else:
        return "Low"

def generate_daily_report(X_test, calibrated_probs, output_path="Daily_Risk_Report.csv"):
    risk_df = X_test.copy()
    risk_df["Probability"] = calibrated_probs
    risk_df["Risk_Tier"] = risk_df["Probability"].apply(assign_risk_tier)
    
    report = risk_df[["Probability", "Risk_Tier"]].sort_values(by="Probability", ascending=False)
    report.to_csv(output_path, index=False)
    print(f"\n[INFO] Daily Risk Report generated successfully at: '{output_path}'")
    print(report.head())