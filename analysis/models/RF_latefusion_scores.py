import pandas as pd
import os
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    f1_score, confusion_matrix, roc_curve
)
# from sklearn.ensemble import RandomForestClassifier
import joblib

seed = 42

###
# LOAD DATA
###


data_files = {
    "X_agg": "../../data/X_agg.csv",
    "X_temp_flat": "../../data/X_temp_flat.csv"
}

model_template = "rf_model_{data_type}.joblib"

have_genetic_data = pd.read_csv("../../data/X_temp_earlyfusion.csv")["PATIENT"]


results = []
for data_type, file_path in data_files.items():
    print(f"\n=== Running Random Forest on {data_type} ===")

    df = pd.read_csv(file_path)

    test_set = df[df["PATIENT"].isin(have_genetic_data)].copy()
    train_set = df[~df["PATIENT"].isin(have_genetic_data)].copy()

    x_train = train_set.drop(columns=["PATIENT", "CVD"])
    y_train = train_set["CVD"]
    x_test = test_set.drop(columns=["PATIENT", "CVD"])
    y_test = test_set["CVD"]

    # Load RF model
    model_file = model_template.format(data_type=data_type)
    print(f"→ Loading model {model_file}")
    model = joblib.load(model_file)

    # make rf model with best hyperparameters
    model.fit(x_train, y_train)

    # Predict
    probs = model.predict_proba(x_test.values)
    y_pred_proba = probs[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Save PATIENT and class probabilities
    logits_df = pd.DataFrame({
        "PATIENT": test_set["PATIENT"].values,
        "EHR_CVD_score": probs[:, 1],
        "CVD": test_set["CVD"].values
    })

    # getting PRS scores and adding to logits_df
    prs_scores = pd.read_csv("../genetic_data/prs_scores.csv", usecols=["PATIENT_ID", "PRS_1e-5"])
    prs_scores.rename(columns={"PATIENT_ID": "PATIENT"}, inplace=True)
    logits_df = logits_df.merge(prs_scores, on="PATIENT" ,how="left")

    # os.makedirs("data/results", exist_ok=True)
    logits_df.to_csv(f'../results/logits_{data_type}_RF_lfscores.csv', index=False)

    # Metrics
    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Compute precision and recall without new imports (safe to avoid ZeroDivisionError)
    y_true = y_test.values if hasattr(y_test, "values") else y_test
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    results.append({
        "Dataset": data_type,
        "AUROC": auroc,
        "AUPRC": auprc,
        "Balanced_Accuracy": bal_acc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall
    })

results_df = pd.DataFrame(results)
results_df.to_csv("../results/RF_metrics_lfscores.csv", index=False)