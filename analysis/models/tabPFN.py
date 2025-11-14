import pandas as pd
import os
import torch
torch.set_default_device("cpu")
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    f1_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tabpfn import TabPFNClassifier

seed = 42

###
# LOAD DATA
###

datasets = {
    "Xagg": pd.read_csv("data/pluggable/X_agg.csv"),
    "Xtemp": pd.read_csv("data/pluggable/X_temp_flat.csv"),
    "Xagg_EF": pd.read_csv("data/pluggable/X_agg_earlyfusion.csv"),
    "Xtemp_EF": pd.read_csv("data/pluggable/X_temp_earlyfusion.csv"),
}

results = []
for name, df in datasets.items():
    print(f"\n=== Running TabPFN on {name} ===")

    # Split TODO - changed test size from 0.2 to 0.8
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df["CVD"])
    x_train = train_set.drop(columns=["PATIENT", "CVD"])
    y_train = train_set["CVD"]
    x_test = test_set.drop(columns=["PATIENT", "CVD"])
    y_test = test_set["CVD"]

    # Initialize TabPFN (CPU mode)
    model = TabPFNClassifier(device="cuda", ignore_pretraining_limits=True)
    model.fit(x_train.values, y_train.values)

    # Predict
    y_pred_proba = model.predict_proba(x_test.values)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

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
        "Dataset": name,
        "AUROC": auroc,
        "AUPRC": auprc,
        "Balanced_Accuracy": bal_acc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall
    })

    # Plot AUROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"TabPFN ROC Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/tabPFN_{name}_AUROC.png")
    plt.close()

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"TabPFN Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"plots/tabPFN_{name}_CM.png")
    plt.close()

results_df = pd.DataFrame(results)
results_df.to_csv("data/results/tabPFN_metrics.csv", index=False)