import pandas as pd
import joblib
import shap
import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# ================================================================
# Location of model files (saved earlier)
# ================================================================
model_files = {
    "scores_x_agg": "logreg_latefusion_scores_x_agg.joblib",
    "scores_x_temp": "logreg_latefusion_scores_x_temp.joblib"
}

# For reconstructing CVD labels from original EHR data
label_files = {
    "scores_x_agg": "/Users/niaabdu/Desktop/X_agg.csv",
    "scores_x_temp": "/Users/niaabdu/Desktop/X_temp_flat.csv"
}

# Late fusion scores files
scores_files = {
    "scores_x_agg": "/Users/niaabdu/Desktop/scores_x_agg.csv",
    "scores_x_temp": "/Users/niaabdu/Desktop/scores_x_temp.csv"
}

output_dir = "latefusion_logreg_analysis_plots"
os.makedirs(output_dir, exist_ok=True)

# ================================================================
# Plot: Logistic Regression Coefficients
# ================================================================
def plot_coefficients(model, feature_names, data_type):

    coef = model.coef_[0]
    idx = np.argsort(np.abs(coef))[::-1]

    sorted_features = [feature_names[i] for i in idx]
    sorted_values = coef[idx]

    plt.figure(figsize=(10, 8))
    plt.barh(sorted_features[:30][::-1], sorted_values[:30][::-1])
    plt.xlabel("Coefficient Value")
    plt.title(f"Top 30 Logistic Regression Coefficients — {data_type}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/coefficients_{data_type}.png", dpi=300)
    plt.close()


# ================================================================
# Plot: Confusion Matrix
# ================================================================
def plot_conf_matrix(model, X_test, y_test, data_type):

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix — {data_type}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_{data_type}.png", dpi=300)
    plt.close()


# ================================================================
# Plot: SHAP (PermutationExplainer)
# ================================================================
def plot_shap(model, X_train, X_test, data_type):

    explainer = shap.PermutationExplainer(
        model.predict_proba,
        X_train,
        feature_names=X_train.columns
    )

    shap_values = explainer(X_test)

    # SHAP values for class 1
    shap_pos = shap_values.values[:, :, 1]

    # Summary (beeswarm)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_pos, X_test, show=False)
    plt.title(f"SHAP Summary — {data_type}")
    plt.savefig(f"{output_dir}/shap_summary_{data_type}.png", dpi=300)
    plt.close()

    # Bar plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_pos, X_test, plot_type="bar", show=False)
    plt.title(f"Mean |SHAP| — {data_type}")
    plt.savefig(f"{output_dir}/shap_bar_{data_type}.png", dpi=300)
    plt.close()

    # Dependence plots for top 5
    mean_abs = np.abs(shap_pos).mean(axis=0)
    top5 = np.argsort(mean_abs)[::-1][:5]

    for idx in top5:
        feat = X_test.columns[idx]
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feat, shap_pos, X_test, show=False
        )
        plt.title(f"SHAP Dependence — {feat} ({data_type})")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_dependence_{data_type}_{feat}.png", dpi=300)
        plt.close()


# ================================================================
# Plot: ROC Curve (test set)
# ================================================================
def plot_roc_curve(model, X_test, y_test, data_type):

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {data_type}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_{data_type}.png", dpi=300)
    plt.close()


# ================================================================
# MAIN LOOP — Run analysis for each late-fusion model
# ================================================================
for data_type in ["scores_x_agg", "scores_x_temp"]:

    print(f"\n=== Analyzing Late Fusion Model: {data_type} ===")

    # Load saved model
    model = joblib.load(model_files[data_type])

    # Load scores + attach CVD label
    scores_df = pd.read_csv(scores_files[data_type])
    label_df = pd.read_csv(label_files[data_type])[["PATIENT", "CVD"]]
    df = scores_df.merge(label_df, on="PATIENT", how="left")

    df = df.dropna(subset=["CVD"])
    X = df.drop(columns=["PATIENT", "CVD"])
    y = df["CVD"]

    # Reconstruct original train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("→ Plotting coefficients...")
    plot_coefficients(model, X.columns, data_type)

    print("→ Plotting confusion matrix...")
    plot_conf_matrix(model, X_test, y_test, data_type)

    print("→ Plotting SHAP values...")
    plot_shap(model, X_train, X_test, data_type)

    print("→ Plotting ROC curve...")
    plot_roc_curve(model, X_test, y_test, data_type)

    print(f"✓ Completed analysis for {data_type}")

print(f"\nAll analysis plots saved to: {output_dir}/\n")
