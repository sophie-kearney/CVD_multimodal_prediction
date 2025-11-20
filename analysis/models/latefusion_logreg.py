import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, classification_report,
    balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ================================================================
# Helper: tune hyperparameters using Optuna + 10-fold CV (F1)
# ================================================================
def tune_hyperparams(X_train, y_train):

    def objective(trial):
        # search space for C only (L2, liblinear fixed)
        C = trial.suggest_float("C", 1e-4, 10.0, log=True)

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        f1_scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = LogisticRegression(
                penalty="l2",
                C=C,
                solver="liblinear",
                max_iter=2000
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            f1_scores.append(f1_score(y_val, preds))

        return np.mean(f1_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("\nBest Hyperparameters:", study.best_params)
    return study.best_params


# ================================================================
# Main function for logistic regression late fusion on one dataset
# ================================================================
def log_reg_latefusion_run(scores_filepath,
                           label_source_filepath,
                           data_name,
                           target_column="CVD"):
    """
    scores_filepath: path to scores_x_agg.csv or scores_x_temp.csv
    label_source_filepath: path to X_agg.csv or X_temp_flat.csv
    data_name: short tag like 'scores_x_agg' or 'scores_x_temp'
    """

    print("\n=====================================================")
    print("Late Fusion Logistic Regression for:", data_name)
    print("=====================================================\n")

    # ------------------------------------------------------------
    # Step 0 — Load scores and add CVD labels from the original X files
    # ------------------------------------------------------------
    scores_df = pd.read_csv(scores_filepath)
    label_df = pd.read_csv(label_source_filepath)[["PATIENT", target_column]]

    # Merge CVD label into scores table
    df = scores_df.merge(label_df, on="PATIENT", how="left")

    # Drop any missing labels (should not happen ideally)
    df = df.dropna(subset=[target_column])

    # Features = all non-target, non-PATIENT columns
    X = df.drop(columns=[target_column, "PATIENT"])
    y = df[target_column].astype(int)

    # ------------------------------------------------------------
    # Step 1 — 80/20 train-test split (test untouched by tuning)
    # ------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ------------------------------------------------------------
    # Step 2 — Hyperparameter tuning with Optuna
    # ------------------------------------------------------------
    best_params = tune_hyperparams(X_train, y_train)
    C_best = best_params["C"]

    # ------------------------------------------------------------
    # Step 3 — Final model training
    # ------------------------------------------------------------
    final_model = LogisticRegression(
        penalty="l2",
        C=C_best,
        solver="liblinear",
        max_iter=2000
    )
    final_model.fit(X_train, y_train)

    # Save the trained model for later analysis
    model_filename = f"logreg_latefusion_{data_name}.joblib"
    joblib.dump(final_model, model_filename)
    print(f"\nSaved model to: {model_filename}")

    # ------------------------------------------------------------
    # Step 4 — Evaluate on the held-out test set
    # ------------------------------------------------------------
    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auroc = auc(fpr, tpr)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    auprc = auc(recall_curve, precision_curve)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print("\n---- TEST SET METRICS ----")
    print(f"AUROC:               {auroc:.4f}")
    print(f"AUPRC:               {auprc:.4f}")
    print(f"Balanced Accuracy:   {bal_acc:.4f}")
    print(f"F1 Score:            {f1:.4f}")
    print(f"Precision:           {prec:.4f}")
    print(f"Recall:              {rec:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ------------------------------------------------------------
    # Step 5 — ROC Curve (PNG)
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auroc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")

    hyper_text = f"penalty = l2\nC = {C_best:.4f}\nsolver = liblinear"
    plt.text(0.60, 0.25, hyper_text,
             fontsize=10,
             bbox=dict(facecolor='white', alpha=0.7),
             transform=plt.gca().transAxes)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {data_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ROC_latefusion_{data_name}.png", dpi=300)
    plt.close()
    print(f"Saved ROC curve: ROC_latefusion_{data_name}.png")

    # ------------------------------------------------------------
    # Step 6 — Confusion Matrix (PNG)
    # ------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(figsize=(6, 6))
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix — {data_name}")
    plt.tight_layout()
    plt.savefig(f"CM_latefusion_{data_name}.png", dpi=300)
    plt.close()
    print(f"Saved confusion matrix: CM_latefusion_{data_name}.png")

    # ------------------------------------------------------------
    # Step 7 — Return metrics for master CSV
    # ------------------------------------------------------------
    return {
        "dataset": data_name,
        "AUROC": auroc,
        "AUPRC": auprc,
        "BalancedAccuracy": bal_acc,
        "F1": f1,
        "Precision": prec,
        "Recall": rec,
        "C_best": C_best
    }


# ================================================================
# Run late fusion logistic regression on both scores files
# ================================================================
scores_files = {
    "scores_x_agg": {
        "scores_path": "/Users/niaabdu/Desktop/scores_x_agg.csv",
        "label_source": "/Users/niaabdu/Desktop/X_agg.csv"
    },
    "scores_x_temp": {
        "scores_path": "/Users/niaabdu/Desktop/scores_x_temp.csv",
        "label_source": "/Users/niaabdu/Desktop/X_temp_flat.csv"
    }
}

all_results = []

for name, paths in scores_files.items():
    metrics = log_reg_latefusion_run(
        scores_filepath=paths["scores_path"],
        label_source_filepath=paths["label_source"],
        data_name=name,
        target_column="CVD"
    )
    all_results.append(metrics)

results_df = pd.DataFrame(all_results)
results_df.to_csv("logreg_latefusion_results_optuna.csv", index=False)

print("\nAll late-fusion logistic regression results saved to: logreg_latefusion_results_optuna.csv\n")
