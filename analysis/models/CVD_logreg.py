import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, auc, roc_curve, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

def log_reg_multirun(filepath, target_column = 'CVD'):
    # load data
    print("logistic regression for:", filepath)
    data = pd.read_csv(filepath)

    # split data into features and target
    X = data.drop(columns=[target_column, 'PATIENT'])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_accuracies = []
    cv_aucs = []
    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        log_reg = LogisticRegression(penalty="l2", C=1, solver="liblinear", max_iter=1000)
        log_reg.fit(X_train_fold, y_train_fold)

        # Compute accuracy for this CV fold
        acc = log_reg.score(X_val_fold, y_val_fold)
        cv_accuracies.append(acc)

        # Compute AUC on the validation fold
        val_pred_proba = log_reg.predict_proba(X_val_fold)[:, 1]
        fpr_val, tpr_val, _ = roc_curve(y_val_fold, val_pred_proba)
        fold_auc = auc(fpr_val, tpr_val)
        cv_aucs.append(fold_auc)

        print(f"Fold {fold} -- Accuracy: {acc:.3f}, AUC: {fold_auc:.3f}")

    # ---- AFTER CROSS-VALIDATION ----
    print("\nAverage CV Accuracy:", np.mean(cv_accuracies))
    print("Average CV AUC:", np.mean(cv_aucs))

    # ---- FINAL MODEL (train on full training set) ----
    final_model = LogisticRegression(penalty="l2", C=1, solver="liblinear", max_iter=1000)
    final_model.fit(X_train, y_train)

    # ---- TEST SET PERFORMANCE ----
    test_pred = final_model.predict(X_test)
    test_pred_proba = final_model.predict_proba(X_test)[:, 1]

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_pred))

    fpr, tpr, _ = roc_curve(y_test, test_pred_proba)
    test_auc = auc(fpr, tpr)
    print("Test Set AUC:", test_auc)

    # ---- ROC Curve Plot ----
    short_name = filepath.split("/")[-1]
    plt.plot(fpr, tpr, label=f'LogReg (AUC = {test_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {short_name}')
    # Add hyperparameters inside the plot
    hyperparams_text = (
        f"penalty = l2\n"
        f"C = 1\n"
        f"solver = liblinear\n"
        f"max_iter = 1000"
    )

    plt.text(
        0.60, 0.25, hyperparams_text,
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7),
        transform=plt.gca().transAxes
    )
    plt.legend()
    plt.show()



# plug and chug datasets
files = ["/Users/niaabdu/Desktop/X_agg.csv",
         "/Users/niaabdu/Desktop/X_temp_flat.csv",
         "/Users/niaabdu/Desktop/X_agg_earlyfusion.csv",
         "/Users/niaabdu/Desktop/X_temp_earlyfusion.csv"]

for file in files:
    log_reg_multirun(file)
