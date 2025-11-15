import pandas as pd
import joblib
import shap
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import numpy as np

data_files = {
    "X_agg": "../../data/X_agg.csv",
    "X_temp_flat": "../../data/X_temp_flat.csv",
    "X_agg_earlyfusion": "../../data/X_agg_earlyfusion.csv",
    "X_temp_earlyfusion": "../../data/X_temp_earlyfusion.csv"
}

model_template = "rf_model_{data_type}.joblib"

output_dir = "model_analysis_plots"
os.makedirs(output_dir, exist_ok=True)

def plot_feature_importance(model, feature_names, data_type):
    importances = model.feature_importances_
    idx = importances.argsort()[::-1]  # descending sort

    sorted_features = [feature_names[i] for i in idx]
    sorted_values = importances[idx]

    plt.figure(figsize=(10, 8))
    plt.barh(sorted_features[:30][::-1], sorted_values[:30][::-1])  # top 30 features
    plt.xlabel("Feature Importance")
    plt.title(f"Top 30 Random Forest Feature Importances — {data_type}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance_{data_type}.png", dpi=300)
    plt.close()


def plot_conf_matrix(model, X_test, y_test, data_type):
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(figsize=(6, 6))
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix — {data_type}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_{data_type}.png", dpi=300)
    plt.close()

def compute_and_plot_shap(model, X_train, X_test, data_type):
    X_test = X_test[model.feature_names_in_].copy()

    explainer = shap.PermutationExplainer(
        model.predict_proba,
        X_test,
        feature_names=X_test.columns,
        max_evals=2 * X_test.shape[1] + 1
    )

    shap_values = explainer(X_test)

    # SHAP values for positive class
    shap_pos = shap_values.values[:, :, 1]   # shape: (samples, features)

    # SHAP Summary Beewswarm
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_pos, X_test, show=False)
    plt.title(f"SHAP Summary Plot — {data_type}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary_{data_type}.png", dpi=300)
    plt.close()

    # SHAP Feature Importance (mean |SHAP|)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_pos, X_test, plot_type="bar", show=False)
    plt.title(f"Mean |SHAP| Values — {data_type}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_bar_{data_type}.png", dpi=300)
    plt.close()

    # SHAP Dependence Plots (Top 5)
    mean_abs_shap = np.abs(shap_pos).mean(axis=0)
    top_features = X_test.columns[np.argsort(mean_abs_shap)[::-1][:5]]

    for feat in top_features:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feat,
            shap_pos,
            X_test,
            show=False
        )
        plt.title(f"SHAP Dependence — {feat} ({data_type})")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_dependence_{data_type}_{feat}.png", dpi=300)
        plt.close()


# loop over data
for data_type, file_path in data_files.items():
    print(f"\n=== Analyzing: {data_type} ===")

    # Load data
    df = pd.read_csv(file_path)
    X = df.drop(columns=["CVD", "PATIENT"])
    y = df["CVD"]

    # Load model
    model_file = model_template.format(data_type=data_type)
    print(f"→ Loading model {model_file}")
    model = joblib.load(model_file)

    X = X[model.feature_names_in_]

    # Use same split as training!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # feature importance plot
    print("→ Plotting feature importance...")
    plot_feature_importance(model, X.columns, data_type)

    # confusion matrix
    print("plotting confusion matrix...")
    plot_conf_matrix(model, X_test, y_test, data_type)

    # 3. SHAP values
    print("computing SHAP values...")
    compute_and_plot_shap(model, X_train, X_test, data_type)

    print(f"completed {data_type}\n")

print(f"all results saved in: {output_dir}/")
