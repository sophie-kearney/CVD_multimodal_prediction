import pandas as pd
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import os

output_dir = "xgb_analysis_plots"
os.makedirs(output_dir, exist_ok=True)

data_files = {
    "X_agg": "../../data/X_agg.csv",
    "X_temp_flat": "../../data/X_temp_flat.csv",
    "X_agg_earlyfusion": "../../data/X_agg_earlyfusion.csv",
    "X_temp_earlyfusion": "../../data/X_temp_earlyfusion.csv"
}

def plot_feature_importance(model, feature_names, data_type):
    importance = model.feature_importances_
    idx = importance.argsort()[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(np.array(feature_names)[idx][:40][::-1], importance[idx][:40][::-1])
    plt.title(f"Top 40 Feature Importances — {data_type}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance_{data_type}.png", dpi=300)
    plt.close()

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

def plot_shap(model, X_train, X_test, data_type):

    explainer = shap.TreeExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)

    # SHAP summary plot (beeswarm)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"SHAP Summary — {data_type}")
    plt.savefig(f"{output_dir}/shap_summary_{data_type}.png", dpi=300)
    plt.close()

    # SHAP bar plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f"SHAP Importance — {data_type}")
    plt.savefig(f"{output_dir}/shap_bar_{data_type}.png", dpi=300)
    plt.close()

    # SHAP dependence plots (top 5)
    mean_abs = np.abs(shap_values).mean(axis=0)
    top5 = np.argsort(mean_abs)[::-1][:5]

    for i in top5:
        feat = X_test.columns[i]
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feat, shap_values, X_test, show=False)
        plt.title(f"SHAP Dependence — {feat} ({data_type})")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_dependence_{data_type}_{feat}.png", dpi=300)
        plt.close()

# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
for data_type, file in data_files.items():

    print(f"\n=== Analyzing {data_type} ===")
    df = pd.read_csv(file)
    X = df.drop(columns=['CVD', 'PATIENT'])
    y = df['CVD']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = joblib.load(f"xgb_model_{data_type}.joblib")

    plot_feature_importance(model, X.columns, data_type)
    plot_conf_matrix(model, X_test, y_test, data_type)
    plot_shap(model, X_train, X_test, data_type)

print(f"\nAll XGBoost analysis plots saved to: {output_dir}/")
