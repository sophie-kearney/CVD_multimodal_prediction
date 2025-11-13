import pandas as pd
import matplotlib.pyplot as plt

def run_rf_performance_plot(all_metrics, title, save_path):
    # histogram showing roc_auc, average_precision, f1, accuracy, precision, recall for each model/data type
    metric_names = ['AUC-ROC', 'AUC-PRC', 'F1 Score', 'Accuracy', 'Precision', 'Recall']
    model_labels = list(all_metrics.keys())

    # setup subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    # plot each metric
    for i, metric in enumerate(metric_names):
        values = [all_metrics[m][metric][0] for m in model_labels]  # mean values
        errors = [all_metrics[m][metric][1] for m in model_labels]  # std deviations

        ax = axes[i]
        ax.bar(model_labels, values, yerr=errors, capsize=5, alpha=0.8)
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_xticklabels(model_labels, rotation=45, ha='right')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=800)

rf_metrics = pd.read_csv('../results/rf_all_metrics.csv', index_col=0).to_dict(orient='index')
print(rf_metrics)
run_rf_performance_plot(rf_metrics, title='Random Forest Performance Across Feature Sets', save_path="../results/rf_performance_comparison.png")

# xgboost_metrics = pd.read_csv('../results/xgboost_all_metrics.csv', index_col=0).to_dict(orient='index')
# run_rf_performance_plot(xgboost_metrics, title='XGBoost Performance Across Feature Sets', save_path="../results/xgboost_performance_comparison.png")