import pandas as pd
import matplotlib.pyplot as plt

def run_rf_performance_plot(all_metrics, title):
    # histogram showing roc_auc, average_precision, f1, accuracy, precision, recall for each model/data type
    metric_names = [col for col in all_metrics.columns if col != "data_type"]
    model_labels = list(all_metrics.index)

    # setup subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    print(all_metrics)

    # plot each metric
    for i, metric in enumerate(metric_names):
        values = all_metrics[metric].values
        # values = [all_metrics[m][metric][0] for m in model_labels]  # metric values
        # errors = [all_metrics[m][metric][1] for m in model_labels]  # std deviations

        ax = axes[i]
        bars = ax.bar(model_labels, values, capsize=5, alpha=0.8)
        # add actual values above bars
        ax.bar_label(bars, fmt='%.2f', padding=3)
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_xticks(range(len(model_labels)))
        ax.set_xticklabels(model_labels, rotation=45, ha='right')

    plt.suptitle(f'Hyperparameter Optimized {title} Performance Across Feature Sets', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'../results/optuna_{title}_performance_comparison.png', dpi=800)

rf_metrics = pd.read_csv('../results/optuna_rf_model_metrics.csv', index_col=0)
print(rf_metrics.index)
run_rf_performance_plot(rf_metrics, title='RF')

xgboost_metrics = pd.read_csv('../results/optuna_xgb_model_metrics.csv', index_col=0)
run_rf_performance_plot(xgboost_metrics, title='XGBoost')