import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
import matplotlib.pyplot as plt

# use optuna for hyperparameter tuning

def run_rf(filepath, target_column='CVD', n_estimators=500, random_state=42, max_depth=100):
    # load data
    data = pd.read_csv(filepath)

    # split data into features and target
    X = data.drop(columns=[target_column, 'PATIENT'])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # fit model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)
    # model.fit(X_train, y_train)

    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']

    # 10-fold cross-validation
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = cross_validate(model, X, y, cv=kf, scoring=scoring, return_train_score=False)
    print("Cross-validation results:")
    metrics = {}
    for metric_name in scoring:
        metric_mean = results[f'test_{metric_name}'].mean()
        metric_std = results[f'test_{metric_name}'].std()
        metrics[metric_name] = [metric_mean, metric_std]
        print(f"{metric_name}: {metric_mean:.4f} (+/- {metric_std:.4f})")

    return metrics

# training all our models!
files = ['../../data/X_agg.csv',
         '../../data/X_temp_flat.csv',
         '../../data/X_agg_earlyfusion.csv',
         '../../data/X_temp_earlyfusion.csv']

all_metrics = {}
for file in files:
    data_type = (file.split('/')[-1]).split('.')[0]
    print(f"\n{file.split('/')[-1]} Random Forest Results:")
    metrics = run_rf(file)
    all_metrics[data_type] = {
        'AUC-ROC': metrics['roc_auc'],
        'AUC-PRC': metrics['average_precision'],
        'F1 Score': metrics['f1'],
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall']}
    
# # save all_metrics to csv
# metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
# metrics_df = metrics_df.map(lambda x: f"{x[0]:.4f},{x[1]:.4f}")
# metrics_df.to_csv('../results/rf_all_metrics.csv')

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

plt.suptitle('Random Forest Performance Across Feature Sets', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("../results/rf_performance_comparison.png", dpi=800)
