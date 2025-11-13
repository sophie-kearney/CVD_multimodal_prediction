import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_validate
import matplotlib.pyplot as plt

def load_xgboost_model(data, n_estimators=500, random_state=42, max_depth=100):

    # split data into features and target
    X = data.drop(columns=['CVD', 'PATIENT'])
    y = data['CVD']

    # fit model
    # no early stopping right now bc we're doing cross-validation -> should add it in later though
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    # model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

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
# all datasets so we can loop through them
files = ['../../data/X_agg.csv',
         '../../data/X_temp_flat.csv',
         '../../data/X_agg_earlyfusion.csv',
         '../../data/X_temp_earlyfusion.csv']

all_metrics = {}
for file in files:
    data_type = (file.split('/')[-1]).split('.')[0]
    print(f"\n{file.split('/')[-1]} XGBoost Results:")
    data = pd.read_csv(file) # loading data here
    metrics = load_xgboost_model(data)
    all_metrics[data_type] = {
        'AUC-ROC': metrics['roc_auc'],
        'AUC-PRC': metrics['average_precision'],
        'F1 Score': metrics['f1'],
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall']}


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
    bars = ax.bar(model_labels, values, yerr=errors, capsize=5, alpha=0.8)
    # add actual values above bars
    ax.bar_label(bars, fmt='%.2f', padding=3)
    ax.set_title(metric)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels, rotation=45, ha='right')

plt.suptitle('XGBoost Performance Across Feature Sets', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("../results/xgboost_performance_comparison.png", dpi=800)
