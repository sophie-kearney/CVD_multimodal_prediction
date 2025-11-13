import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, auc, roc_curve, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import optuna

# with hyperparameter optimization through optuna

def load_xgboost_model(x_test, y_test, best_params):
    # fit model
    # no early stopping right now bc we're doing cross-validation -> should add it in later though
    model = xgb.XGBClassifier(objective='binary:logistic', **best_params)
    # model.fit(x_test, y_test, verbose=False)

    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']

    # 10-fold cross-validation
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = cross_validate(model, x_test, y_test, cv=kf, scoring=scoring, return_train_score=False)
    print("Cross-validation results:")
    metrics = {}
    for metric_name in scoring:
        metric_mean = results[f'test_{metric_name}'].mean()
        metric_std = results[f'test_{metric_name}'].std()
        metrics[metric_name] = [metric_mean, metric_std]
        print(f"{metric_name}: {metric_mean:.4f} (+/- {metric_std:.4f})")

    return metrics

# # using optuna for hyperparameter tuning
def objective(trial, X, y):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)}

    # 10-fold cross-validation
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    model = xgb.XGBClassifier(**param, use_label_encoder=False) # set use_label_encoder=False for newer versions
    # model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision'] # scoring metrics, in case we want to change what we optimize

    # the cross_validate function takes care of splitting the data for me, because I've specified cv=kf with 10 folds
    results = cross_validate(model, X, y, cv=kf, scoring=scoring, return_train_score=False)
    return results['test_f1'].mean() # using f1 as the metric to optimize

def run_hyperparameter_tuning(data):
    # split data into features and target
    X = data.drop(columns=['CVD', 'PATIENT'])
    y = data['CVD']

    # train is now 80% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # x_test and y_test will be used for final evaluation after hyperparameter tuning

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    # hyperparameter optimization with 50 trials, done on the 80% training data only
    study.optimize(lambda trial: objective(trial, x_train, y_train), n_trials=50)
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    # run_xgboost with best hyperparameters on the held out test set (20%)
    return best_params, x_test, y_test

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
    best_params, x_test, y_test = run_hyperparameter_tuning(data)
    metrics = load_xgboost_model(x_test, y_test, best_params)
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

plt.suptitle('Hyperparameter Optimized XGBoost Performance Across Feature Sets', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("../results/optuna_xgboost_performance_comparison.png", dpi=800)
