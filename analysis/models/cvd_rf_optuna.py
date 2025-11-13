import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score
import matplotlib.pyplot as plt
import optuna

# using optuna for hyperparameter tuning
def objective(trial, X, y):
    n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
    max_depth = trial.suggest_int('max_depth', 5, 150, step=10)

    # 10-fold cross-validation
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision'] # scoring metrics, in case we want to change what we optimize

    # the cross_validate function takes care of splitting the data for me, because I've specified cv=kf with 10 folds
    results = cross_validate(model, X, y, cv=kf, scoring=scoring, return_train_score=False)
    return results['test_f1'].mean() # using f1 as the metric to optimize

def run_hyperparameter_tuning(x_train, y_train):
    study = optuna.create_study(direction='maximize')
    # hyperparameter optimization with 50 trials, done on the 80% training data only
    study.optimize(lambda trial: objective(trial, x_train, y_train), n_trials=50)
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    # run_rf with best hyperparameters on the held out test set (20%)
    return best_params

def run_rf(best_params, x_train, y_train, x_test, y_test, random_state=42):
    best_n_estimators = best_params['n_estimators']
    best_max_depth = best_params['max_depth']

    # make rf model with best hyperparameters
    model = RandomForestClassifier(n_estimators=best_n_estimators, random_state=random_state, max_depth=best_max_depth)
    model.fit(x_train, y_train)

    # calculate metrics and return them (thehere will no mean or standard deviation bc no cross validation)
    metrics = {}
    y_pred = model.predict(x_test)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
    metrics['f1'] = f1_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred)
    metrics['recall'] = recall_score(y_test, y_pred)
    y_pred_proba = model.predict_proba(x_test)[:, 1] # Probabilities for the positive class (class 1)
    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    metrics['average_precision'] = average_precision_score(y_test, y_pred_proba)
    
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
    data = pd.read_csv(file) # loading data here so we can pass to optuna

    # split data into features and target
    X = data.drop(columns=['CVD', 'PATIENT'])
    y = data['CVD']

    # train is now 80% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    best_params = run_hyperparameter_tuning(x_train, y_train)
    print(f"Best params for {data_type}: {best_params}")

    metrics = run_rf(best_params, x_train, y_train, x_test, y_test)
    all_metrics[data_type] = {
        'AUC-ROC': metrics['roc_auc'],
        'AUC-PRC': metrics['average_precision'],
        'F1 Score': metrics['f1'],
        'Balanced Accuracy': metrics['balanced_accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall']}
    
# histogram showing roc_auc, average_precision, f1, accuracy, precision, recall for each model/data type
metric_names = ['AUC-ROC', 'AUC-PRC', 'F1 Score', 'Balanced Accuracy', 'Precision', 'Recall']
model_labels = list(all_metrics.keys())

# setup subplot grid
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

# plot each metric
for i, metric in enumerate(metric_names):
    values = [all_metrics[m][metric][0] for m in model_labels]  # metric values
    # errors = [all_metrics[m][metric][1] for m in model_labels]  # std deviations

    ax = axes[i]
    ax.bar(model_labels, values, capsize=5, alpha=0.8)
    ax.set_title(metric)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xticklabels(model_labels, rotation=45, ha='right')

plt.suptitle('Hyperparameter Optimized RF Performance Across Feature Sets', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("../results/optuna_rf_performance_comparison.png", dpi=800)

