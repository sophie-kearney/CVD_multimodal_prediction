# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
# from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score
# import optuna
# import joblib


# # using optuna for hyperparameter tuning
# def objective(trial, X, y):
#     n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
#     max_depth = trial.suggest_int('max_depth', 5, 150, step=10)

#     # 10-fold cross-validation
#     kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#     model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
#     scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision'] # scoring metrics, in case we want to change what we optimize

#     # the cross_validate function takes care of splitting the data for me, because I've specified cv=kf with 10 folds
#     results = cross_validate(model, X, y, cv=kf, scoring=scoring, return_train_score=False)
#     return results['test_f1'].mean() # using f1 as the metric to optimize

# def run_hyperparameter_tuning(x_train, y_train):
#     study = optuna.create_study(direction='maximize')
#     # hyperparameter optimization with 50 trials, done on the 80% training data only
#     study.optimize(lambda trial: objective(trial, x_train, y_train), n_trials=50)
#     best_params = study.best_params
#     print("Best hyperparameters:", best_params)

#     # run_rf with best hyperparameters on the held out test set (20%)
#     return best_params

# def run_rf(best_params, x_train, y_train, x_test, y_test, data_type, random_state=42):
#     best_n_estimators = best_params['n_estimators']
#     best_max_depth = best_params['max_depth']

#     # make rf model with best hyperparameters
#     model = RandomForestClassifier(n_estimators=best_n_estimators, random_state=random_state, max_depth=best_max_depth)
#     model.fit(x_train, y_train)

#     # calculate metrics and return them (thehere will no mean or standard deviation bc no cross validation)
#     metrics = {}
#     y_pred = model.predict(x_test)
#     metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
#     metrics['f1'] = f1_score(y_test, y_pred)
#     metrics['precision'] = precision_score(y_test, y_pred)
#     metrics['recall'] = recall_score(y_test, y_pred)
#     y_pred_proba = model.predict_proba(x_test)[:, 1] # Probabilities for the positive class (class 1)
#     metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
#     metrics['average_precision'] = average_precision_score(y_test, y_pred_proba)
    
#     return metrics

# # training all our models!
# files = ['../../data/X_agg.csv',
#          '../../data/X_temp_flat.csv',
#          '../../data/X_agg_earlyfusion.csv',
#          '../../data/X_temp_earlyfusion.csv']
    
# all_metrics = pd.DataFrame(columns=['data_type', 'balanced_accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'average_precision'])
# for file in files:
#     data_type = (file.split('/')[-1]).split('.')[0]
#     print(f"\n{file.split('/')[-1]} Random Forest Results:")
#     data = pd.read_csv(file) # loading data here so we can pass to optuna

#     # split data into features and target
#     X = data.drop(columns=['CVD', 'PATIENT'])
#     y = data['CVD']

#     # train is now 80% of the entire data set
#     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#     best_params = run_hyperparameter_tuning(x_train, y_train)
#     print(f"Best params for {data_type}: {best_params}")

#     metrics = run_rf(best_params, x_train, y_train, x_test, y_test, data_type)
#     all_metrics = pd.concat([all_metrics, pd.DataFrame(list(metrics.values()))], ignore_index=True)
    
# all_metrics.to_csv('../results/optuna_rf_model_metrics.csv', index=False)

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score, precision_score, recall_score
)
import optuna

# optuna hyperparameter tuning
def objective(trial, X, y):
    n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
    max_depth = trial.suggest_int('max_depth', 5, 145, step=10)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']

    results = cross_validate(model, X, y, cv=kf, scoring=scoring, return_train_score=False)
    return results['test_f1'].mean()

def run_hyperparameter_tuning(x_train, y_train):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, x_train, y_train), n_trials=50)
    return study.best_params

# train and save best model
def run_rf(best_params, x_train, y_train, x_test, y_test, data_type, random_state=42):
    model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        random_state=random_state
    )
    model.fit(x_train, y_train)

    # save model for later feature importance analysis
    joblib.dump(model, f"rf_model_{data_type}.joblib")

    # calc metrics
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]

    metrics = {
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'average_precision': average_precision_score(y_test, y_pred_proba)
    }

    return metrics

# main training loop
files = [
    '../../data/X_agg.csv',
    '../../data/X_temp_flat.csv',
    '../../data/X_agg_earlyfusion.csv',
    '../../data/X_temp_earlyfusion.csv'
]

all_metrics = pd.DataFrame(columns=[
    'data_type', 'balanced_accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'average_precision'
])

for file in files:
    data_type = file.split('/')[-1].split('.')[0]
    print(f"\nRunning Random Forest on: {data_type}")

    data = pd.read_csv(file)

    X = data.drop(columns=['CVD', 'PATIENT'])
    y = data['CVD']

    # use stratified splitting
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    best_params = run_hyperparameter_tuning(x_train, y_train)
    print(f"Best params for {data_type}: {best_params}")

    metrics = run_rf(best_params, x_train, y_train, x_test, y_test, data_type)

    row = {'data_type': data_type}
    row.update(metrics)

    all_metrics = pd.concat([all_metrics, pd.DataFrame([row])], ignore_index=True)

# save metrics
all_metrics.to_csv('../results/optuna_rf_model_metrics.csv', index=False)
