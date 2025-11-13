import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score
import matplotlib.pyplot as plt
import optuna

# with hyperparameter optimization through optuna

def load_xgboost_model(x_train, y_train, x_test, y_test, best_params):
    # fit model
    # no early stopping right now bc we're doing cross-validation -> should add it in later though
    model = xgb.XGBClassifier(objective='binary:logistic', **best_params)
    model.fit(x_train, y_train, verbose=False)
    y_pred = model.predict(x_test)

    # calculate and report metrics
    metrics = {}
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred)
    metrics['f1_score'] = f1_score(y_test, y_pred)
    metrics['recall'] = recall_score(y_test, y_pred)
    y_pred_proba = model.predict_proba(x_test)[:, 1] # Probabilities for the positive class (class 1)
    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    metrics['average_precision'] = average_precision_score(y_test, y_pred_proba)
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

def run_hyperparameter_tuning(x_train, y_train):
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    # hyperparameter optimization with 50 trials, done on the 80% training data only
    study.optimize(lambda trial: objective(trial, x_train, y_train), n_trials=50)
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    # run_xgboost with best hyperparameters on the held out test set (20%)
    return best_params

# training all our models!
# all datasets so we can loop through them
files = ['../../data/X_agg.csv',
         '../../data/X_temp_flat.csv',
         '../../data/X_agg_earlyfusion.csv',
         '../../data/X_temp_earlyfusion.csv']

all_metrics = pd.DataFrame(columns=['data_type', 'balanced_accuracy', 'precision', 'f1_score', 'recall', 'roc_auc', 'average_precision'])
for file in files:
    data_type = (file.split('/')[-1]).split('.')[0]
    print(f"\n{file.split('/')[-1]} XGBoost Results:")
    data = pd.read_csv(file) # loading data here
    # split data into features and target
    X = data.drop(columns=['CVD', 'PATIENT'])
    y = data['CVD']

    # train is now 80% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    best_params = run_hyperparameter_tuning(x_train, y_train)
    metrics = load_xgboost_model(x_train, y_train, x_test, y_test, best_params)
    all_metrics.append(list(metrics.values()), ignore_index=True)

all_metrics.to_csv('../results/optuna_xgboost_model_metrics.csv', index=False)