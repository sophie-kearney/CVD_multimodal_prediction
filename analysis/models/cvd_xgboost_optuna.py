import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score
import optuna
import numpy as np

# with hyperparameter optimization through optuna
def objective(trial, X, y):
    params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'max_depth': trial.suggest_int('max_depth', 3, 10),
    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    'gamma': trial.suggest_float('gamma', 0.0, 0.5),
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    'tree_method': 'hist'}

    # 10-fold cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    f1_scores = []

    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = xgb.XGBClassifier(**params, early_stopping_rounds=50)

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )

        preds = model.predict(X_valid)
        f1_scores.append(f1_score(y_valid, preds))

    return np.mean(f1_scores)

# --------------------------------------------------------------------
# Train final model with early stopping
# --------------------------------------------------------------------
def train_best_model(x_train, y_train, x_test, y_test, best_params, data_type):

    # Create validation split for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    model = xgb.XGBClassifier(**best_params, early_stopping_rounds=50)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # ---- SAVE MODEL ----
    joblib.dump(model, f"xgb_model_{data_type}.joblib")

    # ----- METRICS -----
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    metrics = {
        'data_type': data_type,
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'average_precision': average_precision_score(y_test, y_proba)
    }

    return metrics


# training all our models!
files = [
    '../../data/X_agg.csv',
    '../../data/X_temp_flat.csv',
    '../../data/X_agg_earlyfusion.csv',
    '../../data/X_temp_earlyfusion.csv'
]

all_metrics = []

for file in files:

    data_type = file.split("/")[-1].split(".")[0]
    print(f"\n=== Training XGBoost on {data_type} ===")

    # loading data here
    df = pd.read_csv(file)

    X = df.drop(columns=['CVD', 'PATIENT'])
    y = df['CVD']

    # stratified split. train is now 80% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    study = optuna.create_study(direction='maximize')
    # hyperparameter optimization with 50 trials, done on the 80% training data only
    study.optimize(lambda trial: objective(trial, x_train, y_train), n_trials=50)

    best_params = study.best_params
    print(f"Best params for {data_type}: {best_params}")

    # run_xgboost with best hyperparameters on the held out test set (20%)
    metrics = train_best_model(x_train, y_train, x_test, y_test, best_params, data_type)
    all_metrics.append(metrics)

# save results
pd.DataFrame(all_metrics).to_csv('../results/optuna_xgb_model_metrics.csv', index=False)