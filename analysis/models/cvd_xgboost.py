import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, auc, roc_curve, precision_recall_curve

def load_xgboost_model(filepath, target_column = 'CVD', n_estimators=500, random_state=42, max_depth=100):

    # load data
    data = pd.read_csv(filepath)

    # split data into features and target
    X = data.drop(columns=[target_column, 'PATIENT'])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    eval_set = [(X_test, y_test)] # for early stoppping

    # fit model
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', early_stopping_rounds=10)
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    # evaluate model
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    # calculate and print eval metrics
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])
    prc_auc = auc(recall, precision)
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"AUC-PRC: {prc_auc:.4f}")
    print("Model evaluation report:\n", report)

# training all our models!
# all datasets so we can loop through them
files = ['../../data/X_agg.csv',
         '../../data/X_temp_flat.csv',
         '../../data/X_agg_earlyfusion.csv',
         '../../data/X_temp_earlyfusion.csv']

for file in files:
    print(f"\n{file.split('/')[-1]} XGBoost Results:")
    load_xgboost_model(file)