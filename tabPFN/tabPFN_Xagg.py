import pandas as pd
import tabpfn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import os
import torch
torch.set_default_device("cpu")

seed = 36

###
# LOAD DATA
###

Xagg = pd.read_csv("data/X_agg.csv")

test_set, train_set = train_test_split(Xagg, test_size=0.2, random_state=seed)

x_test = test_set.drop(columns=["CVD", "PATIENT"])
y_test = test_set["CVD"]
x_train = train_set.drop(columns=["PATIENT","CVD"])
y_train = train_set["CVD"]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

###
# LOAD MODEL
###

HF_TOKEN = os.getenv("HF_TOKEN")
model = tabpfn.TabPFNClassifier(n_preprocessing_jobs=1,device="cpu")

###
# RUN MODEL
###

# Fit and predict
model.fit(x_train.values, y_train.values)
y_pred_proba = model.predict_proba(x_test.values)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

# Evaluate
print("AUROC:", roc_auc_score(y_test, y_pred_proba))
print("F1:", f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))