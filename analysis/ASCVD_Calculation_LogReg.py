import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ASCVD_variables_df = pd.read_csv("/Users/niaabdu/Desktop/patient_ASCVD_variables.csv")
x_agg = pd.read_csv("/Users/niaabdu/Desktop/X_agg.csv")
print("X_agg columns", x_agg.columns)

print("df columns:", ASCVD_variables_df.columns)
#Age, Total Cholesterol, HDL-C, Systolic BP are numbers
#Diabetes and Current Smoker are binary (1 for true)

#Sex is M/F and will need to be converted to 1 for male and 0 for female
ASCVD_variables_df["BINARY SEX"] = ASCVD_variables_df["GENDER"].map({"M": 1, "F": 0})
print("Binary sex column", ASCVD_variables_df["BINARY SEX"])
print("Binary sex column numbers", ASCVD_variables_df["BINARY SEX"].value_counts())


#Race is categorical (white, black, asian, other, native) and will need to be 1 for Black and 0 for non-black
print("Race unique inputs: ", ASCVD_variables_df["RACE"].unique())
ASCVD_variables_df["BINARY RACE"] = ASCVD_variables_df["RACE"].map({"white": 0, "black": 1, "asian": 0, "other": 0, "native": 0})
print("Binary race column", ASCVD_variables_df["BINARY RACE"])
print("Binary race column numbers", ASCVD_variables_df["BINARY RACE"].value_counts())

final_ASCVD_values = ASCVD_variables_df[["PATIENT", "AGE", "BINARY SEX", "BINARY RACE", "Total Cholesterol", "HDL-C", "Systolic Blood Pressure", "Diabetes", "Smoker"]].copy()
print("final ASCVD values", final_ASCVD_values)
print("final ASCVD columns", final_ASCVD_values.columns)

#natural log needs to be taken for Age, Total Cholesterol, HDL-C and Systolic BP
final_ASCVD_values["Log Age"] = np.log(final_ASCVD_values["AGE"])
final_ASCVD_values["Log Total Cholesterol"] = np.log(final_ASCVD_values["Total Cholesterol"])
final_ASCVD_values["Log HDL-C"] = np.log(final_ASCVD_values["HDL-C"])
final_ASCVD_values["Log Systolic BP"] = np.log(final_ASCVD_values["Systolic Blood Pressure"])
final_ASCVD_values = final_ASCVD_values.merge(x_agg[["PATIENT", "CVD"]],on="PATIENT",how="left")
print("final_ASCVD_values columns before logistic reg: ", final_ASCVD_values.columns)

X = final_ASCVD_values[["BINARY SEX", "BINARY RACE", "Log Age", "Log Total Cholesterol", "Log HDL-C", "Log Systolic BP", "Diabetes", "Smoker"]]
y = final_ASCVD_values["CVD"]

#fit a logistic regression model to the data
log_reg = LogisticRegression()
log_reg.fit(X, y)

final_ASCVD_values["ASCVD"] = log_reg.predict_proba(X)[:, 1]
print("ASCVD calculations:", final_ASCVD_values["ASCVD"])
print("ASCVD calculations columns", final_ASCVD_values.columns)
# final_ASCVD_values.to_csv("/Users/niaabdu/Desktop/ASCVD_logreg.csv")


#coefficients
coefficients = pd.DataFrame({"Variable": X.columns, "Coefficient": log_reg.coef_[0]})
print("Coefficients", coefficients)
print("Intercept:", log_reg.intercept_[0])
# coefficients.to_csv("/Users/niaabdu/Desktop/ASCVD_logreg_coefficients.csv")

#ROC Curve
y_pred_proba = log_reg.predict_proba(X)[:, 1]

fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
auc = roc_auc_score(y, y_pred_proba)

plt.plot(fpr, tpr, label='Logistic Regression (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for ASCVD Logisitic Regression')
plt.legend(loc='lower right')
plt.show()
