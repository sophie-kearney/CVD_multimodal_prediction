import pandas as pd

scores = pd.read_csv('prs_scores.csv')
patients = pd.read_csv('/Users/ananyara/Github/CVD_multimodal_prediction/coherent-11-07-2022/csv/patients.csv')
encounters = pd.read_csv('/Users/ananyara/Github/CVD_multimodal_prediction/coherent-11-07-2022/csv/encounters.csv')
case_control_status = pd.read_csv('/Users/ananyara/Github/CVD_multimodal_prediction/data/X_agg.csv')

# need to get patient brithday to calculate age at genotype date (but we need to pull first encounter date from the encounters.csv)
# to calculate age, we will add 4 years to age at first encounter date. to get genotyping age
patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'])
encounters['START'] = pd.to_datetime(encounters['START'])
first_encounter = encounters.sort_values('START').groupby('PATIENT').first().reset_index()
first_encounter = first_encounter[['PATIENT', 'START']].rename(columns={'START': 'FIRST_ENCOUNTER_DATE'})
merged = pd.merge(patients, first_encounter, left_on = 'Id', right_on='PATIENT', how='left')
merged['AGE_AT_FIRST_ENCOUNTER'] = (
    (merged['FIRST_ENCOUNTER_DATE'].dt.tz_localize(None) - merged['BIRTHDATE'])
    .dt.days // 365
)
merged['PRS_AGE'] = merged['AGE_AT_FIRST_ENCOUNTER'] + 4

prs_with_age_sex = pd.merge(scores, merged[['Id', 'PRS_AGE', 'GENDER']], left_on='PATIENT_ID', right_on='Id', how='left')
prs_with_status = pd.merge(prs_with_age_sex, case_control_status[['PATIENT', 'CVD']], left_on='PATIENT_ID', right_on='PATIENT', how='left')
prs_with_status['GENDER'] = prs_with_status['GENDER'].map({'M': 1, 'F': 0}) # one hot encoding for gender

# now evaluate prs performance
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.linear_model import LogisticRegression
prs_with_status = prs_with_status.dropna(subset=['CVD'])

# fit logistic regression model

def fit_evaluate_prs(prs_column):
    X = prs_with_status[[prs_column, 'PRS_AGE', 'GENDER']]
    y = prs_with_status['CVD']
    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    f1 = f1_score(y, y_pred_binary)
    precision = precision_score(y, y_pred_binary)
    recall = recall_score(y, y_pred_binary)
    avg_precision_score = average_precision_score(y, y_pred)
    print(f'{prs_column} AUC: {auc:.4f} F1: {f1:.4f} Precision: {precision:.4f} Recall: {recall:.4f} AUPRC: {avg_precision_score:.4f}')

fit_evaluate_prs('PRS_0.05')
fit_evaluate_prs('PRS_1e-5')
fit_evaluate_prs('PRS_5e-8')