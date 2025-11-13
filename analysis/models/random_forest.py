import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, auc, roc_curve, precision_recall_curve

class RandomForest:
    def __init__(self, n_estimators=500, random_state=42, max_depth=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)

    def load_data(self, filepath):
        # Load EHR data from a CSV file
        self.data = pd.read_csv(filepath)
        print("Data loaded successfully.")

    def split_data(self, target_column='CVD'):
        X = self.data.drop(columns=[target_column, 'PATIENT'])
        y = self.data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data split into training and testing sets.")

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully.")

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        report = classification_report(self.y_test, y_pred)
        # calculate and print AUC-ROC, AUC-PRC, etc.
        fpr, tpr, _ = roc_curve(self.y_test, self.model.predict_proba(self.X_test)[:,1])
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(self.y_test, self.model.predict_proba(self.X_test)[:,1])
        prc_auc = auc(recall, precision)
        print(f"AUC-ROC: {roc_auc:.4f}")
        print(f"AUC-PRC: {prc_auc:.4f}")
        print("Model evaluation report:\n", report)
    
    def split_train_evaluate(self, target_column='CVD'):
        self.split_data(target_column)
        self.train_model()
        self.evaluate_model()

# Example usage:
ehr_rf = RandomForest()

# X_agg.csv random forest
print("X_agg.csv Random Forest Results:")
ehr_rf.load_data('../../data/X_agg.csv')
ehr_rf.split_train_evaluate()

# X_temp_flat.csv random forest
print("\nX_temp_flat.csv Random Forest Results:")
ehr_rf.load_data('../../data/X_temp_flat.csv')
ehr_rf.split_train_evaluate()

# early fusion: X_agg_earlyfusion.csv random forest
print("\nX_agg_earlyfusion.csv Random Forest Results:")
ehr_rf.load_data('../../data/X_agg_earlyfusion.csv')
ehr_rf.split_train_evaluate()

# early fusion: X_temp_earlyfusion.csv random forest
print("\nX_temp_earlyfusion.csv Random Forest Results:")
ehr_rf.load_data('../../data/X_temp_earlyfusion.csv')
ehr_rf.split_train_evaluate()