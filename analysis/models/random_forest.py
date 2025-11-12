import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class RandomForest:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def load_data(self, filepath):
        # Load EHR data from a CSV file
        self.data = pd.read_csv(filepath)
        print("Data loaded successfully.")

    # def preprocess_data(self):
    #     # Example preprocessing: handle missing values and encode categorical variables
    #     self.data.fillna(self.data.mean(), inplace=True)
    #     self.data = pd.get_dummies(self.data, drop_first=True)
    #     print("Data preprocessed successfully.")

    def split_data(self, target_column):
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data split into training and testing sets.")

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully.")

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        report = classification_report(self.y_test, y_pred)
        print("Model evaluation report:\n", report)

# Example usage:
ehr_rf = RandomForest()
ehr_rf.load_data('path_to_ehr_data.csv')
ehr_rf.preprocess_data()
ehr_rf.split_data(target_column='target')
ehr_rf.train_model()
ehr_rf.evaluate_model()
