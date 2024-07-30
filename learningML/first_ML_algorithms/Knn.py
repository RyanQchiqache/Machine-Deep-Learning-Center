import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

#Using Knn algorithm to predict wheather a person would have diabetes
class DiabetesPredictor:
    def __init__(self, filename):
        self.dataset = pd.read_csv(filename)
        self.model = None
        self.scaler = None

    def preprocess_data(self):
        # Replace zeros with the mean for specified columns
        zero_not_accept = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                           'DiabetesPedigreeFunction', 'Age', 'Outcome']
        for column in zero_not_accept:
            mean = int(self.dataset[column].mean(skipna=True))
            self.dataset[column] = self.dataset[column].replace(0, mean)

        # Split data into features and target
        X = self.dataset.iloc[:, 0:8]
        y = self.dataset.iloc[:, 8]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Keep DataFrame structure to maintain feature names
        self.scaler = StandardScaler()
        X_train = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns)

        return X_train, X_test, y_train, y_test

    #Training the model using KNeighborsClassifier from sklearn.neighbors
    def train_model(self, X_train, y_train):
        self.model = KNeighborsClassifier(n_neighbors=11, metric='euclidean', p=2)
        self.model.fit(X_train, y_train)

    # Using Matplotlib for evaluation
    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        co_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(co_matrix)

        plt.figure(figsize=(8, 6))
        plt.imshow(co_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix').set_fontsize(20)
        plt.colorbar().set_label('Count')
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Non-diabetic', 'Diabetic'], rotation=45)
        plt.yticks(tick_marks, ['Non-diabetic', 'Diabetic'])
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

        print(classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))

    # Prediction for an example person predictor.predict([2,3,4,5,6,7,7,7 etc])
    def predict(self, person_data):
        person_data_df = pd.DataFrame([person_data], columns=self.dataset.columns[:8])
        person_data_scaled = self.scaler.transform(person_data_df)
        prediction = self.model.predict(person_data_scaled)
        if prediction[0] == 1:
            print("The person is predicted to have diabetes.")
        else:
            print("The person is predicted not to have diabetes.")



if __name__ == '__main__':
    # Usage
    DiabetesPredictor = DiabetesPredictor(
        '/learningML/diabetes.csv')
    X_train, X_test, y_train, y_test = DiabetesPredictor.preprocess_data()
    DiabetesPredictor.train_model(X_train, y_train)
    DiabetesPredictor.evaluate_model(X_test, y_test)
    DiabetesPredictor.predict([2, 130, 76, 25, 60, 23.1, 0.672, 55])  # Example person's data
    p




