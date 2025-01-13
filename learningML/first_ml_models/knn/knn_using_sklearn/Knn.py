import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

#Using Knn algorithm to predict whether a person would have diabetes
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

    # Visualizing the decision boundary in a 2D plot using only two features
    def plot_decision_boundary(self, X, y, feature1, feature2):
        # Only use the two features specified
        X = X[[feature1, feature2]].values

        # Retrain the model using only these two features
        self.model.fit(X, y)

        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
        cmap_bold = ['darkorange', 'c']

        # Create a mesh grid for plotting
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        # Predict the label for each point in the mesh grid
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the contour and training examples
        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(cmap_bold), edgecolor='k', s=20)
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f'Decision Boundary using {feature1} and {feature2}')
        plt.legend(handles=scatter.legend_elements()[0], labels=['Non-diabetic', 'Diabetic'])
        plt.show()


    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        co_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(co_matrix)

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
    predictor = DiabetesPredictor(
        '/learningML/data/diabetes.csv')
    X_train, X_test, y_train, y_test = predictor.preprocess_data()
    predictor.train_model(X_train, y_train)
    predictor.evaluate_model(X_test, y_test)
    predictor.predict([2, 130, 76, 25, 60, 23.1, 0.672, 55])  # Example person's data

    # Plot the decision boundary using 'Glucose' and 'BMI' as features
    predictor.plot_decision_boundary(X_train, y_train, 'Glucose', 'BMI')
