import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000, random_init=False, tol=1e-3):
        """
        Initialize the Perceptron classifier.

        Parameters:
        learning_rate (float): The step size for weight updates.
        n_iters (int): Number of iterations over the training data.
        random_init (bool): Flag to use random initialization for weights.
        tol (float): Tolerance for early stopping based on loss improvement.
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.random_init = random_init
        self.tol = tol
        self.activation_func = self._unit_step_func
        self.weight = None
        self.bias = None

    def fit(self, X, Y):
        """
        Train the Perceptron model on the given dataset.

        Parameters:
        X (ndarray): Training data, shape (n_samples, n_features).
        Y (ndarray): Target values, shape (n_samples,).
        """
        n_samples, n_features = X.shape

        # Initialize weights and bias
        if self.random_init:
            self.weight = np.random.randn(n_features)
            self.bias = np.random.randn(1)
        else:
            self.weight = np.zeros(n_features)
            self.bias = np.zeros(1)

        # Convert target values to binary (0 or 1)
        y_ = np.where(Y > 0, 1, 0)
        last_loss = float('inf')

        # Training loop over the number of iterations
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Compute the linear combination of inputs and weights
                linear_output = np.dot(x_i, self.weight) + self.bias
                # Apply the activation function to get the predicted class
                y_predicted = self.activation_func(linear_output)
                # Update weights and bias based on prediction error
                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weight += update * x_i
                self.bias += update

            # Calculate loss for early stopping
            current_loss = np.mean((y_ - self.predict(X)) ** 2)
            if last_loss - current_loss < self.tol:
                break
            last_loss = current_loss

    def _unit_step_func(self, x):
        """
        Unit step activation function.

        Parameters:
        x (ndarray): Linear combination of inputs and weights.

        Returns:
        ndarray: Binary class predictions.
        """
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        X (ndarray): Input data, shape (n_samples, n_features).

        Returns:
        ndarray: Predicted class labels, shape (n_samples,).
        """
        linear_output = np.dot(X, self.weight) + self.bias
        y_pred = self.activation_func(linear_output)
        return y_pred

    @staticmethod
    def precision(y_true, y_pred):
        """
        Calculate the precision metric.

        Parameters:
        y_true (ndarray): True class labels.
        y_pred (ndarray): Predicted class labels.

        Returns:
        float: Precision score.
        """
        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positive = np.sum(y_pred == 1)
        return true_positive / predicted_positive if predicted_positive > 0 else 0

    @staticmethod
    def recall(y_true, y_pred):
        """
        Calculate the recall metric.

        Parameters:
        y_true (ndarray): True class labels.
        y_pred (ndarray): Predicted class labels.

        Returns:
        float: Recall score.
        """
        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        actual_positive = np.sum(y_true == 1)
        return true_positive / actual_positive if actual_positive > 0 else 0

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        Calculate the F1 score.

        Parameters:
        y_true (ndarray): True class labels.
        y_pred (ndarray): Predicted class labels.

        Returns:
        float: F1 score.
        """
        prec = Perceptron.precision(y_true, y_pred)
        rec = Perceptron.recall(y_true, y_pred)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
