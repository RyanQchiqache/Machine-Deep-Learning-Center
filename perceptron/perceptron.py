import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000, random_init=False, tol=1e-3):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.random_init = random_init
        self.tol = tol
        self.activation_func = self._unit_step_func
        self.weight = None
        self.bias = None

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        if self.random_init:
            self.weight = np.random.randn(n_features)
            self.bias = np.random.randn(1)
        else:
            self.weight = np.zeros(n_features)
            self.bias = np.zeros(1)

        y_ = np.array([1 if i > 0 else 0 for i in Y])
        last_loss = float('inf')

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weight) + self.bias
                y_predicted = self.activation_func(linear_output)
                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weight += update * x_i
                self.bias += update

            # Calculate loss to check for early stopping
            current_loss = np.mean((y_ - self.predict(X)) ** 2)
            if last_loss - current_loss < self.tol:
                break
            last_loss = current_loss

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        linear_output = np.dot(X, self.weight) + self.bias
        y_pred = self.activation_func(linear_output)
        return y_pred

    @staticmethod
    def precision(y_true, y_pred):
        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positive = np.sum(y_pred == 1)
        return true_positive / predicted_positive if predicted_positive > 0 else 0

    @staticmethod
    def recall(y_true, y_pred):
        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        actual_positive = np.sum(y_true == 1)
        return true_positive / actual_positive if actual_positive > 0 else 0

    @staticmethod
    def f1_score(y_true, y_pred):
        prec = Perceptron.precision(y_true, y_pred)
        rec = Perceptron.recall(y_true, y_pred)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
