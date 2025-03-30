import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
import idx2numpy

# Load MNIST data
X_train = idx2numpy.convert_from_file("archive/train-images.idx3-ubyte")
y_train = idx2numpy.convert_from_file("archive/train-labels.idx1-ubyte")
X_test = idx2numpy.convert_from_file("archive/t10k-images.idx3-ubyte")
y_test = idx2numpy.convert_from_file("archive/t10k-labels.idx1-ubyte")

# Preprocess: flatten and normalize
X_train_flat = X_train.reshape(X_train.shape[0], -1).T / 255.0
X_test_flat = X_test.reshape(X_test.shape[0], -1).T / 255.0

# Shuffle and split 10% validation
m = X_train_flat.shape[1]
indices = np.random.permutation(m)
val_size = int(0.1 * m)
val_indices = indices[:val_size]
train_indices = indices[val_size:]

X_val_split = X_train_flat[:, val_indices]
y_val_split = y_train[val_indices]
X_train_split = X_train_flat[:, train_indices]
y_train_split = y_train[train_indices]


@dataclass
class NeuralNetwork:
    def initialize_parameters(self) -> Dict[str, np.ndarray]:
        np.random.seed(42)
        return {
            "W1": np.random.randn(128, 784) * np.sqrt(2 / 784),
            "b1": np.zeros((128, 1)),
            "W2": np.random.randn(64, 128) * np.sqrt(2 / 128),
            "b2": np.zeros((64, 1)),
            "W3": np.random.randn(10, 64) * np.sqrt(2 / 64),
            "b3": np.zeros((10, 1)),
        }

    def relu(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)

    def softmax(self, Z: np.ndarray) -> np.ndarray:
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def forward_pass(self, X: np.ndarray, parameters: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Tuple]:
        W1, b1 = parameters["W1"], parameters["b1"]
        W2, b2 = parameters["W2"], parameters["b2"]
        W3, b3 = parameters["W3"], parameters["b3"]

        Z1 = W1 @ X + b1
        A1 = self.relu(Z1)
        Z2 = W2 @ A1 + b2
        A2 = self.relu(Z2)
        Z3 = W3 @ A2 + b3
        A3 = self.softmax(Z3)

        return A3, (Z1, A1, Z2, A2, Z3, A3)

    def compute_loss(self, A3: np.ndarray, Y: np.ndarray) -> float:
        m = Y.shape[1]
        return -np.sum(Y * np.log(A3 + 1e-8)) / m

    def one_hot_encode(self, y: np.ndarray, num_classes: int = 10) -> np.ndarray:
        one_hot = np.zeros((num_classes, y.size))
        one_hot[y, np.arange(y.size)] = 1
        return one_hot

    def backward_pass(self, X: np.ndarray, Y: np.ndarray, cache: Tuple, parameters: Dict[str, np.ndarray]) -> Dict[
        str, np.ndarray]:
        W1, W2, W3 = parameters["W1"], parameters["W2"], parameters["W3"]
        Z1, A1, Z2, A2, Z3, A3 = cache
        m = X.shape[1]

        dZ3 = A3 - Y
        dW3 = dZ3 @ A2.T / m
        db3 = np.sum(dZ3, axis=1, keepdims=True) / m

        dA2 = W3.T @ dZ3
        dZ2 = dA2 * (Z2 > 0)
        dW2 = dZ2 @ A1.T / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = W2.T @ dZ2
        dZ1 = dA1 * (Z1 > 0)
        dW1 = dZ1 @ X.T / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}

    def update_parameters(self, parameters: Dict[str, np.ndarray], grads: Dict[str, np.ndarray],
                          learning_rate: float) -> Dict[str, np.ndarray]:
        for key in grads:
            param_key = key[1:]  # remove 'd'
            parameters[param_key] -= learning_rate * grads[key]
        return parameters

    def shuffle_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.random.permutation(X.shape[1])
        return X[:, indices], y[indices]

    def get_batches(self, X: np.ndarray, y: np.ndarray, batch_size: int):
        for i in range(0, X.shape[1], batch_size):
            yield X[:, i:i + batch_size], y[i:i + batch_size]

    def train_one_epoch(self, X_train: np.ndarray, y_train: np.ndarray, parameters: Dict[str, np.ndarray],
                        learning_rate: float, batch_size: int) -> Tuple[float, float, Dict[str, np.ndarray]]:
        X_shuffled, y_shuffled = self.shuffle_data(X_train, y_train)
        total_loss, correct = 0, 0
        batches = X_train.shape[1] // batch_size

        for X_batch, y_batch in self.get_batches(X_shuffled, y_shuffled, batch_size):
            Y_batch = self.one_hot_encode(y_batch)
            A3, cache = self.forward_pass(X_batch, parameters)
            loss = self.compute_loss(A3, Y_batch)
            total_loss += loss

            predictions = np.argmax(A3, axis=0)
            correct += np.sum(predictions == y_batch)

            grads = self.backward_pass(X_batch, Y_batch, cache, parameters)
            parameters = self.update_parameters(parameters, grads, learning_rate)

        avg_loss = total_loss / batches
        accuracy = correct / X_train.shape[1] * 100
        return avg_loss, accuracy, parameters

    def evaluate_model(self, X: np.ndarray, y: np.ndarray, parameters: Dict[str, np.ndarray]) -> Tuple[float, float]:
        A3, _ = self.forward_pass(X, parameters)
        loss = self.compute_loss(A3, self.one_hot_encode(y))
        accuracy = np.mean(np.argmax(A3, axis=0) == y) * 100
        return loss, accuracy

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                    parameters: Dict[str, np.ndarray], epochs: int, batch_size: int, learning_rate: float):
        train_losses, val_losses, val_accs = [], [], []

        for epoch in range(epochs):
            train_loss, train_acc, parameters = self.train_one_epoch(X_train, y_train, parameters, learning_rate,
                                                                     batch_size)
            val_loss, val_acc = self.evaluate_model(X_val, y_val, parameters)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        return parameters, train_losses, val_losses, val_accs


# Main logic
if __name__ == "__main__":
    nn = NeuralNetwork()
    params = nn.initialize_parameters()
    epochs, batch_size, learning_rate = 15, 64, 0.1

    params, train_losses, val_losses, val_accs = nn.train_model(
        X_train_split, y_train_split,
        X_val_split, y_val_split,
        params, epochs, batch_size, learning_rate
    )

    import matplotlib.pyplot as plt

    # Loss plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()
