import shutil
print(hasattr(shutil, 'rmtree'))

"""
MaxEnt (Logistic Regression)
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42, max_iter=500, penalty='l1')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

########################################################################################################################

"""
SVM (Support Vector Machine)
"""
from sklearn.svm import SVC

# Load dataset
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

########################################################################################################################

"""
Word Embeddings using Gensim
"""
from gensim.models import Word2Vec, FastText

# Example corpus
sentences = [
    ['human', 'interface', 'computer'],
    ['survey', 'user', 'computer', 'system', 'response', 'human', 'interface'],
    ['eps', 'user', 'interface', 'system']
]

# Train Word2Vec model
word2vec_model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=0)  # CBOW
word2vec_model.save("word2vec.model")

# Load Word2Vec model
word2vec_model = Word2Vec.load("word2vec.model")

# Train FastText model
fasttext_model = FastText(sentences, vector_size=50, window=3, min_count=1, sg=0)  # CBOW
fasttext_model.save("fasttext.model")

# Load FastText model
fasttext_model = FastText.load("fasttext.model")

print(word2vec_model)

########################################################################################################################

"""
SVD for Matrix Factorization
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Example matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Apply SVD
svd = TruncatedSVD(n_components=2)
transformed_matrix = svd.fit_transform(matrix)

print("Transformed Matrix:")
print(transformed_matrix)

########################################################################################################################

"""
Gradient Descent
"""
import numpy as np

# Simple Gradient Descent Example
def gradient_descent(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)
    for _ in range(epochs):
        gradient = (1/m) * X.T.dot(X.dot(theta) - y)
        theta -= lr * gradient
    return theta

# Example data
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.array([6, 8, 9, 11])

# Run gradient descent
theta = gradient_descent(X, y)
print("Theta:", theta)

########################################################################################################################

"""
Word2Vec Skip-Gram (Implemented via Gensim)
"""
from gensim.models import Word2Vec

# Example corpus
sentences = [['human', 'interface', 'computer'],
             ['survey', 'user', 'computer', 'system', 'response', 'human', 'interface'],
             ['eps', 'user', 'interface', 'system']]

# Train Word2Vec Skip-Gram model
skipgram_model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1)  # sg=1 for Skip-Gram
skipgram_model.save("skipgram.model")

# Load Word2Vec Skip-Gram model
skipgram_model = Word2Vec.load("skipgram.model")

########################################################################################################################

"""
NN : Neural Network
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
import os
import ssl
import urllib

# Handling SSL certificate verification error
def download_mnist_data():
    try:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        return (X_train, y_train), (X_test, y_test)
    except Exception as e:
        print(f"Failed to download MNIST dataset: {e}")
        if not os.path.exists('mnist.npz'):
            urllib.request.urlretrieve(
                'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz',
                'mnist.npz'
            )
        with np.load('mnist.npz') as data:
            return (data['x_train'], data['y_train']), (data['x_test'], data['y_test'])

# Load dataset
(X_train, y_train), (X_test, y_test) = download_mnist_data()
X_train, X_test = X_train.reshape(-1, 28*28) / 255.0, X_test.reshape(-1, 28*28) / 255.0

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(28*28,)),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)

########################################################################################################################

"""
RNN = Recurrent Neural Network
"""
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Example data
X = np.array([[1, 2, 3], [4, 5, 0], [6, 0, 0]])
y = np.array([0, 1, 0])

# Pad sequences
X = pad_sequences(X, maxlen=5)

# Build RNN model
model = Sequential([
    Embedding(input_dim=10, output_dim=8, input_length=5),
    SimpleRNN(16),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=1)

# Example prediction
print(model.predict(X))

########################################################################################################################

"""
CNN = Convolutional Neural Network
"""
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist

# Load dataset
(X_train, y_train), (X_test, y_test) = download_mnist_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Build CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)
