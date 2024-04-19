import tensorflow as tf
from tensorflow.keras import layers, models

class DigitRecognizer:
    def __init__(self):
        self.model = None

    def load_data(self):
        mnist = tf.keras.datasets.mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist.load_data()
        self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0

    def build_model(self):
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

    def compile_and_train(self, epochs=5):
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(self.train_images, self.train_labels, epochs=epochs)

    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)
        print(f'Test accuracy: {test_acc}')

    def save_model(self, path='model.h5'):
        self.model.save(path)

    def load_model(self, path='model.h5'):
        self.model = models.load_model(path)

# Usage
recognizer = DigitRecognizer()
recognizer.load_data()
recognizer.build_model()
recognizer.compile_and_train(epochs=10)  # More epochs for potentially better training
recognizer.evaluate()
recognizer.save_model()  # Saves the trained model
