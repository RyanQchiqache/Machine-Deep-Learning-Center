import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import datetime  # For timestamping the log directories


class DigitRecognizer:
    """
    A class used to represent a Convolutional Neural Network for recognizing handwritten digits
    using the MNIST dataset.

    Attributes:
    -----------
    model : tf.keras.Model
        The Keras Sequential model representing the neural network.
    """

    def __init__(self):
        """Initializes the DigitRecognizer with no model defined."""
        self.model = None

    def load_data(self):
        """
        Loads the MNIST dataset and preprocesses it by normalizing the images to the range [0, 1].
        Also, it splits the training data into a training set and a validation set.

        Attributes:
        -----------
        train_images : numpy.ndarray
            The training images normalized and reshaped for the model.
        train_labels : numpy.ndarray
            The labels corresponding to the training images.
        val_images : numpy.ndarray
            The validation images extracted from the training set.
        val_labels : numpy.ndarray
            The labels corresponding to the validation images.
        test_images : numpy.ndarray
            The test images normalized and reshaped for the model.
        test_labels : numpy.ndarray
            The labels corresponding to the test images.
        """
        # Load the MNIST dataset
        mnist = tf.keras.datasets.mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist.load_data()

        # Normalize the images to the range [0, 1]
        self.train_images = self.train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        self.test_images = self.test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

        # Split a validation set from the training data
        self.train_images, self.val_images = self.train_images[:-5000], self.train_images[-5000:]
        self.train_labels, self.val_labels = self.train_labels[:-5000], self.train_labels[-5000:]

    def build_model(self):
        """
        Builds a Convolutional Neural Network (CNN) using the Keras Sequential API.

        The model architecture includes:
        - Convolutional layers with ReLU activation.
        - Max-pooling layers for downsampling.
        - Dropout for regularization.
        - Dense layers with the final softmax activation layer for classification.
        """
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),  # Add dropout for regularization
            layers.Dense(10, activation='softmax')
        ])

    def compile_and_train(self, epochs=10, batch_size=64, use_augmentation=True):
        """
        Compiles and trains the model.

        Parameters:
        -----------
        epochs : int, optional
            Number of epochs to train the model (default is 10).
        batch_size : int, optional
            Number of samples per gradient update (default is 64).
        use_augmentation : bool, optional
            Whether to apply data augmentation to the training images (default is True).

        Returns:
        --------
        history : tf.keras.callbacks.History
            A History object that contains all information collected during training.
        """
        # Compile the model
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        # Define TensorBoard callback
        log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Define other callbacks for early stopping and model checkpointing
        early_stopping_cb = callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        model_checkpoint_cb = callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)

        # Use data augmentation if specified
        if use_augmentation:
            datagen = ImageDataGenerator(
                rotation_range=10,
                zoom_range=0.1,
                width_shift_range=0.1,
                height_shift_range=0.1
            )
            datagen.fit(self.train_images)
            train_generator = datagen.flow(self.train_images, self.train_labels, batch_size=batch_size)
            steps_per_epoch = len(self.train_images) // batch_size
            validation_data = (self.val_images, self.val_labels)
            history = self.model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=validation_data,
                callbacks=[tensorboard_cb, early_stopping_cb, model_checkpoint_cb]
            )
        else:
            history = self.model.fit(
                self.train_images, self.train_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(self.val_images, self.val_labels),
                callbacks=[tensorboard_cb, early_stopping_cb, model_checkpoint_cb]
            )

        return history

    def evaluate(self):
        """
        Evaluates the model on the test dataset and prints the test accuracy.

        Returns:
        --------
        None
        """
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)
        print(f'\nTest accuracy: {test_acc * 100:.2f}%')

    def save_model(self, path='model.h5'):
        """
        Saves the trained model to the specified path.

        Parameters:
        -----------
        path : str, optional
            The file path where the model should be saved (default is 'model.h5').

        Returns:
        --------
        None
        """
        self.model.save(path)
        print(f'Model saved to {path}')

    def load_model(self, path='model.h5'):
        """
        Loads a model from the specified path.

        Parameters:
        -----------
        path : str, optional
            The file path from where the model should be loaded (default is 'model.h5').

        Returns:
        --------
        None
        """
        if os.path.exists(path):
            self.model = models.load_model(path)
            print(f'Model loaded from {path}')
        else:
            print(f'Model path {path} does not exist.')


# Usage example:
# --------------
# Initialize and train the digit recognizer model.
if __name__ == "__main__":
    recognizer = DigitRecognizer()
    recognizer.load_data()
    recognizer.build_model()
    recognizer.compile_and_train(epochs=15)
    recognizer.evaluate()
    recognizer.save_model()
