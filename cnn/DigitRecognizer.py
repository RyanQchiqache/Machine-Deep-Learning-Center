import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import datetime  # For timestamping the log directories


class DigitRecognizer:
    def __init__(self):
        self.model = None

    def load_data(self):
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
        # Build a Sequential model
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
        # Evaluate the model on the test dataset
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)
        print(f'\nTest accuracy: {test_acc * 100:.2f}%')

    def save_model(self, path='model.h5'):
        # Save the trained model to the specified path
        self.model.save(path)
        print(f'Model saved to {path}')

    def load_model(self, path='model.h5'):
        # Load a model from the specified path
        if os.path.exists(path):
            self.model = models.load_model(path)
            print(f'Model loaded from {path}')
        else:
            print(f'Model path {path} does not exist.')


# Usage
if __name__ == "__main__":
    recognizer = DigitRecognizer()
    recognizer.load_data()
    recognizer.build_model()
    recognizer.compile_and_train(epochs=15)  # Train with data augmentation and early stopping
    recognizer.evaluate()
    recognizer.save_model()  # Save the trained model
