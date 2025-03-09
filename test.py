import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
import matplotlib.pyplot as plt



class SignLanguageModelTrainer:
    def __init__(self):
        # Initialize parameters
        self.dataset_dir = "data/raw/asl_alphabet"  # Update to your dataset path
        self.model_path = "models/sign_language_model.h5"
        self.gestures = []  # To be populated with gesture names

    def load_asl_dataset(self):
        """Load the ASL dataset from the specified directory."""
        X = []  # Features
        y = []  # Labels
        gestures = os.listdir(self.dataset_dir)  # List of gesture folders

        for gesture_id, gesture_name in enumerate(gestures):
            gesture_dir = os.path.join(self.dataset_dir, gesture_name)
            for image_name in os.listdir(gesture_dir):
                image_path = os.path.join(gesture_dir, image_name)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (64, 64))  # Resize to a consistent size
                X.append(image)
                y.append(gesture_id)

        X = np.array(X).astype('float32') / 255.0  # Normalize pixel values
        y = np.array(y)

        self.gestures = gestures  # Store gesture names
        return X, y

    def create_model(self):
        """Define the deep learning model architecture."""
        model = Sequential()
        model.add(Flatten(input_shape=(64, 64, 3)))  # Adjust input shape based on image size
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.gestures), activation='softmax'))  # Output layer for multi-class classification
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, X, y):
        """Train the deep learning model on the collected data."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = self.create_model()
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Model accuracy: {accuracy * 100:.2f}%")

        # Save the model
        model.save(self.model_path)
        print(f"Model saved to {self.model_path}")

        # Plot training history
        self.plot_training_history(history)

    def plot_training_history(self, history):
        """Plot training and validation accuracy and loss."""
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    def run(self):
        """Run the complete training pipeline."""
        print("Sign Language Recognition - Model Training")
        print("-----------------------------------------")

        # Load ASL dataset
        X, y = self.load_asl_dataset()

        # Train the model
        self.train_model(X, y)

if __name__ == "__main__":
    trainer = SignLanguageModelTrainer()
    trainer.run()