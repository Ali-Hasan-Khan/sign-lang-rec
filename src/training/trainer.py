import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

class ModelTrainer:
    def __init__(self, model, model_save_path="models/weights-improvement.keras"):
        self.model = model
        self.model_save_path = model_save_path
        self.history = None
        
    def compile_model(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        """Compile the model with specified parameters"""
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self.model
    
    def create_callbacks(self):
        """Create callbacks for model training"""
        lr = ReduceLROnPlateau(
            monitor='val_accuracy', 
            patience=2, 
            factor=0.5, 
            verbose=1
        )
        
        es = EarlyStopping(
            monitor='val_loss', 
            min_delta=0.001,
            patience=5,
            restore_best_weights=True, 
            verbose=0
        )
        
        checkpoint = ModelCheckpoint(
            self.model_save_path,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            mode="max"
        )
        
        return [checkpoint, lr, es]
    
    def train(self, train_gen, valid_gen, epochs=50, batch_size=32, steps_per_epoch=None, validation_steps=None):
        """Train the model"""
        callbacks = self.create_callbacks()
        
        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=valid_gen,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return self.history
    
    def load_best_weights(self):
        """Load the best weights from training"""
        self.model.load_weights(self.model_save_path)
        return self.model
    
    def evaluate(self, test_gen):
        """Evaluate the model on test data"""
        loss, accuracy = self.model.evaluate(test_gen, verbose=0)
        print(f'Test accuracy: {accuracy*100:.2f}%')
        print(f'Test loss: {loss:.4f}')
        return loss, accuracy