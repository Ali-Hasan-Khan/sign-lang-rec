import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.data_loader import DataLoader
from src.models.cnn_model import create_cnn_model
from src.training.trainer import ModelTrainer
from src.visualization.visualizer import Visualizer
from keras.utils import plot_model


# Suppress warnings
warnings.filterwarnings("ignore")

def main():
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    
    # Initialize data loader
    data_loader = DataLoader(data_path="data/raw/asl_dataset", input_size=(224, 224))
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_loader.load_data()
    data_loader.preprocess_data()
    x_train_arr, x_valid_arr, x_test_arr = data_loader.load_images()
    
    # Create data generators
    print("Creating data generators...")
    train_gen, valid_gen, test_gen = data_loader.create_data_generators(
        x_train_arr, x_valid_arr, x_test_arr, batch_size=16
    )

    # Load existing model
    # model_path = "models/weights-improvement.keras"
    # if os.path.exists(model_path):
    #     print(f"Loading existing model from {model_path}")
    #     # Option 1: Load the entire model
    #     from keras.models import load_model
    #     model = load_model(model_path)
    # else:
    #     print("Creating new model")
    #     # Create and compile model
    #     num_classes = data_loader.get_num_classes()
    #     model = create_cnn_model(224, 224, num_classes)
    #     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # # Initialize trainer with the loaded or new model
    # trainer = ModelTrainer(model, model_save_path=model_path)
    
    # # If you loaded the entire model, you don't need to compile it again
    # if not os.path.exists(model_path):
    #     trainer.compile_model()
    
    # Create and compile model
    print("Creating model...")
    num_classes = data_loader.get_num_classes()
    model = create_cnn_model(224, 224, num_classes)
    model.summary()
    # Add the visualization code here
    plot_model(
        model,
        to_file='model_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        dpi=300,  # Higher DPI for better quality
        expand_nested=True
    )
    print("Model architecture image saved as 'model_architecture.png'")

    
    # # Initialize trainer
    # trainer = ModelTrainer(model, model_save_path="models/model-weights.keras")
    # trainer.compile_model()
    
    # # Initialize visualizer with results directory
    # visualizer = Visualizer(data_loader, results_dir="results")
    
    # # Set the iteration number (you can read this from a file or pass as an argument)
    # # For simplicity, we'll use a counter file
    # iteration_counter_file = "iteration_counter.txt"
    # if os.path.exists(iteration_counter_file):
    #     with open(iteration_counter_file, 'r') as f:
    #         iteration = int(f.read().strip())
    # else:
    #     iteration = 1
    
    # # Set the current iteration in the visualizer
    # visualizer.set_iteration(iteration)
    
    # # Save model architecture summary
    # model_name = os.path.basename(trainer.model_save_path).replace(".keras", "")
    # visualizer.save_model_summary(model, model_name=model_name)
    
    # # Visualize data distribution and sample images
    # visualizer.plot_class_distribution()
    # visualizer.plot_sample_images(num_samples=30)
    
    # # Train model
    # print(f"Training model (Iteration {iteration})...")
    # history = trainer.train(
    #     train_gen,
    #     valid_gen,
    #     epochs=50,
    #     batch_size=32,
    #     steps_per_epoch=x_train_arr.shape[0] // 32,
    #     validation_steps=x_valid_arr.shape[0] // 32
    # )
    
    # # Visualize training history
    # visualizer.plot_training_history(history, model_name=model_name)
    
    # # Load best weights and evaluate
    # trainer.load_best_weights()
    # loss, accuracy = trainer.evaluate(test_gen)
    
    # # Plot confusion matrix
    # visualizer.plot_confusion_matrix(model, x_test_arr, data_loader.Y_test, model_name=model_name)
    
    # # Visualize specific classes
    # visualizer.plot_specific_class_samples(['o', '0', '2', '3', 'w'], num_samples=10)
    
    # # Increment and save the iteration counter for next run
    # with open(iteration_counter_file, 'w') as f:
    #     f.write(str(iteration + 1))
    
    # print(f"Completed iteration {iteration}. Results saved to {visualizer.iteration_dir}")
    print("Done!")

if __name__ == "__main__":
    main()