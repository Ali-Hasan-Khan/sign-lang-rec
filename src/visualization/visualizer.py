import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from datetime import datetime

class Visualizer:
    def __init__(self, data_loader, results_dir="results"):
        self.data_loader = data_loader
        self.results_dir = results_dir
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create timestamp for this visualization session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a unique run folder based on timestamp
        self.run_dir = os.path.join(self.results_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize iteration counter
        self.iteration = 0
        
    def set_iteration(self, iteration):
        """Set the current iteration number"""
        self.iteration = iteration
        # Create iteration subfolder
        self.iteration_dir = os.path.join(self.run_dir, f"iteration_{iteration}")
        os.makedirs(self.iteration_dir, exist_ok=True)
        
    def _save_figure(self, fig, name, model_name=None):
        """Save figure with model name in the current iteration folder"""
        if model_name:
            filename = f"{model_name}_{name}.png"
        else:
            filename = f"{name}.png"
        
        filepath = os.path.join(self.iteration_dir, filename)
        fig.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"Saved figure to {filepath}")
        return filepath
        
    def plot_class_distribution(self, save=True):
        """Plot the distribution of classes in the dataset"""
        fig = plt.figure(figsize=(15, 8))
        self.data_loader.df["categories"].sort_values().value_counts(sort=False).plot(kind="bar")
        plt.title("Class Distribution", fontsize=16)
        plt.xlabel("Classes", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        if save:
            self._save_figure(fig, "class_distribution")
        
        # plt.show()
        
    def plot_sample_images(self, num_samples=30, figsize=(40, 40), save=True):
        """Plot sample images from the dataset"""
        fig = plt.figure(figsize=figsize)
        for i in range(num_samples):
            j = np.random.randint(self.data_loader.df.shape[0])
            img = plt.imread(self.data_loader.df["img_paths"].iloc[j])
            plt.subplot(10, 10, i+1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(self.data_loader.df['categories'].iloc[j], fontsize=25)
        plt.tight_layout()
        
        if save:
            self._save_figure(fig, "sample_images")
        
        # plt.show()
        
    def plot_training_history(self, history, model_name=None, save=True):
        """Plot training and validation metrics"""
        hist_ = pd.DataFrame(history.history)
        fig = plt.figure(figsize=(15, 10))
        
        plt.subplot(1, 2, 1)
        plt.plot(hist_['loss'], label='Train_Loss')
        plt.plot(hist_['val_loss'], label='Validation_Loss')
        plt.title('Train_Loss & Validation_Loss', fontsize=20)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(hist_['accuracy'], label='Train_Accuracy')
        plt.plot(hist_['val_accuracy'], label='Validation_Accuracy')
        plt.title('Train_Accuracy & Validation_Accuracy', fontsize=20)
        plt.legend()
        
        plt.tight_layout()
        
        if save:
            self._save_figure(fig, "learning_curves", model_name)
        
        # plt.show()
        
    def plot_confusion_matrix(self, model, x_test_arr, y_test, model_name=None, save=True):
        """Plot confusion matrix for model evaluation"""
        # Get predictions
        result = model.predict(x_test_arr, verbose=0)
        y_pred = np.argmax(result, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Print classification report
        class_names = self.data_loader.get_class_names()
        report = classification_report(y_true, y_pred, target_names=class_names)
        print(report)
        
        # Save classification report to text file
        if save:
            report_filename = f"{model_name}_classification_report.txt" if model_name else "classification_report.txt"
            report_path = os.path.join(self.iteration_dir, report_filename)
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Saved classification report to {report_path}")
        
        # Plot confusion matrix
        con_mat = confusion_matrix(y_true, y_pred)
        fig = plt.figure(figsize=(20, 8), dpi=200)
        sns.heatmap(con_mat, annot=True, cbar=False)
        plt.xticks(np.arange(con_mat.shape[1]), class_names, rotation=90)
        plt.yticks(np.arange(con_mat.shape[0]), class_names)
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('True label', fontsize=14)
        plt.xlabel('Predicted label', fontsize=14)
        plt.tight_layout()
        
        if save:
            self._save_figure(fig, "confusion_matrix", model_name)
        
        # plt.show()
        
    def plot_specific_class_samples(self, class_names, num_samples=10, figsize=(40, 40), save=True):
        """Plot samples from specific classes"""
        fig = plt.figure(figsize=figsize)
        row = 0
        for class_name in class_names:
            for i in range(num_samples):
                class_df = self.data_loader.df[self.data_loader.df['categories'] == class_name]
                if class_df.shape[0] > 0:
                    j = np.random.randint(class_df.shape[0])
                    img = plt.imread(class_df["img_paths"].iloc[j])
                    plt.subplot(len(class_names), num_samples, row*num_samples + i + 1)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(class_name, fontsize=25)
            row += 1
        plt.tight_layout()
        
        if save:
            class_str = '_'.join(class_names) if len(class_names) <= 3 else f"{len(class_names)}_classes"
            self._save_figure(fig, f"specific_classes_{class_str}")
        
        # plt.show()
        
    def save_model_summary(self, model, model_name=None):
        """Save model summary to a text file"""
        # Redirect model.summary() output to a string
        from io import StringIO
        import sys
        
        original_stdout = sys.stdout
        string_io = StringIO()
        sys.stdout = string_io
        model.summary()
        sys.stdout = original_stdout
        model_summary = string_io.getvalue()
        
        # Save to file
        summary_filename = f"{model_name}_model_summary.txt" if model_name else "model_summary.txt"
        summary_path = os.path.join(self.iteration_dir, summary_filename)
        with open(summary_path, 'w') as f:
            f.write(model_summary)
        print(f"Saved model summary to {summary_path}")