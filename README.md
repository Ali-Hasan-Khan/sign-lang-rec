# Sign Language Recognition

This project implements a deep learning model for American Sign Language (ASL) recognition using Convolutional Neural Networks (CNNs).

## Project Structure
```
sign_language_recognition/
│
├── data/
│   └── raw/
│       └── asl_dataset/
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn_model.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── visualization/
│       ├── __init__.py
│       └── visualizer.py
│
├── main.py
├── requirements.txt
└── README.md
```
## Setup

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Download the ASL dataset and place it in the `data/raw/asl_dataset` directory
4. Run the main script:
   ```
   python main.py
   ```

## Dataset

The project uses the American Sign Language dataset, which contains images of hand signs for various letters and numbers.

## Model

The model is a Convolutional Neural Network (CNN) with the following architecture:
- 3 convolutional layers with max pooling
- Fully connected layer with 512 units
- Dropout layer for regularization
- Output layer with softmax activation

## Results

The model achieves 91.72% accuracy on the test set. The confusion matrix and classification report provide detailed performance metrics for each class.
