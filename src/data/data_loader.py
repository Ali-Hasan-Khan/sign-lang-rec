import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataLoader:
    def __init__(self, data_path="data/raw/asl_dataset", input_size=(224, 224)):
        self.data_path = data_path
        self.input_height, self.input_width = input_size
        self.df = None
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.Y_train = None
        self.Y_valid = None
        self.Y_test = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load image paths and categories into a DataFrame"""
        file_names = os.listdir(self.data_path)
        if ".DS_Store" in file_names:
            file_names.remove(".DS_Store")
            
        df = pd.DataFrame(columns=["img_names", "categories"])
        for img_category in file_names:
            img_df = pd.DataFrame()
            img_df["img_names"] = os.listdir(os.path.join(self.data_path, img_category))
            img_df["categories"] = img_category
            df = pd.concat([df, img_df], axis=0, join="inner")
            
        df["img_paths"] = self.data_path + "/" + df["categories"] + "/" + df["img_names"]
        self.df = df
        return df
    
    def preprocess_data(self, test_size=0.25, valid_size=0.5, random_state=42):
        """Split data and preprocess images"""
        if self.df is None:
            self.load_data()
            
        X = self.df["img_paths"].values
        Y = self.df["categories"]
        
        # Encode labels
        Y_encode = self.label_encoder.fit_transform(Y)
        Y_encode = to_categorical(Y_encode)
        
        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encode, test_size=test_size, random_state=random_state)
        X_test, X_valid, Y_test, Y_valid = train_test_split(X_test, Y_test, test_size=valid_size, random_state=random_state)
        
        # Store split data
        self.X_train, self.X_valid, self.X_test = X_train, X_valid, X_test
        self.Y_train, self.Y_valid, self.Y_test = Y_train, Y_valid, Y_test
        
        return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
    
    def load_images(self):
        """Load and preprocess images"""
        if self.X_train is None:
            self.preprocess_data()
            
        # Process test images
        x_test_arr = np.zeros((self.X_test.shape[0], self.input_height, self.input_width, 3))
        for i in range(len(self.X_test)):
            im = load_img(self.X_test[i],
                color_mode="rgb",
                target_size=(self.input_height, self.input_width),
                interpolation="nearest",
                keep_aspect_ratio=False)
            x_test_arr[i] = img_to_array(im)/255
            
        # Process train images
        x_train_arr = np.zeros((self.X_train.shape[0], self.input_height, self.input_width, 3))
        for i in range(len(self.X_train)):
            im = load_img(self.X_train[i],
                color_mode="rgb",
                target_size=(self.input_height, self.input_width),
                interpolation="nearest",
                keep_aspect_ratio=False)
            x_train_arr[i] = img_to_array(im)/255
            
        # Process validation images
        x_valid_arr = np.zeros((self.X_valid.shape[0], self.input_height, self.input_width, 3))
        for i in range(len(self.X_valid)):
            im = load_img(self.X_valid[i],
                color_mode="rgb",
                target_size=(self.input_height, self.input_width),
                interpolation="nearest",
                keep_aspect_ratio=False)
            x_valid_arr[i] = img_to_array(im)/255
            
        return x_train_arr, x_valid_arr, x_test_arr
    
    def create_data_generators(self, x_train_arr, x_valid_arr, x_test_arr, batch_size=16):
        """Create data generators with augmentation"""
        train_datagen = ImageDataGenerator(
            rescale=1.,
            rotation_range=30,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        valid_datagen = ImageDataGenerator(rescale=1.)
        test_datagen = ImageDataGenerator(rescale=1.)
        
        train_gen = train_datagen.flow(x_train_arr, self.Y_train, batch_size=batch_size, shuffle=True, seed=42)
        valid_gen = valid_datagen.flow(x_valid_arr, self.Y_valid, batch_size=batch_size, shuffle=False, seed=42)
        test_gen = test_datagen.flow(x_test_arr, self.Y_test, batch_size=batch_size, shuffle=False, seed=42)
        
        return train_gen, valid_gen, test_gen
    
    def get_num_classes(self):
        """Get the number of classes in the dataset"""
        return len(self.label_encoder.classes_)
    
    def get_class_names(self):
        """Get the class names"""
        return self.label_encoder.classes_