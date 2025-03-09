from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_cnn_model(input_height, input_width, num_classes):
    """Create a CNN model for sign language recognition"""
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), strides=2, padding='same', activation='relu', 
                    input_shape=(input_height, input_width, 3)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
    model.add(Conv2D(64, kernel_size=(3,3), strides=2, activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), 2, padding='same'))
    model.add(Conv2D(128, kernel_size=(3,3), strides=2, activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), 2, padding='same'))
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(units=num_classes, activation='softmax'))
    
    return model