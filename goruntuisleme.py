import os # for file works
import shutil # copy sheet

# folder paths
mysource = "/kaggle/input/animals-with-attributes-2/Animals_with_Attributes2/JPEGImages"
mytarget = "/kaggle/working/FilteredImages"  # new folder path for the image that used for model

# class that used on the model
selected_classes = ["collie", "dolphin", "elephant", "fox", "moose", "rabbit", "sheep", "squirrel", "giant+panda", "polar+bear"]

# create new file for new classes that used on the model
os.makedirs(mytarget, exist_ok=True)

for class_name in selected_classes:
    class_path = os.path.join(mysource, class_name)
    target_path = os.path.join(mytarget, class_name)
    os.makedirs(target_path, exist_ok=True)

    # listing to files on the class_path then get the fir 650 file
    image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))][:650]

    for file_name in image_files:
        full_file_name = os.path.join(class_path, file_name)
        target_file_name = os.path.join(target_path, file_name)
        #print(target_file_name) #used for control

        # copying images from source to target
        shutil.copy2(full_file_name, target_file_name)

print("Dataset created")


import os
from PIL import Image 
import numpy as np

def prepare_images(data_folder, img_size=(128, 128)):
    all_images = []
    all_labels = []

    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)

        if os.path.isdir(folder_path):
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)

                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB") # convort to all images rgb(.,.,3)
                        img_resized = img.resize(img_size) # resized all imaged (128,128)
                        img_array = np.array(img_resized) / 255.0 # normolization(0-1) and convort to array
                        all_images.append(img_array) #add list
                        all_labels.append(folder_name)
                except Exception as e:
                    print(f"Error: {img_path} did not load. {e}")

    return np.array(all_images), np.array(all_labels)

# get to images and labels of images
images, labels = prepare_images(mytarget)

print(f"Datase Size: {images.shape}")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# encoding to labels
label_encoder = OneHotEncoder(sparse_output=False)  
y_encoded = label_encoder.fit_transform(np.array(labels).reshape(-1, 1))

# Divide to dataset to train and test (%70 train, %30 test)
data_train, data_test, labels_train, labels_test = train_test_split(
    images, y_encoded, test_size=0.3, random_state=42
)

print(f"Train Dataset Size: {data_train.shape}, Test Dataset Size: {data_test.shape}")


from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Augmentation
augmentation_params = {
    "rotation_range": 15,
    #"width_shift_range": 0.2,
    #"height_shift_range": 0.2,
    "shear_range": 0.2,
    "zoom_range": 0.2,
    #"horizontal_flip": True,
    "fill_mode": "nearest"
}

data_augmentation = ImageDataGenerator(**augmentation_params)
data_augmentation.fit(data_train)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, Dropout

# Model of CNN
cnn_model = Sequential()  # Initialize a Sequential model for building the CNN architecture.

# Input Layer
cnn_model.add(Input(shape=(128, 128, 3)))  # the input layer with an image size of 128x128 and 3 channels (RGB).

# Convolutional Layers
cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))  # a Conv2D layer with 32 filters, a 3x3 kernel, and ReLU activation.
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))  # a MaxPooling layer to downsample the feature maps by 2x2.
cnn_model.add(Conv2D(64, (3, 3), activation='relu'))  # another Conv2D layer with 64 filters and a 3x3 kernel.
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))  # another MaxPooling layer to further downsample.
cnn_model.add(Conv2D(128, (3, 3), activation='relu'))  # a Conv2D layer with 128 filters and a 3x3 kernel.
cnn_model.add(BatchNormalization())  # a Batch Normalization to normalize activations and improve training stability. (important)

# Flatten Layer
cnn_model.add(Flatten())  # Flatten the feature maps into a 1D vector for input to the fully connected layers.

# Fully Connected Layers
cnn_model.add(Dense(128, activation='relu'))  # a Dense layer with 128 units and ReLU activation for feature extraction.
cnn_model.add(Dropout(0.5))  # a Dropout to reduce overfitting by randomly setting 50% of the inputs to zero during training. (avoid overfitting)
cnn_model.add(Dense(10, activation='softmax'))  # an output Dense layer with 10 units (for 10 animal classes) and softmax activation for classification.

# Model Summary
cnn_model.summary()  # the model summary (the architecture and the number of parameters)


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf 
from tensorflow.data import Dataset  

# Compile the model
cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(),  # the Adam optimization algorithm to optimize the model parameters while training
    loss=tf.keras.losses.CategoricalCrossentropy(),  # Cross-entropy loss function for multi-class classification
    metrics=[tf.keras.metrics.CategoricalAccuracy()]  # Metric to monitor correct classification during training
)

# TensorFlow data augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),  # Randomly rotate images by 10%
    tf.keras.layers.RandomTranslation(0.1, 0.1),  # Randomly translate images horizontally and vertically by 10%
    tf.keras.layers.RandomFlip("horizontal"),  # Randomly flip images horizontally
])

# Preprocessing function for training and validation datasets
def preprocess(image, label):
    image = tf.image.resize(image, (128, 128))  # Resize images to 128x128
    image = augmentation(image)  # Apply data augmentation
    return image, label  # Return the processed image and label

# Function to create tf.data.Dataset objects
def create_dataset(images, labels, batch_size):
    dataset = Dataset.from_tensor_slices((images, labels))  # Create a dataset from image and label tensors
    dataset = dataset.shuffle(buffer_size=len(images))  # Shuffle the data to reduce overfitting (avoid overfitting)
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)  # Apply preprocessing in parallel
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)  # Batch the data and prefetch for efficient data loading
    return dataset  # Return the created tf.data.Dataset object

# Create datasets
batch_size = 32  # the batch size (it should be optimum value to avoid OOM(out of memory))
train_dataset = create_dataset(data_train, labels_train, batch_size)  # Create the training dataset
validation_dataset = create_dataset(data_test, labels_test, batch_size)  # Create the validation dataset to monitor the model

# Train the model
history = cnn_model.fit(
    train_dataset,  # Training data
    validation_data=validation_dataset,  # Validation data
    epochs=20,  # Number of epochs for training (optimum value)
    steps_per_epoch=len(data_train) // batch_size,  # Define the number of steps per epoch
    validation_steps=len(data_test) // batch_size  # Define the number of validation steps

test_loss, test_accuracy = cnn_model.evaluate(data_test, labels_test) # Evaluation of the model
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

import numpy as np  # Importing NumPy for array manipulation
import cv2  # Importing OpenCV for image processing

# Function to adjust brightness and contrast of images
def adjust_brightness(images, alpha=1.5, beta=30):
    """
    Adjust brightness and contrast of a batch of images.
    
    Parameters:
    - images: NumPy array of images to process
    - alpha: Contrast control (higher values increase contrast)
    - beta: Brightness control (higher values increase brightness)
    
    Returns:
    - Processed images as a NumPy array
    """
    return np.array([cv2.convertScaleAbs(image, alpha=alpha, beta=beta) for image in images])

# Create a manipulated test set by adjusting brightness and contrast
data_test_brightness_adjusted = adjust_brightness(data_test)

# Evaluate the model on the manipulated test set
loss_adjusted, accuracy_adjusted = cnn_model.evaluate(data_test_brightness_adjusted, labels_test)

# Print the accuracy of the model on the manipulated test set
print(f"Manipulated Test Accuracy: {accuracy_adjusted * 100:.2f}%")

import cv2  # Importing OpenCV for image processing
import numpy as np  # Importing NumPy for array manipulations

# Function to apply Gray World color correction
def gray_world_correction(image):
    mean_values = cv2.mean(image)[:3]  # Calculate mean values for RGB channels
    gray_mean = np.mean(mean_values)  # Compute the average gray value across channels
    scaling_factors = gray_mean / np.array(mean_values)  # Calculate scaling factors for each channel
    corrected = np.clip(image * scaling_factors, 0, 255).astype(np.uint8)  # Adjust pixel values and clip to valid range
    return corrected

# Apply Gray World color correction to the manipulated test set
data_test_color_corrected = np.array(list(map(gray_world_correction, data_test_brightness_adjusted)))

# Evaluate the model on the color-corrected test set
loss_corrected, accuracy_corrected = cnn_model.evaluate(data_test_color_corrected, labels_test)

# Print the accuracy of the model on the color-corrected test set
print(f"Color-Corrected Test Accuracy: {accuracy_corrected * 100:.2f}%")


import cv2  # Importing OpenCV for image processing
import numpy as np  # Importing NumPy for array manipulations

# Function to perform Gray World normalization
def gray_world_normalization(image):
    """
    Perform Gray World normalization on an image.
    
    Parameters:
    - image: Input image as a NumPy array
    
    Returns:
    - Normalized image with balanced colors as a NumPy array
    """
    channel_means = np.mean(image, axis=(0, 1))  # Calculate the mean for each color channel
    overall_mean = np.mean(channel_means)       # Calculate the overall mean of the color channels
    scale_factors = overall_mean / channel_means  # Calculate scaling factors for each channel

    # Apply scaling factors to normalize the image
    normalized_image = (image * scale_factors).clip(0, 255).astype(np.uint8)
    return normalized_image

# Apply Gray World normalization to the manipulated test set
data_test_normalized = np.array([gray_world_normalization(img) for img in data_test_brightness_adjusted])

# Evaluate the model on the normalized test set
loss_normalized, accuracy_normalized = cnn_model.evaluate(data_test_normalized, labels_test)

print(f"Color-Normalized Test Accuracy: {accuracy_normalized * 100:.2f}%")


# Print accuracy results for every test set
print("Comparison of Test Results:")
print(f"Original Test Set Accuracy: {test_accuracy * 100:.2f}%")  # Accuracy for the original test set
print(f"Manipulated Test Set Accuracy: {accuracy_adjusted * 100:.2f}%")  # Accuracy for the brightness-adjusted test set
print(f"Color-Corrected Test Set Accuracy: {accuracy_corrected * 100:.2f}%")  # Accuracy for the Gray World corrected test set
print(f"Color-Normalized Test Set Accuracy: {accuracy_normalized * 100:.2f}%")  # Accuracy for the Gray World normalized test set



