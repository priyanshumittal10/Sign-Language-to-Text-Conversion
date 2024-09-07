import numpy as np
import matplotlib.image as mimg
import os
from sklearn import model_selection
from tensorflow import keras

# Define the path to the dataset directory
dataset_dir = "D:\\Sign to Text Convertor Project\\Test\\dataset"  # Update this path to your dataset location

# Define the total number of samples
tot_samples = 26 * 100  # 26 letters, each with 100 images

# Read a sample image to determine the image shape
sample_image_path = os.path.join(dataset_dir, 'A', os.listdir(os.path.join(dataset_dir, 'A'))[0])
sample_image = mimg.imread(sample_image_path)
image_shape = sample_image.shape

# Initialize arrays to store the data, labels, and images
data = np.zeros((tot_samples, image_shape[0], image_shape[1], image_shape[2]))
labels = np.zeros((tot_samples))

indx = -1
for label_index, label in enumerate(sorted(os.listdir(dataset_dir))):
    label_dir = os.path.join(dataset_dir, label)
    image_files = os.listdir(label_dir)
    for image_file in image_files:
        indx += 1
        path = os.path.join(label_dir, image_file)
        im = mimg.imread(path)
        if im.shape != image_shape:
            print(f"Skipping image {image_file} due to shape mismatch: {im.shape} != {image_shape}")
            continue
        data[indx] = im
        labels[indx] = label_index
        print(f"Label {label} (index {label_index}), image {image_file} processed...")

print("All images processed.")

# Normalize data
data = data / 255.0

# Example of how to use these arrays
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, labels, test_size=0.3, random_state=42)
print(xtrain.shape, ytrain.shape)
print(xtest.shape, ytest.shape)

# Define the model
face_model = keras.Sequential()

# Input layer
face_model.add(keras.layers.InputLayer(input_shape=image_shape))

# Hidden layers
face_model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
face_model.add(keras.layers.MaxPooling2D((2, 2)))
face_model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
face_model.add(keras.layers.MaxPooling2D((2, 2)))
face_model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
face_model.add(keras.layers.MaxPooling2D((2, 2)))
face_model.add(keras.layers.Flatten())
face_model.add(keras.layers.Dense(512, activation='relu'))
face_model.add(keras.layers.Dropout(0.3))
face_model.add(keras.layers.Dense(512, activation='relu'))
face_model.add(keras.layers.Dropout(0.3))
face_model.add(keras.layers.Dense(512, activation='relu'))
face_model.add(keras.layers.Dropout(0.3))

# Output layer
face_model.add(keras.layers.Dense(26, activation='softmax'))  # 26 classes

# Compile the model
face_model.compile(optimizer='adam',  # Changed to 'adam'
                   loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # Updated for softmax
                   metrics=['accuracy'])

# Print model summary
print(face_model.summary())

# Train the model
history = face_model.fit(xtrain, ytrain, epochs=30, validation_data=(xtest, ytest))

# Save the trained model
model_save_path = "D:\\Sign to Text Convertor Project\\Test\\orl_face_model.h5"
face_model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Evaluate the test data
loss, accNN = face_model.evaluate(xtest, ytest)
print(f"Testing Accuracy of Neural Network (NN) is: {accNN}")
