import numpy as np
import matplotlib.image as mimg
import os


# Define the path to the dataset directory
dataset_dir = "D:\\Sign to Text Convertor Project\\Test\\dataset"  # Update this path to your dataset location

# Define the total number of samples
tot_samples = 26 * 100  # 26 letters, each with 100 images

# Read a sample image to determine the image shape
sample_image_path = os.path.join(dataset_dir, 'A', os.listdir(os.path.join(dataset_dir, 'A'))[0])
sample_image = mimg.imread(sample_image_path)
image_shape = sample_image.shape

# Initialize arrays to store the data, labels, and images
data = np.zeros((tot_samples, image_shape[0] * image_shape[1] * image_shape[2]))
labels = np.zeros((tot_samples))
images = np.zeros((tot_samples, *image_shape))

indx = -1
for label_index, label in enumerate(sorted(os.listdir(dataset_dir))):
    label_dir = os.path.join(dataset_dir, label)
    image_files = os.listdir(label_dir)
    for image_file in image_files:
        indx += 1
        path = os.path.join(label_dir, image_file)
        im = mimg.imread(path)
        feat = im.reshape(1, -1)
        if feat.shape[1] != data.shape[1]:
            print(f"Skipping image {image_file} due to shape mismatch: {feat.shape} != {data.shape[1]}")
            continue
        data[indx, :] = feat
        labels[indx] = label_index
        images[indx, :, :, :] = im
        print(f"Label {label} (index {label_index}), image {image_file} processed...")

print("All images processed.")

# Example of how to use these arrays
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)
print("Images shape:", images.shape)
