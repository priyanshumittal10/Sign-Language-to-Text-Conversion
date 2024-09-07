import os
import pandas as pd

# Path to the dataset
dataset_path = r"D:\\Sign to Text Convertor Project\\Test\\dataset"

# Complete mapping of folders to sign labels
label_mapping = {
    'A': 'love',
    'B': 'thank_you',
    'C': 'hello',
    'D': 'goodbye',
    'E': 'yes',
    'F': 'no',
    'G': 'please',
    'H': 'sorry',
    'I': 'help',
    'J': 'stop',
    'K': 'go',
    'L': 'wait',
    'M': 'come',
    'N': 'good_morning',
    'O': 'good_night',
    'P': 'congratulations',
    'Q': 'birthday',
    'R': 'holiday',
    'S': 'vacation',
    'T': 'party',
    'U': 'anniversary',
    'V': 'graduation',
    'W': 'success',
    'X': 'failure',
    'Y': 'try_again',
    'Z': 'winner'
}

# List to store image names and their labels
data = []

# Iterate through each folder
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        # Get the corresponding label for the folder
        label = label_mapping.get(folder)
        if label:
            # Iterate through each image in the folder
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                if os.path.isfile(image_path):
                    # Append the image name and label to the data list
                    data.append((image_name, label))
 
# Create a DataFrame from the data list
df = pd.DataFrame(data, columns=['image_name', 'label'])

# Save the DataFrame to a CSV file
output_csv_path = "labeled_dataset.csv"
df.to_csv(output_csv_path, index=False)

# Display the DataFrame as a table
print(df)

# If you prefer to see the first few rows as a preview
print(df.head())
