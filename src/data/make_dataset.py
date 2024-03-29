import os
import sys
# Set the working directory to src root
os.chdir(os.path.dirname(sys.path[0]))
sys.path.append(os.path.normcase(os.getcwd()))
import numpy as np
from PIL import Image
import torch
from data.config import image_size
from data.unpack_data import unpack_raw_data


def load_images_and_labels(base_path, folder_names, standard_size):
    all_images = []
    labels = []

    for label, folder_name in enumerate(folder_names):
        folder_path = os.path.join(base_path, folder_name)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".png")):  # check for image files
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize(standard_size)
                img_array = np.array(img)
                all_images.append(img_array)
                labels.append(label)

    return all_images, labels


# Pull and unpack data
unpack_raw_data()

# Define the paths to the training and testing folders
base_path_training = os.path.normpath("../data/raw/Training")
base_path_testing = os.path.normpath("../data/raw/Testing")
folder_names = ["glioma", "meningioma", "notumor", "pituitary"]
standard_size = image_size
# Load training and testing data
training_images, training_labels = load_images_and_labels(base_path_training, folder_names, standard_size)
testing_images, testing_labels = load_images_and_labels(base_path_testing, folder_names, standard_size)

# Convert to PyTorch tensors
training_images_tensor = torch.tensor(training_images)
training_labels_tensor = torch.tensor(training_labels)
testing_images_tensor = torch.tensor(testing_images)
testing_labels_tensor = torch.tensor(testing_labels)

# Define the path for the new folder to store the tensors
tensor_storage_path = os.path.normpath("../data/processed")

# Create the folder if it doesn't exist
os.makedirs(tensor_storage_path, exist_ok=True)

# Define file names for the tensors
training_images_file = os.path.join(tensor_storage_path, "training_images.pt")
training_labels_file = os.path.join(tensor_storage_path, "training_labels.pt")
testing_images_file = os.path.join(tensor_storage_path, "testing_images.pt")
testing_labels_file = os.path.join(tensor_storage_path, "testing_labels.pt")

# Save the tensors
torch.save(training_images_tensor, training_images_file)
torch.save(training_labels_tensor, training_labels_file)
torch.save(testing_images_tensor, testing_images_file)
torch.save(testing_labels_tensor, testing_labels_file)
