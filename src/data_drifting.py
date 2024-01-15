from scipy.stats import ks_2samp
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import os
import numpy as np

# Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Local Image Folders
image_folder_path_1 = "data/data_features/set1"
image_folder_path_2 = "data/data_features/set2"

# Function to extract features from a folder
def extract_features_from_folder(folder_path):
    image_files = os.listdir(folder_path)
    image_files = [os.path.join(folder_path, file) for file in image_files]

    features = []
    for image_file in image_files:
        image = Image.open(image_file)
        inputs = processor(images=image, return_tensors="pt", padding=True)
        img_features = model.get_image_features(inputs['pixel_values'])
        features.append(img_features.squeeze().detach().numpy())

    return np.array(features)

# Extract features from two sets of images
image_features_set1 = extract_features_from_folder(image_folder_path_1)
image_features_set2 = extract_features_from_folder(image_folder_path_2)

# Convert the NumPy arrays to PyTorch tensors
image_features_set1 = torch.tensor(image_features_set1)
image_features_set2 = torch.tensor(image_features_set2)

# Flatten the 2D arrays
image_features_set1_flat = image_features_set1.flatten()
image_features_set2_flat = image_features_set2.flatten()

# KS test for flattened arrays
ks_statistic, p_value = ks_2samp(image_features_set1_flat, image_features_set2_flat)

print("KS Statistic for Each Feature Dimension:")
print(ks_statistic)

# Choose a significance level (alpha) for the test
alpha = 0.05

# Find the maximum KS statistic across all feature dimensions
max_ks_statistic = np.max(ks_statistic)

# Compare with the threshold based on the significance level
threshold = max_ks_statistic / (np.sqrt(image_features_set1.shape[0] + image_features_set2.shape[0]) * (1.0 + 0.12 + 0.11 / (np.sqrt(image_features_set1.shape[0] + image_features_set2.shape[0]))))
print(f"\nThreshold based on Significance Level ({alpha}): {threshold.item()}")

# Compare with the threshold and detect data drift
if max_ks_statistic > threshold:
    print("\nData Drifting Detected!")
else:
    print("\nNo Significant Drift Detected.")
