import torch

def dataloader():
    """Return train and test dataloaders for your MRI dataset."""
    # Load training and testing data
    train_data = torch.load(r"../data/proces/training_images.pt").float()
    train_labels = torch.load(r"../data/proces/training_labels.pt")
    test_data = torch.load(r"../data/proces/testing_images.pt").float()
    test_labels = torch.load(r"../data/proces/testing_labels.pt")

    # Unsqueeze the data tensors to add a channel dimension
    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    # Create TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

    return train_dataset, test_dataset

