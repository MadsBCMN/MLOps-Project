import torch

def mnist():
    """Return train and test dataloaders for your MRI dataset."""
    # Load training and testing data
    train_data = torch.load(r"/Users/helenehjort/Library/Mobile Documents/com~apple~CloudDocs/Human-centered AI/9. semester/02476 Machine Learning Operations/Project/MLOps-Project/data/proces/training_images.pt").float()
    train_labels = torch.load(r"/Users/helenehjort/Library/Mobile Documents/com~apple~CloudDocs/Human-centered AI/9. semester/02476 Machine Learning Operations/Project/MLOps-Project/data/proces/training_labels.pt")
    test_data = torch.load(r"/Users/helenehjort/Library/Mobile Documents/com~apple~CloudDocs/Human-centered AI/9. semester/02476 Machine Learning Operations/Project/MLOps-Project/data/proces/testing_images.pt").float()
    test_labels = torch.load(r"/Users/helenehjort/Library/Mobile Documents/com~apple~CloudDocs/Human-centered AI/9. semester/02476 Machine Learning Operations/Project/MLOps-Project/data/proces/testing_labels.pt")

    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)

    # Unsqueeze the data tensors to add a channel dimension
    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    print(train_data.shape)
    print(test_data.shape)

    # Create TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

    return train_dataset, test_dataset

