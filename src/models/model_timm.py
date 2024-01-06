import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import timm
from dataloader import mnist

model_name = 'resnet18'  # Use a lighter model for faster training
num_classes = 4  # Replace with the number of classes in your dataset

# Load the pretrained model
model = timm.create_model(model_name, pretrained=True, in_chans=1)

# Modify the classifier (assuming the last layer is named 'fc')
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load your datasets
train_dataset, test_dataset = mnist()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 4  # Define the number of training epochs

for epoch in range(num_epochs):
    # Training Phase
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}")

    # Evaluation Phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set after epoch {epoch+1}: {accuracy}%')

torch.save(model.state_dict(), 'model.pt')
