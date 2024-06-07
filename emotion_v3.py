import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image

# Custom loader to convert images to grayscale
def gray_loader(path):
    return Image.open(path).convert('L')

# Data Preparation
train_dir = 'train'
test_dir = 'test'

transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.Resize(48),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define the neural network with the best hyperparameters
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.2)  # Increased dropout

        self.conv2 = nn.Conv2d(96, 192, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(192)
        self.dropout2 = nn.Dropout(0.3)  # Increased dropout

        self.conv3 = nn.Conv2d(192, 256, kernel_size=5, padding=2)  # Increased filters
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.4)  # Increased dropout

        self.fc1 = nn.Linear(256 * 6 * 6, 512)  # Increased neurons
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.5)  # Increased dropout

        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == '__main__':
    trainset = ImageFolder(root=train_dir, transform=transform_train, loader=gray_loader)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    testset = ImageFolder(root=test_dir, transform=transform_test, loader=gray_loader)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    classes = trainset.classes
    num_classes = len(classes)
    print("Training classes:", trainset.class_to_idx)
    print("Testing classes:", testset.class_to_idx)
    print("Number of classes:", num_classes)

    net = Net(num_classes)

    # Check if CUDA is available and move the model to GPU if it is
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2, min_lr=0.001)

    # Train the network
    num_epochs = 15
    early_stopping_patience = 5
    early_stopping_counter = 0
    best_val_loss = float('inf')

    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        net.train()  # Set the model to training mode
        for data in trainloader:
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move to GPU if available

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # accumulate the loss
            running_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = running_loss / len(trainloader)
        train_loss_history.append(avg_loss)

        # Calculate training accuracy
        net.eval()  # Set the model to evaluation mode
        train_accuracy = calculate_accuracy(trainloader, net)
        train_acc_history.append(train_accuracy)

        # Calculate test accuracy and loss
        val_loss = 0.0
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

        avg_val_loss = val_loss / len(testloader)
        val_loss_history.append(avg_val_loss)
        val_accuracy = calculate_accuracy(testloader, net)
        val_acc_history.append(val_accuracy)

        # Print epoch statistics
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, '
              f'Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%, '
              f'Validation Loss: {avg_val_loss:.4f}')

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Check early stopping condition
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            best_model_state_dict = net.state_dict()
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping")
                break

    # Restore the best model
    net.load_state_dict(best_model_state_dict)

    # Save the Model
    torch.save(net.state_dict(), 'emotion_v3.pth')

    # Download the Model
    # files.download('emotion_v3.pth')

    # Output Training Summary
    print("Training Summary after", epoch + 1, "epochs:")
    print(f"Training Accuracy: {train_acc_history[-1]:.4f}")
    print(f"Validation Accuracy: {val_acc_history[-1]:.4f}")
    print(f"Training Loss: {train_loss_history[-1]:.4f}")
    print(f"Validation Loss: {val_loss_history[-1]:.4f}")

    # Plot training and validation accuracy and loss
    epochs_range = range(1, len(train_acc_history) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc_history, label='Training Accuracy')
    plt.plot(epochs_range, val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss_history, label='Training Loss')
    plt.plot(epochs_range, val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()
