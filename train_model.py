import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
from Net import Net


# Set device
def set_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Define transformations
def initialize_transforms():
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomResizedCrop(48, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return transform_train, transform_test


# Load datasets
def load_datasets(transform_train, transform_test):
    trainset = torchvision.datasets.ImageFolder(root='./fer2013/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(root='./fer2013/test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    return trainloader, testloader


def train_model(net, trainloader, optimizer, criterion, device):
    net.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(trainloader)
    train_accuracy = 100 * correct / total

    return train_loss, train_accuracy


def test_model(net ,testloader, criterion, device):
    net.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(testloader)
    test_accuracy = 100 * correct / total

    return test_loss, test_accuracy


def save_checkpoint(state_dict, filename='model_checkpoint.pth'):
    torch.save(state_dict, filename)


def plot_and_save_stats(train_losses, train_accuracies, test_losses, test_accuracies, epoch):
    epochs = range(1, epoch + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'r', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 1)
    plt.plot(epochs, test_losses, 'r', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, test_accuracies, 'g', label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(f'./plots/training_plot_epoch_{epoch}.png')
    plt.close()


def main():
    device = set_device()
    print("Using device:", device)

    transform_train, transform_test = initialize_transforms()
    trainloader, testloader = load_datasets(transform_train, transform_test)
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)

    train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []

    print('Starting training. . .')

    for epoch in range(100):
        start_time = time.time()

        train_loss, train_accuracy = train_model(net, trainloader, optimizer, criterion, device)
        test_loss, test_accuracy = test_model(net, testloader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        save_checkpoint(net.state_dict())
        plot_and_save_stats(train_losses, train_accuracies, test_losses, test_accuracies, epoch)

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.2f}%, Time: {epoch_time:.2f}s')

    print('Finished Training')


if __name__ == '__main__':
    main()
