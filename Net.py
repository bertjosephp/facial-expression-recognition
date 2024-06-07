import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(num_features=64)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(num_features=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.25)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(num_features=128)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(num_features=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.25)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(num_features=256)
        self.drop3 = nn.Dropout(0.25)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(num_features=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(0.25)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(num_features=512)
        self.drop5 = nn.Dropout(0.25)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(num_features=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop6 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512 * 3 * 3, 1024)
        self.drop7 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 7)

    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = self.drop3(x)
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.drop4(x)

        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = self.drop5(x)
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)
        x = self.drop6(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.drop7(x)
        x = self.fc2(x)
        return x
