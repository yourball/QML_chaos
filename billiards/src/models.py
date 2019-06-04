import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net1(nn.Module):
    def __init__(self, debug=False):
        super(Net1, self).__init__()
        self.debug = debug
        self.conv1 = nn.Conv2d(1, 5, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(5, 3, kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(8, 50)
        self.fc1 = None
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        if self.debug:
            print('in:', x.shape)

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        if self.debug:
            print('>:', x.shape)

        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        if self.debug:
            print('>:', x.shape)
#         x = x.view(-1, 8)
        num_elements = np.product(x.shape[1:])
        x = x.view(-1, num_elements)
        if self.debug:
            print('>:', x.shape)
        if self.fc1 is None:
            self.fc1 = nn.Linear(num_elements, 50)

        x = F.relu(self.fc1(x))
        if self.debug:
            print('>:', x.shape)

        x = F.dropout(x, training=self.training)

        x = self.fc2(x)
        if self.debug:
            print('out:', x.shape)
            assert 1 == 0

        return F.log_softmax(x, dim=1)


class Net2(nn.Module):
    def __init__(self, debug=False):
        super(Net2, self).__init__()
        self.debug = debug
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(5, 4, kernel_size=3, stride=2)
        self.fc1 = None
        self.fc2 = nn.Linear(50, 2)
        self.bn1 = nn.BatchNorm1d(50)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        num_elements = np.product(x.shape[1:])
        x = x.view(-1, num_elements)
        if self.fc1 is None:
            self.fc1 = nn.Linear(num_elements, 50)

        x = self.bn1(F.relu(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


class Net3(nn.Module):
    def __init__(self, debug=False):
        super(Net3, self).__init__()
        self.debug = debug
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1)
        # self.conv2 = nn.Conv2d(4, 5, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(4, 3, kernel_size=2, stride=1)
        self.fc1 = None
        self.fc2 = nn.Linear(200, 2)
        self.bn1 = nn.BatchNorm1d(200)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))

        num_elements = np.product(x.shape[1:])
        x = x.view(-1, num_elements)
        if self.fc1 is None:
            self.fc1 = nn.Linear(num_elements, 200)

        x = self.bn1(F.relu(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


if __name__ == '__main__':
    raise RuntimeError('Not a main file')
