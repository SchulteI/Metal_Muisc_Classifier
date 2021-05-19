import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import mysql.connector


database = mysql.connector.connect(
    host='localhost',
    user='metallibrary',
    passwd='password',
    database='metal_features'
)

device = torch.device('cpu')

# Create model to identify metal sub-genres based off mfccs
class MfccConvNet(nn.Module):
    def __init__(self):
        super(MfccConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (3, 3))
        self.pool1 = nn.MaxPool2d((3, 3), stride=(2, 2), padding='same')
        self.batch_norm1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, (3, 3))
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2), padding='same')
        self.conv3 = nn.Conv2d(16, 32, (3, 3))
        self.batch_norm3 = nn.BatchNorm2d(32)
        # First linear input value is very likely wrong, further calculations needed
        self.fc1 = nn.Linear(32*8*8, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 50)
        self.fc4 = nn.Linear(50, 7)

    def forward(self, u):
        u = self.pool1(F.leaky_relu(self.conv1(u)))
        u = self.batch_norm1(u)
        u = self.pool1(F.leaky_relu(self.conv2(u)))
        u = self.batch_norm2(u)
        u = self.pool2(F.leaky_relu(self.conv3(u)))
        u = self.batch_norm3(u)
        # value of flattened tensor still needs to be adjusted
        u = u.view(-1, 32*8*8)
        u = F.leaky_relu(self.fc1(u))
        u = F.leaky_relu(self.fc2(u))
        u = F.leaky_relu(self.fc3(u))
        u = self.fc4(u)
        return u


model = MfccConvNet().to(device)

def data_preparation():
    # transform data stored as np.array into tensor
    mfcc_cursor = database.cursor()



