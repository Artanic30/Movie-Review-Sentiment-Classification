import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
import pandas as pd
from ast import literal_eval
import numpy as np


class Net(nn.Module):
    def __init__(self, vector_dimension=270):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(vector_dimension, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 20)
        self.fc4 = nn.Linear(vector_dimension + 100 + 50 + 20, 1)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        x5 = torch.cat((x, x1, x2, x3), dim=0)
        x = self.fc4(x5)
        return x

    def predict(self, x):
        if x.float() >= 0.5:
            return 1
        else:
            return 0
