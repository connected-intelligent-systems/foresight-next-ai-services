import torch
import torch.nn.functional as F
from torch import nn


class NILMTKModel(nn.Module):
    def __init__(self, window_size, drop_out):
        super(NILMTKModel, self).__init__()
        self.original_len = window_size
        self.dropout_rate = drop_out

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=30, kernel_size=10, stride=2)

        self.conv2 = nn.Conv1d(in_channels=30, out_channels=30, kernel_size=8, stride=2)
        self.conv3 = nn.Conv1d(in_channels=30, out_channels=40, kernel_size=6, stride=1)

        self.conv4 = nn.Conv1d(in_channels=40, out_channels=50, kernel_size=5, stride=1)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)
        self.conv5 = nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5, stride=1)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)

        self.flatten = nn.Flatten()

        tmp_x = torch.randn(1, 1, self.original_len)
        tmp_x = self.flatten(
            self.dropout2(self.conv5(self.dropout1(self.conv4(self.conv3(self.conv2(self.conv1(tmp_x))))))))
        num_flattened = tmp_x.shape[1]

        self.fc1 = nn.Linear(in_features=num_flattened, out_features=1024)
        self.dropout3 = nn.Dropout(p=self.dropout_rate)
        self.fc2 = nn.Linear(in_features=1024, out_features=self.original_len)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout1(x)
        x = F.relu(self.conv5(x))
        x = self.dropout2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        x = x.unsqueeze(1)
        return x

    def summary(self):
        print("Model Summary:")
        print("Number of parameters:", sum(p.numel() for p in self.parameters()))
        for layer in self.children():
            print(layer)
