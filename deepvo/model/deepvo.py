import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size, stride, padding, dropout=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout)
    )


class DeeoVO(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv2 = conv(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.conv3 = conv(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.conv3_1 = conv(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = conv(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv4_1 = conv(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = conv(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv5_1 = conv(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = conv(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.rnn = nn.LSTM(input_size=20 * 6 * 1024, hidden_size=100, num_layers=2)
        self.fc = nn.Linear(in_features=100, out_features=6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv6(x)
        x = x.view(x.size(0), 20 * 6 * 1024)
        out, hidden = self.rnn(x)

        out = self.fc(out)
        return out
