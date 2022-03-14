import torch.nn as nn


class DANGERDIST_CNN(nn.Module):
    def __init__(self, N=1, size=16 * 4 * 4):
        super(DANGERDIST_CNN, self).__init__()
        self.size = size
        self.sigmoid = nn.Sigmoid()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),     # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),    # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),     # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, N),
        )

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x


class DANGERORDR_CNN(nn.Module):
    def __init__(self, N=1, size=16 * 4 * 4):
        super(DANGERORDR_CNN, self).__init__()
        self.size = size
        self.sigmoid = nn.Sigmoid()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),     # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),    # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),     # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, N),
        )

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x
