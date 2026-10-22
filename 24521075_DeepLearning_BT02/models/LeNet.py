import torch 
from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=(5, 5),
            stride=1,
            padding=(2, 2)
        )
        self.pool1 = nn.AvgPool2d(
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0)
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=(5, 5),
            padding=0
        )
        self.pool2 = nn.AvgPool2d(
            stride=(2, 2),
            kernel_size=(2, 2),
            padding=(0, 0)
        )
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=(5, 5),
            padding=(0, 0)
        )
        self.fc = nn.Linear(
            in_features=120,
            out_features=84
        )
        self.output = nn.Linear(
            in_features=84,
            out_features=10
        )

    def forward(self, images: torch.Tensor):
        images = images.unsqueeze(1)
        features = F.sigmoid(self.conv1(images))
        features = self.pool1(features)

        features = F.sigmoid(self.conv2(features))
        features = self.pool2(features)

        features = F.sigmoid(self.conv3(features))
        features = features.squeeze(-1).squeeze(-1)                                                                                                                                                                    
        
        features = F.sigmoid(self.fc(features))
        logits = self.output(features)

        return logits

