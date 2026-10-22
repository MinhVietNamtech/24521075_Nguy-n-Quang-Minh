import torch 
from torch import nn
import torch.nn.functional as F

class ResBlock_basic(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock_basic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if (in_channels!=out_channels or stride!=1) else nn.Identity()

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, out_channels=21):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False)
        self.in_channels = 64
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1,ceil_mode=True)
        self.relu = nn.ReLU()

        self.phase2 = self._make_layer(ResBlock_basic,64,2,stride=1)
        self.phase3 = self._make_layer(ResBlock_basic,128,2,stride=2)
        self.phase4 = self._make_layer(ResBlock_basic,256,2,stride=2)
        self.phase5 = self._make_layer(ResBlock_basic,512,2,stride=2)[:-1] # Ignore the last MaxPool layer

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,out_channels)
        # self.softmax = nn.Softmax()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        strides = [stride] + [1] * (num_blocks-1)
        for s in strides:
            layers.extend([block(self.in_channels, out_channels, s),
                           self.pool])
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.pool(self.relu(self.bn1(self.conv1(x))))
        out = self.phase2(out)
        out = self.phase3(out)
        out = self.phase4(out)
        out = self.phase5(out)

        out = self.global_pool(out)
        out = out.view(out.shape[0],-1)
        out = self.fc(out)
        return out