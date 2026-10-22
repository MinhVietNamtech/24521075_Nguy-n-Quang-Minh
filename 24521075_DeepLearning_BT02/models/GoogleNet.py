import torch
from torch import nn

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(InceptionBlock, self).__init__()
        self.first_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=(1,1)),
            nn.ReLU()
        )
        self.second_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=(3,3), padding=1),
            nn.ReLU()
        )
        self.third_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=(5,5), padding=2),
            nn.ReLU()
        )
        self.fourth_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=1,ceil_mode=True),
            nn.Conv2d(in_channels=in_channels,out_channels=c4, kernel_size=(1,1)),
            nn.ReLU()
        )

    def forward(self, x):
        first = self.first_branch(x)
        second = self.second_branch(x)
        third = self.third_branch(x)
        fourth = self.fourth_branch(x)
        return torch.cat((first, second, third, fourth), dim=1)

class GoogleNet(nn.Module):
    def __init__(self, in_channels, out_channels=1000):
        super(GoogleNet, self).__init__()
        self.in_channels = in_channels

        self.p1 = self.phase_1()
        self.p2 = self.phase_2()
        self.p3 = self.phase_3()
        self.p4 = self.phase_4()
        self.p5 = self.phase_5()
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(in_features=1024, out_features=out_channels)
        self.softmax = nn.Softmax()

    def phase_1(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=(7,7), stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=2, ceil_mode=True)            
        )

    def phase_2(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=2,ceil_mode=True)
        )

    def phase_3(self):
        return nn.Sequential(
            InceptionBlock(in_channels=192, c1=64, c2=(96,128), c3=(16,32), c4=32),
            InceptionBlock(in_channels=256, c1=128, c2=(128,192), c3=(32,96), c4=64),
            nn.MaxPool2d(kernel_size=(3,3), stride=2, ceil_mode=True)
        )

    def phase_4(self):
        return nn.Sequential(
            InceptionBlock(in_channels=480, c1=192, c2=(96,208), c3=(16,48), c4=64),
            InceptionBlock(in_channels=512, c1=160, c2=(112,224), c3=(24,64), c4=64),
            InceptionBlock(in_channels=512, c1=128, c2=(128,256), c3=(24,64), c4=64),
            InceptionBlock(in_channels=512, c1=112, c2=(144,288), c3=(32,64), c4=64),
            InceptionBlock(in_channels=528, c1=256, c2=(160,320), c3=(32,128), c4=128),
            nn.MaxPool2d(kernel_size=(3,3), stride=2, ceil_mode=True)
        )

    def phase_5(self):
        return nn.Sequential(
            InceptionBlock(in_channels=832, c1=256, c2=(160,320), c3=(32,128), c4=128),
            InceptionBlock(in_channels=832, c1=384, c2=(192,384), c3=(48,128), c4=128),
            nn.AvgPool2d(kernel_size=(6,6), stride=1)
        )
    
    def forward(self, x):
        out = self.p1(x)
        out = self.p2(out)
        out = self.p3(out)
        out = self.p4(out)
        out = self.p5(out)
        out = out.squeeze()
        out = self.dropout(out)
        return self.softmax(self.linear(out))

