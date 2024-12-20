import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride = 1,padding=0,bias = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding,bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class DepthConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride = 1,padding=0,groups = None):
        super().__init__()
        if groups is None:
            groups = in_channel
        self.depwise_conv = nn.Sequential(nn.Conv2d(in_channel,in_channel,kernel_size,stride,padding,groups=groups),
                            nn.BatchNorm2d(in_channel),
                            nn.ReLU6())
        self.conv = nn.Sequential(nn.Conv2d(groups,out_channel,1,1,bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU6())

    def forward(self,x):
        x = self.depwise_conv(x)
        x = self.conv(x)
        return x



class MobileNet(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        '3*224*224 -> 32*112*112'
        self.conv1 = ConvBlock(in_channel,32,kernel_size=3,stride=2,padding=1)
        '32*112*112 -> 64*112*112'
        '64*112*112 -> 128*56*56'
        '128*56*56 -> 128*56*56'
        '128*56*56 -> 256*28*28'
        '256*28*28 -> 256*28*28'
        '256*28*28 -> 512*14*14'
        self.dep_conv1 = nn.Sequential(DepthConvBlock(32,64,kernel_size=3,stride=1,padding=1),
                                       DepthConvBlock(64,128,kernel_size=3,stride=2,padding=1),
                                       DepthConvBlock(128,128,kernel_size=3,stride=1,padding=1),
                                       DepthConvBlock(128,256,kernel_size=3,stride=2,padding=1),
                                       DepthConvBlock(256,256,kernel_size=3,stride=1,padding=1),
                                       DepthConvBlock(256,512,kernel_size=3,stride=2,padding=1))

        '5个 512*14*14 -> 512*14*14'
        self.dep_conv2 = nn.Sequential(DepthConvBlock(512,512,kernel_size=3,stride=1,padding=1),
                                       DepthConvBlock(512, 512, kernel_size=3, stride=1, padding=1),
                                       DepthConvBlock(512, 512, kernel_size=3, stride=1, padding=1),
                                       DepthConvBlock(512, 512, kernel_size=3, stride=1, padding=1),
                                       DepthConvBlock(512, 512, kernel_size=3, stride=1, padding=1))

        '512*14*14 -> 1024*7*7'
        '1024*7*7 -> 1024*7*7'
        self.dep_conv3 = nn.Sequential(DepthConvBlock(512,1024,kernel_size=3,stride=2,padding=1),
                                       DepthConvBlock(1024,1024,kernel_size=3,stride=1,padding=1))

        self.avg_pool = nn.AvgPool2d(kernel_size=(7,7))

        self.fc1 = nn.Sequential(nn.Flatten(start_dim=1),
                                 nn.Linear(1024,out_channel))

        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.dep_conv1(x)
        x = self.dep_conv2(x)
        x = self.dep_conv3(x)
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    inputs = torch.randn(5,3,224,224)
    net = MobileNet(3,1000)
    net(inputs)
