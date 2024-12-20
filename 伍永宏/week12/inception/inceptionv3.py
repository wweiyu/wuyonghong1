import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride = 1,padding=0,bn:bool=True):
        super().__init__()
        self.is_bn = True if bn else False
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        if self.is_bn:
            x = self.bn(x)
        return self.relu(x)


class InceptionA(nn.Module):
    def __init__(self,in_channel,c1x1_out,c3x3_outs,c5xd_outs,pool_out):
        super().__init__()
        c3x3_in, c3x3_out = c3x3_outs
        c5x5_in, c5x5_out1, c5x5_out2 = c5xd_outs
        self.branch1x1 = ConvBlock(in_channel,c1x1_out,1,1)

        self.branch3x3 = nn.Sequential(ConvBlock(in_channel,c3x3_in,1,1),
                                       ConvBlock(c3x3_in,c3x3_out,3,1,1))

        self.branch5x5 = nn.Sequential(ConvBlock(in_channel,c5x5_in,1,1),
                                       ConvBlock(c5x5_in,c5x5_out1,3,1,padding=1),
                                       ConvBlock(c5x5_out1,c5x5_out2,3,1,padding=1))

        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
                                         ConvBlock(in_channel,pool_out,1,1))

    def forward(self,x):
        branch1 = self.branch1x1(x)
        branch2 = self.branch3x3(x)
        branch3 = self.branch5x5(x)
        branch4 = self.branch_pool(x)
        return torch.cat([branch1,branch2,branch3,branch4],dim=1)


class InceptionB(nn.Module):
    def __init__(self,in_channel,c1x1_out,c7x7_outs,c7x7dbl_outs,pool_out):
        super().__init__()
        c5x5_in1, c5x5_in2, c5x5_out = c7x7_outs
        c7x7_in, c7x7_in1, c7x7_in2, c7x7_in3, c7x7_out = c7x7dbl_outs

        self.branch1x1 = ConvBlock(in_channel,c1x1_out,1,1)

        self.branch7x7 = nn.Sequential(ConvBlock(in_channel,c5x5_in1,1,1),
                                       ConvBlock(c5x5_in1,c5x5_in2,(1,7),padding=(0,3)),
                                       ConvBlock(c5x5_in2,c5x5_out,(7,1),padding=(3,0)))

        self.branch7x7dbl = nn.Sequential(ConvBlock(in_channel,c7x7_in,1,1),
                                       ConvBlock(c7x7_in,c7x7_in1,(1,7),padding=(0,3)),
                                       ConvBlock(c7x7_in1,c7x7_in2,(7,1),padding=(3,0)),
                                       ConvBlock(c7x7_in2,c7x7_in3,(1,7),padding=(0,3)),
                                       ConvBlock(c7x7_in3,c7x7_out,(7,1),padding=(3,0)),)

        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
                                         ConvBlock(in_channel,pool_out,1,1))

    def forward(self, x):
        branch1 = self.branch1x1(x)
        branch2 = self.branch7x7(x)
        branch3 = self.branch7x7dbl(x)
        branch4 = self.branch_pool(x)
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class InceptionC(nn.Module):
    def __init__(self,in_channel,c1x1_out,c3x3_outs,c5x5_outs,pool_out):
        super().__init__()
        c3x3_in1, c3x3_out1, c3x3_out2 = c3x3_outs
        c5x5_in, c5x5_in1, c5x5_out1, c5x5_out2 = c5x5_outs

        self.branch1x1 = ConvBlock(in_channel,c1x1_out,1,1)

        self.branch3x3_0 = ConvBlock(in_channel,c3x3_in1,1,1)
        self.branch3x3_1 = ConvBlock(c3x3_in1,c3x3_out1,(1,3),padding=(0,1))
        self.branch3x3_2 = ConvBlock(c3x3_in1,c3x3_out2,(3,1),padding=(1,0))

        self.branch5x5_0 = nn.Sequential(ConvBlock(in_channel,c5x5_in,1,1),
                                         ConvBlock(c5x5_in,c5x5_in1,3,1,padding=1))

        self.branch5x5_1 = ConvBlock(c5x5_in1,c5x5_out1,(1,3),padding=(0,1))
        self.branch5x5_2 = ConvBlock(c5x5_in1,c5x5_out2,(3,1),padding=(1,0))

        self.branch_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                         ConvBlock(in_channel, pool_out, 1, 1))

    def forward(self,x):
        branch1 = self.branch1x1(x)

        branch2_0 = self.branch3x3_0(x)
        branch2_1 = self.branch3x3_1(branch2_0)
        branch2_2 = self.branch3x3_2(branch2_0)

        branch3_0 = self.branch5x5_0(x)
        branch3_1 = self.branch5x5_1(branch3_0)
        branch3_2 = self.branch5x5_2(branch3_0)

        branch_pool = self.branch_pool(x)

        return torch.cat([branch1,branch2_1,branch2_2,branch3_1,branch3_2,branch_pool],dim=1)


class InceptionD(nn.Module):
    def __init__(self,in_channel,c3x3_outs,c5xd_outs,stride):
        super().__init__()
        c3x3_in, c3x3_out = c3x3_outs
        c5x5_in, c5x5_out1, c5x5_out2 = c5xd_outs

        self.branch3x3 = nn.Sequential(ConvBlock(in_channel,c3x3_in,1,1),
                                       ConvBlock(c3x3_in,c3x3_out,3,stride=stride))

        self.branch5x5 = nn.Sequential(ConvBlock(in_channel,c5x5_in,1,1),
                                       ConvBlock(c5x5_in,c5x5_out1,3,1,padding=1),
                                       ConvBlock(c5x5_out1,c5x5_out2,3,stride=stride))

        self.branch_pool = nn.MaxPool2d(kernel_size=3,stride=stride)

    def forward(self,x):
        branch2 = self.branch3x3(x)
        branch3 = self.branch5x5(x)
        branch4 = self.branch_pool(x)
        return torch.cat([branch2, branch3, branch4], dim=1)


class InceptionV3(nn.Module):
    def __init__(self,in_channel,classes):
        super(InceptionV3,self).__init__()
        # self.bn1 = ConvBlock(in_channel,32,3,2)
        # self.bn2 = ConvBlock(32,32,3,1)
        # self.bn3 = ConvBlock(32,64,3,1,padding=1)
        # '64*73*73'
        # self.pool1 = nn.MaxPool2d(3,2)
        # '80*71*71'
        # self.bn4 = ConvBlock(64,80,3,1)
        # '192*35*35'
        # self.bn5 = ConvBlock(80,192,3,2)
        # '288*35*35'
        # self.bn6 = ConvBlock(192,288,3,1,padding=1)
        self.conv1 = nn.Sequential(ConvBlock(in_channel,32,3,2),
                                   ConvBlock(32, 32, 3, 1),
                                   ConvBlock(32, 64, 3, 1, padding=1),
                                   nn.MaxPool2d(3, 2),
                                   ConvBlock(64, 80, 3, 1),
                                   ConvBlock(80, 192, 3, 2),
                                   ConvBlock(192, 288, 3, 1, padding=1))

        '35*35*288 -> 17*17*768'
        '3个inception'
        self.inception_1 = nn.Sequential(InceptionA(288,64,[48,64],[64,96,96],64),
                                         InceptionA(288,64,[48,64],[64,96,96],64),
                                         InceptionD(288,[384,384],[64,96,96],stride=2))

        '768*17*17 -> 1280*8*8'

        '5个inception'
        self.inception_2 = nn.Sequential(InceptionB(768, 192, [128, 128, 192], [128, 128, 128, 128, 192], 192),
                                     InceptionB(768, 192, [128, 128, 192], [128, 128, 128, 128, 192], 192),
                                     InceptionB(768, 192, [128, 128, 192], [128, 128, 128, 128, 192], 192),
                                     InceptionB(768, 192, [192, 192, 192], [192, 192, 192, 192, 192], 192),
                                     InceptionD(768, [192, 320], [192, 192, 192], stride=2))


        '2个inception'
        '1280*8*8 -> 2048*8*8'
        self.inception_3 = nn.Sequential(InceptionC(1280,320,[384,384,384],[448,384,384,384],192),
                                         InceptionC(2048,320,[384,384,384],[448,384,384,384],192))

        self.avg_pool = nn.AvgPool2d(kernel_size=(8,8))

        '2048->1000'
        self.fc1 = nn.Sequential(nn.Flatten(start_dim=1),
                                 nn.Linear(2048,classes))

        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.inception_1(x)
        x = self.inception_2(x)
        x = self.inception_3(x)
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    inputs = torch.randn(4,3,299,299)
    net = InceptionV3(3,1000)
    outputs = net(inputs)
