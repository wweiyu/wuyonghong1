import torch
from torch.nn import Conv2d,MaxPool2d,BatchNorm2d,ReLU,ConvTranspose2d
import torch.nn.functional as F

class Double_Conv(torch.nn.Module):
    def __init__(self,in_channel, out_channel):
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            Conv2d(in_channel, out_channel, kernel_size=3, stride=1,padding=1),
            BatchNorm2d(out_channel),
            ReLU(inplace=True),
            Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_channel),
            ReLU(inplace=True)
        )

    def forward(self,x):
        return self.double_conv(x)


class Unet_Down(torch.nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            MaxPool2d(kernel_size=2, stride=2),
            Double_Conv(in_channel, out_channel)
        )

    def forward(self,x):
        return self.maxpool_conv(x)

class Unet_Up(torch.nn.Module):
    def __init__(self,in_channel,out_channel,bilinear=True):
        super().__init__()
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2,mode = "bilinear",align_corners = True)
        else:
            self.up = ConvTranspose2d(in_channel//2,in_channel//2,kernel_size =2,stride=2)
        self.conv = Double_Conv(in_channel,out_channel)

    def forward(self,x1,x2):
        x1 = self.up(x1)
        'nchw'
        diffx = torch.tensor(x2.shape[3] - x1.shape[3])
        diffy = torch.tensor(x2.shape[2] - x1.shape[2])
        x1 = F.pad(x1, (diffy // 2, diffy - diffy // 2, diffx // 2, diffx - diffx // 2))
        x = torch.concat([x2,x1],dim=1)
        x = self.conv(x)
        return x

class OutConv(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self,x):
        return self.conv(x)

class Unet(torch.nn.Module):
    def __init__(self,in_channels,class_num,bilinear=True):
        super().__init__()
        self.inc = Double_Conv(in_channels,64)
        self.down1 = Unet_Down(64,128)
        self.down2 = Unet_Down(128,256)
        self.down3 = Unet_Down(256,512)
        self.down4 = Unet_Down(512, 512)

        self.up1 = Unet_Up(1024,256,bilinear)
        self.up2 = Unet_Up(512,128,bilinear)
        self.up3 = Unet_Up(256, 64, bilinear)
        self.up4 = Unet_Up(128, 64, bilinear)

        self.outc = OutConv(64,class_num)

    def forward(self,x):
        x = self.inc(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        out = self.up1(x4,x3)
        out = self.up2(out,x2)
        out = self.up3(out,x1)
        out = self.up4(out,x)
        out = self.outc(out)
        return out

# if __name__ == "__main__":
#     inputs = torch.randn(size=(1,3,572,572))
#     net = Unet(3,2)
#     inputs = torch.randn(size=(1,3,572,572))
#     out = net(inputs)
#     print(out.size())
