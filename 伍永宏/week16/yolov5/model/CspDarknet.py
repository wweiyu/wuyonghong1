import torch
import torch.nn as nn
from torch.nn.modules import Conv2d,BatchNorm2d,MaxPool2d
import torchsummary

def autopad(kernel,padding = None):
    if padding is None:
        return kernel//2 if isinstance(kernel,int) else [x//2 for x in kernel]
    return padding


class SiLU(nn.Module):
    def forward(self,x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    def __init__(self,c1,c2,kernel=1,stride=1,padding = None,act = True):
        super().__init__()
        self.conv = Conv2d(c1,c2,kernel,stride,padding=autopad(kernel,padding),bias=False)
        self.bn = BatchNorm2d(c2,eps=0.001, momentum=0.03)
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self,x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self,c1,c2,shortcut=True,e = 0.5):
        super().__init__()
        c_mid = int(c2*e)
        self.cv1 = Conv(c1,c_mid,1,1)
        self.cv2 = Conv(c_mid,c2,3,1)
        self.add = shortcut and c1 == c2

    def forward(self,x):
        return x + self.cv2(self.cv1(x)) if self.add else  self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self,c1,c2,n=1,shortcut=True,e = 0.5):
        super().__init__()
        c_mid = int(c2 * e)
        self.cv1 = Conv(c1,c_mid,1,1)
        self.cv2 = Conv(c1,c_mid,1,1)
        self.cv3 = Conv(c_mid*2,c2,1,1)
        self.m = nn.Sequential(*[Bottleneck(c_mid,c_mid,shortcut,e=1.0) for _ in range(n)])

    def forward(self,x):
        return  self.cv3(torch.concat([self.m(self.cv1(x)),self.cv2(x)],dim=1))

class SPFF(nn.Module):
    def __init__(self,c1,c2,kernel = 5):
        super().__init__()
        c_mid = c2 // 2
        self.cv1 = Conv(c1,c_mid,1,1)
        self.cv2 = Conv(4*c_mid,c2,1,1)
        self.m = MaxPool2d(kernel_size=kernel,padding=kernel//2,stride=1)

    def forward(self,x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.concat([x,y1,y2,self.m(y2)],dim=1))

class CSPDarknet(nn.Module):

    def __init__(self,base_channels,base_depth,phi, pretrained):
        super().__init__()
        ' 640, 640, 3 -> 320, 320, 64'
        self.stem = Conv(3,base_channels,6,2,2)

        ' 320, 320, 64 -> 128 160 160'
        self.dark2 = nn.Sequential(
            Conv(base_channels,base_channels*2,3,2),
            C3(base_channels*2,base_channels*2,base_depth)
        )
        '128 160 160 -> 256 80 80'
        'out1'
        self.dark3 = nn.Sequential(
            Conv(base_channels*2, base_channels * 4, 3, 2),
            C3(base_channels * 4, base_channels * 4, base_depth*2)
        )

        '256 80 80-> 512 40 40'
        'out2'
        self.dark4 = nn.Sequential(
            Conv(base_channels*4,base_channels*8,3,2),
            C3(base_channels * 8, base_channels * 8, base_depth * 3)
        )

        '512 40 40 -> 1024 20 20'
        'out3'
        self.dark5 = nn.Sequential(
            Conv(base_channels*8,base_channels*16,3,2),
            C3(base_channels * 16, base_channels * 16, base_depth),
            SPFF(base_channels * 16,base_channels * 16)
        )

        if pretrained:
            backbone = "cspdarknet_" + phi
            url = {
                "cspdarknet_n": 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_n_v6.1_backbone.pth',
                "cspdarknet_s": 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_s_v6.1_backbone.pth',
                'cspdarknet_m': 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_m_v6.1_backbone.pth',
                'cspdarknet_l': 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_l_v6.1_backbone.pth',
                'cspdarknet_x': 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_x_v6.1_backbone.pth',
            }[backbone]
            checkpoint = torch.hub.load_state_dict_from_url(url=url,model_dir='../model_data',map_location='cpu')
            self.load_state_dict(checkpoint,strict=False)
            print("Load weights from ", url.split('/')[-1])

    def forward(self,x):
        x = self.stem(x)
        x = self.dark2(x)
        out1 = self.dark3(x)
        out2 = self.dark4(out1)
        out3 = self.dark5(out2)
        return out1,out2,out3


if __name__ == "__main__":
    net = CSPDarknet(64,1,'s',False)
    from torchsummary import summary
    summary(net,(3,640,640))
