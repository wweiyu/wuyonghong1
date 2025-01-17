import torch
import torch.nn as nn
from yolov5.model.CspDarknet import *

class YoloV5(nn.Module):
    def __init__(self,anchor_mask,num_class,phi, pretrained = False):
        super().__init__()
        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        dep_mul,wid_mul = depth_dict[phi],width_dict[phi]

        base_depth = max(round(3*dep_mul),1)
        base_channels = int(64*wid_mul)

        self.backbone = CSPDarknet(base_channels,base_depth,phi,pretrained)

        self.upsample = nn.Upsample(scale_factor =2,mode='nearest')

        self.conv_for_feat3 = Conv(base_channels*16,base_channels*8,1,1)
        self.conv3_for_upsample1 = C3(base_channels*16,base_channels*8,base_depth,False)

        self.conv_for_feat2 = Conv(base_channels*8,base_channels*4,1,1)
        self.conv3_for_upsample2 = C3(base_channels*8,base_channels*4,base_depth,False)

        self.down_sample1 = Conv(base_channels*4,base_channels*4,3,2)
        self.conv3_for_downsample1 = C3(base_channels*8,base_channels*8,base_depth,False)

        self.down_sample2 =  Conv(base_channels*8,base_channels*8,3,2)
        self.conv3_for_downsample2 = C3(base_channels*16,base_channels*16,base_depth,False)

        self.yolo_head_P3 = nn.Conv2d(base_channels*4,(num_class+5) * len(anchor_mask[2]),1,1)
        self.yolo_head_P4 = nn.Conv2d(base_channels*8,(num_class+5) * len(anchor_mask[1]),1,1)
        self.yolo_head_P5 = nn.Conv2d(base_channels*16,(num_class+5) * len(anchor_mask[0]),1,1)

    def forward(self,x):
        feat1,feat2,feat3 = self.backbone(x)
        p5 = self.conv_for_feat3(feat3)
        p5_upsample = self.upsample(p5)
        p4 = torch.cat((p5_upsample,feat2),dim=1)
        p4 = self.conv3_for_upsample1(p4)
        p4 = self.conv_for_feat2(p4)

        p4_upsample = self.upsample(p4)
        p3 = torch.concat([p4_upsample,feat1],dim=1)

        p3 = self.conv3_for_upsample2(p3)
        p3_out = self.yolo_head_P3(p3)

        p3 = self.down_sample1(p3)
        p4 = torch.concat([p3,p4],dim=1)
        p4 = self.conv3_for_downsample1(p4)
        p4_out = self.yolo_head_P4(p4)

        p4 = self.down_sample2(p4)
        p5 = torch.cat([p4,p5],dim=1)
        p5 = self.conv3_for_downsample2(p5)
        p5_out = self.yolo_head_P5(p5)

        return p5_out,p4_out,p3_out
