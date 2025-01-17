import torch
import torch.nn as nn
from collections import OrderedDict

def mask_layers(block,no_relu_layers):
    layers = []
    for layer_name,vals in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=vals[0],stride=vals[1],padding=vals[2])
            layers.append([layer_name,layer])
        else:
            conv = nn.Conv2d(vals[0],vals[1], kernel_size=vals[2],stride=vals[3],padding=vals[4])
            layers.append((layer_name,conv))
            if layer_name not in no_relu_layers:
                relu = nn.ReLU(inplace=True)
                layers.append(('relu_'+layer_name, relu))
    return nn.Sequential(OrderedDict(layers))


class bodypose_model(nn.Module):
    def __init__(self):
        super().__init__()
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',
                          'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',
                          'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        'backbone'
        block0 = OrderedDict([
                      ('conv1_1', [3, 64, 3, 1, 1]),
                      ('conv1_2', [64, 64, 3, 1, 1]),
                      ('pool1_stage1', [2, 2, 0]),
                      ('conv2_1', [64, 128, 3, 1, 1]),
                      ('conv2_2', [128, 128, 3, 1, 1]),
                      ('pool2_stage1', [2, 2, 0]),
                      ('conv3_1', [128, 256, 3, 1, 1]),
                      ('conv3_2', [256, 256, 3, 1, 1]),
                      ('conv3_3', [256, 256, 3, 1, 1]),
                      ('conv3_4', [256, 256, 3, 1, 1]),
                      ('pool3_stage1', [2, 2, 0]),
                      ('conv4_1', [256, 512, 3, 1, 1]),
                      ('conv4_2', [512, 512, 3, 1, 1]),
                      ('conv4_3_CPM', [512, 256, 3, 1, 1]),
                      ('conv4_4_CPM', [256, 128, 3, 1, 1])
                  ])

        blocks = {}
        blocks['block1_1'] = OrderedDict([('conv5_1_CPM_L1',[128,128,3,1,1]),
                                ('conv5_2_CPM_L1', [128, 128, 3, 1, 1]),
                                ('conv5_3_CPM_L1', [128, 128, 3, 1, 1]),
                                ('conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
                                ('conv5_5_CPM_L1', [512, 38, 1, 1, 0])
                                ])

        blocks['block1_2'] = OrderedDict([('conv5_1_CPM_L2',[128,128,3,1,1]),
                                ('conv5_2_CPM_L2', [128, 128, 3, 1, 1]),
                                ('conv5_3_CPM_L2', [128, 128, 3, 1, 1]),
                                ('conv5_4_CPM_L2', [128, 512, 1, 1, 0]),
                                ('conv5_5_CPM_L2', [512, 19, 1, 1, 0])
                                ])

        for i in range(2, 7):
            blocks['block%d_1' % i] = OrderedDict([
                ('Mconv1_stage%d_L1' % i, [185, 128, 7, 1, 3]),
                ('Mconv2_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv3_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv4_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv5_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv6_stage%d_L1' % i, [128, 128, 1, 1, 0]),
                ('Mconv7_stage%d_L1' % i, [128, 38, 1, 1, 0])
            ])

            blocks['block%d_2' % i] = OrderedDict([
                ('Mconv1_stage%d_L2' % i, [185, 128, 7, 1, 3]),
                ('Mconv2_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv3_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv4_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv5_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv6_stage%d_L2' % i, [128, 128, 1, 1, 0]),
                ('Mconv7_stage%d_L2' % i, [128, 19, 1, 1, 0])])

        self.model0 = mask_layers(block0, no_relu_layers)
        self.model1_1 = mask_layers(blocks['block1_1'],no_relu_layers)
        self.model1_2 = mask_layers(blocks['block1_2'],no_relu_layers)

        self.model2_1 = mask_layers(blocks['block2_1'], no_relu_layers)
        self.model2_2 = mask_layers(blocks['block2_2'], no_relu_layers)

        self.model3_1 = mask_layers(blocks['block3_1'],no_relu_layers)
        self.model3_2 = mask_layers(blocks['block3_2'],no_relu_layers)

        self.model4_1 = mask_layers(blocks['block4_1'],no_relu_layers)
        self.model4_2 = mask_layers(blocks['block4_2'],no_relu_layers)

        self.model5_1 = mask_layers(blocks['block5_1'], no_relu_layers)
        self.model5_2 = mask_layers(blocks['block5_2'], no_relu_layers)

        self.model6_1 = mask_layers(blocks['block6_1'], no_relu_layers)
        self.model6_2 = mask_layers(blocks['block6_2'], no_relu_layers)

    def forward(self,x):
        x = self.model0(x)
        out_1 = self.model1_1(x)
        out_2 = self.model1_2(x)
        in_0 = torch.cat([out_1,out_2,x],dim=1)
        out_1 = self.model2_1(in_0)
        out_2 = self.model2_2(in_0)

        in_0 = torch.cat([out_1,out_2,x],dim=1)
        out_1 = self.model3_1(in_0)
        out_2 = self.model3_2(in_0)

        in_0 = torch.cat([out_1, out_2,x], dim=1)
        out_1 = self.model4_1(in_0)
        out_2 = self.model4_2(in_0)

        in_0 = torch.cat([out_1, out_2,x], dim=1)
        out_1 = self.model5_1(in_0)
        out_2 = self.model5_2(in_0)

        in_0 = torch.cat([out_1, out_2,x], dim=1)
        out_1 = self.model6_1(in_0)
        out_2 = self.model6_2(in_0)

        return out_1,out_2

if __name__ == "__main__":
    model =  bodypose_model()
    from torchsummary import summary
    summary(model,(3,224,224))
