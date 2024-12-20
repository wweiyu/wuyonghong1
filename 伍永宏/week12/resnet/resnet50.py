import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utils as utils

class IdentityBlock(nn.Module):
    def __init__(self,in_channel,fileters):
        super(IdentityBlock,self).__init__()
        filter1,filter2,filter3 = fileters
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channel,filter1,(1,1),stride=1),
                                    nn.BatchNorm2d(filter1),
                                    nn.ReLU()
                                  )
        self.conv_2 =  nn.Sequential(nn.Conv2d(filter1,filter2,(3,3),stride=1,padding=1),
                                    nn.BatchNorm2d(filter2),
                                    nn.ReLU()
                                  )
        self.conv_3 = nn.Sequential(nn.Conv2d(filter2,filter3,(1,1),stride=1),
                                    nn.BatchNorm2d(filter3),
                                  )

    def forward(self,x):
        part_1 = self.conv_1(x)
        part_1 = self.conv_2(part_1)
        part_1 = self.conv_3(part_1)
        return F.relu(part_1 + x)

class ConvBlock(nn.Module):
    def __init__(self,in_channel,fileters,stride):
        super(ConvBlock,self).__init__()
        filter1,filter2,filter3 = fileters
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channel,filter1,(1,1),stride=stride),
                                    nn.BatchNorm2d(filter1),
                                    nn.ReLU()
                                  )
        self.conv_2 =  nn.Sequential(nn.Conv2d(filter1,filter2,(3,3),stride=1,padding=1),
                                    nn.BatchNorm2d(filter2),
                                    nn.ReLU()
                                  )
        self.conv_3 = nn.Sequential(nn.Conv2d(filter2,filter3,(1,1),stride=1),
                                    nn.BatchNorm2d(filter3),
                                  )

        self.shortcut = nn.Sequential(nn.Conv2d(in_channel,filter3,(1,1),stride=stride),
                                      nn.BatchNorm2d(filter3))

    def forward(self,x):
        part_1 = self.conv_1(x)
        part_1 = self.conv_2(part_1)
        part_1 = self.conv_3(part_1)
        part_2 = self.shortcut(x)
        return F.relu(part_1 + part_2)


class Resnet50(nn.Module):
    def __init__(self,in_channel, classes):
        super(Resnet50,self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channel,64,(7,7),padding=3,stride=2),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(3,3),stride=2))

        self.block1 = nn.Sequential(ConvBlock(64,[64,64,256],stride=1),
                                    IdentityBlock(256,[64,64,256]),
                                    IdentityBlock(256,[64,64,256]))

        self.block2 = nn.Sequential(ConvBlock(256,[128,128,512],stride=2),
                                    IdentityBlock(512,[128,128,512]),
                                    IdentityBlock(512,[128,128,512]),
                                    IdentityBlock(512,[128,128,512]))

        self.block3 = nn.Sequential(ConvBlock(512, [256,256,1024], stride=2),
                                    IdentityBlock(1024, [256, 256, 1024]),
                                    IdentityBlock(1024, [256, 256, 1024]),
                                    IdentityBlock(1024, [256, 256, 1024]),
                                    IdentityBlock(1024, [256, 256, 1024]),
                                    IdentityBlock(1024, [256, 256, 1024]))

        self.block4 = nn.Sequential(ConvBlock(1024,[512,512,2048],stride=2),
                                    IdentityBlock(2048,[512,512,2048]),
                                    IdentityBlock(2048,[512,512,2048]))

        self.fc1 = nn.Linear(2048,classes)

    def forward(self,x):
        x = self.conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = nn.AvgPool2d(kernel_size=(7,7))(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc1(x)
        x = F.softmax(x,dim=1)
        return x



if __name__ == "__main__":
    model = Resnet50(3,10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    epochs = 5
    train_loader,test_loader = utils.get_cifar10_data((224,224))
    for epoch in range(epochs):
        i = 0
        for data in train_loader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            i += 1
            if i % 100 == 0:
                print(f' epoch :[ {epoch + 1} ,{i / len(train_loader) * 100. :.0f}%] loss = {loss.item() :4f},')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            predicted = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'evaluate accuracy :{correct / total * 100 :.2f}%')
