import torch
from unet.model.unet import Unet
from unet.train.dataset import ISBI_Dataset
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

def train_net(model,device,epoch = 50,batch_size =1,lr=0.00001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(model.parameters(),lr=lr,momentum=0.9)
    dataset = ISBI_Dataset('../data/train/')
    train_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    best_loss = float('inf')
    for _ in range(epoch):
        model.train()
        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.to(device,dtype = torch.float32)
            label = label.to(device,dtype = torch.float32)
            result = model(image)
            loss = criterion(result,label)
            print(type(loss))
            print('Loss/train', loss.item())
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(),'best_model.path')
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = Unet(in_channels=1,class_num=1)
    model.to(device)
    train_net(model,device)
