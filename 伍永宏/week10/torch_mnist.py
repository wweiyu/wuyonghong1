import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision

class MnisetNet(torch.nn.Module):
    def __init__(self):
        super(MnisetNet,self).__init__()
        self.fc1 = torch.nn.Linear(28*28,512)
        # self.fc2 = torch.nn.Linear(512,100)
        self.fc3 = torch.nn.Linear(512,10)

    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return x

class Model:
    def __init__(self,net,cost,optimize,lr = 0.1):
        self.net = net
        self.cost = self.get_cost(cost)
        self.lr = lr
        self.optimize = self.get_optimize(optimize)


    def get_cost(self,cost):
        support_cost = {
            "MSE": torch.nn.MSELoss(),
            "CROSS_ENTROPY":torch.nn.CrossEntropyLoss()
        }
        return support_cost[cost]

    def get_optimize(self,optimize,**kwargs):
        support_optimize = {
            'SGD':torch.optim.SGD(self.net.parameters(),0.1,**kwargs),
            'RMSP':torch.optim.RMSprop(self.net.parameters(),0.01,**kwargs),
            "ADAM":torch.optim.Adam(self.net.parameters(),0.01,**kwargs)
        }
        return support_optimize[optimize]

    def train(self,train_loader,epochs = 5):
        for epoch in range(epochs):
            i = 0
            for data in train_loader:
                inputs,labels = data
                self.optimize.zero_grad()
                outputs = self.net(inputs)
                one_hot = F.one_hot(labels,10).float()
                loss = self.cost(outputs,one_hot)
                loss.backward()
                self.optimize.step()
                i += 1
                if i % 100 == 0 :
                    print(f' epoch :[ {epoch +1 } ,{i/len(train_loader) * 100. :.0f}%] loss = {loss.item() :4f},')
        print('train finish')

    def evaluate(self,test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs,labels = data
                outputs = self.net(inputs)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'evaluate accuracy :{correct/total *100 :.2f}%')

def get_mniset_data():
    trans = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST('./data',train= True,download=True,transform = trans)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,shuffle=True)

    test_set = torchvision.datasets.MNIST('./data',train=False,download=True,transform = trans)

    test_loader = torch.utils.data.DataLoader(test_set,batch_size = 32,shuffle = True)
    return train_loader,test_loader

if __name__ == "__main__":
    train_loader,test_loader = get_mniset_data()
    net = MnisetNet()
    model = Model(net,"MSE",'ADAM')
    model.train(train_loader,5)
    model.evaluate(test_loader)
