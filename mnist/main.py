# Imports
from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets , transforms
from torch.optim.lr_scheduler import StepLR

# Neural Net
class Net(nn.Module):
    def __init__(self):
        super(Net , self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,1)
        self.conv2 = nn.Conv2d(32, 64 ,3,1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216,128)
        self.fc2 = nn.Linear(128, 10)


    def forward(self , x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output= F.log_softmax(x , dim=1)
        return output


def train(model , device , train_loader  ,optimizer , epoch):
    model.train()
    for batch_idx , (data , target) in enumerate(train_loader):
        data , target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output , target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]/tLoss: {:.6f}'.format(epoch , batch_idx * len(data) , 
                len(train_loader.dataset), 100. * batch_idx/ len(train_loader) , loss.item()))

            dry_run = False
            if dry_run:
                break

def test(model , device , test_loader ):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data , target in test_loader:
            data, target  = data.to(device) , target.to(device)
            output = model(data)

            test_loss += F.nll_loss(output , target , reduction = 'sum').item()
            pred = output.argmax(dim= 1 , keepdim= True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set : Average loss : {:.4f} , Accuracy :{}/{} ({:.0f}%)\n '.format(
        test_loss , correct , len(test_loader.dataset) , 100. * correct / len(test_loader.dataset)
    ))

def main():
    torch.manual_seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    kwargs = {'batch_size': 64}
    if torch.cuda.is_available():
        kwargs.update({
            'num_workers' : 1,
            'pin_memory':True,
            'shuffle': True
        })

    transform = transforms.Compose([transforms.ToTensor() , transforms.Normalize((0.1307,) , (0.3081,))])

    trainset = datasets.MNIST('../data', train=True ,download=True, transform = transform)
    testset = datasets.MNIST('../data' , train= False , download = True, transform = transform)

    train_loader = torch.utils.data.DataLoader(trainset , **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, **kwargs)

    model = Net().to(device)
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    scheduler = StepLR(optimizer, step_size= 1 , gamma=0.7)
    epochs = 12
    for epoch in range(1, epochs+1):
        train(model , device , train_loader , optimizer , epoch)
        test(model , device , test_loader)
        scheduler.step()

    save_model = True
    if save_model:
        torch.save(model.state_dict() , 'mnist_cnn_v1.pt')

if __name__ == '__main__':
    main()