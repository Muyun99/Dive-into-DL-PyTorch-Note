from IPython import display
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import time
import sys
import d2lzh as d2l
import numpy as np
from torch import nn
from torch.nn import init
from collections import OrderedDict

batch_size = 256
num_workers = 4
num_features = 784
num_classes = 10
num_epochs = 5
torch.manual_seed(1)

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

class softmax_model(nn.Module):
    def __init__(self):
        super(softmax_model,self).__init__()
        self.model = nn.Sequential(
            # FlattenLayer(),
            # nn.Linear(num_features,num_classes)
            OrderedDict([
           ('flatten', FlattenLayer()),
           ('linear', nn.Linear(num_features,num_classes))]) # 或者写成我们自己定义的 LinearNet(num_inputs, num_outputs) 也可以
        )
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        init.normal_(self.model.linear.weight, mean=0, std=0.01)
        init.constant_(self.model.linear.bias, val=0)
        
    def predict(self,x):
        return self.model(x)

def evaluate_accuracy(data_iter,model):
    acc_sum,n=0.0,0
    for X,y in data_iter:
        acc_sum += (model.predict(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def get_dataset(down):
    # 若没有数据集，将down改为True
    mnist_train = torchvision.datasets.FashionMNIST(root='./FashionMNIST', train=True, download=down, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='./FashionMNIST', train=False, download=down, transform=transforms.ToTensor())
    return mnist_train,mnist_test

def show_dataset(mnist_train,mnist_test):
    X, y = [], []
    for i in range(5):
        X.append(mnist_train[i][0]) # 将第i个feature加到X中
        y.append(mnist_train[i][1]) # 将第i个label加到y中
    d2l.show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y))
    plt.show()

def get_dataset_iter(mnist_train,mnist_test):
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter,test_iter

def train(dataset_train,dataset_test):
    train_iter,test_iter = get_dataset_iter(dataset_train,dataset_test)
    model = softmax_model()
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X,y in train_iter:
            y_hat = model.predict(X)
            batch_loss = model.loss(y_hat,y).sum()
            
            model.optimizer.zero_grad()
            batch_loss.backward()
            model.optimizer.step()

            train_loss_sum += batch_loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, model)
        print("epoch %d, train_loss %.3f, train_acc %.3f, test acc %.3f" %(epoch + 1,train_loss_sum / n,train_acc_sum / n,test_acc))            

if __name__ == "__main__":
    mnist_train,mnist_test = get_dataset(down=False)
    # show_dataset(mnist_train,mnist_test)
    train(dataset_train = mnist_train,dataset_test = mnist_test)
    