from IPython import display
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import time
import sys
import d2lzh as d2l
import numpy as np

batch_size = 256
num_workers = 4
num_features = 784
num_classes = 10
num_epochs = 5
torch.manual_seed(1)

class softmax_model():
    def __init__(self):
        self.W = torch.tensor(np.random.normal(0,0.01,(num_features,num_classes)),dtype = torch.float32)
        self.b = torch.zeros(num_classes,dtype = torch.float32)
        self.W.requires_grad_(requires_grad=True)
        self.b.requires_grad_(requires_grad=True)
    
    def loss(self,y_hat,y):
        return - torch.log(y_hat.gather(1, y.view(-1, 1)))

    def optimizer(self):
        self.learning_rate = 0.1
        d2l.sgd([self.W,self.b], self.learning_rate, batch_size)

    def softmax(self,X):
        X_exp = X.exp()
        partition = X_exp.sum(dim=1, keepdim=True)
        return X_exp / partition

    def predict(self,X):
        return self.softmax(torch.mm(X.view((-1, num_features)), self.W) + self.b)

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
            batch_loss.backward()

            model.optimizer()
            model.W.grad.data.zero_()
            model.b.grad.data.zero_()

            train_loss_sum += batch_loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, model)
        print("epoch %d, train_loss %.3f, train_acc %.3f, test acc %.3f" %(epoch + 1,train_loss_sum / n,train_acc_sum / n,test_acc))            

if __name__ == "__main__":
    mnist_train,mnist_test = get_dataset(down=False)
    # show_dataset(mnist_train,mnist_test)
    train(dataset_train = mnist_train,dataset_test = mnist_test)
    