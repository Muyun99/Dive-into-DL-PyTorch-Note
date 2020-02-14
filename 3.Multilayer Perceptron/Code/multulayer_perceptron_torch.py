import torch
import numpy as np
import sys
sys.path.append("../../utils")
import d2lzh as d2l
from torch import nn
from torch.nn import init

batch_size = 256
num_features = 784
num_hiddens = 256
num_classes = 10
num_epochs = 5

class multilayer_perceptron_model(nn.Module):
    def __init__(self):
        super(multilayer_perceptron_model, self).__init__()
        self.model = nn.Sequential(
            d2l.FlattenLayer(),
            nn.Linear(num_features, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, num_classes), 
        )
        for params in self.model.parameters():
            init.normal_(params, mean=0, std=0.01)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.5)
    
    def predict(self,X):
        return self.model(X)

def evaluate_accuracy(data_iter,model):
    acc_sum,n=0.0,0
    for X,y in data_iter:
        acc_sum += (model.predict(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def train():
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size,root='../../datasets/FashionMNIST')
    model = multilayer_perceptron_model()
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = model.predict(X)
            batch_loss = model.loss(y_hat,y).sum()

            model.optimizer.zero_grad()
            batch_loss.backward()
            model.optimizer.step()

            train_loss_sum += batch_loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

        test_acc = evaluate_accuracy(test_iter, model)
        print("epoch %d, train_loss %.3f, train_acc %.3f, test acc %.3f" %(epoch + 1,train_loss_sum / n, train_acc_sum / n,test_acc))     

if __name__ == "__main__":
    train()