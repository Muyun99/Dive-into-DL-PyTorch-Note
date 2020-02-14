import torch
import numpy as np
import sys
sys.path.append("../../utils")
import d2lzh as d2l

batch_size = 256
num_features = 784
num_classes = 10
num_hiddens = 256
num_epochs = 5


class multilayer_perceptron_model():
    def __init__(self):
        self.W1 = torch.tensor(np.random.normal(0, 0.01, (num_features, num_hiddens)), dtype=torch.float)
        self.b1 = torch.zeros(num_hiddens, dtype=torch.float)
        self.W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_classes)), dtype=torch.float)
        self.b2 = torch.zeros(num_classes, dtype=torch.float)
        

        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.loss = torch.nn.CrossEntropyLoss()
        for param in self.params:
            param.requires_grad_(requires_grad=True)

    
    def optimizer(self):
        self.learning_rate = 100
        d2l.sgd(self.params,self.learning_rate,batch_size)
    
    def relu(self,X):
        return torch.max(input=X, other=torch.tensor(0.0))

    def predict(self,X):
        X = X.view(-1,num_features)
        H = self.relu(torch.matmul(X,self.W1) + self.b1)
        return torch.matmul(H,self.W2) + self.b2

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
            batch_loss.backward()

            model.optimizer()
            for param in model.params:
                param.grad.data.zero_()

            train_loss_sum += batch_loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

        test_acc = evaluate_accuracy(test_iter, model)
        print("epoch %d, train_loss %.3f, train_acc %.3f, test acc %.3f" %(epoch + 1,train_loss_sum / n, train_acc_sum / n,test_acc))            

if __name__ == "__main__":
    train()
    