import random

import numpy as np
import torch
import torch.utils.data as Data
from torch import nn

print(torch.__version__)
torch.manual_seed(1)
torch.set_default_tensor_type('torch.FloatTensor')

num_features = 2
num_samples = 1000
num_epochs = 5
batch_size = 10

true_w = [2.5, 5.1]
true_b = 1.5

class linear_regresion_model():
    def __init__(self,features,labels):
        self.w = torch.tensor(np.random.normal(0,0.01,(num_features,1)),dtype = torch.float32)
        self.b = torch.zeros(1,dtype=torch.float32)
        self.w.requires_grad_(requires_grad=True)
        self.b.requires_grad_(requires_grad=True)


    def sgd(self):
        self.learning_rate = 0.03
        for param in [self.w,self.b]:
            param.data -= self.learning_rate * param.grad / batch_size

    def squared_loss(self,label_hat,label):
        return (label_hat - label.view(label_hat.size())) ** 2 / 2

    def predict(self,X):
        return torch.mm(X,self.w) + self.b

def data_generate():
    features = torch.randn(num_samples,num_features,dtype = torch.float32)
    labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
    labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype = torch.float32)
    return features,labels

def data_iter(features,labels):
    num_samples = len(features)
    indices = list(range(num_samples))
    random.shuffle(indices)
    for i in range(0,num_samples,batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size,num_samples)])
        yield features.index_select(0,j), labels.index_select(0,j)

def train(features,labels):
    model = linear_regresion_model(features,labels)
    for epoch in range(num_epochs):
        for X, y in data_iter(features,labels):
            batch_loss = model.squared_loss(model.predict(X),y).sum()
            batch_loss.backward()
            
            model.sgd()
            model.w.grad.data.zero_()
            model.b.grad.data.zero_()

        train_loss = model.squared_loss(model.predict(features),labels)
        print("epoch %d, loss %f" %(epoch+1, train_loss.mean().item()))


    print("-" * 13)
    print("training finish!")
    print("-" * 13)
    print("learn_w: ",model.w)
    print("learn_b: ",model.b)
    print("true_w: ",true_w)
    print("true_b: ",true_b)

if __name__ == "__main__":
    features,labels = data_generate()
    train(features,labels)
