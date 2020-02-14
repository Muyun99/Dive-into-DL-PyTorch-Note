import torch
import random
import numpy as np
import torch.utils.data as Data
import torch.optim as optim
from torch import nn
from torch.nn import init

# print(torch.__version__)
torch.manual_seed(1)
torch.set_default_tensor_type('torch.FloatTensor')

num_features = 2
num_samples = 1000
num_epochs = 5
batch_size = 10

true_w = [2.5, 5.1]
true_b = 1.5

class linear_regresion_model(nn.Module):
    def __init__(self,num_features):
        super(linear_regresion_model,self).__init__()
        self.model = nn.Linear(num_features,1)
        self.loss = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(),lr=0.03)

        init.normal_(self.model.weight, mean=0.0, std=0.01)
        init.constant_(self.model.bias, val=0.0)
        
    def forward(self,x):
        return self.model(x)

def data_generate():
    features = torch.randn(num_samples,num_features,dtype = torch.float32)
    labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
    labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype = torch.float32)
    return features,labels

def get_data_iter(features,labels):
    dataset = Data.TensorDataset(features,labels)
    data_iter = Data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 2,
    )
    return data_iter

def train(features,labels):
    data_iter = get_data_iter(features,labels)
    model = linear_regresion_model(num_features)
    for epoch in range(num_epochs):
        for X,y in data_iter:
            output = model(X)
            batch_loss = model.loss(output,y.view(output.size()))

            model.optimizer.zero_grad()
            batch_loss.backward()
            model.optimizer.step()
        print('epoch %d, loss: %f' % (epoch, batch_loss.item()))

    print("-" * 13)
    print("training finish!")
    print("-" * 13)
    print("learn_w: ",model.model.weight.data)
    print("learn_b: ",model.model.bias.data)
    print("true_w: ",true_w)
    print("true_b: ",true_b)
        
    

if __name__ == "__main__":
    features,labels = data_generate()
    train(features,labels)