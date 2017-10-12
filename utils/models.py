import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.autograd import Variable
from torchvision import transforms

class VAE(nn.Module):
    def __init__(self, shape, layer1, layer2, isCuda):
        '''
        shape: tuple (shape[1], shape[2]) of c_train
        layer1: dim of hidden layer1
        layer2: dim of hidden layer2
        
        '''
        super(VAE, self).__init__()
        #input size (batch, seq, fea)
        #400: random, 20: random
        self.shape = shape
        self.isCuda = isCuda
        
        self.fc1 = nn.Linear(shape[1]*shape[2], layer1) 
        self.fc21 = nn.Linear(layer1, layer2) #encode
        self.fc22 = nn.Linear(layer1, layer2) #encode
        self.fc3 = nn.Linear(layer2, layer1) #decode
        self.fc4 = nn.Linear(layer1, shape[1]*shape[2]) #decode
    
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        if self.isCuda:
            self.cuda()

        #initialize weights
        nn.init.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc21.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc22.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc3.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc4.weight, gain=np.sqrt(2))

    def encode(self, x):
        #x --> fc1 --> relu --> fc21
        #x --> fc1 --> relu --> fc22
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
# This is a method proposed by https://arxiv.org/pdf/1312.6114.pdf
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(logvar/2)
        if self.isCuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps*std + mu

    def decode(self, z, x):
        #z --> fc3 --> relu --> fc4 --> sigmoid
        h3 = self.relu(self.fc3(z))
        #return self.sigmoid(self.fc4(h3))
        return self.sigmoid(self.fc4(h3)).view(x.size())
    
    def forward(self, x):
        #flatten input and pass to encode
        #mu= mean
        #logvar = log variational
        mu, logvar = self.encode(x.view(-1, self.shape[1]*self.shape[2]))
        z = self.reparametrize(mu, logvar)
        return self.decode(z, x), mu, logvar
    

class AE(nn.Module):
    def __init__(self, shape, h_dim, z_dim):
        super(AE, self).__init__()
        self.shape = shape
        
        # X --> fc1 --> Z --> fc2 --> X'
        self.fc1 = nn.Linear(shape[1]*shape[2], h_dim) #encode
        self.fc2 = nn.Linear(h_dim, z_dim) #decode
        
        self.fc3 = nn.Linear(z_dim, h_dim) #decode
        self.fc4 = nn.Linear(h_dim, shape[1]*shape[2]) #decode
    
        #self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        #initialize weights
        nn.init.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc2.weight, gain=np.sqrt(2))

    def encode(self, x):
        #x --> fc1 --> fc2 --> sigmoid --> z
        h = self.fc1(x)
        z = self.sigmoid(self.fc2(h))
        return z

    def decode(self, z, x):
        #z --> fc3 --> fc4 --> sigmoid
        h = self.fc3(z)
        recon_x = self.sigmoid(self.fc4(h))
        return recon_x.view(x.size())
    
    def forward(self, x):
        #flatten input and pass to encode
        z = self.encode(x.view(-1, self.shape[1]*self.shape[2]))
        return self.decode(z, x)
    
    
'''
class VAE(nn.Module):
    def __init__(self, c_train, args.layer1, args.layer2):
        super(VAE, self).__init__()
        #input size (batch, seq, fea)
        #400: random, 20: random
        self.fc1 = nn.Linear(c_train.shape[1]*c_train.shape[2], args.layer1) 
        self.fc21 = nn.Linear(args.layer1, args.layer2) #encode
        self.fc22 = nn.Linear(args.layer1, args.layer2) #encode
        self.fc3 = nn.Linear(args.layer2, args.layer1) #decode
        self.fc4 = nn.Linear(args.layer1, c_train.shape[1]*c_train.shape[2]) #decode
    
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        #initialize weights
        nn.init.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc21.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc22.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc3.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc4.weight, gain=np.sqrt(2))

    def encode(self, x):
        #x --> fc1 --> relu --> fc21
        #x --> fc1 --> relu --> fc22
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
# This is a method proposed by https://arxiv.org/pdf/1312.6114.pdf
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(logvar/2)
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps*std + mu

    def decode(self, z, x):
        #z --> fc3 --> relu --> fc4 --> sigmoid
        h3 = self.relu(self.fc3(z))
        #return self.sigmoid(self.fc4(h3))
        return self.sigmoid(self.fc4(h3)).view(x.size())
    
    def forward(self, x):
        #flatten input and pass to encode
        #mu= mean
        #logvar = log variational
        mu, logvar = self.encode(x.view(-1, c_train.shape[1]*c_train.shape[2]))
        z = self.reparametrize(mu, logvar)
        return self.decode(z, x), mu, logvar

'''