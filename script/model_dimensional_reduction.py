
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Conv2d,ReLU,MaxPool2d,Linear,BatchNorm2d,LayerNorm

#########################################################

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()


        self.cnn = nn.Sequential(
            Conv2d(1, 16, kernel_size=(5,5), stride=2, padding=0)
            ,nn.LeakyReLU(0.2, inplace=False) #from 101X101 to 49X49
            ,Conv2d(16, 32, kernel_size=(5, 5), stride=2, padding=0)
            ,nn.LeakyReLU(0.2, inplace=False) #from 49X49 to 23X23
            ,Conv2d(32, 64, kernel_size=(5, 5), stride=2, padding=0)
            ,nn.LeakyReLU(0.2, inplace=False) #from 23X23 to 10X10
            ,Conv2d(64, 128, kernel_size=(4, 4), stride=2, padding=0)
            ,nn.LeakyReLU(0.2, inplace=False) #from 10X10 to 4X4
            ,Conv2d(128, 256, kernel_size=(4, 4), stride=2, padding=0)
            ,nn.Tanh() #from 4X4 to 1X1
        )

        self.fc1 = nn.Linear(256,10)
        self.fc2 = nn.Linear(256,10)

        
    def forward(self,x):
        
        out = self.cnn(x)
        out = torch.flatten(out,1)
        out1 = self.fc1(out)
        out2 = self.fc2(out)
        
        return out1, out2


################################################

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(256,128, kernel_size=(4, 4), stride=2, padding=0)
            ,nn.LeakyReLU(0.2, inplace=False)
            ,nn.ConvTranspose2d(128,64, kernel_size=(4, 4), stride=2, padding=0)
            ,nn.LeakyReLU(0.2, inplace=False)
            ,nn.ConvTranspose2d(64,32, kernel_size=(5, 5), stride=2, padding=0)
            ,nn.LeakyReLU(0.2, inplace=False)
            ,nn.ConvTranspose2d(32,16, kernel_size=(5, 5), stride=2, padding=0)
            ,nn.LeakyReLU(0.2, inplace=False)
            ,nn.ConvTranspose2d(16, 1, kernel_size=(5, 5), stride=2, padding=0)
            ,nn.Tanh()
        )

        self.fc = nn.Linear(10, 256)
        
        
    def forward(self,x):
        
        out = self.fc(x)
        out = out.view(len(x),256,1,1)
        out = self.cnn(out)
        
        return out


###################################################

class VAE2(nn.Module):
    def __init__(self):
        super(VAE2, self).__init__()

        self.enc = Encoder()
        self.dec = Decoder()


    def pick_random(self, mu, logvar):
        
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return mu + eps*std

    def forward(self, x):


        # forward the encoder, we get: mu and logvar for each time step: -> tensor([:,40])
        enc_out = [self.enc(x[:,t].unsqueeze(1)) for t in range(4)]
        mu = torch.cat([out[0] for out in enc_out],1)
        logvar = torch.cat([out[0] for out in enc_out],1) 

        # make a random new disition of z: -> tensor([:,4,10])
        z = self.pick_random(mu, logvar) 
        z = z.view(-1,4,10) 
    
        # forward thorow the decoder, each time step separaly and concat the output: #tensor([:,4,101,101])
        out = torch.cat([self.dec(z[:,t].unsqueeze(1)) for t in range(4)],1)

        return out, mu, logvar


###################################################
################################################
    
class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(in_dim,100)
            ,nn.ReLU(inplace=False)
            ,nn.Linear(100,out_dim)
        )

    def forward(self, x):
        
        # out = self.norm(x)
        out = self.fc(x)
        
        return out

###################################################

class Z_Classifier(nn.Module):
    def __init__(self):
        super(Z_Classifier, self).__init__()

        self.classifier_velocity = Classifier(40,3)
        self.classifier_time = Classifier(10,4)


    def forward(self, mu):

        # classifier set property: ->tensor([:,3])
        out40 = self.classifier_velocity(mu)

        # classifier image property: ->tensor([:,4,4])
        mu = mu.view(-1,10)
        out10 = self.classifier_time(mu)
        out10 = out10.view(-1,4,4)
       
        return out40, out10

###################################################
