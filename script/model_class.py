
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Conv2d,ReLU,MaxPool2d,Linear,BatchNorm2d, LayerNorm





#########################


class Classifier_Time(nn.Module):
    def __init__(self):
        super(Classifier_Time,self).__init__()

        self.classifier = Classifier_Skeleton(1,4)
        self.norm = LayerNorm([101,101], elementwise_affine=False)
        
    def forward(self,x):

        out = self.norm(x)
        out = out.view(-1,1,101,101)
        out = self.classifier(out)
        out = out.view(-1,4,4)

        return out

#########################


class Classifier_Index(nn.Module):
    def __init__(self):
        super(Classifier_Index,self).__init__()

        self.classifier = Classifier_Skeleton(4,3)
        
    def forward(self,x):

        out = self.classifier(x)
        
        return out


######################
######################


class Classifier_Skeleton(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Classifier_Skeleton,self).__init__()

        self.cnn = nn.Sequential(
            Conv2d(in_channels,10,kernel_size=(5,5), stride=1, padding=2)
            ,MaxPool2d(kernel_size=(3,3),stride=2)#from 101X101 to 50X50
            ,ReLU(inplace=True)
            ,Conv2d(10, 20, kernel_size=(3, 3), stride=1, padding=1)
            ,MaxPool2d(kernel_size=(2,2),stride=2)#from 50X50 to 25X25
            ,ReLU(inplace=True)
            ,Conv2d(20, 50, kernel_size=(3, 3), stride=1, padding=1)
            ,MaxPool2d(kernel_size=(3,3),stride=2)#from 25X25 to 12X12
            ,ReLU(inplace=True)
            ,Conv2d(50, 100, kernel_size=(3, 3), stride=1, padding=1)
            ,MaxPool2d(kernel_size=(2,2),stride=2)#from 12X12 to 6X6
            ,ReLU(inplace=True)
            ,Conv2d(100, 200, kernel_size=(3, 3), stride=1, padding=1)
            ,MaxPool2d(kernel_size=(2,2),stride=2)#from 6X6 to 3X3
        )
        self.fc = nn.Sequential(
            nn.Linear(200*3*3,100)
            ,ReLU(inplace=True)
            ,nn.Linear(100,50)
            ,ReLU(inplace=True)
            ,nn.Linear(50,out_channels)
        )
        
    def forward(self,x):
        
        out = self.cnn(x)
        out = torch.flatten(out,1)
        out = self.fc(out)
        
        return out



############################