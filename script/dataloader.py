from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import numpy as np
import torch
import glob
import re
import csv
import math

class CustomDataset(Dataset):
    def __init__(self,dir_path):
        
        
        self.dir_path = dir_path
        file_x_list = glob.glob(self.dir_path+'xdata/xdata*.csv')
        file_y_list = glob.glob(self.dir_path+'ydata/ydata*.csv')
        file_x_id_list = []
        file_y_id_list = []
        for file_name in file_x_list:
            file_x_id_list.append(re.findall("xdata(\d+).csv", file_name))
        for file_name in file_x_list:
            file_y_id_list.append(re.findall("xdata(\d+).csv", file_name))
        self.match_list_id = [x for x in file_x_id_list if x in file_y_id_list]
        self.match_list_id = np.array(self.match_list_id).flatten()





    def __len__(self):
        
        return len(self.match_list_id)


    def __getitem__(self, idx):
        
        #we want to turn it into a pytorch tensor, 
        #and flatten it (since we want to train a fully connceted network)


        id_x_path = self.dir_path+'xdata/xdata' + self.match_list_id[idx] + '.csv'
        id_y_path = self.dir_path+'ydata/ydata' + self.match_list_id[idx] + '.csv'
        with open(id_x_path) as f:
          x=list(csv.reader(f,delimiter=","))
        x = np.array(x).astype(np.float32)
        x = torch.tensor(x)
        #x = pd.read_csv(id_x_path, header=None, dtype=np.float32)
        #x = torch.tensor( np.array( x ) )
        x = x.view(4,101,101) 
        # y = pd.read_csv(id_y_path, header=None, dtype=np.float32)
        # y = torch.tensor( np.array( y ) ).view(-1)
        with open(id_y_path) as f:
          y=list(csv.reader(f,delimiter=","))
        y = np.array(y).astype(np.float32)
        y = torch.tensor(y).view(-1)
        col=-round(math.log10(y[1]))-2 #choose {0,1,2} to temp e^{-2,-3,-4}
        row=round(int(y[0])/3)-4  #choose {0,1,2} to gamma {12,15,18}
        #label=np.array([[1,2,3],[4,5,6],[7,8,9]]) #class labels
        y=torch.tensor([row,col])    #label[row][col]
        

        
        return x, y #x[0].unsqueeze(0), y

