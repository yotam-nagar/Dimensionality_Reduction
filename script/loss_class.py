import torch.nn as nn
import torch
from tqdm import tqdm

########################################################

def lost_in_time(dataloader,net):

    """
    Calculate the loss of the time-classifier for convergence plot (loss vs ephoc)
    Arguments:
        dataloader - DataLoader object
        net - Classifier
    Returns:
        [mean of the loss over the all data, the accuracy (between 0 to 1)] 
    """

    total = 0
    correct = 0
    loss = 0
    n_batches = 0
    loss_class = nn.CrossEntropyLoss()
    norm = nn.LayerNorm([101,101], elementwise_affine=False)

    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    
    with torch.no_grad():
        for x,y in dataloader: #tqdm(dataloader):
          if torch.cuda.is_available():
              x = x.cuda()
              y  = y.cuda()
          for t in range(4):
              n_batches+=1
              t_tensor = t*torch.ones(len(x),dtype=int).to(x.device)
              pred = net.classifier(norm(x[:,t].unsqueeze(1)))
              loss+= loss_class(pred,t_tensor).item()
              pred = torch.argmax(pred,dim=1)
              correct += len(torch.where(pred==t_tensor)[0])
              total+=len(y)
              
    
    return [loss/n_batches, correct/total]

##################################################


def compute_loss_class(index,dataloader,net):
    """
    Calculate the loss of the velocity-classifier for convergence plot (loss vs ephoc)
    Arguments:
        index - 0 for velocity-classifier, 1 for temperature-classifier
        dataloader - DataLoader object
        net - Classifier
    Returns:
        [mean of the loss over the all data, the accuracy (between 0 to 1)]
    """
    total = 0
    correct = 0
    loss = 0
    loss_class = nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    n_batches = 0
    with torch.no_grad():
        for x,y in dataloader:
            n_batches+=1
            
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            pred = net(x)
            loss+= loss_class(pred, y[:,index]).item()
            pred = torch.argmax(pred,dim=1)
            correct += len(torch.where(pred==y[:,index])[0])
            total+=len(y)
    
    return [loss/n_batches, correct/total]


#############################################