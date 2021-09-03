import torch.nn as nn
import torch

#############################################
class Loss_Func(nn.Module):

  def __init__(self,net_time,net_velocity):
    super(Loss_Func, self).__init__()
    
    self.net_time = net_time.eval()
    self.net_velocity = net_velocity.eval()
    self.perc_time = PerceptualLoss(net_time.eval())
    self.perc_velocity = PerceptualLoss(net_velocity.eval())

    if torch.cuda.is_available():
      self.net_time.cuda()
      self.net_velocity.cuda()
      self.perc_time.cuda()
      self.perc_velocity.cuda()
      self.device = torch.device("cuda")
    else:
      self.device = torch.device("cpu")
    


    # initial set of loss-elements coefficients (all set to one)
    self.hyperparam = {'mse': 1, 'kld': 1, 'z_time': 1, 'z_vel': 1,\
                       'perc_time': 1, 'perc_vel': 1, 'end_time': 1, 'end_vel': 1}

    self.cross_entropy_func = nn.CrossEntropyLoss()
    self.mse_func = nn.MSELoss(reduction='sum')
    self.norm = nn.LayerNorm([101,101], elementwise_affine=False)

    self.loss_list = {}

  def clear(self):
    self.loss_list = {}

  def set_mse(self, x, pred_x):
    mse = self.mse_func(self.norm(x), self.norm(pred_x))
    self.loss_list['mse'] = mse*self.hyperparam['mse']/100000 #rescale by /1000000
    return self.loss_list['mse'].data.item()

  def set_kld(self, mu, logvar):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    self.loss_list['kld'] = kld*self.hyperparam['kld']/100 #rescale by /100
    return self.loss_list['kld'].data.item()

  def set_z(self, z_time, z_vel, y):
    target = (torch.arange(4).repeat(z_time.shape[0],1)).long()
    target = target.to(self.device)
    z_time_loss = self.cross_entropy_func(z_time,target)    
    z_vel_loss = self.cross_entropy_func(z_vel,y[:,0])
    self.loss_list['z_time'] = z_time_loss*self.hyperparam['z_time']
    self.loss_list['z_vel'] = z_vel_loss*self.hyperparam['z_vel']

  def set_perc(self, x, pred_x):
    x_2 = self.norm(x.view(-1,1,101,101))
    pred_x_2 = self.norm(pred_x.view(-1,1,101,101))
    perc_time_loss = self.perc_time(x_2,pred_x_2)/4
    perc_vel_loss = self.perc_velocity(x,pred_x) 
    self.loss_list['perc_time'] = perc_time_loss*self.hyperparam['perc_time']
    self.loss_list['perc_vel'] = perc_vel_loss*self.hyperparam['perc_vel']

  def set_end(self, pred_x, y):
    out_time = self.net_time(pred_x)
    target = (torch.arange(4).repeat(out_time.shape[0],1)).long()
    target = target.to(self.device)
    out_time_loss = self.cross_entropy_func(out_time,target)
    out_vel = self.net_velocity(pred_x)
    out_vel_loss = self.cross_entropy_func(out_vel,y[:,0])
    self.loss_list['end_time'] = out_time_loss*self.hyperparam['end_time']
    self.loss_list['end_vel'] = out_vel_loss*self.hyperparam['end_vel']

  def get_acc_z_vel(self,z_vel,y):
    pred_vel = torch.argmax(z_vel,dim=1)
    correct_vel = len(torch.where(pred_vel==y[:,0])[0])
    acc_vel = correct_vel/z_vel.shape[0]
    return acc_vel

  def get_acc_z_time(self,z_time):
    pred_time = torch.argmax(z_time,dim=2)
    target = (torch.arange(4).repeat(z_time.shape[0],1))
    target = target.to(self.device)
    correct_time = len(torch.where((target == pred_time).view(-1))[0])
    acc_time = correct_time/z_time.shape[0]/4
    return acc_time

  def get_acc_end_vel(self,pred_x,y):
    out_vel = self.net_velocity(pred_x)
    pred_vel = torch.argmax(out_vel,dim=1)
    correct_vel = len(torch.where(pred_vel==y[:,0])[0])
    acc_vel = correct_vel/out_vel.shape[0]
    return acc_vel

  def get_acc_end_time(self,pred_x):
    out_time = self.net_time(pred_x)
    pred_time = torch.argmax(out_time,dim=2)
    target = (torch.arange(4).repeat(out_time.shape[0],1))
    target = target.to(self.device)
    correct_time = len(torch.where((target == pred_time).view(-1))[0])
    acc_time = correct_time/out_time.shape[0]/4
    return acc_time


  def forward(self):
    loss_list = []
    for key in self.loss_list:
      loss_list.append(self.loss_list[key])

    return torch.stack(loss_list)

################################################

def Loss_Func_Info(dataloader, loss_cal, net_VAE2, net_z):


      
  net_VAE2.eval()
  net_z.eval()
  if torch.cuda.is_available(): net_VAE2.cuda(); net_z.cuda(); loss_cal.cuda()
  loss_cal.clear()
  loss_list={}
  n_batch=0
  key_list = {'loss',
                'acc_z_time',
                'acc_z_vel',
                'acc_end_time',
                'acc_end_vel',
                'mse',
                'kld'}
  for key in key_list:
        loss_list[key] = 0
        
    
  for x,y in dataloader:
        n_batch += 1
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        
        pred_x, mu, logvar = net_VAE2(x)
        z_vel, z_time = net_z(mu)
        
        # calculate all loss elements & backward:
        loss_cal.clear()
        loss_list['mse'] += loss_cal.set_mse(x, pred_x)
        loss_list['kld'] += loss_cal.set_kld(mu, logvar)
        loss_list['acc_end_time'] += loss_cal.get_acc_end_time(pred_x)
        loss_list['acc_end_vel'] += loss_cal.get_acc_end_vel(pred_x,y)
        loss_list['acc_z_time'] += loss_cal.get_acc_z_time(z_time)
        loss_list['acc_z_vel'] += loss_cal.get_acc_z_vel(z_vel,y)
        loss_cal.set_perc(x, pred_x)
        loss_cal.set_z(z_time, z_vel, y)
        loss_list['loss'] += torch.sum(loss_cal()).data.item()

       

  for key in loss_list:
        loss_list[key] = loss_list[key]/n_batch

  return loss_list   



#####################################################
"""
input: normalize two [N,1,101,101] normalize! data
output: scalar that represent the preceptioal loss
"""

class PerceptualLoss(nn.Module):
    def __init__(self, net_class):
        super(PerceptualLoss, self).__init__()
        
        self.activ1 = net_class.classifier.cnn[0]
        self.activ2 = net_class.classifier.cnn[1:4]
        self.activ3 = net_class.classifier.cnn[4:7]
        self.activ4 = net_class.classifier.cnn[7:10]
        self.activ5 = net_class.classifier.cnn[10:13]


    def get_activ(self,x):
        
        activ1 = self.activ1(x)
        activ2 = self.activ2(activ1)
        activ3 = self.activ3(activ2)
        activ4 = self.activ4(activ3)
        activ5 = self.activ5(activ4)

        return activ1 ,activ2 ,activ3 ,activ4 ,activ5
        
    def forward(self, x, xhat):
        
        a1 ,a2 ,a3 ,a4 ,a5 = self.get_activ(x)
        ap1 ,ap2 ,ap3 ,ap4 ,ap5= self.get_activ(xhat)

        return torch.nn.functional.l1_loss(a1, ap1)+torch.nn.functional.l1_loss(a2, ap2)+\
                        torch.nn.functional.l1_loss(a3, ap3)+torch.nn.functional.l1_loss(a4, ap4)+\
                        torch.nn.functional.l1_loss(a5, ap5)

################################################