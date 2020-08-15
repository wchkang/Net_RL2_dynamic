import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import sys 
import os
import time

#from models.cifar100 import resnet
from models.cifar10 import resnet_skipmiddle
#from models.ilsvrc import resnet
import utils

import numpy as np

from ptflops import get_model_complexity_info

model = resnet_skipmiddle.ResNet32_DoubleShared

#model = resnet.ResNet34_SingleShared
#file_weight = './checkpoint/CIFAR100-ResNet34_SingleShared-S32-U1-L10.0-phase5-3.pth'
#model = resnet.ResNet34_DoubleShared
#file_weight = './checkpoint/ILSVRC2012-ft-ResNet34_DoubleShared-S32-U1-L10.0-0,1,2,3.pth'

#testloader = utils.get_testdata('CIFAR100',"./data",batch_size=256)


# parallelize
class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

with torch.cuda.device(0):
  #net = model(32, 1)
  net = model(16, 1)
  #net = MyDataParallel(net)
  net = net.to('cuda')
  #checkpoint = torch.load(file_weight)
  #net.load_state_dict(checkpoint['net_state_dict'])

  macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True,
  #macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=False)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))
