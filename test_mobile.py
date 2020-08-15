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

import utils
import datetime

import numpy as np

from models.cifar10 import mobilenetv2_skip

from ptflops import get_model_complexity_info

model = mobilenetv2_skip.MobileNetV2_skip

with torch.cuda.device(0):
  net = model()
  net = net.to('cuda')

  macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True,
                                           print_per_layer_stat=True, verbose=False)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))

  x = torch.randn(16,3,32,32, device='cuda')
  y = net(x)
  t_start = time.time()
  for i in range(100):
      y = net(x)
  t_end = time.time()
  print('time: {:.3f} seconds'.format(t_end - t_start))
  
