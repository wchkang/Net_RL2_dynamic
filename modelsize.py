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

#from models.cifar100 import mobilenetv2_skip
#from models.cifar10 import mobilenetv2_cifar10_skip
#from models.cifar100 import resnet_skip
#from models.ilsvrc import mobilenetv2_skip
from models.ilsvrc import mobilenetv2_skip_pytorch
from models.ilsvrc import resnet_skip_imagenet

from ptflops import get_model_complexity_info

#args.visible_device sets which cuda devices to be used"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device='cuda'

#model = mobilenetv2_skip.MobileNetV2_skip
#model = mobilenetv2_skip.MobileNetV2
#model = mobilenetv2_cifar10_skip.MobileNetV2_skip
#model = mobilenetv2_skip.MobileNetV2_skip
model = mobilenetv2_skip_pytorch.MobileNetV2_skip
#model = resnet_skip_imagenet.ResNet34_skip
#model = torchvision.models.resnet34(pretrained=False) 

with torch.cuda.device(0):
  net = model(num_classes=1000)
  net = net.to('cuda')
  #inputsize = (3,32,32)
  inputsize = (3,224,224)
  macs, params = get_model_complexity_info(net, inputsize, as_strings=True,
                                           print_per_layer_stat=True, verbose=False)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))

  #x = torch.randn(256,3,32,32, device='cuda')
  #x = torch.randn(1,3,32,32, device='cuda')
  x = torch.randn(1,3,224,224, device='cuda')
  y = net(x)
  t_start = time.time()
  for i in range(1000):
      y = net(x)
  t_end = time.time()
  print('time: {:.3f} seconds per inference'.format((t_end - t_start)/100.0))
