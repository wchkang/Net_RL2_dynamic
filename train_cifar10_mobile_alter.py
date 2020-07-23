from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import sys
import os
import argparse

import utils
import timeit
from random import randint

#Possible arguments
parser = argparse.ArgumentParser(description='Following arguments are used for the script')
parser.add_argument('--lr', default=0.1, type=float, help='Learning Rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--batch_size', default=256, type=int, help='Batch_size')
parser.add_argument('--visible_device', default="0", help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--pretrained', default=None, help='Path of a pretrained model file')
parser.add_argument('--starting_epoch', default=0, type=int, help='An epoch which model training starts')
parser.add_argument('--dataset_path', default="./data/", help='A path to dataset directory')
parser.add_argument('--model', default="MobileNetV2_skip", help='MobileNetV2_skip')
args = parser.parse_args()

from models.cifar10 import mobilenetv2_skip
dic_model = {'MobileNetV2_skip': mobilenetv2_skip.MobileNetV2_skip}
    
if args.model not in dic_model:
    print("The model is currently not supported")
    sys.exit()

trainloader = utils.get_traindata('CIFAR10',args.dataset_path,batch_size=args.batch_size,download=True)
testloader = utils.get_testdata('CIFAR10',args.dataset_path,batch_size=args.batch_size)

#args.visible_device sets which cuda devices to be used"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=args.visible_device
device='cuda'

net = dic_model[args.model]()
net = net.to(device)
                    
#CrossEntropyLoss for accuracy loss criterion
criterion = nn.CrossEntropyLoss()


#Training for standard models
def train_alter(epoch):    
    print('\nCuda ' + args.visible_device + ' Epoch: %d' % epoch)
    net.train()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
    
        optimizer.zero_grad()

        for skip in  (True, False):
            outputs = net(inputs,skip)
            
            _, pred = outputs.topk(5, 1, largest=True, sorted=True)

            label_e = targets.view(targets.size(0), -1).expand_as(pred)
            correct = pred.eq(label_e).float()

            correct_top5 += correct[:, :5].sum()
            correct_top1 += correct[:, :1].sum()        
            total += targets.size(0)
                            
            loss = criterion(outputs, targets)
            if (batch_idx == 0):
                print("accuracy_loss: %.6f" % loss)
            loss.backward()

        optimizer.step()
    
    acc_top1 = 100.*correct_top1/total
    acc_top5 = 100.*correct_top5/total
    
    print("Training_Acc_Top1/5 = %.3f\t%.3f" % (acc_top1, acc_top5))

    
def test(epoch, skip=False, update_best=True):
    """ Train roultine for SingleShared CIFAR10 models. 

    Arguments:
        skip: if True, skip some blocks
        update_best: if True, update best_acc and save the model when a best model is found
    """
    global best_acc
    global best_acc_top5
    net.eval()
    test_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs, skip)
            
            _, pred = outputs.topk(5, 1, largest=True, sorted=True)

            label_e = targets.view(targets.size(0), -1).expand_as(pred)
            correct = pred.eq(label_e).float()

            correct_top5 += correct[:, :5].sum()
            correct_top1 += correct[:, :1].sum()
            
            total += targets.size(0)
            
    # Save checkpoint.
    acc_top1 = 100.*correct_top1/total
    acc_top5 = 100.*correct_top5/total
    print("Test_Acc_Top1/5 = %.3f\t%.3f" % (acc_top1, acc_top5))

    if update_best == True and acc_top1 > best_acc:
        #print('Saving..')
        state = {
            'net_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': acc_top1,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + 'CIFAR10-' + args.model + "-" + args.visible_device + '.pth')
        best_acc = acc_top1
        best_acc_top5 = acc_top5
        print("Best_Acc_top1 = %.3f" % acc_top1)
        #print("Best_Acc_top5 = %.3f" % acc_top5)


def freeze_highperf_model(net):
    """ Freeze the high-performance model while enabling the training of the low-perf model. """
    
    # freeze params of only being used by the high-performance model
    for i, i_base in enumerate(net.skip_layers):
        net.layers[i_base].conv3.weight.requires_grad = False
        net.layers[i_base].bn3.eval()
        for j in range(net.skip_distance[i]):
            for param in net.layers[i_base+1+j].parameters():
                param.requires_grad = False
            net.layers[i_base+1+j].eval()

    # freeze params of high-perf FC layer
    net.linear.weight.requires_grad = False
    net.linear.bias.requires_grad = False

def freeze_lowperf_model(net):
    """ Freeze parts of low-performance model while enabling the training of high-perf model """
    for i, i_base in enumerate(net.skip_layers):
        net.layers[i_base].conv3_skip.weight.requires_grad = False
        net.layers[i_base].bn3_skip.eval()
    # freeze params of high-perf FC layer
    net.linear_skip.weight.requires_grad = False
    net.linear_skip.bias.requires_grad = False


def freeze_lowperf_model_all(net):
    """ Freeze the low-performance model while enabling the training of high-perf model """

    # bn layers need to be freezed explicitly since they cannot be freezed via '.requires_grad=False'
    for module in net.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            module.eval()

    # freeze all parameters
    for param in net.parameters():
        param.requires_grad = False

    # defreeze params of only being used by the high-performance model
    for i, i_base in enumerate(net.skip_layers):
        net.layers[i_base].conv3.weight.requires_grad = True
        net.layers[i_base].bn3.train()
        for j in range(net.skip_distance[i]):
            for param in net.layers[i_base+1+j].parameters():
                param.requires_grad = True
            net.layers[i_base+1+j].train()

    # defreeze params of high-perf FC layer
    net.linear.weight.requires_grad = True
    net.linear.bias.requires_grad = True

def freeze_all_but_lowperf_fc(net):
    """ Used to freeze all parameters except 'bn2_skip' and 'fc_skip' layers. """
    # bn layers need to be freezed explicitly since they cannot be freezed via '.requires_grad=False'
    for module in net.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            module.eval()
    
    # freeze all parameters
    for param in net.parameters():
        param.requires_grad = False
    
    for i, i_base in enumerate(net.skip_layers):
        net.layers[i_base].conv3_skip.weight.requires_grad = True
        net.layers[i_base].bn3_skip.train()

    # defreeze params of high-perf FC layer
    net.linear_skip.weight.requires_grad = True
    net.linear_skip.bias.requires_grad = True

def defreeze_model(net):
    """ Defreeze all parameters and enable training. Must be called to enable training. """
    # defreeze all parameters
    for param in net.parameters():
        param.requires_grad = True
    # make the whole network trainable
    net.train()


def adjust_learning_rate(optimizer, epoch, args_lr):
    lr = args_lr
    if epoch > 150:
        lr = lr * 0.1
    if epoch > 250:
        lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_long(optimizer, epoch, args_lr):
    # cifar10 requires particularlly long training epoches.
    lr = args_lr
    if epoch > 250:
        lr = lr * 0.1
    if epoch > 375: 
        lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_finetune(optimizer, epoch, args_lr):
    lr = args_lr
    if epoch > 50:
        lr = lr * 0.1
    if epoch > 100:
        lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

best_acc = 0
best_acc_top5 = 0

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
if args.pretrained != None:
    checkpoint = torch.load(args.pretrained)
    net.load_state_dict(checkpoint['net_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_acc = checkpoint['acc']

net.train()
for i in range(args.starting_epoch, 350):
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    start = timeit.default_timer()
    
    adjust_learning_rate(optimizer, i+1, args.lr)
    
    train_alter(i+1)
    
    stop = timeit.default_timer()
    
    test(i+1, skip=True)
    test(i+1, skip=False)
        
    print('Time: {:.3f}'.format(stop - start))

print("Best_Acc_top1 = %.3f" % best_acc)
print("Best_Acc_top5 = %.3f" % best_acc_top5)

'''
## finetuning
best_acc = 92.61
best_acc_top5 = 0
net.train()
for i in range(args.starting_epoch, 30):
    #freeze_all_but_lowperf_fc(net)
    freeze_lowperf_model_all(net)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    start = timeit.default_timer()
    #adjust_learning_rate(optimizer, i+1, args.lr)
    func_train(i+1, skip=False)
    stop = timeit.default_timer()
    
    test(i+1, skip=True, update_best=False)
    test(i+1, skip=False)
        
    defreeze_model(net)
    print('Time: {:.3f}'.format(stop - start))

print("Best_Acc_top1 = %.3f" % best_acc)
print("Best_Acc_top5 = %.3f" % best_acc_top5)
'''
