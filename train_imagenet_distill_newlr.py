from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import sys
import os
import argparse

import utils
import timeit
from random import randint

from LRschedule_simple import MyLRScheduler

#Possible arguments
parser = argparse.ArgumentParser(description='Following arguments are used for the script')
parser.add_argument('--lr', default=0.1, type=float, help='Learning Rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--weight_decay', default=4e-5, type=float, help='Weight decay')
parser.add_argument('--batch_size', default=512, type=int, help='Batch_size')
parser.add_argument('--visible_device', default="0", help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--pretrained', default=None, help='Path of a pretrained model file')
parser.add_argument('--starting_epoch', default=0, type=int, help='An epoch which model training starts')
parser.add_argument('--dataset_path', default="/media/data/ILSVRC2012/", help='A path to dataset directory')
parser.add_argument('--model', default="MobileNetV2_skip", help='MobileNetV2_skip')
args = parser.parse_args()

from models.ilsvrc import mobilenetv2_skip
#from models.cifar100 import mobilenetv2_skip
dic_model = {'MobileNetV2_skip': mobilenetv2_skip.MobileNetV2_skip}
    
if args.model not in dic_model:
    print("The model is currently not supported")
    sys.exit()

trainloader = utils.get_traindata('ILSVRC2012',args.dataset_path,batch_size=args.batch_size,download=True, num_workers=8)
testloader = utils.get_testdata('ILSVRC2012',args.dataset_path,batch_size=args.batch_size, num_workers=8)

#args.visible_device sets which cuda devices to be used"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=args.visible_device
device='cuda'

net = dic_model[args.model](num_classes=1000)
net = net.to(device)

#load teacher network

# teacher from pytorch pretrained
net_teacher = torchvision.models.mobilenet_v2(pretrained=True)  # from torchvision
net_teacher = net_teacher.to(device)

# teacher from home-trained
#net_teacher = dic_model[args.model](num_classes=1000)
#teacher_pretrained='./checkpoint/CIFAR100-MobileNetV2_skip-75.35H-noskip.pth'
#checkpoint = torch.load(teacher_pretrained)
#net_teacher.load_state_dict(checkpoint['net_state_dict'], strict=False)


# parallelize 
class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs

net = MyDataParallel(net)
net_teacher = MyDataParallel(net_teacher)


#CrossEntropyLoss for accuracy loss criterion
criterion = nn.CrossEntropyLoss()

# Kullback Leibler divergence loss
criterion_kd = nn.KLDivLoss(reduction='batchmean')


#Training for standard models
def train_alter(epoch):    
    print('\nCuda ' + args.visible_device + ' Epoch: %d' % epoch)
    net.train()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        net.train()
        optimizer.zero_grad()

        # listen what the teach says
        with torch.no_grad():
            net_teacher.eval()
            #outputs_teacher = net_teacher(inputs, skip=False)
            outputs_teacher = net_teacher(inputs)

        alpha = 1.0# 0.7 # 0.1 #0.5 # 1.0 #0.1 #1.0 #1.0
        T = 4
        # forward for the full model
        outputs_full = net(inputs, skip=False)
        _, pred = outputs_full.topk(5, 1, largest=True, sorted=True)
        label_e = targets.view(targets.size(0), -1).expand_as(pred)
        correct = pred.eq(label_e).float()
        correct_top5 += correct[:, :5].sum()
        correct_top1 += correct[:, :1].sum()        
        total += targets.size(0)
        loss_acc = criterion(outputs_full, targets) 
        loss_kd = criterion_kd(F.log_softmax(outputs_full/T, dim=1), F.softmax(outputs_teacher.clone().detach()/T, dim=1)) * T*T
        loss = loss_kd * alpha + loss_acc * (1. - alpha)
        loss.backward()

        alpha = 1.0 # 0.9 #1.0 # 0.1 # 1.0 #1.0 # 0.9
        T = 4 #1 #4
        # forward/backward for the skipped model
        outputs_skip = net(inputs,skip=True)
        _, pred = outputs_skip.topk(5, 1, largest=True, sorted=True)
        label_e = targets.view(targets.size(0), -1).expand_as(pred)
        correct = pred.eq(label_e).float()
        correct_top5 += correct[:, :5].sum()
        correct_top1 += correct[:, :1].sum()        
        total += targets.size(0)
        # learn either from an external teacher or an internal teacher 
        # external teacher
        loss_skip_kd = criterion_kd(F.log_softmax(outputs_skip/T, dim=1), F.softmax(outputs_teacher.detach()/T, dim=1)) * T*T
        # internal teacher
        #loss_skip_kd = criterion_kd(F.log_softmax(outputs_skip/T, dim=1), F.softmax(outputs_full.clone().detach()/T, dim=1)) * T*T
        loss_skip_acc = criterion(outputs_skip, targets) 
        loss_skip = loss_skip_kd * alpha + loss_skip_acc * (1. - alpha)

        if (batch_idx == 0):
            print("kd acc loss: %.6f\t%.6f\t%.6f" % (loss_kd, loss_acc, loss))
            print("kd_skip acc_skip loss_skip: %.6f\t%.6f\t%.6f" % (loss_skip_kd, loss_skip_acc, loss_skip))
            # print(loss_skip_kd.requires_grad)
            # print(loss_skip_acc.requires_grad)
            # print(loss_skip.requires_grad)
        loss_skip.backward()

        # update parameters
        optimizer.step()
    
    acc_top1 = 100.*correct_top1/total
    acc_top5 = 100.*correct_top5/total
    
    print("Training_Acc_Top1/5 = %.3f\t%.3f" % (acc_top1, acc_top5))


#Training for standard models
def train(epoch, skip=False):    
    print('\nCuda ' + args.visible_device + ' Epoch: %d' % epoch)
    #net.train()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
    
        optimizer.zero_grad()
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
            print(loss.requires_grad)
        loss.backward()
        optimizer.step()
    
    acc_top1 = 100.*correct_top1/total
    acc_top5 = 100.*correct_top5/total

    if skip==True:
        print("[skip] Training_Acc_Top1/5 = %.3f\t%.3f" % (acc_top1, acc_top5))
    else:
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

    if update_best == True:
        if acc_top1 > best_acc:
            print('Saving Best..')
            state = {
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': acc_top1,
                'epoch': epoch,
            }
            if not os.path.isdir('./checkpoint'):
                os.mkdir('./checkpoint')
            torch.save(state, './checkpoint/' + 'ILSVRC-' + args.model + "-" + args.visible_device + '.pth')
            best_acc = acc_top1
            best_acc_top5 = acc_top5
            print("Best_Acc_top1/5 = %.3f\t%.3f" % (best_acc, best_acc_top5))
        if epoch % 5 == 0:
            print('Saving..')
            state = {
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': acc_top1,
                'epoch': epoch,
            }
            if not os.path.isdir('./checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/' + 'ILSVRC-' + args.model + "-" + args.visible_device + '-epoch-' + str(epoch) + '.pth')
   

def adjust_learning_rate(optimizer, epoch, args_lr):
    lr = args_lr
    if epoch > 30:
        lr = lr * 0.1
    if epoch > 60:
        lr = lr * 0.1
    if epoch > 90:
        lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


best_acc = 0
best_acc_top5 = 0

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

lrSched_60 = MyLRScheduler(max_lr=0.5, cycle_len=5, warm_up_interval=1)
lrSched_120 = MyLRScheduler(max_lr=0.1, cycle_len=60, warm_up_interval=0)
lrSched_finetune = MyLRScheduler(max_lr=0.01, cycle_len=30, warm_up_interval=0)

if args.pretrained != None:
    checkpoint = torch.load(args.pretrained)
    net.load_state_dict(checkpoint['net_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_acc = checkpoint['acc']

if args.starting_epoch != 0:
    for i in range(0, args.starting_epoch):
        if i < 60:
            lrSched_60.get_lr(i)
        else:
            lrSched_120.get_lr(i)
   
#'''
for i in range(args.starting_epoch, 120):
    start = timeit.default_timer()
    
    if i < 60:
        lr_log = lrSched_60.get_lr(i)
    else:
        lr_log = lrSched_120.get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_log
    
    train_alter(i+1)
    
    stop = timeit.default_timer()
    
    test(i+1, skip=True)
    test(i+1, skip=False)
    print("LR for epoch {} = {:.5f}".format(i+1, lr_log))
    print('Time: {:.3f}'.format(stop - start))

print("Best_Acc_top1 = %.3f" % best_acc)
print("Best_Acc_top5 = %.3f" % best_acc_top5)

# save & load a best performing model
checkpoint = torch.load('./checkpoint/' + 'ILSVRC-' + args.model + "-" + args.visible_device + '.pth')
torch.save(checkpoint, './checkpoint/' + 'ILSVRC-' + args.model + "-" + args.visible_device + '-nofinetuned'+'.pth')
net.load_state_dict(checkpoint['net_state_dict'], strict=False)
#'''
