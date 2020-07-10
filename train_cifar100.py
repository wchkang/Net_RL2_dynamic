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
parser.add_argument('--lambdaR', default=10, type=float, help='Lambda (Basis regularization)')
parser.add_argument('--shared_rank', default=16, type=int, help='Number of shared base)')
parser.add_argument('--unique_rank', default=1, type=int, help='Number of unique base')
parser.add_argument('--batch_size', default=256, type=int, help='Batch_size')
parser.add_argument('--visible_device', default="0", help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--pretrained', default=None, help='Path of a pretrained model file')
parser.add_argument('--starting_epoch', default=0, type=int, help='An epoch which model training starts')
parser.add_argument('--dataset_path', default="./data/", help='A path to dataset directory')
parser.add_argument('--model', default="ResNet34_SingleShared", help='ResNet18, ResNet34, ResNet34_SingleShared, ResNet34_NonShared, ResNet34_SharedOnly, DenseNet121, DenseNet121_SingleShared, ResNext50, ResNext50_SingleShared')
args = parser.parse_args()

from models.cifar100 import resnet, densenet, resnext
dic_model = {'ResNet18': resnet.ResNet18, 'ResNet34':resnet.ResNet34, 'ResNet34_SingleShared':resnet.ResNet34_SingleShared, 'ResNet34_NonShared':resnet.ResNet34_NonShared, 'ResNet34_SharedOnly':resnet.ResNet34_SharedOnly, 'DenseNet121':densenet.DenseNet121, 'DenseNet121_SingleShared':densenet.DenseNet121_SingleShared, 'ResNext50':resnext.ResNext50_32x4d, 'ResNext50_SingleShared':resnext.ResNext50_32x4d_SingleShared}
    
if args.model not in dic_model:
    print("The model is currently not supported")
    sys.exit()

trainloader = utils.get_traindata('CIFAR100',args.dataset_path,batch_size=args.batch_size,download=True)
testloader = utils.get_testdata('CIFAR100',args.dataset_path,batch_size=args.batch_size)

#args.visible_device sets which cuda devices to be used
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=args.visible_device
device='cuda'

if 'SingleShared' in args.model:
    net = dic_model[args.model](args.shared_rank, args.unique_rank)
elif 'SharedOnly' in args.model:
    net = dic_model[args.model](args.shared_rank)
elif 'NonShared' in args.model:
    net = dic_model[args.model](args.unique_rank)
else:
    net = dic_model[args.model]()
    
net = net.to(device)
                    
#CrossEntropyLoss for accuracy loss criterion
criterion = nn.CrossEntropyLoss()

#Training for standard models
def train(epoch):
    print('\nCuda ' + args.visible_device + ' Epoch: %d' % epoch)
    net.train()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
    
        optimizer.zero_grad()
        outputs = net(inputs)
        
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
    
    print("Training_Acc_Top1 = %.3f" % acc_top1)
    print("Training_Acc_Top5 = %.3f" % acc_top5)

def train_basis(epoch, skip=False):
    print('\nCuda ' + args.visible_device + ' Basis Epoch: %d' % epoch)

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
    
        optimizer.zero_grad()
        outputs = net(inputs, skip)

        _, pred = outputs.topk(5, 1, largest=True, sorted=True)

        label_e = targets.view(targets.size(0), -1).expand_as(pred)
        correct = pred.eq(label_e).float()

        correct_top5 += correct[:, :5].sum()
        correct_top1 += correct[:, :1].sum()        
        total += targets.size(0)
        
        # get similarity of basis filters
        cnt_sim = 0 
        sim = 0
        for gid in range(1, 5):  # all models have 4 groups
            shared_basis = getattr(net,"shared_basis_"+str(gid))

            num_shared_basis = shared_basis.weight.shape[0]
            num_all_basis = num_shared_basis 

            all_basis =(shared_basis.weight,)

            B = torch.cat(all_basis).view(num_all_basis, -1)
            #print("B size:", B.shape)

            # compute orthogonalities btwn all baisis  
            D = torch.mm(B, torch.t(B)) 

            # make diagonal zeros
            D = (D - torch.eye(num_all_basis, num_all_basis, device=device))**2
            
            sim += torch.sum(D[0:num_shared_basis,0:num_shared_basis])
            cnt_sim += num_shared_basis**2

        #average similarity
        avg_sim = sim / cnt_sim

        #acc loss
        loss = criterion(outputs, targets)

        if (batch_idx == 0):
            print("accuracy_loss: %.6f" % loss)
            print("similarity loss: %.6f" % avg_sim)

        #apply similarity loss, multiplied by args.lambdaR
        loss = loss + avg_sim * args.lambdaR
        loss.backward()
        optimizer.step()
        
    acc_top1 = 100.*correct_top1/total
    acc_top5 = 100.*correct_top5/total
    
    if (skip==False):
        print("Training_Acc_Top1 = %.3f" % acc_top1)
        #print("Training_Acc_Top5 = %.3f" % acc_top5)
    else:
        print("[Skip] Training_Acc_top1 = %.3f" % acc_top1)
        #print("[Skip] Training_Acc_top5 = %.3f" % acc_top5)
                                
#Test for models
def test(epoch, skip=False, update_best=True):
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
    print("Test_Acc_top1 = %.3f" % acc_top1)
    #print("Test_Acc_top5 = %.3f" % acc_top5)

    if update_best==True and acc_top1 > best_acc:
        #print('Saving..')
        state = {
            'net_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': acc_top1,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + 'CIFAR100-' + args.model + "-S" + str(args.shared_rank) + "-U" + str(args.unique_rank) + "-L" + str(args.lambdaR) + "-" + args.visible_device + '.pth')
        best_acc = acc_top1
        best_acc_top5 = acc_top5
        print("Best_Acc_top1 = %.3f" % acc_top1)
        #print("Best_Acc_top5 = %.3f" % acc_top5)


def freeze_lowperf_model(net):
    '''Freeze low-performance mode while enabling the training of high-perf model'''
    # bn layers need to be freezed explicitly since they cannot be freezed via '.requires_grad=False'
    for module in net.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            module.eval()

    # freeze all parameters
    for param in net.parameters():
        param.requires_grad = False

    # defreeze params of only being used by the high-performance model
    num_blocks =[0, 3, 4, 6, 3]
    for i in range(1,5): # Layers. Skip the first layer
        layer = getattr(net,"layer"+str(i))
        num_skip_blocks = int(len(layer)/2)
        for j in range(num_skip_blocks, num_blocks[i]): # blocks. Skip the first block
            #print("layer: %s, block: %s" %(i, j))
            layer[j].coeff_conv1.weight.requires_grad = True
            layer[j].coeff_conv2.weight.requires_grad = True
            layer[j].basis_bn1.train()
            layer[j].basis_bn2.train()
            layer[j].bn1.train()
            layer[j].bn2.train()
            if num_skip_blocks == 1: 
            # basis is used only for high-perf models. Hence needs retraining.
                layer[j].shared_basis.weight.requires_grad = True

    # defreeze params of FC layer
    net.fc.weight.requires_grad = True
    net.fc.bias.requires_grad = True        

def freeze_highperf_model(net):
    '''Freeze high-performance mode while enabling the training of low-perf model'''
    # freeze params of only being used by the high-performance model
    num_blocks =[0, 3, 4, 6, 3]
    for i in range(1,5): # Layers. Skip the first layer
        layer = getattr(net,"layer"+str(i))
        num_skip_blocks = int(len(layer)/2)
        for j in range(num_skip_blocks, num_blocks[i]): # blocks. Skip the first block
            #print("layer: %s, block: %s" %(i, j))
            layer[j].coeff_conv1.weight.requires_grad = False
            layer[j].coeff_conv2.weight.requires_grad = False
            layer[j].basis_bn1.eval()
            layer[j].basis_bn2.eval()
            layer[j].bn1.eval()
            layer[j].bn2.eval()
            if num_skip_blocks == 1: 
            # basis is used only for high-perf models. Hence needs retraining.
                layer[j].shared_basis.weight.requires_grad = False
    # freeze params of FC layer
    net.fc.weight.requires_grad = False
    net.fc.bias.requires_grad = False

def defreeze_model(net, freeze_bn=True):
    # defreeze all parameters
    for param in net.parameters():
        param.requires_grad = True
    # make the whole network trainable
    net.train()

best_acc = 0
best_acc_top5 = 0

func_train = train
if 'SingleShared' in args.model or 'SharedOnly' in args.model:
    func_train = train_basis

if args.pretrained != None:
    checkpoint = torch.load(args.pretrained)
    net.load_state_dict(checkpoint['net_state_dict'])
    best_acc = checkpoint['acc']

print('\n######### Alternate Training Low- and High-Performance Model ###########\n')

train_epochs = (150,75,75)
base_epoch = [1,]
for i in range(1, len(train_epochs)):
    base_epoch.append(base_epoch[i-1]+train_epochs[i-1])

net.train()
for i in range(args.starting_epoch, train_epochs[0]):
    if (randint(0,1) == 0):
        skip = True
        freeze_highperf_model(net)
    else:
        skip = False
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    start = timeit.default_timer()
    func_train(i+1, skip)
    test(base_epoch[0]+i, skip=True)
    test(base_epoch[0]+i, skip=False)
    stop = timeit.default_timer()
    defreeze_model(net)
    print('skip:', skip)
    print('Time: {:.3f}'.format(stop - start))  

    #============
    
checkpoint = torch.load('./checkpoint/' + 'CIFAR100-' + args.model + "-S" + str(args.shared_rank) + "-U" + str(args.unique_rank) + "-L" + str(args.lambdaR) + "-" + args.visible_device + '.pth')
net.load_state_dict(checkpoint['net_state_dict'])
best_acc = checkpoint['acc']

for i in range(args.starting_epoch, train_epochs[1]):
    if (randint(0,1) == 0):
        skip = True
        freeze_highperf_model(net)
    else:
        skip = False
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr*0.1, momentum=args.momentum, weight_decay=args.weight_decay)
    start = timeit.default_timer()
    func_train(i+151, skip)
    test(base_epoch[1]+i, skip=True)
    test(base_epoch[1]+i, skip=False)
    stop = timeit.default_timer()
    defreeze_model(net)
    print('skip:', skip)
    print('Time: {:.3f}'.format(stop - start))  
    
    #============
    
checkpoint = torch.load('./checkpoint/' + 'CIFAR100-' + args.model + "-S" + str(args.shared_rank) + "-U" + str(args.unique_rank) + "-L" + str(args.lambdaR) + "-" + args.visible_device + '.pth')
net.load_state_dict(checkpoint['net_state_dict'])
best_acc = checkpoint['acc']

for i in range(args.starting_epoch, train_epochs[2]):
    if (randint(0,1) == 0):
        skip = True
        freeze_highperf_model(net)
    else:
        skip = False
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr*0.01, momentum=args.momentum, weight_decay=args.weight_decay)
    start = timeit.default_timer()
    func_train(i+226, skip)
    test(base_epoch[2]+i, skip=True)
    test(base_epoch[2]+i, skip=False)
    stop = timeit.default_timer()
    defreeze_model(net)
    print('skip:', skip)
    print('Time: {:.3f}'.format(stop - start))  

print("Best_Acc_top1 = %.3f" % best_acc)
print("Best_Acc_top5 = %.3f" % best_acc_top5)

# save the model with the best performance. 
checkpoint = torch.load('./checkpoint/' + 'CIFAR100-' + args.model + "-S" + str(args.shared_rank) + "-U" + str(args.unique_rank) + "-L" + str(args.lambdaR) + "-" + args.visible_device + '.pth')
net.load_state_dict(checkpoint['net_state_dict'])
best_acc = checkpoint['acc']
torch.save(checkpoint, './checkpoint/' + 'CIFAR100-' + args.model + "-S" \
    + str(args.shared_rank) + "-U" + str(args.unique_rank) + "-L" + str(args.lambdaR) + "-" \
    + 'nofinetuned-' + args.visible_device + '.pth')

print('\n######### Finetuning High-Performance Model ###########\n')

best_acc = 0
best_acc_top5 = 0

defreeze_model(net)
freeze_lowperf_model(net)

optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr*0.1, momentum=args.momentum, weight_decay=args.weight_decay)

for i in range(args.starting_epoch, 75):
    start = timeit.default_timer()
    func_train(1+i, skip=False)
    test(1+i, skip=True, update_best=False)
    test(1+i, skip=False)
    stop = timeit.default_timer()

    print('Time: {:.3f}'.format(stop - start))  

checkpoint = torch.load('./checkpoint/' + 'CIFAR100-' + args.model + "-S" + str(args.shared_rank) + "-U" + str(args.unique_rank) + "-L" + str(args.lambdaR) + "-" + args.visible_device + '.pth')
net.load_state_dict(checkpoint['net_state_dict'])
best_acc = checkpoint['acc']

optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr*0.01, momentum=args.momentum, weight_decay=args.weight_decay)

for i in range(args.starting_epoch, 75):
    start = timeit.default_timer()
    func_train(76+i, skip=False)
    test(76+i, skip=True, update_best=False)
    test(76+i, skip=False)
    stop = timeit.default_timer()

    print('Time: {:.3f}'.format(stop - start))  

checkpoint = torch.load('./checkpoint/' + 'CIFAR100-' + args.model + "-S" + str(args.shared_rank) + "-U" + str(args.unique_rank) + "-L" + str(args.lambdaR) + "-" + args.visible_device + '.pth')
net.load_state_dict(checkpoint['net_state_dict'])
best_acc = checkpoint['acc']

optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr*0.005, momentum=args.momentum, weight_decay=args.weight_decay)

for i in range(args.starting_epoch, 75):
    start = timeit.default_timer()
    func_train(151+i, skip=False)
    test(151+i, skip=True, update_best=False)
    test(151+i, skip=False)
    stop = timeit.default_timer()

    print('Time: {:.3f}'.format(stop - start))  


print("Best_Acc_top1 = %.3f" % best_acc)
print("Best_Acc_top5 = %.3f" % best_acc_top5)