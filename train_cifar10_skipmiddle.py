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
parser.add_argument('--model', default="ResNet56_DoubleShared", help='ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, ResNet1202, ResNet56_DoubleShared, ResNet32_DoubleShared, ResNet56_SingleShared, ResNet32_SingleShared, ResNet56_SharedOnly, ResNet32_SharedOnly, ResNet56_NonShared, ResNet32_NonShared')
args = parser.parse_args()

#from models.cifar10 import resnet
#dic_model = {'ResNet20': resnet.ResNet20, 'ResNet32':resnet.ResNet32,'ResNet44':resnet.ResNet44,'ResNet56':resnet.ResNet56, 'ResNet110':resnet.ResNet110, 'ResNet1202':resnet.ResNet1202, 'ResNet56_DoubleShared':resnet.ResNet56_DoubleShared, 'ResNet32_DoubleShared':resnet.ResNet32_DoubleShared, 'ResNet56_SingleShared':resnet.ResNet56_SingleShared, 'ResNet32_SingleShared':resnet.ResNet32_SingleShared, 'ResNet56_SharedOnly':resnet.ResNet56_SharedOnly, 'ResNet32_SharedOnly':resnet.ResNet32_SharedOnly, 'ResNet56_NonShared':resnet.ResNet56_NonShared, 'ResNet32_NonShared':resnet.ResNet32_NonShared}
from models.cifar10 import resnet_skipmiddle
dic_model = {'ResNet56_DoubleShared':resnet_skipmiddle.ResNet56_DoubleShared, 'ResNet32_DoubleShared':resnet_skipmiddle.ResNet32_DoubleShared}


if args.model not in dic_model:
    print("The model is currently not supported")
    sys.exit()

trainloader = utils.get_traindata('CIFAR10',args.dataset_path,batch_size=args.batch_size,download=True)
testloader = utils.get_testdata('CIFAR10',args.dataset_path,batch_size=args.batch_size)

#args.visible_device sets which cuda devices to be used"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=args.visible_device
device='cuda'

if 'DoubleShared' in args.model or 'SingleShared' in args.model:
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

def train_basis(epoch, skip=False, include_unique_basis=True):
    """Train roultine for DoubleShared CIFAR10 models.

    Arguments:
        skip: if True, skip some blocks
        include_unique_basis: if True, include unique basis for calculating similarity loss
    """
    print('\nCuda ' + args.visible_device + ' Basis Epoch: %d' % epoch)
    #net.train()  % 
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
    
        optimizer.zero_grad()
        outputs = net(inputs, skip=skip)
           
        _, pred = outputs.topk(5, 1, largest=True, sorted=True)

        label_e = targets.view(targets.size(0), -1).expand_as(pred)
        correct = pred.eq(label_e).float()

        correct_top5 += correct[:, :5].sum()
        correct_top1 += correct[:, :1].sum()        
        total += targets.size(0)
        
        # get similarity of basis filters
        cnt_sim = 0 
        sim = 0
        for gid in range(1, 4):  # ResNet for CIFAR10 has 3 groups
            layer = getattr(net, "layer"+str(gid))
            shared_basis_1 = getattr(net,"shared_basis_"+str(gid)+"_1")
            shared_basis_2 = getattr(net,"shared_basis_"+str(gid)+"_2")

            num_shared_basis = shared_basis_2.weight.shape[0] + shared_basis_1.weight.shape[0]
            num_all_basis = num_shared_basis 

            all_basis =(shared_basis_1.weight, shared_basis_2.weight, )
            if (include_unique_basis == True):  
                num_unique_basis = layer[1].basis_conv1.weight.shape[0] 
                num_all_basis += (num_unique_basis * 2 * (len(layer) -1))
                for i in range(1, len(layer)):
                    all_basis += (layer[i].basis_conv1.weight, layer[i].basis_conv2.weight,)

            B = torch.cat(all_basis).view(num_all_basis, -1)
            #print("B size:", B.shape)

            # compute orthogonalities btwn all baisis  
            D = torch.mm(B, torch.t(B)) 

            # make diagonal zeros
            D = (D - torch.eye(num_all_basis, num_all_basis, device=device))**2
            
            #print("D size:", D.shape)
           
            if (include_unique_basis == True):  
                # orthogonalities btwn shared<->(shared/unique)
                sim += torch.sum(D[0:num_shared_basis,:])  
                cnt_sim += num_shared_basis*num_all_basis
                
                # orthogonalities btwn unique<->unique in the same layer
                for i in range(1, len(layer)):
                    for j in range(2):  # conv1 & conv2
                         idx_base = num_shared_basis   \
                          + (i-1) * (num_unique_basis) * 2 \
                          + num_unique_basis * j 
                         sim += torch.sum(\
                                 D[idx_base:idx_base + num_unique_basis, \
                                 idx_base:idx_base+num_unique_basis])
                         cnt_sim += num_unique_basis ** 2 
            else: # orthogonalities only btwn shared basis
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
    
    if skip==True:
        print("[skip] Training_Acc_Top1 = %.3f" % acc_top1)
    else:
        print("Training_Acc_Top1 = %.3f" % acc_top1)
    #print("Training_Acc_Top5 = %.3f" % acc_top5)
    
def train_basis_single(epoch, include_unique_basis=True):
    """ Train roultine for SingleShared CIFAR10 models. 

        TODO (wchkang/20200713): update to make it skippable
    """
    print('\nCuda ' + args.visible_device + ' Basis Epoch: %d' % epoch)
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
        
        # get similarity of basis filters
        cnt_sim = 0 
        sim = 0
        for gid in range(1, 4):  # ResNet for CIFAR10 has 3 groups
            layer = getattr(net, "layer"+str(gid))
            shared_basis = getattr(net,"shared_basis_"+str(gid))

            num_shared_basis = shared_basis.weight.shape[0]
            num_all_basis = num_shared_basis 

            all_basis =(shared_basis.weight, )
            if (include_unique_basis == True):  
                num_unique_basis = layer[1].basis_conv1.weight.shape[0] 
                num_all_basis += (num_unique_basis * 2 * (len(layer) -1))
                for i in range(1, len(layer)):
                    all_basis += (layer[i].basis_conv1.weight, layer[i].basis_conv2.weight,)

            B = torch.cat(all_basis).view(num_all_basis, -1)
            #print("B size:", B.shape)

            # compute orthogonalities btwn all baisis  
            D = torch.mm(B, torch.t(B)) 

            # make diagonal zeros
            D = (D - torch.eye(num_all_basis, num_all_basis, device=device))**2
            
            #print("D size:", D.shape)
           
            if (include_unique_basis == True):  
                # orthogonalities btwn shared<->(shared/unique)
                sim += torch.sum(D[0:num_shared_basis,:])  
                cnt_sim += num_shared_basis*num_all_basis
                
                # orthogonalities btwn unique<->unique in the same layer
                for i in range(1, len(layer)):
                    for j in range(2):  # conv1 & conv2
                         idx_base = num_shared_basis   \
                          + (i-1) * (num_unique_basis) * 2 \
                          + num_unique_basis * j 
                         sim += torch.sum(\
                                 D[idx_base:idx_base + num_unique_basis, \
                                 idx_base:idx_base+num_unique_basis])
                         cnt_sim += num_unique_basis ** 2 
            else: # orthogonalities only btwn shared basis
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
    
    print("Training_Acc_Top1 = %.3f" % acc_top1)
    print("Training_Acc_Top5 = %.3f" % acc_top5)
    
def train_basis_sharedonly(epoch):
    """ Train roultine for ShareOnly CIFAR10 models. 

    TODO (wchkang/20200713): update to make it skippable
    """
    print('\nCuda ' + args.visible_device + ' Basis Epoch: %d' % epoch)
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
        
        # get similarity of basis filters
        cnt_sim = 0 
        sim = 0
        for gid in range(1, 4):  # ResNet for CIFAR10 has 3 groups
            layer = getattr(net, "layer"+str(gid))
            shared_basis_1 = getattr(net,"shared_basis_"+str(gid)+"_1")
            shared_basis_2 = getattr(net,"shared_basis_"+str(gid)+"_2")

            num_shared_basis = shared_basis_2.weight.shape[0] + shared_basis_1.weight.shape[0]
            num_all_basis = num_shared_basis 

            all_basis =(shared_basis_1.weight, shared_basis_2.weight, )

            B = torch.cat(all_basis).view(num_all_basis, -1)
            #print("B size:", B.shape)

            # compute orthogonalities btwn all baisis  
            D = torch.mm(B, torch.t(B)) 

            # make diagonal zeros
            D = (D - torch.eye(num_all_basis, num_all_basis, device=device))**2
            
            #print("D size:", D.shape)
          
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
    
    print("Training_Acc_Top1 = %.3f" % acc_top1)
    print("Training_Acc_Top5 = %.3f" % acc_top5)
    
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
    print("Test_Acc_top1 = %.3f" % acc_top1)

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
        torch.save(state, './checkpoint/' + 'CIFAR10-' + args.model + "-S" + str(args.shared_rank) + "-U" + str(args.unique_rank) + "-L" + str(args.lambdaR) + "-" + args.visible_device + '.pth')
        best_acc = acc_top1
        best_acc_top5 = acc_top5
        print("Best_Acc_top1 = %.3f" % acc_top1)
        #print("Best_Acc_top5 = %.3f" % acc_top5)


def freeze_highperf_model(net):
    """ Freeze the high-performance model while enabling the training of the low-perf model. """
    
    # freeze params of only being used by the high-performance model
    for i in range(1,4): # CIFAR10 layers. Skip the first layer
        layer = getattr(net,"layer"+str(i))
        #num_skip_blocks = round(len(layer)/2)
        num_skip_blocks = len(layer)//2
        layer[1].bn2.eval()
        layer[1].coeff_conv2.weight.requires_grad = False
        for j in range(2, 2+num_skip_blocks): # CIFAR10 blocks of the high-perf model
            #print("layer: %s, block: %s" %(i, j))
            layer[j].coeff_conv1.weight.requires_grad = False
            layer[j].coeff_conv2.weight.requires_grad = False
            layer[j].basis_bn1.eval()
            layer[j].basis_bn2.eval()
            layer[j].bn1.eval()
            layer[j].bn2.eval()
    # freeze params of high-perf FC layer
    net.fc.weight.requires_grad = False
    net.fc.bias.requires_grad = False

def freeze_lowperf_model(net):
    """ Freeze parts of low-performance model while enabling the training of high-perf model """
    for i in range(1,4): # Layers. Skip the first layer
        layer = getattr(net,"layer"+str(i))
        layer[1].bn2_skip.eval()
        layer[1].coeff_conv2_skip.weight.requires_grad = False
    # freeze params of low-perf FC layer
    net.fc_skip.weight.requires_grad = False
    net.fc_skip.bias.requires_grad = False

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
    for i in range(1,4): # Layers. Skip the first layer
        layer = getattr(net,"layer"+str(i))
        num_skip_blocks = len(layer)//2
        for j in range(2, 2+num_skip_blocks): # CIFAR10 blocks of the high-perf model
            layer[j].coeff_conv1.weight.requires_grad = True
            layer[j].coeff_conv2.weight.requires_grad = True
            layer[j].basis_bn1.train()
            layer[j].basis_bn2.train()
            layer[j].bn1.train()
            layer[j].bn2.train()
    # defreeze params of high-perf FC layer
    net.fc.weight.requires_grad = True
    net.fc.bias.requires_grad = True  

def freeze_all_but_lowperf_fc(net):
    """ Used to freeze all parameters except 'bn2_skip' and 'fc_skip' layers. """
    # bn layers need to be freezed explicitly since they cannot be freezed via '.requires_grad=False'
    for module in net.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            module.eval()
    
    # freeze all parameters
    for param in net.parameters():
        param.requires_grad = False
    
    # make intermediate BNs trainable
    for i in range(1,4):
        layer = getattr(net, "layer"+str(i))
        n_skip = len(layer)//2 
        layer[1].coeff_conv2_skip.weight.requires_grad=True
        layer[1].bn2_skip.train()
        for param in layer[1].bn2_skip.parameters():
            param.requires_grad = True
        
    net.fc_skip.weight.requires_grad = True
    net.fc_skip.bias.requires_grad = True

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
    if epoch > 225:
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

func_train = train
if 'DoubleShared' in args.model:
    func_train = train_basis
elif 'SingleShared' in args.model:
    func_train = train_basis_single
elif 'SharedOnly' in args.model:
    func_train = train_basis_sharedonly

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
if args.pretrained != None:
    checkpoint = torch.load(args.pretrained)
    net.load_state_dict(checkpoint['net_state_dict'], strict=False)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_acc = checkpoint['acc']
'''
net.train()
for i in range(args.starting_epoch, 500):
    if (randint(0,2) == 0): # give more chance to high-perf model
        skip = True
        freeze_highperf_model(net)
    else:
        skip = False     
        freeze_lowperf_model(net)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    start = timeit.default_timer()
    
    adjust_learning_rate_long(optimizer, i+1, args.lr)
    func_train(i+1, skip=skip)
    
    stop = timeit.default_timer()
    
    test(i+1, skip=True)
    test(i+1, skip=False)
        
    defreeze_model(net)

    print("Skip:", skip)
    print('Time: {:.3f}'.format(stop - start))

print("Best_Acc_top1 = %.3f" % best_acc)
print("Best_Acc_top5 = %.3f" % best_acc_top5)

'''
## finetuning
best_acc = 0
best_acc_top5 = 0
net.train()
for i in range(args.starting_epoch, 150):
    freeze_all_but_lowperf_fc(net)
    #freeze_lowperf_model_all(net)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr*0.01, momentum=args.momentum, weight_decay=args.weight_decay)

    start = timeit.default_timer()
    adjust_learning_rate_finetune(optimizer, i+1, args.lr)
    func_train(i+1, skip=True)
    stop = timeit.default_timer()
    
    test(i+1, skip=True, update_best=True)
    test(i+1, skip=False, update_best=False)
        
    defreeze_model(net)
    print('Time: {:.3f}'.format(stop - start))

print("Best_Acc_top1 = %.3f" % best_acc)
print("Best_Acc_top5 = %.3f" % best_acc_top5)
