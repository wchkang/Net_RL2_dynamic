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

from models.cifar import resnet, resnet_basis
import utils

#Possible arguments
parser = argparse.ArgumentParser(description='TODO')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--lambda2', default=0.5, type=float, help='lambda2 (for basis loss)')
parser.add_argument('--shared_rank', default=16, type=int, help='number of shared base)')
parser.add_argument('--dataset', default="CIFAR100", help='CIFAR10, CIFAR100')
parser.add_argument('--batch_size', default=256, type=int, help='batch_size')
parser.add_argument('--model', default="ResNet34", help='ResNet152, ResNet101, ResNet50, ResNet34, ResNet18, ResNet34_Basis, ResNet18_Basis')
parser.add_argument('--visible_device', default="0", help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--unique_rank', default=16, type=int, help='number of unique base')
parser.add_argument('--pretrained', default=None, help='path of a pretrained model file')
parser.add_argument('--starting_epoch', default=0, type=int, help='an epoch which model training starts')
args = parser.parse_args()

lr = args.lr
momentum = args.momentum
weight_decay = args.weight_decay
lambda2 = args.lambda2
shared_rank = args.shared_rank
unique_rank = args.unique_rank

dic_dataset = {'CIFAR100':100, 'CIFAR10':10}
dic_model = {'ResNet152':resnet.ResNet152,'ResNet101':resnet.ResNet101,'ResNet50':resnet.ResNet50,'ResNet34':resnet.ResNet34,'ResNet18':resnet.ResNet18,'ResNet34_Basis':resnet_basis.ResNet34_Basis,'ResNet18_Basis':resnet_basis.ResNet18_Basis, 'ResNet34_Unique':resnet_basis.ResNet34_Unique}

if args.dataset not in dic_dataset:
    print("The dataset is currently not supported")
    sys.exit()

if args.model not in dic_model:
    print("The model is currently not supported")
    sys.exit()

trainloader = utils.get_traindata(args.dataset,"./data",batch_size=args.batch_size,download=True)
testloader = utils.get_testdata(args.dataset,"./data",batch_size=args.batch_size)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=args.visible_device
device='cuda'
#args.visible_device sets which cuda devices to be used"

if 'Basis' in args.model:
    net = dic_model[args.model](dic_dataset[args.dataset], shared_rank, unique_rank)
elif 'Unique' in args.model:
    net = dic_model[args.model](dic_dataset[args.dataset], unique_rank)
else:
    net = dic_model[args.model](dic_dataset[args.dataset])
    
net = net.to(device)

if args.pretrained != None:
    checkpoint = torch.load(args.pretrained)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
                    
#CrossEntropyLoss for accuracy loss criterion
criterion = nn.CrossEntropyLoss()

#Unused - reserved for different LR schedulers
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#Training for standard models
def train(epoch):
    if epoch < args.starting_epoch:
        return
    
    print('\nCuda ' + args.visible_device + ' Epoch: %d' % epoch)
    net.train()
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
    
        optimizer.zero_grad()
        outputs = net(inputs)
                        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
#Training for models with unique base only    
def train_unique(epoch):
    if epoch < args.starting_epoch:
        return
    
    print('\nCuda ' + args.visible_device + ' Epoch: %d' % epoch)
    net.train()
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
    
        optimizer.zero_grad()
        outputs = net(inputs)
                        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
#Training for parameter shared models
def train_basis(epoch):
    if epoch < args.starting_epoch:
        return
    
    print('\nCuda ' + args.visible_device + ' Basis Epoch: %d' % epoch)
    net.train()
    
    cos_simil= nn.CosineSimilarity(dim=-1)
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
    
        optimizer.zero_grad()
        outputs = net(inputs)
        
        sum_simil=0
        sum_cnt=0
        
        #absolute sum of cos similarity between every shared base
        #CosineSimilarity calculates cosine similarity of tensors along dim=1
        
        #group 1
        conv1_tuple = (net.shared_basis_1.weight,)
        conv2_tuple = (net.shared_basis_1.weight,)
        if unique_rank != 0:
            for i in range(1,len(net.layer1)):
                conv1_tuple = conv1_tuple + (net.layer1[i].basis_conv1.weight,)
                conv2_tuple = conv2_tuple + (net.layer1[i].basis_conv2.weight,)
        conv1_all_basis = torch.cat(conv1_tuple)
        conv2_all_basis = torch.cat(conv2_tuple)
        #conv1_all_basis and conv2_all_basis contains every base of residual blocks in group 1
        
        len_shared = net.shared_basis_1.weight.shape[0]
        if unique_rank != 0:
            len_unique = net.layer1[1].basis_conv1.weight.shape[0]
        else:
            len_unique = 0
        
        #for every basis in conv1_all_basis and conv2_all_basis, calculates similarities to every other basis in the convx_all_basis tensor
        #convx_abssum_simil[i+1:]=
        #skips calculating itself (which the result is always 1.0)
        #skips already calculated similarity
        for i in range(len_shared):
            if len_unique == 0 and i == len_shared-1:
                break
            conv1_abssum_simil = abs(cos_simil(
                conv1_all_basis[i+1:].view(conv1_all_basis.shape[0]-1-i,-1),
                conv1_all_basis[i].view(-1)
            ))
            conv2_abssum_simil = abs(cos_simil(
                conv2_all_basis[i+1:].view(conv2_all_basis.shape[0]-1-i,-1),
                conv2_all_basis[i].view(-1)
            ))
            #sum_simil contains every absolute sum of cos similarity
            #sum_cnt contains total number of base
            sum_simil=sum_simil + torch.sum(conv1_abssum_simil)
            sum_cnt=sum_cnt + conv1_abssum_simil.shape[0]
            sum_simil=sum_simil + torch.sum(conv2_abssum_simil)
            sum_cnt=sum_cnt + conv2_abssum_simil.shape[0]
            
        #similarity between unique base in a same residual block
        if unique_rank != 0:
            for i in range(1,len(net.layer1)):
                for j in range(len_unique-1):
                    #index of unique basis in convx_all_basis
                    idx = len_shared+(i-1)*len_unique + j
                    #index of last unique basis within the same residual block in convx_all_basis
                    idx_end = len_shared+i*len_unique

                    #similarity between unique base in a same residual block
                    conv1_abssum_simil = abs(cos_simil(
                        conv1_all_basis[idx+1:idx_end].view(idx_end-idx-1,-1),
                        conv1_all_basis[idx].view(-1)
                    ))
                    conv2_abssum_simil = abs(cos_simil(
                        conv2_all_basis[idx+1:idx_end].view(idx_end-idx-1,-1),
                        conv2_all_basis[idx].view(-1)
                    ))
                    sum_simil=sum_simil + torch.sum(conv1_abssum_simil)
                    sum_cnt=sum_cnt + conv1_abssum_simil.shape[0]
                    sum_simil=sum_simil + torch.sum(conv2_abssum_simil)
                    sum_cnt=sum_cnt + conv2_abssum_simil.shape[0]
            
        #group 2
        conv1_tuple = (net.shared_basis_2.weight,)
        conv2_tuple = (net.shared_basis_2.weight,)
        if unique_rank != 0:
            for i in range(1,len(net.layer2)):
                conv1_tuple = conv1_tuple + (net.layer2[i].basis_conv1.weight,)
                conv2_tuple = conv2_tuple + (net.layer2[i].basis_conv2.weight,)
        conv1_all_basis = torch.cat(conv1_tuple)
        conv2_all_basis = torch.cat(conv2_tuple)
        #conv1_all_basis and conv2_all_basis contains every basis of residual blocks in group 2
        
        len_shared = net.shared_basis_2.weight.shape[0]
        if unique_rank != 0:
            len_unique = net.layer2[1].basis_conv1.weight.shape[0]
        else:
            len_unique = 0
        
        #for every basis in conv1_all_basis and conv2_all_basis, calculates similarities to every other basis in the convx_all_basis tensor
        #convx_all_basis[i+1:]=
        #skips calculating itself (which the result is always 1.0)
        #skips already calculated similarity

        for i in range(len_shared):
            if len_unique == 0 and i == len_shared-1:
                break
            conv1_abssum_simil = abs(cos_simil(
                conv1_all_basis[i+1:].view(conv1_all_basis.shape[0]-1-i,-1),
                conv1_all_basis[i].view(-1)
            ))
            conv2_abssum_simil = abs(cos_simil(
                conv2_all_basis[i+1:].view(conv2_all_basis.shape[0]-1-i,-1),
                conv2_all_basis[i].view(-1)
            ))
            #sum_simil contains every absolute sum of cos similarity
            #sum_cnt contains total number of base
            sum_simil=sum_simil + torch.sum(conv1_abssum_simil)
            sum_cnt=sum_cnt + conv1_abssum_simil.shape[0]
            sum_simil=sum_simil + torch.sum(conv2_abssum_simil)
            sum_cnt=sum_cnt + conv2_abssum_simil.shape[0]
            
        #similarity between unique base in a same residual block
        if unique_rank != 0:
            for i in range(1,len(net.layer2)):
                for j in range(len_unique-1):
                    #index of unique basis in convx_all_basis
                    idx = len_shared+(i-1)*len_unique + j
                    #index of last unique basis within the same residual block in convx_all_basis
                    idx_end = len_shared+i*len_unique

                    #similarity between unique base in a same residual block
                    conv1_abssum_simil = abs(cos_simil(
                        conv1_all_basis[idx+1:idx_end].view(idx_end-idx-1,-1),
                        conv1_all_basis[idx].view(-1)
                    ))
                    conv2_abssum_simil = abs(cos_simil(
                        conv2_all_basis[idx+1:idx_end].view(idx_end-idx-1,-1),
                        conv2_all_basis[idx].view(-1)
                    ))
                    sum_simil=sum_simil + torch.sum(conv1_abssum_simil)
                    sum_cnt=sum_cnt + conv1_abssum_simil.shape[0]
                    sum_simil=sum_simil + torch.sum(conv2_abssum_simil)
                    sum_cnt=sum_cnt + conv2_abssum_simil.shape[0]
            
        #group 3
        conv1_tuple = (net.shared_basis_3.weight,)
        conv2_tuple = (net.shared_basis_3.weight,)
        if unique_rank != 0:
            for i in range(1,len(net.layer3)):
                conv1_tuple = conv1_tuple + (net.layer3[i].basis_conv1.weight,)
                conv2_tuple = conv2_tuple + (net.layer3[i].basis_conv2.weight,)
        conv1_all_basis = torch.cat(conv1_tuple)
        conv2_all_basis = torch.cat(conv2_tuple)
        #conv1_all_basis and conv2_all_basis contains every basis of residual blocks in group 3
        
        len_shared = net.shared_basis_3.weight.shape[0]
        if unique_rank != 0:
            len_unique = net.layer3[1].basis_conv1.weight.shape[0]
        else:
            len_unique = 0
        
        #for every basis in conv1_all_basis and conv2_all_basis, calculates similarities to every other basis in the convx_all_basis tensor
        #convx_all_basis[i+1:]=
        #skips calculating itself (which the result is always 1.0)
        #skips already calculated similarity

        for i in range(len_shared):
            if len_unique == 0 and i == len_shared-1:
                break
            conv1_abssum_simil = abs(cos_simil(
                conv1_all_basis[i+1:].view(conv1_all_basis.shape[0]-1-i,-1),
                conv1_all_basis[i].view(-1)
            ))
            conv2_abssum_simil = abs(cos_simil(
                conv2_all_basis[i+1:].view(conv2_all_basis.shape[0]-1-i,-1),
                conv2_all_basis[i].view(-1)
            ))
            #sum_simil contains every absolute sum of cos similarity
            #sum_cnt contains total number of base
            sum_simil=sum_simil + torch.sum(conv1_abssum_simil)
            sum_cnt=sum_cnt + conv1_abssum_simil.shape[0]
            sum_simil=sum_simil + torch.sum(conv2_abssum_simil)
            sum_cnt=sum_cnt + conv2_abssum_simil.shape[0]
            
        #similarity between unique base in a same residual block
        if unique_rank != 0:
            for i in range(1,len(net.layer3)):
                for j in range(len_unique-1):
                    #index of unique basis in convx_all_basis
                    idx = len_shared+(i-1)*len_unique + j
                    #index of last unique basis within the same residual block in convx_all_basis
                    idx_end = len_shared+i*len_unique
                    #index of last unique basis within the same residual block in convx_all_basis
                    conv1_abssum_simil = abs(cos_simil(
                        conv1_all_basis[idx+1:idx_end].view(idx_end-idx-1,-1),
                        conv1_all_basis[idx].view(-1)
                    ))
                    conv2_abssum_simil = abs(cos_simil(
                        conv2_all_basis[idx+1:idx_end].view(idx_end-idx-1,-1),
                        conv2_all_basis[idx].view(-1)
                    ))
                    sum_simil=sum_simil + torch.sum(conv1_abssum_simil)
                    sum_cnt=sum_cnt + conv1_abssum_simil.shape[0]
                    sum_simil=sum_simil + torch.sum(conv2_abssum_simil)
                    sum_cnt=sum_cnt + conv2_abssum_simil.shape[0]
            
        #group 4
        conv1_tuple = (net.shared_basis_4.weight,)
        conv2_tuple = (net.shared_basis_4.weight,)
        if unique_rank != 0:
            for i in range(1,len(net.layer4)):
                conv1_tuple = conv1_tuple + (net.layer4[i].basis_conv1.weight,)
                conv2_tuple = conv2_tuple + (net.layer4[i].basis_conv2.weight,)
        conv1_all_basis = torch.cat(conv1_tuple)
        conv2_all_basis = torch.cat(conv2_tuple)
        #conv1_all_basis and conv2_all_basis contains every basis of residual blocks in group 4
        
        len_shared = net.shared_basis_4.weight.shape[0]
        if unique_rank != 0:
            len_unique = net.layer4[1].basis_conv1.weight.shape[0]
        else:
            len_unique = 0
        
        #for every basis in conv1_all_basis and conv2_all_basis, calculates similarities to every other basis in the convx_all_basis tensor
        #convx_all_basis[i+1:]=
        #skips calculating itself (which the result is always 1.0)
        #skips already calculated similarity
        
        for i in range(len_shared):
            if len_unique == 0 and i == len_shared-1:
                break
            conv1_abssum_simil = abs(cos_simil(
                conv1_all_basis[i+1:].view(conv1_all_basis.shape[0]-1-i,-1),
                conv1_all_basis[i].view(-1)
            ))
            conv2_abssum_simil = abs(cos_simil(
                conv2_all_basis[i+1:].view(conv2_all_basis.shape[0]-1-i,-1),
                conv2_all_basis[i].view(-1)
            ))
            #sum_simil contains every absolute sum of cos similarity
            #sum_cnt contains total number of base
            sum_simil=sum_simil + torch.sum(conv1_abssum_simil)
            sum_cnt=sum_cnt + conv1_abssum_simil.shape[0]
            sum_simil=sum_simil + torch.sum(conv2_abssum_simil)
            sum_cnt=sum_cnt + conv2_abssum_simil.shape[0]
            
        #similarity between unique base in a same residual block
        if unique_rank != 0:
            for i in range(1,len(net.layer4)):
                for j in range(len_unique-1):
                    #index of unique basis in convx_all_basis
                    idx = len_shared+(i-1)*len_unique + j
                    #index of last unique basis within the same residual block in convx_all_basis
                    idx_end = len_shared+i*len_unique

                    #similarity between unique base in a same residual block
                    conv1_abssum_simil = abs(cos_simil(
                        conv1_all_basis[idx+1:idx_end].view(idx_end-idx-1,-1),
                        conv1_all_basis[idx].view(-1)
                    ))
                    conv2_abssum_simil = abs(cos_simil(
                        conv2_all_basis[idx+1:idx_end].view(idx_end-idx-1,-1),
                        conv2_all_basis[idx].view(-1)
                    ))
                    sum_simil=sum_simil + torch.sum(conv1_abssum_simil)
                    sum_cnt=sum_cnt + conv1_abssum_simil.shape[0]
                    sum_simil=sum_simil + torch.sum(conv2_abssum_simil)
                    sum_cnt=sum_cnt + conv2_abssum_simil.shape[0]

        #average of sum_simil across every base
        sum_simil = sum_simil/sum_cnt

        #acc loss
        loss = criterion(outputs, targets)
        if (batch_idx == 0):
            print("accuracy_loss: %.10f" % loss)
        #apply similarity loss, multiplied by lambda2
        loss = loss - lambda2*torch.log(1 - sum_simil)
        if (batch_idx == 0):
            print("simililarity_loss: %.10f" % torch.log(1 - sum_simil))
            print("similarity:%.3f" % sum_simil)
        loss.backward()
        optimizer.step()
    
#Test for models
def test(epoch):
    if epoch < args.starting_epoch:
        return
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
            outputs = net(inputs)
            
            _, pred = outputs.topk(5, 1, largest=True, sorted=True)

            label_e = targets.view(targets.size(0), -1).expand_as(pred)
            correct = pred.eq(label_e).float()

            correct_top5 += correct[:, :5].sum()
            correct_top1 += correct[:, :1].sum()
            
            total += targets.size(0)
            
    # Save checkpoint.
    acc_top1 = 100.*correct_top1/total
    acc_top5 = 100.*correct_top5/total
    if acc_top1 > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc_top1,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt' + args.visible_device + '.pth')
        best_acc = acc_top1
        best_acc_top5 = acc_top5
        print("Best_Acc_top1 = %.3f" % acc_top1)
        print("Best_Acc_top5 = %.3f" % acc_top5)
        
best_acc = 0
best_acc_top5 = 0

#For parameter shared models
if 'Basis' in args.model:
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    for i in range(150):
        train_basis(i+1)
        test(i+1)
    
    #============
    
    checkpoint = torch.load('./checkpoint/ckpt' + args.visible_device + '.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    optimizer = optim.SGD(net.parameters(), lr=lr*0.1, momentum=momentum, weight_decay=weight_decay)
    for i in range(75):
        train_basis(i+151)
        test(i+151)
    
    #============
    
    checkpoint = torch.load('./checkpoint/ckpt' + args.visible_device + '.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    optimizer = optim.SGD(net.parameters(), lr=lr*0.01, momentum=momentum, weight_decay=weight_decay)
    for i in range(75):
        train_basis(i+226)
        test(i+226)

    print("Best_Acc_top1 = %.3f" % best_acc)
    print("Best_Acc_top5 = %.3f" % best_acc_top5)
    
#placeholder
elif 'Unique' in args.model:
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    for i in range(150):
        train(i+1)
        test(i+1)
    
    #============
    
    checkpoint = torch.load('./checkpoint/ckpt' + args.visible_device + '.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    optimizer = optim.SGD(net.parameters(), lr=lr*0.1, momentum=momentum, weight_decay=weight_decay)
    for i in range(75):
        train(i+151)
        test(i+151)
    
    #============
    
    checkpoint = torch.load('./checkpoint/ckpt' + args.visible_device + '.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    optimizer = optim.SGD(net.parameters(), lr=lr*0.01, momentum=momentum, weight_decay=weight_decay)
    for i in range(75):
        train(i+226)
        test(i+226)

    print("Best_Acc_top1 = %.3f" % best_acc)
    print("Best_Acc_top5 = %.3f" % best_acc_top5)
    
#for models without parameter sharing
else:
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    for i in range(150):
        train(i+1)
        test(i+1)
    
    #============
    
    checkpoint = torch.load('./checkpoint/ckpt' + args.visible_device + '.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    optimizer = optim.SGD(net.parameters(), lr=lr*0.1, momentum=momentum, weight_decay=weight_decay)
    for i in range(75):
        train(i+151)
        test(i+151)
    
    #============
    
    checkpoint = torch.load('./checkpoint/ckpt' + args.visible_device + '.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    optimizer = optim.SGD(net.parameters(), lr=lr*0.01, momentum=momentum, weight_decay=weight_decay)
    for i in range(75):
        train(i+226)
        test(i+226)

    print("Best_Acc_top1 = %.3f" % best_acc)
    print("Best_Acc_top5 = %.3f" % best_acc_top5)
