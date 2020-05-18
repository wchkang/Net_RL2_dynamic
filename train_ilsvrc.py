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

#Possible arguments
parser = argparse.ArgumentParser(description='TODO')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--lambda2', default=0.5, type=float, help='lambda2 (for basis loss)')
parser.add_argument('--shared_rank', default=16, type=int, help='number of shared base)')
parser.add_argument('--dataset', default="CIFAR100", help='CIFAR10, CIFAR100, ILSVRC2012')
parser.add_argument('--batch_size', default=256, type=int, help='batch_size')
parser.add_argument('--model', default="ResNet34", help='ResNet50, ResNet34, ResNet18, ResNet34_Basis, ResNet34_Unique, ResNext50, ResNext101')
parser.add_argument('--visible_device', default="0", help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--unique_rank', default=16, type=int, help='number of unique base')
parser.add_argument('--pretrained', default=None, help='path of a pretrained model file')
parser.add_argument('--starting_epoch', default=0, type=int, help='an epoch which model training starts')
parser.add_argument('--dataset_path', default="./data", help='dataset path')
args = parser.parse_args()

if 'CIFAR' in args.dataset:
    from models.cifar import resnet, resnet_basis, resnext, resnext_basis
    dic_model = {'ResNet50': resnet.ResNet50, 'ResNet34':resnet.ResNet34,'ResNet18':resnet.ResNet18,'ResNet34_Basis':resnet_basis.ResNet34_Basis, 'ResNet34_Unique':resnet_basis.ResNet34_Unique, 'ResNext50':resnext.ResNext50_32x4d, 'ResNext101':resnext.ResNext101_32x8d, 'ResNext50_Basis':resnext_basis.ResNext50_32x4d_Basis}
if 'ILSVRC' in args.dataset:
    from models.ilsvrc import resnet, resnet_basis
    dic_model = {'ResNet34':resnet.ResNet34,'ResNet18':resnet.ResNet18,'ResNet34_Basis':resnet_basis.ResNet34_Basis}

lr = args.lr
momentum = args.momentum
weight_decay = args.weight_decay
lambda2 = args.lambda2
shared_rank = args.shared_rank
unique_rank = args.unique_rank
    
dic_dataset = {'ILSVRC2012':1000, 'CIFAR100':100, 'CIFAR10':10}

#dic_model = {'ResNet50': resnet.ResNet50, 'ResNet34':resnet.ResNet34,'ResNet18':resnet.ResNet18,'ResNet34_Basis':resnet_basis.ResNet34_Basis, 'ResNet34_Unique':resnet_basis.ResNet34_Unique, 'ResNext50':resnext.ResNext50_32x4d, 'ResNext101':resnext.ResNext101_32x8d, 'ResNext50_Basis':resnext_basis.ResNext50_32x4d_Basis}

if args.dataset not in dic_dataset:
    print("The dataset is currently not supported")
    sys.exit()

#if 'CIFAR' in args.dataset:
#    from models.cifar import resnet, resnet_basis
#elif 'ILSVRC' in args.dataset:
    
if args.model not in dic_model:
    print("The model is currently not supported")
    sys.exit()

trainloader = utils.get_traindata(args.dataset,args.dataset_path,batch_size=args.batch_size,download=True)
testloader = utils.get_testdata(args.dataset,args.dataset_path,batch_size=args.batch_size)


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
criterion = nn.CrossEntropyLoss().cuda()

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
def train_basis_orig(epoch):
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
        
# Training for parameter shared models
# Use the property of orthogonal matrices;
# e.g.: AxA.T = I if A is orthogonal 
def train_basis(epoch, include_unique_basis=False):
    if epoch < args.starting_epoch:
        return
    
    print('\nCuda ' + args.visible_device + ' Basis Epoch: %d' % epoch)
    net.train()
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
    
        optimizer.zero_grad()
        outputs = net(inputs)

        # get similarity of basis filters
        cnt_sim = 0 
        sim = 0
        for gid in range(1, 5):  # ResNet has 4 groups
            layer = getattr(net, "layer"+str(gid))
            shared_basis = getattr(net,"shared_basis_"+str(gid))

            num_shared_basis = shared_basis.weight.shape[0]
            num_all_basis = num_shared_basis 

            all_basis =(shared_basis.weight,)
            if (include_unique_basis == True):  
                num_unique_basis = layer[1].basis_conv1.weight.shape[0] 
                num_all_basis += (num_unique_basis * 2 * (len(layer) -1))
                for i in range(1, len(layer)):
                    all_basis += (layer[i].basis_conv1.weight, \
                            layer[i].basis_conv2.weight,)

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
            #print("similarity loss: %.6f" % (-torch.log(1.0-avg_sim)))
            print("similarity loss: %.6f" % avg_sim)

        #apply similarity loss, multiplied by lambda2
        #loss = loss - lambda2 * torch.log(1.0 - avg_sim)
        loss = loss + lambda2 * avg_sim
        loss.backward()
        optimizer.step()

# Training for parameter shared models
# Use the property of orthogonal matrices;
# e.g.: AxA.T = I if A is orthogonal 
def train_basis_resnext(epoch, include_unique_basis=False):
    if epoch < args.starting_epoch:
        return
    
    print('\nCuda ' + args.visible_device + ' Basis Epoch: %d' % epoch)
    net.train()
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
    
        optimizer.zero_grad()
        outputs = net(inputs)
        
        # get similarity of basis filters
        cnt_sim = 0 
        sim = 0
        for gid in range(1, 5):  # ResNet has 4 groups
            layer = getattr(net, "layer"+str(gid))
            shared_basis = getattr(net,"shared_basis_"+str(gid))

            num_shared_basis = shared_basis.weight.shape[0]
            num_all_basis = num_shared_basis 

            all_basis =(shared_basis.weight,)
            if (include_unique_basis == True):  
                num_unique_basis = layer[1].basis_conv2.weight.shape[0] 
                num_all_basis += (num_unique_basis * 2 * (len(layer) -1))
                for i in range(1, len(layer)):
                    all_basis += (layer[i].basis_conv2.weight,)

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
                    for j in range(1):  # conv1 & conv2
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
            #print("similarity loss: %.6f" % (-torch.log(1.0-avg_sim)))
            print("similarity loss: %.6f" % avg_sim)

        #apply similarity loss, multiplied by lambda2
        #loss = loss - lambda2 * torch.log(1.0 - avg_sim)
        loss = loss + lambda2 * avg_sim
        loss.backward()
        optimizer.step()
        
def adjust_learning_rate(optimizer, epoch, args_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
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
    
    for i in range(90):
        adjust_learning_rate(optimizer, i, lr)
        train_basis(i)
        test(i)

    print("Best_Acc_top1 = %.3f" % best_acc)
    print("Best_Acc_top5 = %.3f" % best_acc_top5)
    
#for models without parameter sharing
else:
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    for i in range(90):
        adjust_learning_rate(optimizer, i, lr)
        train(i)
        test(i)

    print("Best_Acc_top1 = %.3f" % best_acc)
    print("Best_Acc_top5 = %.3f" % best_acc_top5)