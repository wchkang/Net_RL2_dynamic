import torch
import torch.nn as nn
import torch.nn.functional as F

#Basic block with prameter sharing
#in_planes: Integer, number of input channels
#planes: Integer, number of output channels
#shared_rank: Integer, size of rank for model's shared basis layers
#unique_rank: Integer, size of rank for model's unique basis layers
#shared_basis: tensor, tensor for shared base
#stride: Integer, stride

class BasicBlock_Basis(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, unique_rank, shared_basis, stride=1):
        super(BasicBlock_Basis, self).__init__()
        
        self.unique_rank = unique_rank
        #shared tensor - shared across basic blocks in a same group
        self.shared_basis = shared_basis

        #total channel size of share and unique base
        self.total_rank = unique_rank+shared_basis.weight.shape[0]
        
        #decomposed form of CONV1
        if unique_rank == 0:
            self.basis_conv1 = nn.Sequential()
        else:
            self.basis_conv1 = nn.Conv2d(in_planes, unique_rank, kernel_size=3, stride=stride, padding=1, bias=False)
        self.basis_bn1 = nn.BatchNorm2d(self.total_rank)
        self.coeff_conv1 = nn.Conv2d(self.total_rank, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        #decomposed form of CONV2
        if unique_rank == 0:
            self.basis_conv2 = nn.Sequential()
        else:
            self.basis_conv2 = nn.Conv2d(planes, unique_rank, kernel_size=3, stride=stride, padding=1, bias=False)
        self.basis_bn2 = nn.BatchNorm2d(self.total_rank)
        self.coeff_conv2 = nn.Conv2d(self.total_rank, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        #Identity
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x): 
        #merge feature maps from shared basis and unique basis into a single feature map by torch.cat()
        #X -> shared/unique basis -> BN -> coeff -> BN -> ReLU -> shared/uniquebasis -> BN -> coeff -> BN + Shortcut -> ReLU = Out 
        if self.unique_rank ==0:
            out = self.basis_bn1( self.shared_basis(x) )
            out = F.relu(self.bn1(self.coeff_conv1(out)),inplace=True)
            out = self.bn2( self.coeff_conv2( self.basis_bn2( self.shared_basis(x) ) ))
            out += self.shortcut(x)
            out = F.relu(out,inplace=True)
        else:
            out = self.basis_bn1(torch.cat((self.basis_conv1(x), self.shared_basis(x)),dim=1))
            out = F.relu(self.bn1(self.coeff_conv1(out)),inplace=True)
            out = self.bn2( self.coeff_conv2( self.basis_bn2(torch.cat((self.basis_conv2(out), self.shared_basis(out)),dim=1) ) ))
            out += self.shortcut(x)
            out = F.relu(out,inplace=True)
        return out
            
#Basic block without prameter sharing
#in_planes: Integer, number of input channels
#planes: Integer, number of output channels
#stride: Integer, stride
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)),inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out,inplace=True)
        return out
                           
#Resnet with prameter sharing
#block: Class, Residual block with parameter sharing
#blcok_without_basis: Class, Residual block without parameter sharing
#num_blocks: 4-elements list, number of blocks per group
#num_classes: Integer, total number of dataset's classes
#shared_rank: Integer, size of rank for model's shared basis layers
#unique_rank: Integer, size of rank for model's unique basis layers

class ResNet_Basis(nn.Module):
    def __init__(self, block, block_without_basis, num_blocks, num_classes, shared_rank, unique_rank):
        super(ResNet_Basis, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #shared_basis_1 is shared across all residual blocks in layer1(=group 1 of residual blocks)
        #As a channel size for a group is multiplied by 2, channel sizes of unique basis and shared basis are multiplied by 2, too.
        self.shared_basis_1 = nn.Conv2d(64, shared_rank, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, block_without_basis, 64, num_blocks[0], unique_rank, self.shared_basis_1, stride=1)
        
        #shared_basis_2 is shared across all residual blocks in layer2
        self.shared_basis_2 = nn.Conv2d(128, shared_rank*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2 = self._make_layer(block, block_without_basis, 128, num_blocks[1], unique_rank*2, self.shared_basis_2, stride=2)
        
        #shared_basis_3 is shared across all residual blocks in layer2
        self.shared_basis_3 = nn.Conv2d(256, shared_rank*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3 = self._make_layer(block, block_without_basis, 256, num_blocks[2], unique_rank*4, self.shared_basis_3, stride=2)
        
        #shared_basis_4 is shared across all residual blocks in layer2
        self.shared_basis_4 = nn.Conv2d(512, shared_rank*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4 = self._make_layer(block, block_without_basis, 512, num_blocks[3], unique_rank*8, self.shared_basis_4, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, block_without_basis, planes, num_blocks, shared_rank, basis, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        
        #There is no parameter shraing for a first block of the group
        layers.append(block_without_basis(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
            
        #After the first block, parameter shraing happens in every blocks in the group
        for stride in strides[1:]:
            layers.append(block(self.in_planes, planes, shared_rank, basis, stride))
            self.in_planes = planes * block.expansion
            
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x)),inplace=True))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out
        #relu inplace?

#Parameter shared ResNet models
#num_classes: Integer, total number of dataset's classes
#shared_rank: Integer, size of rank for model's shared basis layers
#unique_rank: Integer, size of rank for model's unique basis layers

def ResNet34_Unique(num_classes, unique_rank):
    print("Placeholder")
    return None

def ResNet18_Basis(num_classes, shared_rank, unique_rank):
    return ResNet_Basis(BasicBlock_Basis, BasicBlock, [2,2,2,2],num_classes,shared_rank, unique_rank)

def ResNet34_Basis(num_classes, shared_rank, unique_rank):
    return ResNet_Basis(BasicBlock_Basis, BasicBlock, [3,4,6,3],num_classes,shared_rank, unique_rank)