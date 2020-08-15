import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import copy

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
    
# BasicBlock for models with unique basis only
class BasicBlock_NonShared(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, unique_rank, stride=1, option='A'):
        super(BasicBlock_NonShared, self).__init__()
        
        self.unique_rank = unique_rank
        
        self.total_rank_1 = unique_rank
        self.total_rank_2 = unique_rank
        
        self.basis_conv1 = nn.Conv2d(in_planes, unique_rank, kernel_size=3, stride=stride, padding=1, bias=False)
        self.basis_bn1 = nn.BatchNorm2d(self.total_rank_1)
        self.coeff_conv1 = nn.Conv2d(self.total_rank_1, planes, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.basis_conv2 = nn.Conv2d(planes, unique_rank, kernel_size=3, stride=1, padding=1, bias=False)
        self.basis_bn2 = nn.BatchNorm2d(self.total_rank_2)
        self.coeff_conv2 = nn.Conv2d(self.total_rank_2, planes, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                LambdaLayer implementation is imported from https://github.com/akamaster/pytorch_resnet_cifar10/
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):           
        out = self.basis_conv1(x)
        out = self.basis_bn1(out)
        out = self.coeff_conv1(out)
        
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.basis_conv2(out)
        out = self.basis_bn2(out)
        out = self.coeff_conv2(out)
        
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        
        return out
    
# BasicBlock for models with shared basis only
class BasicBlock_SharedOnly(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, shared_basis_1, shared_basis_2, stride=1, option='A'):
        super(BasicBlock_SharedOnly, self).__init__()
        
        self.shared_basis_1 = shared_basis_1
        self.shared_basis_2 = shared_basis_2
        
        self.total_rank_1 = shared_basis_1.weight.shape[0]
        self.total_rank_2 = shared_basis_2.weight.shape[0]
        
        self.basis_bn1 = nn.BatchNorm2d(self.total_rank_1)
        self.coeff_conv1 = nn.Conv2d(self.total_rank_1, planes, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.basis_bn2 = nn.BatchNorm2d(self.total_rank_2)
        self.coeff_conv2 = nn.Conv2d(self.total_rank_2, planes, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                LambdaLayer implementation is imported from https://github.com/akamaster/pytorch_resnet_cifar10/
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):           
        out = self.shared_basis_1(x)
        out = self.basis_bn1(out)
        out = self.coeff_conv1(out)
        
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.shared_basis_2(out)
        out = self.basis_bn2(out)
        out = self.coeff_conv2(out)
        
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        
        return out
    
# BasicBlock for single basis models
class BasicBlock_SingleShared(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, unique_rank, shared_basis, stride=1, option='A'):
        super(BasicBlock_SingleShared, self).__init__()
        
        self.unique_rank = unique_rank
        self.shared_basis = shared_basis
        
        self.total_rank = unique_rank+shared_basis.weight.shape[0]
        
        self.basis_conv1 = nn.Conv2d(in_planes, unique_rank, kernel_size=3, stride=stride, padding=1, bias=False)
        self.basis_bn1 = nn.BatchNorm2d(self.total_rank)
        self.coeff_conv1 = nn.Conv2d(self.total_rank, planes, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.basis_conv2 = nn.Conv2d(planes, unique_rank, kernel_size=3, stride=1, padding=1, bias=False)
        self.basis_bn2 = nn.BatchNorm2d(self.total_rank)
        self.coeff_conv2 = nn.Conv2d(self.total_rank, planes, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                LambdaLayer implementation is imported from https://github.com/akamaster/pytorch_resnet_cifar10/
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):           
        out = torch.cat((self.basis_conv1(x), self.shared_basis(x)),dim=1)
        out = self.basis_bn1(out)
        out = self.coeff_conv1(out)
        
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = torch.cat((self.basis_conv2(out), self.shared_basis(out)),dim=1)
        out = self.basis_bn2(out)
        out = self.coeff_conv2(out)
        
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        
        return out
    
# BasicBlock for double basis models (proposed models)
class BasicBlock_DoubleShared(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, unique_rank, shared_basis_1, shared_basis_2, stride=1, option='A', skippable=True):
        super(BasicBlock_DoubleShared, self).__init__()
        
        self.unique_rank = unique_rank
        self.shared_basis_1 = shared_basis_1
        self.shared_basis_2 = shared_basis_2
        
        self.total_rank_1 = unique_rank+shared_basis_1.weight.shape[0]
        self.total_rank_2 = unique_rank+shared_basis_2.weight.shape[0]
        
        self.basis_conv1 = nn.Conv2d(in_planes, unique_rank, kernel_size=3, stride=stride, padding=1, bias=False)
        self.basis_bn1 = nn.BatchNorm2d(self.total_rank_1)
        self.coeff_conv1 = nn.Conv2d(self.total_rank_1, planes, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.basis_conv2 = nn.Conv2d(planes, unique_rank, kernel_size=3, stride=1, padding=1, bias=False)
        self.basis_bn2 = nn.BatchNorm2d(self.total_rank_2)
        self.coeff_conv2 = nn.Conv2d(self.total_rank_2, planes, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bn2 = nn.BatchNorm2d(planes)

        self.skippable = skippable
        if (self.skippable == True): # these layers are used only by the low-perf model
            self.coeff_conv2_skip = nn.Conv2d(self.total_rank_2, planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn2_skip = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                LambdaLayer implementation is imported from https://github.com/akamaster/pytorch_resnet_cifar10/
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x, skip=False):           
        out = torch.cat((self.basis_conv1(x), self.shared_basis_1(x)),dim=1)
        out = self.basis_bn1(out)
        out = self.coeff_conv1(out)
        
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = torch.cat((self.basis_conv2(out), self.shared_basis_2(out)),dim=1)
        out = self.basis_bn2(out)

        # The last layers of the low-perf model is not shared with the high-perf model
        # to make them learn specific representaion. 
        if self.skippable == True and skip ==True:
            out = self.coeff_conv2_skip(out)
            out = self.bn2_skip(out)
        else:
            out = self.coeff_conv2(out)
            out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out, inplace=True)

        return out
    
# Original BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                LambdaLayer implementation is imported from https://github.com/akamaster/pytorch_resnet_cifar10/
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        
        return out
    
# ResNet for unique basis only models
class ResNet_NonShared(nn.Module):
    def __init__(self, block_basis, block_original, num_blocks, unique_rank, num_classes=10):
        super(ResNet_NonShared, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(block_basis, block_original, 16, num_blocks[0], unique_rank*1, stride=1)
        
        self.layer2 = self._make_layer(block_basis, block_original, 32, num_blocks[1], unique_rank*2, stride=2)
        
        self.layer3 = self._make_layer(block_basis, block_original, 64, num_blocks[2], unique_rank*4, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #initialize every con2d first, then initialize shared basis again later
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block_basis, block_original, planes, blocks, unique_rank, stride=1):
        layers = []
        
        layers.append(block_original(self.in_planes, planes, stride))
        
        self.in_planes = planes * block_original.expansion
        for _ in range(1, blocks):
            layers.append(block_basis(self.in_planes, planes, unique_rank))

        return nn.Sequential(*layers)
    
    def _make_layer_original(self, block_original, planes, blocks, stride=1):
        layers = []
        
        layers.append(block_original(self.in_planes, planes, stride))
        
        self.in_planes = planes * block_original.expansion
        for _ in range(1, blocks):
            layers.append(block_original(self.in_planes, planes, stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
     
        return x
    
# ResNet for shared basis only models
class ResNet_SharedOnly(nn.Module):
    def __init__(self, block_basis, block_original, num_blocks, shared_rank, num_classes=10):
        super(ResNet_SharedOnly, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.shared_basis_1_1 = nn.Conv2d(16, shared_rank*1, kernel_size=3, stride=1, padding=1, bias=False)
        self.shared_basis_1_2 = nn.Conv2d(16, shared_rank*1, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block_basis, block_original, 16, num_blocks[0], self.shared_basis_1_1, self.shared_basis_1_2, stride=1)
        
        self.shared_basis_2_1 = nn.Conv2d(32, shared_rank*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.shared_basis_2_2 = nn.Conv2d(32, shared_rank*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2 = self._make_layer(block_basis, block_original, 32, num_blocks[1], self.shared_basis_2_1, self.shared_basis_2_2, stride=2)
        
        self.shared_basis_3_1 = nn.Conv2d(64, shared_rank*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.shared_basis_3_2 = nn.Conv2d(64, shared_rank*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3 = self._make_layer(block_basis, block_original, 64, num_blocks[2], self.shared_basis_3_1, self.shared_basis_3_2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #initialize every con2d first, then initialize shared basis again later
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Two shared bases are orthogonal-initialized together. 
        for i in range(1,len(num_blocks)+1):
            basis_1 = getattr(self, "shared_basis_"+str(i)+"_1")
            basis_2 = getattr(self, "shared_basis_"+str(i)+"_2")
            out_chan = basis_1.weight.shape[0]
            in_chan =  basis_1.weight.shape[1]
            X = torch.empty(out_chan*2, in_chan, 3, 3)
            torch.nn.init.orthogonal_(X)
            basis_1.weight.data = copy.deepcopy(X[:out_chan,:])
            basis_2.weight.data = copy.deepcopy(X[out_chan:,:])

    def _make_layer(self, block_basis, block_original, planes, blocks, shared_basis_1, shared_basis_2, stride=1):
        layers = []
        
        layers.append(block_original(self.in_planes, planes, stride))
        
        self.in_planes = planes * block_original.expansion
        for _ in range(1, blocks):
            layers.append(block_basis(self.in_planes, planes, shared_basis_1, shared_basis_2))

        return nn.Sequential(*layers)
    
    def _make_layer_original(self, block_original, planes, blocks, stride=1):
        layers = []
        
        layers.append(block_original(self.in_planes, planes, stride))
        
        self.in_planes = planes * block_original.expansion
        for _ in range(1, blocks):
            layers.append(block_original(self.in_planes, planes, stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
     
        return x

class SkippableSequential_orig(nn.Sequential):
    def forward(self, input, skip=False):
        if skip == True:
            #n_modules = int(len(self)/2)
            n_modules = round(len(self)/2)
        else:
            n_modules = len(self)
        for i in range(n_modules):
            input = self[i](input)
        return input

class SkippableSequential(nn.Sequential):
    """ Skip some blocks if requested. """
    def forward(self, input, skip=False):
        if skip == True:
            n_skip = round(len(self)/2) 
            for i in range(0, n_skip -1):   # execute first-half blocks 
                input = self[i](input)
            
            input = self[n_skip-1](input,skip=True)  # execute the last block of the low-perf model.
        else:
            for b in self:  # execute all blocks
                input = b(input)
        return input

class ResNet_SingleShared(nn.Module):
    def __init__(self, block_basis, block_original, num_blocks, shared_rank, unique_rank, num_classes=10):
        super(ResNet_SingleShared, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.shared_basis_1 = nn.Conv2d(16, shared_rank*1, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block_basis, block_original, 16, num_blocks[0], unique_rank*1, self.shared_basis_1, stride=1)
        
        self.shared_basis_2 = nn.Conv2d(32, shared_rank*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2 = self._make_layer(block_basis, block_original, 32, num_blocks[1], unique_rank*2, self.shared_basis_2, stride=2)
        
        self.shared_basis_3 = nn.Conv2d(64, shared_rank*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3 = self._make_layer(block_basis, block_original, 64, num_blocks[2], unique_rank*4, self.shared_basis_3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.fc_skip = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #initialize every con2d first, then initialize shared basis again later
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Each share basis is orthogonal-initialized separately
        
        torch.nn.init.orthogonal_(self.shared_basis_1.weight)
        torch.nn.init.orthogonal_(self.shared_basis_2.weight)
        torch.nn.init.orthogonal_(self.shared_basis_3.weight)
        

    def _make_layer(self, block_basis, block_original, planes, blocks, unique_rank, shared_basis, stride=1):
        layers = []
        
        layers.append(block_original(self.in_planes, planes, stride))
        
        self.in_planes = planes * block_original.expansion
        for _ in range(1, blocks):
            layers.append(block_basis(self.in_planes, planes, unique_rank, shared_basis))

        return SkippableSequential(*layers)
    
    def forward(self, x, skip=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.layer1(x, skip)
        x = self.layer2(x, skip)
        x = self.layer3(x, skip)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if (skip == True):
            x = self.fc_skip(x)
        else:
            x = self.fc(x)
     
        return x

# Proposed ResNet sharing 2 bases for each residual block group.
class ResNet_DoubleShared(nn.Module):
    def __init__(self, block_basis, block_original, num_blocks, shared_rank, unique_rank, num_classes=10):
        super(ResNet_DoubleShared, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.shared_basis_1_1 = nn.Conv2d(16, shared_rank*1, kernel_size=3, stride=1, padding=1, bias=False)
        self.shared_basis_1_2 = nn.Conv2d(16, shared_rank*1, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block_basis, block_original, 16, num_blocks[0], unique_rank*1, self.shared_basis_1_1, self.shared_basis_1_2, stride=1)
        
        self.shared_basis_2_1 = nn.Conv2d(32, shared_rank*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.shared_basis_2_2 = nn.Conv2d(32, shared_rank*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2 = self._make_layer(block_basis, block_original, 32, num_blocks[1], unique_rank*2, self.shared_basis_2_1, self.shared_basis_2_2, stride=2)
        
        self.shared_basis_3_1 = nn.Conv2d(64, shared_rank*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.shared_basis_3_2 = nn.Conv2d(64, shared_rank*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3 = self._make_layer(block_basis, block_original, 64, num_blocks[2], unique_rank*4, self.shared_basis_3_1, self.shared_basis_3_2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)      # FC layer for the high-perf model  
        self.fc_skip = nn.Linear(64, num_classes) # FC layer for the low-perf model
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #initialize every con2d first, then initialize shared basis again later
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Two shared bases are orthogonal-initialized together. 
        for i in range(1,len(num_blocks)+1):
            basis_1 = getattr(self, "shared_basis_"+str(i)+"_1")
            basis_2 = getattr(self, "shared_basis_"+str(i)+"_2")
            out_chan = basis_1.weight.shape[0]
            in_chan =  basis_1.weight.shape[1]
            X = torch.empty(out_chan*2, in_chan, 3, 3)
            torch.nn.init.orthogonal_(X)
            basis_1.weight.data = copy.deepcopy(X[:out_chan,:])
            basis_2.weight.data = copy.deepcopy(X[out_chan:,:])

        # Each share basis is orthogonal-initialized separately
        # @@ obsolote woochul@2020.05.25
        '''
        torch.nn.init.orthogonal_(self.shared_basis_1_1.weight)
        torch.nn.init.orthogonal_(self.shared_basis_1_2.weight)
        torch.nn.init.orthogonal_(self.shared_basis_2_1.weight)
        torch.nn.init.orthogonal_(self.shared_basis_2_2.weight)
        torch.nn.init.orthogonal_(self.shared_basis_3_1.weight)
        torch.nn.init.orthogonal_(self.shared_basis_3_2.weight)
        '''

    def _make_layer(self, block_basis, block_original, planes, blocks, unique_rank, shared_basis_1, shared_basis_2, stride=1):
        layers = []
        
        layers.append(block_original(self.in_planes, planes, stride))
        
        self.in_planes = planes * block_original.expansion
        n_skip = round(blocks/2)
        for i in range(1, n_skip -1):  
            layers.append(block_basis(self.in_planes, planes, unique_rank, shared_basis_1, shared_basis_2, skippable=False))
        # Only this block needs to have params for skipping
        layers.append(block_basis(self.in_planes, planes, unique_rank, shared_basis_1, shared_basis_2, skippable=True))
        for _ in range(n_skip, blocks):
            layers.append(block_basis(self.in_planes, planes, unique_rank, shared_basis_1, shared_basis_2, skippable=False))

        return SkippableSequential(*layers)

    def forward(self, x, skip=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.layer1(x, skip=skip)
        x = self.layer2(x, skip=skip)
        x = self.layer3(x, skip=skip)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if (skip == True):
            x = self.fc_skip(x)
        else:
            x = self.fc(x)
        
        # share single FC both for low- and high-perf
        # x = self.fc(x)
        
        return x

# Original resnet
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.fc_skip = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return SkippableSequential(*layers)

    def forward(self, x, skip=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.layer1(x, skip)
        x = self.layer2(x, skip)
        x = self.layer3(x, skip)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if (skip == True):
            x = self.fc_skip(x)
        else:
            x = self.fc(x)
         
        return x

# Original ResNet
def ResNet20():
    return ResNet(BasicBlock, [3, 3, 3])

# Original ResNet
def ResNet32():
    return ResNet(BasicBlock, [5, 5, 5])

# Original ResNet
def ResNet44():
    return ResNet(BasicBlock, [7, 7, 7])

# Original ResNet
def ResNet56():
    return ResNet(BasicBlock, [9, 9, 9])

# Original ResNet
def ResNet110():
    return ResNet(BasicBlock, [18, 18, 18])

# Original ResNet
def ResNet1202():
    return ResNet(BasicBlock, [200, 200, 200])

# A model with 2 shared bases in each residual block group.
def ResNet56_DoubleShared(shared_rank, unique_rank):
    return ResNet_DoubleShared(BasicBlock_DoubleShared, BasicBlock, [9, 9, 9], shared_rank, unique_rank)

# A model with 2 shared bases in each residual block group.
def ResNet32_DoubleShared(shared_rank, unique_rank):
    return ResNet_DoubleShared(BasicBlock_DoubleShared, BasicBlock, [5, 5, 5], shared_rank, unique_rank)

# A model with a single shared basis in each residual block group.
def ResNet56_SingleShared(shared_rank, unique_rank):
    return ResNet_SingleShared(BasicBlock_SingleShared, BasicBlock, [9, 9, 9], shared_rank, unique_rank)

# A model with a single shared basis in each residual block group.
def ResNet32_SingleShared(shared_rank, unique_rank):
    return ResNet_SingleShared(BasicBlock_SingleShared, BasicBlock, [5, 5, 5], shared_rank, unique_rank)

# A model with a shared basis in each residual block group, without any unique basis.
def ResNet56_SharedOnly(shared_rank):
    return ResNet_SharedOnly(BasicBlock_SharedOnly, BasicBlock, [9, 9, 9], shared_rank)

# A model with a shared basis in each residual block group, without any unique basis.
def ResNet32_SharedOnly(shared_rank):
    return ResNet_SharedOnly(BasicBlock_SharedOnly, BasicBlock, [5, 5, 5], shared_rank)

# A model without shared bases in each residual block group. Only unique bases used are in each block.
def ResNet56_NonShared(unique_rank):
    return ResNet_NonShared(BasicBlock_NonShared, BasicBlock, [9, 9, 9], unique_rank)

#A model without shared basis in each residual block group. Only unique base are used in the block.
def ResNet32_NonShared(unique_rank):
    return ResNet_NonShared(BasicBlock_NonShared, BasicBlock, [5, 5, 5], unique_rank)
