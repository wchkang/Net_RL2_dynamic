import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, skippable=False):
        super(BasicBlock, self).__init__()
        
        self.skippable = skippable 

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if (self.skippable==True):
            self.conv2_skip = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2_skip = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x, skip=False):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)


        if (self.skippable == True and skip==True):
            out = self.conv2_skip(out)
            out = self.bn2_skip(out)
        else:            
            out = self.conv2(out)
            out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, skippable=False):
        super(Bottleneck, self).__init__()

        self.skippable = skippable
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        if (self.skippable==True):
            self.conv3_skip = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, stride=1, bias=False)
            self.bn3_skip = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x, skip=False):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True) 

        if (self.skippable == True and skip==True):
            out = self.conv3_skip(out)
            out = self.bn3_skip(out)
        else:            
            out = self.conv3(out)
            out = self.bn3(out)

        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        
        return out

class SkippableSequential(nn.Sequential):
    """ Skip first half blocks if requested. """
    def forward(self, input, skip=False):
        out = self[0](input, skip=skip)

        #n_skip = 0 if skip==False else round(len(self)/2.0)
        n_skip = 0 if skip==False else len(self)//2

        #print("n_skip:", n_skip)

        for i in range(n_skip+1, len(self)):
            out = self[i](out)
            #print(" "+str(i), end='')
        #print('')

        return out


class ResNet_skip(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet_skip, self).__init__()
        self.in_planes = 64

        if issubclass(block, BasicBlock):
            #print("Block")
            self.block_type = 'BasicBlock'
        elif issubclass(block, Bottleneck):
            #print("Bottleneck")
            self.block_type = 'Bottleneck'
        else:
            print("unknow block type")

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.in_planes, planes, stride, skippable=True))
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
        x = self.layer4(x, skip)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
     
        return x


    def freeze_model(self):
        """ freeze all layers and BNs """
        # BN layers need to be freezed explicitly since they cannot be freezed via '.requires_grad=False'
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                module.eval()
        
        # freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def defreeze_model(self):
        """ Defreeze all parameters and enable training. . """
        # defreeze all parameters
        for param in self.parameters():
            param.requires_grad = True
        # make the whole network trainable
        self.train()

    def freeze_highperf(self):
        """ freeze high-performace-exclusive and shared layers """
        
        self.freeze_model()

        # defreeze low-perf-exclusive parameters and BNs
        for i in range(1,5):
            layer = getattr(self, "layer"+str(i))
            if self.block_type == 'Bottleneck':
                layer[0].conv3_skip.weight.requires_grad = True
                layer[0].bn3_skip.train()
            elif self.block_type == 'BasicBlock':
                layer[0].conv2_skip.weight.requires_grad = True
                layer[0].bn2_skip.train()
            else:
                print("[Error] Unknown block type")

    def freeze_lowperf(self):
        """ Freeze low-performance-exclusive-exclusive and shared layers """
        
        self.freeze_model()

        # defreeze params of only being used by the high-performance model
        for i in range(1,5):
            layer = getattr(self, "layer"+str(i))
            if self.block_type == 'Bottleneck':
                layer[0].conv3.weight.requires_grad = True
                layer[0].bn3.train()
            elif self.block_type == 'BasicBlock':
                layer[0].conv2.weight.requires_grad = True
                layer[0].bn2.train()
            else:
                print("[Error] Unknown block type")


            num_skip = len(layer)//2
            for j in range(1, num_skip+1):
                for param in layer[j].parameters():
                    param.requires_grad = True
                layer[j].train()


#def ResNet18(num_classes=100):
#     return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# def ResNet34(num_classes=100):
#     return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def ResNet34_skip(num_classes=100):
    return ResNet_skip(BasicBlock, [3, 4, 6, 3], num_classes)

def ResNet50_skip(num_classes=100):
    return ResNet_skip(Bottleneck, [3, 4, 6, 3], num_classes)

def ResNet101_skip(num_classes=100):
    return ResNet_skip(Bottleneck, [3, 4, 23, 3], num_classes)

def test():
    #net = ResNet50_skip()
    net = ResNet34_skip()
    x = torch.randn(256,3,32,32)
    #x = torch.randn(16,3,224,224)
    y = net(x, False)
    #y = net(x, True)
    
    print(net)
    print(y.size())

test()
