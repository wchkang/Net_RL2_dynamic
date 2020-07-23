'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, skippable=False):
        super(Block, self).__init__()
        self.stride = stride
        self.skippable = skippable

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
       
        if (self.skippable ==True):
            self.conv3_skip = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn3_skip = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x, skip=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if (self.skippable ==True and skip==True):
            out = self.bn3_skip(self.conv3_skip(out))
        else:
            out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2_skip(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2_skip, self).__init__()

        #num_layers = sum([group[2] for group in self.cfg])
        #self.basic_layers=[i for i in range(num_layers)]
        self.basic_layers=[]
        self.skip_layers=[]
        self.skip_distance=[]

        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.linear_skip = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        idx_basic_layers = []
        idx_skip_layers=[]
        idx_skip_distance=[]
        idx =0
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for sid, stride in enumerate(strides):
                if (num_blocks >= 3):
                    if sid ==0: 
                        layers.append(Block(in_planes, out_planes, expansion, stride, skippable=True))
                        idx_basic_layers.append(idx)
                        idx_skip_layers.append(idx)
                        idx_skip_distance.append(num_blocks//2)
                    elif sid > 0 and sid <= round(num_blocks//2):
                        layers.append(Block(in_planes, out_planes, expansion, stride))
                    else:
                        layers.append(Block(in_planes, out_planes, expansion, stride))
                        idx_basic_layers.append(idx)
                else:
                   layers.append(Block(in_planes, out_planes, expansion, stride))
                   idx_basic_layers.append(idx)
                in_planes = out_planes
                idx = idx + 1
        print(idx_basic_layers)
        print(idx_skip_layers)
        print(idx_skip_distance)
        self.basic_layers = idx_basic_layers
        self.skip_layers = idx_skip_layers
        self.skip_distance = idx_skip_distance
        return nn.Sequential(*layers)

    def forward(self, x, skip=True):
        out = F.relu(self.bn1(self.conv1(x)))

        if skip==True:
            for i in self.basic_layers:
                #print(i)
                out = self.layers[i](out, skip=True)
        else:
            out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if (skip == True):
           out = self.linear_skip(out)
        else:
           out = self.linear(out)
        return out


def test():
    net = MobileNetV2_skip()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
