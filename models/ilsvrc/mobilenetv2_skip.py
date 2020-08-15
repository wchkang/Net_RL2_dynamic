import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, expand_ratio, stride, norm_layer=None, skippable=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.skippable = skippable
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.append(
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer))
        
        self.conv = nn.Sequential(*layers)

        # pw-linear
        self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3 = norm_layer(oup)
        if (self.skippable == True):
            self.conv3_skip = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
            self.bn3_skip = norm_layer(oup)
      

    def forward(self, x, skip=False):
        out = self.conv(x)
        if (self.skippable==True and skip==True):
            out = self.bn3_skip(self.conv3_skip(out))
        else:
            out = self.bn3(self.conv3(out))
  
        #print("out shape:", out.shape)
        if self.use_res_connect == True:
            return x + out
        else:
            return out


class MobileNetV2_skip(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
    ]

    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2_skip, self).__init__()

        self.basic_layers=[]
        self.skip_layers=[]
        self.skip_distance=[]

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_input_channel = 320
        last_channel = 1280

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        print('input_channel:', input_channel)
        self.conv1 = ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer) 
        self.layers = self._make_layers(in_planes = input_channel)
        self.conv2 = ConvBNReLU(last_input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _make_layers(self, in_planes):
        layers = []
        idx_basic_layers = []
        idx_skip_layers=[]
        idx_skip_distance=[]
        idx =0
        for expansion, out_planes, num_blocks, stride in self.inverted_residual_setting:
            strides = [stride] + [1]*(num_blocks-1)
            for sid, stride in enumerate(strides):
                if (num_blocks >= 3):
                    if sid ==0: 
                        layers.append(InvertedResidual(in_planes, out_planes, expansion, stride, skippable=True))
                        idx_basic_layers.append(idx)
                        idx_skip_layers.append(idx)
                        #idx_skip_distance.append(num_blocks//2)
                        idx_skip_distance.append(round(num_blocks/2))
                    #elif sid > 0 and sid <= round(num_blocks//2):
                    elif sid > 0 and sid <= round(num_blocks/2):
                        layers.append(InvertedResidual(in_planes, out_planes, expansion, stride))
                    else:
                        layers.append(InvertedResidual(in_planes, out_planes, expansion, stride))
                        idx_basic_layers.append(idx)
                else:
                   layers.append(InvertedResidual(in_planes, out_planes, expansion, stride))
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
        out = self.conv1(x)
        #print(out.shape)
        if skip==True:
            for i in self.basic_layers:
                #print(i)
                out = self.layers[i](out, skip=True)
                #print(out.shape)
        else:
            out = self.layers(out)
            #print(out.shape)
        #print(out.shape)
        out = self.conv2(out)
        out = nn.functional.adaptive_avg_pool2d(out, 1).reshape(x.shape[0], -1)
        out = self.classifier(out)
        return out

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
        for param in net.parameters():
            param.requires_grad = True
        # make the whole network trainable
        self.train()

    def freeze_highperf(self):
        """ freeze high-performace-exclusive layers """
        
        self.freeze_model()

        # defreeze low-perf-exclusive parameters and BNs
        for i, i_base in enumerate(self.skip_layers):
            self.layers[i_base].conv3_skip.weight.requires_grad = True
            self.layers[i_base].bn3_skip.train()

        # defreeze params of low-perf FC layer
        #self.linear_skip.weight.requires_grad = True
        #self.linear_skip.bias.requires_grad = True
        #self.linear_skip.train()

    def freeze_lowperf(self):
        """ Freeze low-performance-exclusive layers """
        
        self.freeze_model()

        # defreeze params of only being used by the high-performance model
        for i, i_base in enumerate(self.skip_layers):
            self.layers[i_base].conv3.weight.requires_grad = True
            self.layers[i_base].bn3.train()
            for j in range(self.skip_distance[i]):
                for param in self.layers[i_base+1+j].parameters():
                    param.requires_grad = True
                self.layers[i_base+1+j].train()

        # defreeze params of high-perf FC layer
        #self.linear.weight.requires_grad = True
        #self.linear.bias.requires_grad = True
        #self.linear.train()


def test():
    net = MobileNetV2_skip(num_classes=1000)
    x = torch.randn(256,3,224,224)
    y = net(x, skip=False)
    print(y.size())
    #print(net)

#test()
