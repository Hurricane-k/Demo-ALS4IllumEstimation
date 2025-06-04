#%% python package import
import math
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.ops
# %% function
# copy and paste from ucc

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def fuse_conv_bn(conv, bn):

    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean) / var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                           conv.out_channels,
                           conv.kernel_size,
                           conv.stride,
                           conv.padding,
                           bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv

#%% convolution
# copy and paste from ucc

class Conv(nn.Module):

    def __init__(
            self,
            c1,  # in channels
            c2,  # out channels 
            k=1,  # kernel size 
            s=1,  # stride
            p=None,  # padding
            d=1,  # dilation
            act_type='lrelu',  # activation
            depthwise=False,
            groups=1,
            bias=False,
            deformable=False,
            norm="bn"):
        super(Conv, self).__init__()
        self.convs = nn.Sequential()
        self.act_type = act_type
        self.norm = norm
        self.depthwise = depthwise
        self.quant = False
        self.deformable = deformable
        act = self.get_activation()

        if depthwise:
            # depthwise conv
            self.convs.add_module(
                'conv1',
                nn.Conv2d(c1,
                          c1,
                          kernel_size=k,
                          stride=s,
                          padding=autopad(k, p),
                          dilation=d,
                          groups=c1,
                          bias=bias))
            self.convs.add_module('bn1', nn.BatchNorm2d(c1))
            self.convs.add_module('act1', act)
            # pointwise conv
            self.convs.add_module(
                'conv2',
                nn.Conv2d(c1,
                          c2,
                          kernel_size=1,
                          stride=s,
                          padding=0,
                          dilation=d,
                          groups=1,
                          bias=bias))
            self.convs.add_module('bn2', nn.BatchNorm2d(c2))
            self.convs.add_module('act2', act)

        else:
            if self.deformable:
                self.convs.add_module(
                    'deconv',
                    DeformableConv2d(c1,
                                     c2,
                                     kernel_size=k,
                                     stride=s,
                                     padding=autopad(k, p),
                                     dilation=d,
                                     bias=bias))
            else:
                self.convs.add_module(
                    'conv',
                    nn.Conv2d(c1,
                              c2,
                              kernel_size=k,
                              stride=s,
                              padding=autopad(k, p),
                              dilation=d,
                              groups=groups,
                              bias=bias))
            self.convs.add_module('act', act)
            if self.norm == "bn":
                norm = nn.BatchNorm2d(c2)
            elif self.norm == "in":
                norm = nn.InstanceNorm2d(c2)
            elif self.norm is None:
                norm = nn.Identity()
            self.convs.add_module('norm', norm)

    def forward(self, x):
        return self.convs(x)

    def get_activation(self):
        if self.act_type == 'relu':
            act = nn.ReLU(inplace=True)
        elif self.act_type == 'lrelu':
            act = nn.LeakyReLU(inplace=True)
        else:
            act = nn.Identity()
        return act

    def fuse_params(self):
        if (not self.quant) and hasattr(
                self.convs,
                'conv') and self.norm == 'bn' and (not self.deformable):
            convs = fuse_conv_bn(self.convs._modules['conv'],
                                 self.convs._modules['bn'])
            act = self.get_activation()
            self.convs = nn.Sequential()
            self.convs.add_module('conv', convs)
            self.convs.add_module('act', act)

    def get_fused_kernel(self):
        if (not self.quant) and hasattr(self.convs, 'conv') and hasattr(
                self.convs, 'bn'):
            convs = fuse_conv_bn(self.convs._modules['conv'],
                                 self.convs._modules['bn'])
            return convs
        else:
            return self.convs._modules['conv']

    def quant_convert(self):
        if not self.depthwise:
            if self.act_type == 'relu' and hasattr(self.convs, 'bn'):
                torch.quantization.fuse_modules(self.convs,
                                                ['conv', 'bn', 'act'],
                                                inplace=True)
            elif hasattr(self.convs, 'bn'):
                torch.quantization.fuse_modules(self.convs, ['conv', 'bn'],
                                                inplace=True)
            elif self.act_type == 'relu':
                torch.quantization.fuse_modules(self.convs, ['conv', 'act'],
                                                inplace=True)

            self.quant = True

#%% deconvolution
# copy and paste from ucc

class DeformableConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 v2=False,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (
            kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation
        self.v2 = v2

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        if self.v2:
            self.modulator_conv = nn.Conv2d(in_channels,
                                            1 * kernel_size[0] *
                                            kernel_size[1],
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=self.padding,
                                            dilation=self.dilation,
                                            bias=True)
            nn.init.constant_(self.modulator_conv.weight, 0.)
            nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = min(h, w) / 4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        # offset = offset.clamp(-max_offset, max_offset)
        if self.v2:
            modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        else:
            modulator = None
        # op = (n - (k * d - 1) + 2p / s)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x

