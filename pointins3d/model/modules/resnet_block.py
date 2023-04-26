# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import torch.nn as nn

from MinkowskiEngine import MinkowskiReLU
from pointins3d.model.modules.common import ConvType, NormType, conv, get_norm, sum_pool, avg_pool


class BasicBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = NormType.BATCH_NORM

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        shortcut=None,
        conv_type=ConvType.HYPERCUBE,
        bn_momentum=0.1,
        D=3,
    ):
        super().__init__()

        self.relu = MinkowskiReLU(inplace=True)
        self.residual_function = nn.Sequential(
            conv(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D, ),
            get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum),
            self.relu,
            conv(planes, planes, kernel_size=3, stride=1, dilation=dilation, bias=False, conv_type=conv_type, D=D, ),
            get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        out = self.residual_function(x)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out += residual
        out = self.relu(out)
        return out

class BasicBlock(BasicBlockBase):
    NORM_TYPE = NormType.BATCH_NORM


class BasicBlockIN(BasicBlockBase):
    NORM_TYPE = NormType.INSTANCE_NORM


class BasicBlockINBN(BasicBlockBase):
    NORM_TYPE = NormType.INSTANCE_BATCH_NORM

class BottleneckBase(nn.Module):
    expansion = 4
    NORM_TYPE = NormType.BATCH_NORM

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        shortcut=None,
        conv_type=ConvType.HYPERCUBE,
        bn_momentum=0.1,
        D=3,
        heads=4,
        mhsa=False,
        resolution=None
    ):
        super().__init__()
        super().__init__()
        self.relu = MinkowskiReLU(inplace=True)
        self.conv1 = conv(inplanes, planes, kernel_size=1, D=D)
        self.bn1 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        if mhsa:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads, D=D))
            if stride == 2:
                self.conv2.append(avg_pool(2, 2))
            self.conv2 = nn.Sequential(*self.conv2)
        else:
            self.conv2 = conv(planes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
        self.bn2 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, D=D)
        self.bn3 = get_norm(self.NORM_TYPE, planes * self.expansion, D, bn_momentum=bn_momentum)
        self.residual_function = nn.Sequential(
            self.conv1,
            self.bn3,
            self.relu,
            self.conv2,
            self.bn2,
            self.relu,
            self.conv3,
            self.bn3
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        out = self.residual_function(x)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(BottleneckBase):
    NORM_TYPE = NormType.BATCH_NORM


class BottleneckIN(BottleneckBase):
    NORM_TYPE = NormType.INSTANCE_NORM


class BottleneckINBN(BottleneckBase):
    NORM_TYPE = NormType.INSTANCE_BATCH_NORM
