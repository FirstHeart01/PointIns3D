import functools
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule
from spconv.pytorch import SubMConv3d, SparseAvgPool3d, SparseInverseConv3d, SparseConv3d
from pointins3d.model.model import Model
from pointins3d.model.modules.shared_transformer import SharedTransformer
from pointins3d.model.modules.transformer_encoder import Norm, TransformerEncoder
from pointins3d.ops import voxelization, global_avg_pool


class MLP(nn.Sequential):

    def __init__(self, in_channels, out_channels, norm_fn=None, num_layers=2):
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, in_channels))
            if norm_fn:
                modules.append(norm_fn(in_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_channels, out_channels))
        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)


class BasicBlock(SparseModule):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            dilation=1,
            shortcut=None,
            bn_momentum=0.1,
            norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
            indice_key=None):
        super().__init__()
        self.residual_function = spconv.SparseSequential(
            SubMConv3d(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, indice_key=indice_key),
            norm_fn(planes, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            SubMConv3d(planes, planes, kernel_size=3, stride=1, dilation=dilation, indice_key=indice_key),
            norm_fn(planes, momentum=bn_momentum)
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = shortcut
        # self.attention = AFF(planes, planes, norm_fn) #(n, c)

    def forward(self, x):
        residual = x
        out = self.residual_function(x)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        # out.features: (n, c)
        # residual: (n, c)
        out_feats = out.features + residual.features
        # out_feats = self.attention(out, residual) # (n, c)
        out = out.replace_feature(out_feats)
        out_feats = self.relu(out.features)
        out = out.replace_feature(out_feats)
        return out

class AFF(nn.Module):
    """
    多特征融合AFF
    """
    def __init__(self, in_channels, out_channels, norm_fn):
        super().__init__()  
        mid_channels = in_channels // 4
        # self.local_att = spconv.SparseSequential(
        #     spconv.SubMConv3d(in_channels, mid_channels, kernel_size=1),
        #     norm_fn(mid_channels),
        #     nn.ReLU(),
        #     spconv.SubMConv3d(mid_channels, out_channels, kernel_size=1),
        #     norm_fn(out_channels)
        # )
        self.local_att = nn.Sequential(
            nn.Linear(in_channels, mid_channels), 
            norm_fn(mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, out_channels),
            norm_fn(out_channels)
        )
        # self.global_att = spconv.SparseSequential(
        #     spconv.SparseAvgPool3d(1),
        #     spconv.SubMConv3d(in_channels, mid_channels, kernel_size=1),
        #     norm_fn(mid_channels),
        #     nn.ReLU(),
        #     spconv.SubMConv3d(mid_channels, out_channels, kernel_size=1),
        #     norm_fn(out_channels)
        # )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_att = nn.Sequential(
            nn.Linear(in_channels, mid_channels), 
            Norm(mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, out_channels),
            Norm(out_channels)# (1, out_channel, 1)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, residual):
        # x: 底层   residual: 高层
        # x: (n1, 128)  residual: (n2, 256)
        
        x_feats = x.features
        res_feats = residual.features
        batch_ids = x.indices[:, 0]
        batch_size = batch_ids.max().item() + 1
        output_feats = torch.zeros_like(x_feats)
        for i in range(batch_size):
            batch_id = (batch_ids==i).nonzero().squeeze(dim=1)
            if batch_id.size(0) == 0:
                continue
            batch_out_features = x_feats[batch_id] # n',c
            batch_res_features = res_feats[batch_id] # n',c
            # batch_indices = batch_ids[batch_id]
            xa = batch_out_features + batch_res_features # (n', c)
            xl = self.local_att(xa)
            x_pool = self.global_pool(xa.transpose(0, 1)).transpose(0,1) #(c, n') => (c, 1) => (1, c)
            xg = self.global_att(x_pool)
            xlg = xl + xg
            weight = self.sigmoid(xlg)
            out_feats = 2 * batch_out_features * weight +  2 * batch_res_features * (1 - weight)
            output_feats[batch_id] = out_feats #(n, c)
        return output_feats

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int, norm_fn):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Linear(F_g, F_int),
            norm_fn(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Linear(F_l, F_int),
            norm_fn(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class MSC(SparseModule):
    def __init__(self, in_channel, norm_fn): # 3 or 6
        super().__init__()
        self.tiny_unet = UBlock([in_channel, 32, 64], norm_fn, 1, ResidualBlock, indice_key_id=8)
        self.conv_smoothing = spconv.SubMConv3d(in_channel, in_channel, kernel_size=1)
    def forward(self, x, aux):
        return self.conv_smoothing(self.tiny_unet(x, aux))
        
class Res16UNetBase(Model):
    BLOCK = None
    PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    OUT_PIXEL_DIST = 1

    def __init__(self, in_channels, out_channels, train_cfg,
                 norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 out_fpn=True, with_transformer=False, **kwargs):
        assert self.BLOCK is not None
        assert self.OUT_PIXEL_DIST > 0
        super().__init__(in_channels, out_channels, train_cfg, norm_fn, **kwargs)
        self.out_fpn = out_fpn
        self.with_transformer = with_transformer
        dilations = self.DILATIONS
        bn_momentum = train_cfg.bn_momentum
        
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.cs = []
        self.cs.append(in_channels)
        self.point_stem = nn.ModuleList()
        self.conv0p1s1 = SubMConv3d(
            in_channels, # 3 or 6
            self.inplanes, # INI_DIM 32
            kernel_size=train_cfg.conv1_kernel_size,
            stride=1,
            dilation=1,
            indice_key='subm1',
        )
        self.bn0 = norm_fn(self.inplanes, momentum=bn_momentum)
        self.MSC = MSC(self.inplanes, norm_fn)
        self.cs.append(self.inplanes)
        self.conv1p1s2 = SparseConv3d(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            indice_key="spconv1"
        )
        self.bn1 = norm_fn(self.inplanes, momentum=bn_momentum)
        self.block1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            dilation=dilations[0],
            bn_momentum=bn_momentum,
            indice_key='subm2'
        )
        self.conv2p2s2 = SparseConv3d(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            indice_key="spconv2"
        )
        self.bn2 = norm_fn(self.inplanes, momentum=bn_momentum)
        self.block2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            dilation=dilations[1],
            bn_momentum=bn_momentum,
            indice_key="subm3"
        )
        self.conv3p4s2 = SparseConv3d(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            indice_key="spconv3"
        )
        self.bn3 = norm_fn(self.inplanes, momentum=bn_momentum)
        self.block3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            dilation=dilations[2],
            bn_momentum=bn_momentum,
            indice_key="subm4"
        )
        self.conv4p8s2 = SparseConv3d(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            indice_key="spconv4"
        )
        self.bn4 = norm_fn(self.inplanes, momentum=bn_momentum)
        self.block4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            dilation=dilations[3],
            bn_momentum=bn_momentum,
            indice_key="subm5"
        )
        self.cs.append(self.PLANES[3])
        self.convtr4p16s2 = SparseInverseConv3d(
            self.inplanes,
            self.PLANES[4],
            kernel_size=2,
            bias=False,
            indice_key='spconv4'
        )
        self.bntr4 = norm_fn(self.PLANES[4], momentum=bn_momentum)
        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(
            self.BLOCK,
            self.PLANES[4],
            self.LAYERS[4],
            dilation=dilations[4],
            bn_momentum=bn_momentum,
            indice_key='subm4'
        )
        self.convtr5p8s2 = SparseInverseConv3d(
            self.inplanes,
            self.PLANES[5],
            kernel_size=2,
            bias=False,
            indice_key='spconv3'
        )
        self.bntr5 = norm_fn(self.PLANES[5], momentum=bn_momentum)
        self.cs.append(self.PLANES[5])
        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(
            self.BLOCK,
            self.PLANES[5],
            self.LAYERS[5],
            dilation=dilations[5],
            bn_momentum=bn_momentum,
            indice_key='subm3'
        )
        self.convtr6p4s2 = SparseInverseConv3d(
            self.inplanes,
            self.PLANES[6],
            kernel_size=2,
            bias=False,
            indice_key='spconv2'
        )
        self.bntr6 = norm_fn(self.PLANES[6], momentum=bn_momentum)
        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(
            self.BLOCK,
            self.PLANES[6],
            self.LAYERS[6],
            dilation=dilations[6],
            bn_momentum=bn_momentum,
            indice_key='subm2'
        )
        self.convtr7p2s2 = SparseInverseConv3d(
            self.inplanes,
            self.PLANES[7],
            kernel_size=2,
            bias=False,
            indice_key='spconv1'
        )
        self.bntr7 = norm_fn(self.PLANES[7], momentum=bn_momentum)
        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(
            self.BLOCK,
            self.PLANES[7],
            self.LAYERS[7],
            dilation=dilations[7],
            bn_momentum=bn_momentum,
            indice_key='subm1'
        )
        self.cs.append(self.PLANES[7])
        # self.final = SparseConv3d(
        #     self.PLANES[7], out_channels, kernel_size=1, stride=1, bias=True
        # )
        self.relu = nn.ReLU(inplace=True)
        
        self.input_layer = spconv.SparseSequential(
            self.conv0p1s1, self.bn0, self.relu
        )
        self.encoder1 = spconv.SparseSequential(
            self.conv1p1s2, self.bn1, self.relu, self.block1
        )
        self.encoder2 = spconv.SparseSequential(
            self.conv2p2s2, self.bn2, self.relu, self.block2
        )
        self.encoder3 = spconv.SparseSequential(
            self.conv3p4s2, self.bn3, self.relu, self.block3
        )
        self.encoder4 = spconv.SparseSequential(
            self.conv4p8s2, self.bn4, self.relu, self.block4
        )
        if self.with_transformer:
            d_model = 256
            self.input_trans_layer = nn.Linear(self.PLANES[3], d_model)
            self.transformer = TransformerEncoder(d_model=d_model, N=2, heads=16, d_ff=512)
            self.output_trans_layer = nn.Linear(d_model, self.PLANES[3])
        # self.output_trans_layer = nn.Sequential(
            # nn.Linear(d_model, self.PLANES[3]),
            # norm_fn(self.PLANES[3]),
            # nn.ReLU()
        # )
        # self.shared_transformer = SharedTransformer(d_model, self.PLANES[3])
        # 直接特征相加不好
        # 首先decoder4上采样过后得到x特征，
        # 然后来自skip connection的特征
        # 此时两个特征的维度一样，进行加权求和
        self.decoder4 = spconv.SparseSequential(
            self.convtr4p16s2, self.bntr4, self.relu
        )
        self.decoder3 = spconv.SparseSequential(
            self.convtr5p8s2, self.bntr5, self.relu
        )
        self.decoder2 = spconv.SparseSequential(
            self.convtr6p4s2, self.bntr6, self.relu
        )
        self.decoder1 = spconv.SparseSequential(
            self.convtr7p2s2, self.bntr7, self.relu
        )
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(
            self,
            block,
            planes,
            blocks,
            stride=1,
            dilation=1,
            bn_momentum=0.1,
            indice_key=None
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = spconv.SparseSequential(
                SubMConv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                self.norm_fn(
                    planes * block.expansion,
                    momentum=bn_momentum
                )
            )
        layers = []
        # for i in range(blocks):
        #     layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation,
        #                         shortcut=downsample, bn_momentum=0.1, norm_fn=self.norm_fn))
        #     self.inplanes = planes * block.expansion
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                shortcut=downsample,
                bn_momentum=0.1, 
                norm_fn=self.norm_fn,
                indice_key = indice_key
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=dilation,
                    bn_momentum=0.1, 
                    norm_fn=self.norm_fn,
                    indice_key = indice_key
                )
            )
        blocks_ = {
            'block{}'.format(i):
                layers[i] for i in range(blocks)
        }
        blocks_ = OrderedDict(blocks_)
        return spconv.SparseSequential(blocks_)

    def forward(self, x, feature_maps):
        # x: (n, 3)
        out_p1 = self.input_layer(x) # 32
        # 先经过一个上下文单元，获取不同尺寸的上下文特征，输出通道为为32试试
        aux = []
        out_p1 = self.MSC(out_p1, aux)
        out_b1p2 = self.encoder1(out_p1) # 32
        out_b2p4 = self.encoder2(out_b1p2) # 64
        out_b3p8 = self.encoder3(out_b2p4) # 128
        out = self.encoder4(out_b3p8) # 256
        # TODO
        # 在这里再加入一个transformer模块
        # 转化数据
        if self.with_transformer:
            batch_ids = out.indices[:, 0]
            xyz = out.indices[:, 1:].float()
            feats = out.features
            input_trans_feats = self.input_trans_layer(feats)
            feats = self.transformer(xyz=xyz, feats=input_trans_feats, batch_ids=batch_ids)
            feats = self.output_trans_layer(feats) 
            out = out.replace_feature(feats)
        feature_maps.append(out)
        # pixel_dist=8
        out = self.decoder4(out) # 256
        # 这里cat不太好?
        # 将高层特征也就是out和低层特征out_b3p8进行特征融合
        # AFF处理方式是：
        # 对高层特征和底层特征相加？
        # 先对底层特征1x1的conv
        # 然后再相加，经过3x3卷积
        # attention gate
        # 对out_b3p8(n,128)下采样得到g(n/2, 256)，out作为x(n/2, 256)
        # 将g和x经过1的卷积，然后relu，然后psi，然后加权得到out_b3p8，此时维度是(n, 256)
        out_feats = torch.cat((out.features, out_b3p8.features), dim=1) # (n, plane[2]+plane[4]) (n, 128+256)
        out = out.replace_feature(out_feats)
        out = self.block5(out)
        feature_maps.append(out)
        # pixel_dist=4
        out = self.decoder3(out) # 128
        out_feats = torch.cat((out.features, out_b2p4.features), dim=1)
        out = out.replace_feature(out_feats)
        out = self.block6(out)
        feature_maps.append(out)
        # pixel_dist=2
        out = self.decoder2(out) # 96
        out_feats = torch.cat((out.features, out_b1p2.features), dim=1)
        out = out.replace_feature(out_feats)
        out = self.block7(out)
        feature_maps.append(out)
        # pixel_dist=1
        out = self.decoder1(out) # 96
        out_feats = torch.cat((out.features, out_p1.features), dim=1)
        out = out.replace_feature(out_feats)       
        out = self.block8(out)
        feature_maps.append(out)
        # 对feature_maps中的各个特征进行上下文结合？
        # 
        # output = self.final(out)
        return out
    


class Res16UNet34(Res16UNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    
class Res16UNet34C(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


class Custom1x1Subm3d(spconv.SparseConv3d):

    def forward(self, input):
        features = torch.mm(input.features, self.weight.view(self.out_channels, self.in_channels).T)
        if self.bias is not None:
            features += self.bias
        out_tensor = spconv.SparseConvTensor(features, input.indices, input.spatial_shape,
                                             input.batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


class ResidualBlock(SparseModule):

    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                Custom1x1Subm3d(in_channels, out_channels, kernel_size=1, bias=False))

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels), nn.ReLU(),
            spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key), norm_fn(out_channels), nn.ReLU(),
            spconv.SubMConv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key))

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape,
                                           input.batch_size)
        output = self.conv_branch(input)
        out_feats = output.features + self.i_branch(identity).features
        output = output.replace_feature(out_feats)

        return output


class UBlock(nn.Module):

    def __init__(self, nPlanes, norm_fn, block_reps, block,with_transformer=False, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes
        
        blocks = {
            'block{}'.format(i):
                block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) <= 1 and with_transformer:
            d_model = 128
            self.input_trans_layer = nn.Linear(nPlanes[0], d_model)
            self.transformer = TransformerEncoder(d_model=d_model, N=2, heads=16, d_ff=512)
            self.output_trans_layer = nn.Linear(d_model, nPlanes[0])
            # self.output_trans_layer = nn.Sequential(
            #     nn.Linear(d_model, nPlanes[0]),
            #     nn.BatchNorm1d(nPlanes[0],eps=1e-4, momentum=0.1),
            #     nn.ReLU()
            # )
        else:
            self.input_trans_layer = None
            self.transformer = None
            self.output_trans_layer = None
        
        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]), nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            self.u = UBlock(
                nPlanes[1:], norm_fn, block_reps, block, with_transformer, indice_key_id=indice_key_id + 1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]), nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1],
                    nPlanes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input, feature_maps):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape,
                                           output.batch_size)
        if self.input_trans_layer:
            batch_ids = output.indices[:, 0]
            xyz = output.indices[:, 1:].float()
            feats = output.features
            before_params_feats = self.input_trans_layer(feats)
            feats = self.transformer(xyz=xyz, feats=before_params_feats, batch_ids=batch_ids)
            feats = self.output_trans_layer(feats)
            output = output.replace_feature(feats)
        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder, feature_maps)
            output_decoder = self.deconv(output_decoder)
            out_feats = torch.cat((identity.features, output_decoder.features), dim=1)
            output = output.replace_feature(out_feats)
            output = self.blocks_tail(output)
            feature_maps.append(output)
        else:
            feature_maps.append(output)
        return output
