from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule
from pointins3d.ops import functions
import pdb


# current 1x1 conv in spconv2x has a bug. It will be removed after the bug is fixed
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


def conv3x3(in_planes, out_planes, stride=1):
    return spconv.SubMConv3d(in_channels=in_planes, out_channels=out_planes,
                             kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return spconv.SubMConv3d(in_channels=in_planes, out_channels=out_planes,
                             kernel_size=1, stride=stride, padding=0, bias=False)


# 可以先用nn对输入的sparsetensor变量中的feature进行变换，然后再创建一个sparsetensor
class depthwise_separable_conv(SparseModule):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = spconv.SubMConv3d(in_ch, in_ch, kernel_size=kernel_size, padding=padding,
                                           groups=in_ch, bias=bias, stride=stride)
        self.pointwise = spconv.SubMConv3d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, input):
        output = self.depthwise(input)
        output = self.pointwise(output)

        return output


class BasicBlock(SparseModule):

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, indice_key=None):
        super().__init__()
        self.residual_function = spconv.SparseSequential(
            norm_fn(inplanes),
            nn.ReLU(),
            spconv.SubMConv3d(
                in_channels=inplanes,
                out_channels=planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                indice_key=indice_key),
            norm_fn(planes),
            nn.ReLU(),
            spconv.SubMConv3d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                indice_key=indice_key),
        )
        self.shortcut = spconv.SparseSequential(nn.Identity())
        if stride != 1 or inplanes != planes:
            self.shortcut = spconv.SparseSequential(
                norm_fn(inplanes),
                nn.ReLU(),
                Custom1x1Subm3d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices,
                                           input.spatial_shape, input.batch_size)
        output = self.residual_function(input)
        out_feats = output.features + self.shortcut(identity).features
        output = output.replace_feature(out_feats)
        return output


class BottleneckBlock(SparseModule):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes // 4, stride=1)
        self.bn1 = nn.BatchNorm1d(inplanes, eps=1e-4, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes // 4, planes // 4, stride=stride)
        self.bn2 = nn.BatchNorm1d(planes // 4, eps=1e-4, momentum=0.1)

        self.conv3 = conv1x1(planes // 4, planes, stride=1)
        self.bn3 = nn.BatchNorm1d(planes // 4, eps=1e-4, momentum=0.1)
        self.residual_function = spconv.SparseSequential(
            self.bn1, self.relu, self.conv1,
            self.bn2, self.relu, self.conv2,
            self.bn3, self.relu, self.conv3,
        )
        self.shortcut = spconv.SparseSequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = spconv.SparseSequential(
                nn.BatchNorm1d(inplanes, eps=1e-4, momentum=0.1),
                self.relu,
                spconv.SparseConv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, input):
        identity = input
        output = self.residual_function(input)
        out_feats = output.features + self.shortcut(identity).features
        output = output.replace_feature(out_feats)
        return output


class BasicTransBlock(SparseModule):

    def __init__(self, in_ch, heads, dim_head, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp',
                 rel_pos=True, norm_fn=None, indice_key=None):
        super().__init__()
        self.bn1 = norm_fn(in_ch)

        self.attn = LinearAttention(in_ch, heads=heads, dim_head=in_ch // heads, attn_drop=attn_drop,
                                    proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                    rel_pos=rel_pos, norm_fn=norm_fn)

        self.bn2 = norm_fn(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = spconv.SubMConv3d(in_ch, in_ch, kernel_size=1, bias=False, indice_key=indice_key)
        # conv1x1 has not difference with mlp in performance

    def forward(self, input):
        output = input.replace_feature(self.bn1(input.features))
        # output = self.bn1(input)
        # output_tensor = output.dense()
        output, q_k_attn = self.attn(output)

        output = output + input
        residue = output

        output = output.replace_feature(self.bn2(output.features))
        output = output.replace_feature(self.relu(output.features))
        output = output.replace_feature(self.mlp(output.features))
        # output = self.bn2(output)
        # output = self.relu(output)
        # output = self.mlp(output)
        output = output.replace_feature(output.features + residue.features)
        # output += residue
        return output


class BasicTransDecoderBlock(SparseModule):

    def __init__(self, in_ch, out_ch, heads, dim_head, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp',
                 rel_pos=True, norm_fn=None):
        super().__init__()

        self.bn_l = norm_fn(in_ch)
        self.bn_h = norm_fn(out_ch)

        self.conv_ch = spconv.SubMConv3d(in_ch, out_ch, kernel_size=1)
        # 这里采用window-attention，减少计算量
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.attn = LinearAttentionDecoder(in_ch, out_ch, heads=heads, dim_head=out_ch // heads, attn_drop=attn_drop,
                                           proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                           rel_pos=rel_pos)

        self.bn2 = norm_fn(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = spconv.SubMConv3d(out_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, input1, input2):
        residue = F.interpolate(self.conv_ch(input1), size=input2.shape[-2:], mode='bilinear', align_corners=True)
        # x1: low-res, x2: high-res
        input1 = self.bn_l(input1)
        input2 = self.bn_h(input2)
        
        output, q_k_attn = self.attn(input2, input1)

        output = output + residue
        residue = output

        output = self.bn2(output)
        output = self.relu(output)
        output = self.mlp(output)

        output += residue

        return output


########################################################################
# Transformer components

class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp',
                 rel_pos=True, norm_fn=None):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos
        
        # depthwise conv is slightly better than conv1x1
        # self.to_qkv = spconv.SubMConv3d(dim, self.inner_dim * 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.linear_q = spconv.SubMConv3d(dim, self.inner_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.linear_k = spconv.SubMConv3d(dim, self.inner_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.linear_v = spconv.SubMConv3d(dim, self.inner_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.linear_p = spconv.SparseSequential(

        )
        self.linear_w = spconv.SparseSequential(

        )
        self.to_out = spconv.SubMConv3d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_qkv = nn.Conv3d(dim, self.inner_dim * 3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv3d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        # self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            # 2D input-independent relative position encoding is a little bit better than
            # 1D input-denpendent counterpart
            self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
            self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, x):
        # 对x的indices进行
        # B, C, H, W = x.shape
        # B, inner_dim, H, W
        # 需要获取到这样的数据
        # 可以理解为我在第b个batch中，第i个channel，从深度d宽度w高度h的地方，获取到点的特征
        B, C, D, H, W = x.dense().shape
        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n,c)
        q_feats, k_feats, v_feats = q.features, k.features, v.features
        # q = rearrange(q, 'b (dim_head heads) d h w -> b heads d (h w) dim_head', dim_head=self.dim_head,
        #               heads=self.heads,
        #               h=H, w=W, d=D)
        # k, v = map(
        #     lambda t: rearrange(t, 'b (dim_head heads) d h w -> b heads d (h w) dim_head', dim_head=self.dim_head,
        #                         heads=self.heads, h=self.reduce_size, w=self.reduce_size, d=self.reduce_size), (k, v))
        # q_k_attn = torch.einsum('bhid,bhjd->bhij', q_feats, k_feats)
        q_k_attn = torch.matmul(q_feats, k_feats.T)  # (n, n)
        # if self.rel_pos:
        #     relative_position_bias = self.relative_position_encoding(H, W)
        #     q_k_attn += relative_position_bias
        # rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, H, W, self.dim_head)
        # q_k_attn = q_k_attn + rel_attn_h + rel_attn_w

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        # q_k_attn = self.attn_drop(q_k_attn)

        out_feats = torch.einsum('nn,ni->ni', q_k_attn, v)
        # out_feats = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        # out_feats = rearrange(out_feats, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W,
        #                       dim_head=self.dim_head,
        #                       heads=self.heads)
        # 获取了自注意力机制的特征，要将其转化为sparsetensor
        # 但不知道其他数据是否改变，例如：indices、spatial_shape
        out = spconv.SparseConvTensor(out_feats, x.indices, x.spatial_shape, x.batch_size)
        out = self.to_out(x - out)
        out = out.replace_feature(self.proj_drop(out).features)

        return x + out, q_k_attn

class LinearAttentionDecoder(SparseModule):

    def __init__(self, in_dim, out_dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16,
                 projection='interp', rel_pos=True):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        self.to_kv = spconv.SubMConv3d(in_dim, self.inner_dim * 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.to_q = spconv.SubMConv3d(out_dim, self.inner_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.to_out = spconv.SubMConv3d(self.inner_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True)

        # self.to_kv = depthwise_separable_conv(in_dim, self.inner_dim * 2)
        # self.to_q = depthwise_separable_conv(out_dim, self.inner_dim)
        # self.to_out = depthwise_separable_conv(self.inner_dim, out_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
            # self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, q, x):

        B, C, H, W = x.shape  # low-res feature shape
        BH, CH, HH, WH = q.shape  # high-res feature shape

        k, v = self.to_kv(x).chunk(2, dim=1)  # B, inner_dim, H, W
        q = self.to_q(q)  # BH, inner_dim, HH, WH

        if self.projection == 'interp' and H != self.reduce_size:
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))

        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=HH, w=WH)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(HH, WH)
            q_k_attn += relative_position_bias
            # rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, HH, WH, self.dim_head)
            # q_k_attn = q_k_attn + rel_attn_h + rel_attn_w

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=HH, w=WH, dim_head=self.dim_head,
                        heads=self.heads)

        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn


class RelativePositionEmbedding(SparseModule):
    # input-dependent relative position
    def __init__(self, dim, shape):
        super().__init__()

        self.dim = dim
        self.shape = shape

        self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dim)) * 0.02)
        self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dim)) * 0.02)

        coords = torch.arange(self.shape)
        relative_coords = coords[None, :] - coords[:, None]  # h, h
        relative_coords += self.shape - 1  # shift to start from 0

        self.register_buffer('relative_position_index', relative_coords)

    def forward(self, q, Nh, H, W, dim_head):
        # q: B, Nh, HW, dim
        B, _, _, dim = q.shape

        # q: B, Nh, H, W, dim_head
        q = rearrange(q, 'b heads (h w) dim_head -> b heads h w dim_head', b=B, dim_head=dim_head, heads=Nh, h=H, w=W)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, 'w')

        rel_logits_h = self.relative_logits_1d(q.permute(0, 1, 3, 2, 4), self.key_rel_h, 'h')

        return rel_logits_w, rel_logits_h

    def relative_logits_1d(self, q, rel_k, case):

        B, Nh, H, W, dim = q.shape

        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)  # B, Nh, H, W, 2*shape-1

        if W != self.shape:
            # self_relative_position_index origin shape: w, w
            # after repeat: W, w
            relative_index = torch.repeat_interleave(self.relative_position_index, W // self.shape, dim=0)  # W, shape
        relative_index = relative_index.view(1, 1, 1, W, self.shape)
        relative_index = relative_index.repeat(B, Nh, H, 1, 1)

        rel_logits = torch.gather(rel_logits, 4, relative_index)  # B, Nh, H, W, shape
        rel_logits = rel_logits.unsqueeze(3)
        rel_logits = rel_logits.repeat(1, 1, 1, self.shape, 1, 1)

        if case == 'w':
            rel_logits = rearrange(rel_logits, 'b heads H h W w -> b heads (H W) (h w)')

        elif case == 'h':
            rel_logits = rearrange(rel_logits, 'b heads W w H h -> b heads (H W) (h w)')

        return rel_logits


class RelativePositionBias(SparseModule):
    # input-independent relative position attention
    # As the number of parameters is smaller, so use 2D here
    # Borrowed some code from SwinTransformer: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    def __init__(self, num_heads, h, w):
        super().__init__()
        self.num_heads = num_heads
        self.h = h
        self.w = w

        self.relative_position_bias_table = nn.Parameter(
            torch.randn((2 * h - 1) * (2 * w - 1), num_heads) * 0.02)

        coords_h = torch.arange(self.h)
        coords_w = torch.arange(self.w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, h, w
        coords_flatten = torch.flatten(coords, 1)  # 2, hw

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.h - 1
        relative_coords[:, :, 1] += self.w - 1
        relative_coords[:, :, 0] *= 2 * self.h - 1
        relative_position_index = relative_coords.sum(-1)  # hw, hw

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, H, W):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.h,
                                                                                                               self.w,
                                                                                                               self.h * self.w,
                                                                                                               -1)  # h, w, hw, nH
        relative_position_bias_expand_h = torch.repeat_interleave(relative_position_bias, H // self.h, dim=0)
        relative_position_bias_expanded = torch.repeat_interleave(relative_position_bias_expand_h, W // self.w,
                                                                  dim=1)  # HW, hw, nH

        relative_position_bias_expanded = relative_position_bias_expanded.view(H * W, self.h * self.w, self.num_heads) \
            .permute(2, 0, 1).contiguous().unsqueeze(0)

        return relative_position_bias_expanded


###########################################################################
# Unet Transformer building block

class down_block_trans(SparseModule):
    def __init__(self, in_ch, out_ch, num_block, bottleneck=False, maxpool=True, heads=4, dim_head=64, attn_drop=0.,
                 proj_drop=0., reduce_size=16, projection='interp', rel_pos=True, indice_key_id=1, norm_fn=None):

        super().__init__()

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock

        attn_block = BasicTransBlock
        if maxpool:
            block_list = {
                'block0':
                    spconv.SparseSequential(
                        spconv.SparseMaxPool3d(2),
                        block(in_ch, out_ch, stride=1, norm_fn=norm_fn, indice_key='subm{}'.format(indice_key_id))
                    )
            }
        else:
            block_list = {
                'block0':
                    block(in_ch, out_ch, stride=2, norm_fn=norm_fn, indice_key='subm{}'.format(indice_key_id))
            }
        # block_list = {
        #     'block0':
        #         block(in_ch, out_ch, norm_fn=norm_fn, indice_key='subm{}'.format(indice_key_id))
        # }
        # if maxpool:
        #     block_list.append(spconv.SparseMaxPool3d(2))
        #     block_list.append(block(in_ch, out_ch, stride=1, indice_key='subm{}'.format(indice_key_id)))
        # else:
        #     block_list.append(block(in_ch, out_ch, stride=2, indice_key='subm{}'.format(indice_key_id)))

        assert num_block > 0
        for i in range(1, num_block):
            block_list['block{}'.format(i)] = \
                attn_block(out_ch, heads, dim_head, attn_drop=attn_drop,
                           proj_drop=proj_drop, reduce_size=reduce_size,
                           projection=projection, rel_pos=rel_pos, norm_fn=norm_fn,
                           indice_key='subm{}'.format(indice_key_id))
        block_list = OrderedDict(block_list)
        self.blocks = spconv.SparseSequential(block_list)

    def forward(self, input):
        # 需要改变p，o
        output = self.blocks(input)

        return output


class up_block_trans(SparseModule):
    def __init__(self, in_ch, out_ch, num_block, bottleneck=False, heads=4, dim_head=64, attn_drop=0., proj_drop=0.,
                 reduce_size=16, projection='interp', rel_pos=True, norm_fn=None):
        super().__init__()

        self.attn_decoder = BasicTransDecoderBlock(in_ch, out_ch, heads=heads, dim_head=dim_head, attn_drop=attn_drop,
                                                   proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                                   rel_pos=rel_pos, norm_fn=norm_fn)

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock
        attn_block = BasicTransBlock

        block_list = []

        for i in range(num_block):
            block_list.append(
                attn_block(out_ch, heads, dim_head, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                           projection=projection, rel_pos=rel_pos, norm_fn=norm_fn))

        block_list.append(block(2 * out_ch, out_ch, stride=1, norm_fn=norm_fn))

        self.blocks = spconv.SparseSequential(*block_list)

    def forward(self, input1, input2):
        # x1: low-res feature, x2: high-res feature
        output = self.attn_decoder(input1, input2)
        output = torch.cat([input2, output], dim=1)
        output = self.blocks(output)

        return output


class block_trans(SparseModule):
    def __init__(self, in_ch, num_block, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16,
                 projection='interp', rel_pos=True):
        super().__init__()

        block_list = []

        attn_block = BasicTransBlock

        assert num_block > 0
        for i in range(num_block):
            block_list.append(
                attn_block(in_ch, heads, dim_head, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                           projection=projection, rel_pos=rel_pos))
        self.blocks = spconv.SparseSequential(*block_list)

    def forward(self, input):
        output = self.blocks(input)

        return output


###########################################################################
# Unet  building block

class down_block(SparseModule):
    def __init__(self, in_ch, out_ch, scale, num_block, bottleneck=False, pool=True):
        super().__init__()

        block_list = []

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock

        if pool:
            block_list.append(spconv.SparseMaxPool3d(scale))
            block_list.append(block(in_ch, out_ch))
        else:
            block_list.append(block(in_ch, out_ch, stride=2))

        for i in range(num_block - 1):
            block_list.append(block(out_ch, out_ch, stride=1))

        self.conv = spconv.SparseSequential(*block_list)

    def forward(self, input):
        return self.conv(input)


class up_block(SparseModule):
    def __init__(self, in_ch, out_ch, num_block, norm_fn=None, scale=(2, 2), bottleneck=False):
        super().__init__()
        self.scale = scale

        self.conv_ch = spconv.SubMConv3d(in_ch, out_ch, kernel_size=1)

        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock

        block_list = []
        block_list.append(block(2 * out_ch, out_ch, norm_fn=norm_fn))

        for i in range(num_block - 1):
            block_list.append(block(out_ch, out_ch, norm_fn=norm_fn))

        self.conv = spconv.SparseSequential(*block_list)

    def forward(self, input1, input2):
        input1 = F.interpolate(input1, scale_factor=self.scale, mode='bilinear', align_corners=True)
        input1 = self.conv_ch(input1)

        out = torch.cat([input2, input1], dim=1)
        out = self.conv(out)

        return out
