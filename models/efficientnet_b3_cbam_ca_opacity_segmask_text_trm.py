import copy
import math
import torch
import argparse
import torch.nn.functional as F
from models import CBAMBlock
from models import CoordAtt
from functools import partial
from torch import nn
from typing import Any, Callable, List, Optional
from torch.nn.modules.utils import _pair
from torchvision.ops import StochasticDepth


__all__ = ["EfficientNet", "efficientnet_b3"]


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: bool = True,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                  dilation=dilation, groups=groups, bias=norm_layer is None)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels


class Opa_Att(nn.Module):
    def __init__(self, channel, reduction=16, activation=None):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            # nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.Conv2d(channel, reduction, 1, bias=False),
            activation(),
            # nn.Conv2d(channel // reduction, channel, 1, bias=False)
            nn.Conv2d(reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # [2, 65, 64, 64]
        max_result = self.maxpool(x)  # [2, 65, 1, 1]
        avg_result = self.avgpool(x)  # [2, 65, 1, 1]
        max_out = self.se(max_result)  # [2, 64, 1, 1]
        avg_out = self.se(avg_result)  # [2, 64, 1, 1]
        output = self.sigmoid(max_out + avg_out)  # [2, 64, 1, 1]
        return output


class Attention(nn.Module):
    def __init__(self, ch_in=1, ch_mid=64):
        super(Attention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_mid, kernel_size=_pair(3), padding=1, stride=_pair(1)),  # 256
            nn.Conv2d(ch_mid, ch_in, kernel_size=_pair(3), padding=1, stride=_pair(1)),  # 128
            nn.BatchNorm2d(ch_in),
            nn.ReLU(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seg):
        x = self.conv(x_seg)
        att = self.sigmoid(x)

        return att


class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(self,
                 expand_ratio: float, kernel: int, stride: int,
                 input_channels: int, out_channels: int, num_layers: int,
                 width_mult: float, depth_mult: float) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'expand_ratio={expand_ratio}'
        s += ', kernel={kernel}'
        s += ', stride={stride}'
        s += ', input_channels={input_channels}'
        s += ', out_channels={out_channels}'
        s += ', num_layers={num_layers}'
        s += ')'
        return s.format(**self.__dict__)

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    def __init__(self, cnf: MBConvConfig, stochastic_depth_prob: float, norm_layer: Callable[..., nn.Module],
                 se_layer: Callable[..., nn.Module] = CBAMBlock) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(ConvNormActivation(cnf.input_channels, expanded_channels, kernel_size=1,
                                             norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        layers.append(ConvNormActivation(expanded_channels, expanded_channels, kernel_size=cnf.kernel,
                                         stride=cnf.stride, groups=expanded_channels,
                                         norm_layer=norm_layer, activation_layer=activation_layer))

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(ConvNormActivation(expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
                                         activation_layer=None))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNet(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: List[MBConvConfig],
            dropout: float,
            stochastic_depth_prob: float = 0.2,
            in_channels: int = 3,
            num_classes: int = 1000,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.stochastic_depth_prob = stochastic_depth_prob

        layers: List[nn.Module] = []

        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(in_channels, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                               activation_layer=nn.SiLU))

        self.total_stage_blocks = sum([cnf.num_layers for cnf in inverted_residual_setting])
        self.stage_block_id = 0

        layers.append(self._make_block(inverted_residual_setting[0], MBConv, norm_layer))
        layers.append(self._make_block(inverted_residual_setting[1], MBConv, norm_layer))
        layers.append(self._make_block(inverted_residual_setting[2], MBConv, norm_layer))
        layers.append(self._make_block(inverted_residual_setting[3], MBConv, norm_layer))
        layers.append(self._make_block(inverted_residual_setting[4], MBConv, norm_layer))
        layers.append(self._make_block(inverted_residual_setting[5], MBConv, norm_layer))
        layers.append(self._make_block(inverted_residual_setting[6], MBConv, norm_layer))
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 4 * lastconv_input_channels

        layers.append(ConvNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                         norm_layer=norm_layer, activation_layer=nn.SiLU))
        self.features = nn.Sequential(*layers)
        self.ca = CoordAtt(lastconv_output_channels, lastconv_output_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        self.opa_att = Opa_Att(channel=385, activation=partial(nn.ReLU, inplace=True))
        self.adjust_channels = nn.Conv2d(385, 384, kernel_size=1)

        self.seg_att1 = Attention()
        self.seg_att2 = Attention()
        self.seg_att3 = Attention()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _make_block(self, cnf, block, norm_layer):
        stage: List[nn.Module] = []
        for _ in range(cnf.num_layers):
            # copy to avoid modifications. shallow copy is enough
            block_cnf = copy.copy(cnf)
            # overwrite info if not the first conv in the stage
            if stage:
                block_cnf.input_channels = block_cnf.out_channels
                block_cnf.stride = 1
            # adjust stochastic depth probability based on the depth of the stage block
            sd_prob = self.stochastic_depth_prob * float(self.stage_block_id) / self.total_stage_blocks

            stage.append(block(block_cnf, sd_prob, norm_layer))
            self.stage_block_id += 1
        return nn.Sequential(*stage)

    def _forward_impl(self, via, opacity, segmask):
        opacity = torch.unsqueeze(opacity, dim=1)
        segmask = torch.unsqueeze(segmask, dim=1)

        opacity = F.interpolate(opacity, scale_factor=0.03125)
        segmask_128 = F.interpolate(segmask, scale_factor=0.25)
        segmask_32 = F.interpolate(segmask, scale_factor=0.0625)
        segmask_16 = F.interpolate(segmask, scale_factor=0.03125)
        x = self.features[0](via)  # [2, 40, 256, 256]
        x = self.features[1](x)  # [2, 24, 256, 256]
        x = self.features[2](x)  # [2, 32, 128, 128]
        seg_att1 = self.seg_att1(segmask_128)
        x = x + x * seg_att1

        x = self.features[3](x)  # [2, 48, 64, 64]
        x = self.features[4](x)  # [2, 96, 32, 32]
        seg_att2 = self.seg_att2(segmask_32)
        x = x + x * seg_att2

        x = self.features[5](x)
        x = self.features[6](x)
        seg_att3 = self.seg_att3(segmask_16)
        x = x + x * seg_att3  # [2, 232, 16, 16]

        x = self.features[7](x)  # [2, 384, 16, 16]

        x_fuse = torch.cat([x, opacity], dim=1)  # [2, 385, 16, 16]
        x_att = self.opa_att(x_fuse)  # 得到每一个通道的权重 [2, 385, 1, 1]
        x = x_fuse + x_fuse * x_att
        x = self.adjust_channels(x)
        x = self.features[8](x)  # [2, 1536, 16, 16]
        x = self.ca(x)  # [2, 1536, 16, 16]
        x = self.avgpool(x)  # [2, 1536, 1, 1]
        x = torch.flatten(x, 1)  # [2, 1536]
        x = self.classifier(x)
        return x

    def forward(self, via, opacity, segmask):
        return self._forward_impl(via, opacity, segmask)


def _efficientnet_conf(width_mult: float, depth_mult: float, **kwargs: Any) -> List[MBConvConfig]:
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),  # These are the MBConv1 MBConv6 layers
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    return inverted_residual_setting


def _efficientnet_model(
        arch: str,
        inverted_residual_setting: List[MBConvConfig],
        dropout: float,
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> EfficientNet:
    model = EfficientNet(inverted_residual_setting, dropout, **kwargs)
    return model


def efficientnet_b3(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    inverted_residual_setting = _efficientnet_conf(width_mult=1.2, depth_mult=1.4, **kwargs)
    return _efficientnet_model("efficientnet_b3", inverted_residual_setting, 0.3, pretrained, progress, **kwargs)


class Embeddings(nn.Module):
    def __init__(self, ch_in, hidden_size, dropout_rate):
        """
        Transformer这里输入的特征图是[32,32]
        :param ch_in: 256
        :param hidden_size: 768
        :param dropout_rate:
        """
        super(Embeddings, self).__init__()
        self.embeddings = nn.Conv1d(in_channels=ch_in,
                                    out_channels=hidden_size,
                                    kernel_size=_pair(1), stride=_pair(1))
        # n_patches = ch_in
        # self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))  # 生成对应所有序列长度的位置编码

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)  # [16, 768, 1]
        x = torch.squeeze(x, -1)
        # x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        # embeddings = x + self.position_embeddings
        embeddings = self.dropout(x)
        return embeddings


class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)  # mlp_dim   3072
        self.fc2 = nn.Linear(mlp_dim, hidden_size)  # hidden_size 768
        self.act_fn = nn.GELU()  # 激活函数
        self.dropout = nn.Dropout(dropout_rate)  # dropout_rate 0.1

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)  # 一种初始化权重和偏置的方法  基本思想是：通过网络层时，输入和输出的方差相同，包括前向传播和后向传播
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)  # std代表标准差
        nn.init.normal_(self.fc2.bias, std=1e-6)  # 正态分布

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MHAttention(nn.Module):
    def __init__(self, num_heads, hidden_size, attention_dropout_rate):
        super(MHAttention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(
            hidden_size / self.num_attention_heads)  # hidden_size = 768  num_attention_heads = 12  每一个注意力头能分配到多少的数据
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 所有头的尺寸  等于每个头的尺寸*头的数量  768

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)  # Q K 一开始初始化的尺寸就是所有头加起来的
        self.value = nn.Linear(hidden_size, self.all_head_size)  # 这三句话中，每句话代表的是一个线性层，输入是768维度的特征，输出也是768维度的特征

        self.out = nn.Linear(hidden_size, hidden_size)  # 线性层，也就是全连接层，输入是768，输出是768

        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)  # 配置中的值是0.0，也就是说注意力参数的值没有失活的，都是起作用的

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # return x.permute(0, 2, 1, 3)  # 将tensor中的维度换位，也就是说，现在x是一个四维的
        return x  # 将tensor中的维度换位，也就是说，现在x是一个四维的

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)  # self.query就是WQ
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # tensor的乘法，输入可以是高维的  Q和K相乘 点积
        # 当输入是都是二维时，就是普通的矩阵乘法，和tensor.mm函数用法相同。
        # 当输入有多维时，把多出的一维作为batch提出来，其他部分做矩阵乘法。

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # 除以一个数值是为了在反向传播时，求梯度更加稳定
        attention_probs = self.softmax(attention_scores)
        # weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # attention * values
        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.permute(0, 2, 1).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output  # , weights


class Block(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate, num_heads, attention_dropout_rate):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)  # 层归一化  eps=1e-6：为保证数值稳定性而加在分母上的一种值。默认值:1 e-5
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size, mlp_dim, dropout_rate)
        self.attn = MHAttention(num_heads, hidden_size, attention_dropout_rate)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)  # 多头注意力
        x = x + h

        h = x
        x = self.ffn_norm(x)  # 层归一化
        x = self.ffn(x)  # MLP 前向传播
        x = x + h
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_size, mlp_dim, dropout_rate, num_heads, attention_dropout_rate):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(hidden_size, mlp_dim, dropout_rate, num_heads, attention_dropout_rate)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for num, layer_block in enumerate(self.layer):
            hidden_states = layer_block(hidden_states)  # 编码器里面有6层，hidden_states就是每一层出来的东西
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights  # 返回的是层归一化以后的数据，注意力权重
        # trm_features是具有三个元素的一个列表，其中每一个列表都是transformer的中间层出来的特征


class Transformer(nn.Module):
    def __init__(self, ch_in, num_layers, hidden_size, mlp_dim, dropout_rate, num_heads,
                 attention_dropout_rate):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(ch_in, hidden_size, dropout_rate)
        self.encoder = Encoder(num_layers, hidden_size, mlp_dim, dropout_rate, num_heads, attention_dropout_rate)

    def forward(self, input_ids):  # input
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded  # , attn_weights


class CrossModal(nn.Module):
    def __init__(self, args):
        super(CrossModal, self).__init__()
        self.checkpoints = None
        self.load_weights_dict = None
        self.weights = None
        self.args = args
        self.img_feature_extract = efficientnet_b3()
        self.transformer = Transformer(self.args.ch_in,
                                       self.args.num_layers,
                                       self.args.hidden_size,
                                       self.args.mlp_dim,
                                       self.args.dropout_rate,
                                       self.args.num_heads,
                                       self.args.attention_dropout_rate)
        self.final = nn.Linear(self.args.hidden_size, self.args.num_classes)

    def forward(self, via, opacity, segmask, text):
        x_img = self.img_feature_extract(via, opacity, segmask)  # [B, 1000]
        x = torch.cat((x_img, text), dim=1)  # [B, 1014]
        x = torch.unsqueeze(torch.unsqueeze(x, -1), -1)  # [B, 1014, 1, 1]
        x = self.transformer(x)
        x = self.final(x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ----------------Transformer-------------------#
    parser.add_argument('--ch_in', default=1014, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--mlp_dim', default=256, type=int)
    parser.add_argument('--dropout_rate', default=0.0, type=int)
    parser.add_argument('--num_heads', default=2, type=int)
    parser.add_argument('--attention_dropout_rate', default=0.0, type=int)
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--weights', default=None, type=str)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossModal(args).to(device)
    Acetic_image = torch.rand(size=(2, 3, 512, 512)).to(device)  # 醋酸图
    opacity = torch.rand(size=(2, 512, 512)).to(device)
    segmask = torch.rand(size=(2, 512, 512)).to(device)
    text = torch.rand(size=(2, 14)).to(device)
    x = model(Acetic_image, opacity, segmask, text)
    print(x.size())
