from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, normal_

# from MAE_Transformer.vit_3d import Transformer
# from models import Transformer,Embeddings


# import torch
# from MAE_Transformer.vit_3d import ViT3D



import copy
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm
from torch.nn.modules.utils import _pair, _triple
import Models_configs as configs
from torch.distributions.normal import Normal

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

# vit_3d = ViT3D(img_size=(16,64,64),
#             patch_size=8,
#             dim=252,
#             depth=6,
#             heads=16,
#             mlp_dim=3072,
#             dropout=0.1,
#             emb_dropout=0.1)
# print(vit_3d)

class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor


class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = nnf.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x

        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads # 16 * 8 = 128
        self.heads = heads
        self.scale = dim ** -0.5
        self.se_layer = SELayer(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        # print("b:",b)
        # print("n:",n)
        # print("_:",_)
        # print("h:",h)
        x = self.se_layer(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Vision_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads,
                                                dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(
                    dim, mlp_dim, dropout=dropout)))
            ]))
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim)
            # nn.Linear(dim, 512)
        )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x


# 3D MAE
class MAE_Transformer(nn.Module):
    def __init__(self, config, img_size, masking_ratio = 0.75):
        super(MAE_Transformer, self).__init__()
        self.config = config
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio
        patch_size = _triple(config.patches["size"])
        # n_patches = int((img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2]))
        n_patches = int((img_size[0] / 2 ** 2 // patch_size[0]) * (img_size[1] / 2 ** 2 // patch_size[1]) * (
                    img_size[2] / 2 ** 2 // patch_size[2]))
        in_channels = config['encoder_channels'][-1]  # 32
        patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]
        self.patch_embeddings = nn.Sequential(
            Rearrange('b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)', p1=patch_size[0], p2=patch_size[1],p3=patch_size[2]),
            nn.Linear(patch_dim, config.hidden_size),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches, config.hidden_size))
        self.to_patch, self.patch_to_emb = self.patch_embeddings[:2]
        # print("patch_embeddings:",self.patch_embeddings)
        # print(self.to_patch)
        # print(self.patch_to_emb)
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]  # pixel_values_per_patch = 16384
        # print(pixel_values_per_patch)
        # print("n_patches:",n_patches) # n_patches = 128

        self.encoder = Encoder(config)


        # decoder parameters

        encoder_dim = config.hidden_size
        decoder_dim = config.transformer["MAE_decoder_dim"]
        decoder_depth = config.transformer["MAE_decoder_depth"]
        decoder_heads = config.transformer["MAE_decoder_heads"]
        decoder_dim_head = config.transformer["MAE_decoder_dim_head"]
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        # print("enc_to_dec:",self.enc_to_dec) # 252 --> 512
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        # print("encoder_dim:",encoder_dim) # 252
        # print("decoder_dim:",decoder_dim) # 512
        self.Transformer_decoder = Vision_Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head,
                                   mlp_dim=decoder_dim * 4)
        # self.Transformer_decoder = DeformableTransformer(d_model=256,nhead=8,num_encoder_layers=6,dim_feedforward=1024,dropout=0.1,activation='relu',num_feature_levels=4,enc_n_points=4)
        self.decoder_pos_emb = nn.Embedding(n_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        self.dropout = Dropout(config.transformer["dropout_rate"])
        # self.transformer = Vision_Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head,
        #                            mlp_dim=decoder_dim * 4,dropout=0.)

    def forward(self, x):
        # embedding_output, features = self.embeddings(input_ids)
        device = x.device

        # get patches
        patches = self.to_patch(x)

        batch,n_patches,*_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)  # [2,128,252]
        # print("encoder.pos_embedding:",self.pos_embedding)
        tokens += self.pos_embedding[:, :(n_patches)]

        tokens, attn_weights = self.encoder(tokens)

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        # 计算哪些patch需要被mask，获得一些随机的值

        num_masked = int(self.masking_ratio * n_patches)  # 75%的patch(96个)需要被mask掉 (剩32个patch)
        # print("75%（即{}个patch）需要被mask掉--num_masked = {}".format(num_masked,num_masked))
        # 给这128个patch获得一组随机索引
        rand_indices = torch.rand(batch, n_patches, device=device).argsort(dim=-1)
        # print(rand_indices)
        # print("rand_indices.shape:",rand_indices.shape) # [2,128]
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        # print("masked_indices = ",masked_indices.shape) # 96
        # print("unmasked_indices = ",unmasked_indices.shape) # 32

        # get the unmasked tokens to be encoded 获得剩下的token

        batch_range = torch.arange(batch, device=device)[:, None]
        # print(batch_range.shape) # [2,1]
        tokens = tokens[batch_range, unmasked_indices]
        # print(tokens.shape) # [2,32,252]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]
        # print("masked_patches.shape:",masked_patches.shape) # [2,96,16384=8*8*8*32]

        # attend with vision transformer

        encoded_tokens = self.Transformer_decoder(tokens)
        # print(encoded_tokens.shape) # [2,32,252]

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)
        # print(decoder_tokens.shape) # [2,32,512] # 252 --> 512

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        # print("mask_tokens.shape:",mask_tokens.shape) # [2,96,512]
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)
        # print("mask_tokens.shape:",mask_tokens.shape)

        # concat the masked tokens to the decoder tokens and attend with decoder

        decoder_tokens = torch.cat((decoder_tokens, mask_tokens), dim=1)
        # print("decoder_tokens.shape:",decoder_tokens.shape) # [2,128,512]
        decoded_tokens = self.Transformer_decoder(decoder_tokens)
        # print("decoder_tokens.shape:", decoder_tokens.shape)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[:, -num_masked:]
        # print("mask_tokens.shape:", mask_tokens.shape)
        # pred_pixel_values = self.to_pixels(mask_tokens)
        # print("pred_pixel_values.shape:", pred_pixel_values.shape)

        # decoded_tokens = self.dropout(decoded_tokens)



        # calculate reconstruction loss

        recon_loss = F.mse_loss(self.to_pixels(mask_tokens), masked_patches)
        return decoded_tokens,recon_loss


class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderCup(nn.Module):
    def __init__(self, config, img_size):
        super().__init__()
        self.config = config
        self.down_factor = config.down_factor
        head_channels = config.conv_first_channel
        self.img_size = img_size
        self.conv_more = Conv3dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        self.patch_size = _triple(config.patches["size"])
        skip_channels = self.config.skip_channels
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        # print("hidden_states.shape:",hidden_states.shape)
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # print("B:",B)
        # print("n_patch:",n_patch)
        # print("hidden:",hidden)
        l, h, w = (self.img_size[0]//2**self.down_factor//self.patch_size[0]), (self.img_size[1]//2**self.down_factor//self.patch_size[0]), (self.img_size[2]//2**self.down_factor//self.patch_size[0])
        # print("l = ",l)
        # print("h = ",h)
        # print("w = ",w)
        x = hidden_states.permute(0, 2, 1)
        # print(x.shape)
        x = x.contiguous().view(B, hidden, l, h, w)
        # print(x.shape)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
                #print(skip.shape)
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class saliency_map_attention_block(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels):
        super().__init__()

        self.att_block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm3d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1=self.att_block(x)
        #print("x  x1",x.shape,x1.shape)
        # x2=torch.multiply(x,x1)
        x2=torch.mul(x,x1)
        #print("x2", x2.shape)
        return torch.add(x,x2)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            saliency_map_attention_block(out_channels),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            saliency_map_attention_block(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class CNNEncoder(nn.Module):
    def __init__(self, config, n_channels=2):
        super(CNNEncoder, self).__init__()
        self.n_channels = n_channels
        decoder_channels = config.decoder_channels
        encoder_channels = config.encoder_channels
        self.down_num = config.down_num
        self.inc = DoubleConv(n_channels, encoder_channels[0])
        self.down1 = Down(encoder_channels[0], encoder_channels[1])
        self.down2 = Down(encoder_channels[1], encoder_channels[2])
        self.width = encoder_channels[-1]
        self.up = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
        self.se_3D_layer = ChannelSpatialSELayer3D(32)


    def forward(self, x):
        features = []
        x1 = self.inc(x)
        features.append(x1)
        x2 = self.down1(x1)
        features.append(x2)
        x3 = self.down2(x2)
        x3 = self.convnext_3D(x3)
        x3 = self.up(x3)
        x3 = self.se_3D_layer(x3)
        features.append(x3)
        feats_down = x3
        for i in range(self.down_num):
            feats_down = nn.MaxPool3d(2)(feats_down)  # 3D MaxPooling
            features.append(feats_down)
        return x3, features[::-1]

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class MAE_TransRNet(nn.Module):
    def __init__(self, config, img_size=(64, 256, 256), int_steps=7):
        super(MAE_TransRNet, self).__init__()
        self.cnn_encoder = CNNEncoder(config,n_channels=2)
        self.mae_transformer = MAE_Transformer(config, img_size, masking_ratio = 0.75)
        self.decoder = DecoderCup(config, img_size)

        self.reg_head = RegistrationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config['n_dims'],
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(img_size)
        self.config = config
        self.integrate = VecInt(img_size, int_steps)

    def forward(self, x):

        source = x[:,0:1,:,:]
        cnn_output,features = self.cnn_encoder(x)
        x, recon_loss = self.mae_transformer(cnn_output)  # (B, n_patch, hidden)
        # print("After_MAE_Transformer_shape:",x.shape)
        x = self.decoder(x, features)
        # print(x.shape)
        flow = self.reg_head(x)
        flow = self.integrate(flow)
        out = self.spatial_trans(source, flow)
        return out, flow

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

CONFIGS = {
    'MAE_TransRNet': configs.get_3DReg_config(),
}


# vv = MAE_TransRNet(CONFIGS['MAE_TransRNet'],(64,256,256))
# print(vv)
# img = torch.randn(1,2,64,256,256)
# # output,flow = vv(img)
# output,flow = vv(img)
# print("MAE_SE_Saliary_Map_Attention_Output:",output.shape)
# print("Flow:",type(flow))
# print("Flow_shape:",flow.shape)