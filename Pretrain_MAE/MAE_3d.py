import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
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


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads,
                                                dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(
                    dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

class MAE(nn.Module):
    def __init__(
        self,
        image_size=(64,128,128),
        patch_size=16,
        encoder_dim=768,
        mlp_dim=3072,
        channels=1,
        decoder_dim = 512,
        masking_ratio = 0.75,
        decoder_depth = 6,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        num_patches = (image_size[0] // patch_size) * \
                      (image_size[1] // patch_size) * (image_size[2] // patch_size)

        patch_dim = channels * patch_size * patch_size * patch_size
        self.patch_embeddings = nn.Sequential(
            Rearrange('b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)', p1 = patch_size, p2 = patch_size,
                      p3 = patch_size),
            nn.Linear(patch_dim, encoder_dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, encoder_dim))
        self.to_patch, self.patch_to_emb = self.patch_embeddings[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        # decoder parameters

        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        # print("encoder_dim:",encoder_dim) # 768
        # print("decoder_dim:",decoder_dim) # 512
        self.encoder = Transformer(dim = encoder_dim, depth=32, heads=16, dim_head=decoder_dim_head,mlp_dim=decoder_dim * 4, dropout=0.1)
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = mlp_dim,dropout=0.1)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(decoder_dim)
        )
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        new_patch_size = (4, 4, 4)
        self.conv3d_transpose = nn.ConvTranspose3d(decoder_dim, 16, kernel_size=new_patch_size, stride=new_patch_size)
        self.conv3d_transpose_1 = nn.ConvTranspose3d(
            in_channels=16, out_channels=1, kernel_size=new_patch_size, stride=new_patch_size
        )
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        # print("Img_size:",img.shape)
        # print("Patches:",patches.shape)
        batch, num_patches, *_ = patches.shape
        # print("Num_patches:",num_patches)

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches) # [1, 216, 768]
        # print("Token:",tokens.shape) # [1, 216, 768]
        # print("pos_embedding:",self.pos_embedding.shape) # [1, 216, 768]
        tokens = tokens + self.pos_embedding[:, :(num_patches)] # [1, 216, 768]
        # print("After pos_embedding Token:", tokens.shape)  # [1, 216, 768]

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        # print("75%（即{}个patch）需要被mask掉--num_masked = {}".format(num_masked,num_masked))

        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        # print("rand_indices.shape:",rand_indices.shape) # [1,216]
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        # print("batch_range.shape:",batch_range.shape) # [1,1]
        tokens = tokens[batch_range, unmasked_indices]
        # print("token_shape:",tokens.shape) # [1,54,768]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]
        # print("masked_patches.shape:", masked_patches.shape) # [1, 162, 4096=16*16*16*1]

        # attend with vision transformer

        # encoder to decoder

        tokens = self.encoder(tokens)

        decoder_tokens = self.enc_to_dec(tokens)
        # print("E_2_D decoder_tokens.shape", decoder_tokens.shape)
        # decoder_tokens = self.decoder(decoder_tokens)
        # print("E_2_D decoder_tokens.shape",decoder_tokens.shape)

        decoder_tokens = self.to_latent(decoder_tokens)
        decoder_tokens = self.mlp_head(decoder_tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder


        # reapply decoder position embedding to unmasked tokens

        decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)
        # print("decoder_tokens.shape:", decoder_tokens.shape)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        # print("mask_tokens.shape:", mask_tokens.shape)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)
        # print("mask_tokens.shape:", mask_tokens.shape)

        # concat the masked tokens to the decoder tokens and attend with decoder

        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim = 1)
        # print("decoder_tokens.shape:", decoder_tokens.shape)
        decoded_tokens = self.decoder(decoder_tokens)
        # print("decoder_tokens.shape:", decoder_tokens.shape)

        decoder_tokens = decoder_tokens.transpose(1, 2)
        # print("decoder_tokens.shape:", decoder_tokens.shape)
        cuberoot = round(math.pow(decoder_tokens.size()[2], 1 / 3))
        x_shape = decoder_tokens.size()
        x = torch.reshape(decoder_tokens, [x_shape[0], x_shape[1], cuberoot, cuberoot, cuberoot])
        x = self.conv3d_transpose(x)
        x = self.conv3d_transpose_1(x)

        # splice out the mask tokens and project to pixel values

        # mask_tokens = decoded_tokens[:, :num_masked]
        mask_tokens = decoded_tokens[:, -num_masked:]
        # print("mask_tokens.shape:", mask_tokens.shape)
        # pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss

        recon_loss = F.mse_loss(self.to_pixels(mask_tokens), masked_patches)
        return x,recon_loss

# mae = MAE(
#     image_size=(64,128,128),
#     patch_size=16,
#     encoder_dim=768,
#     mlp_dim=3072,
#     masking_ratio = 0.75,   # the paper recommended 75% masked patches
#     decoder_dim = 512,      # paper showed good results with just 512
#     decoder_depth = 6       # anywhere from 1 to 8
# )
#
# images = torch.randn(1,1,64,128,128)
# print(mae)
# Output,loss = mae(images)
#
# loss.backward()
# print("Output_shape:",Output.shape)
# print("Loss:",loss)
#
# # that's all!
# # do the above in a for loop many times with a lot of images and your vision transformer will learn
#
# # save your improved vision transformer
# torch.save(v.state_dict(), './trained-vit.pt')

# from thop import profile, clever_format
# net = MAE(
#     image_size=(64,128,128),
#     patch_size=16,
#     encoder_dim=768,
#     mlp_dim=3072,
#     masking_ratio = 0.75,   # the paper recommended 75% masked patches
#     decoder_dim = 512,      # paper showed good results with just 512
#     decoder_depth = 6       # anywhere from 1 to 8
# )
# flops, params = profile(net, inputs=(images,))
# macs, params = clever_format([flops, params], "%.3f") # 格式化输出
# print('params:',params) # 模型参数量

'''
MAE-base
MAE(
    image_size=(64,128,128),
    patch_size=16,
    encoder_dim=768,
    mlp_dim=3072,
    masking_ratio = 0.75,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6       # anywhere from 1 to 8
)

MAE-large
MAE(
    image_size=(64,128,128),
    patch_size=16,
    encoder_dim=1024,
    mlp_dim=4096,
    masking_ratio = 0.125,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6       # anywhere from 1 to 8
)
'''