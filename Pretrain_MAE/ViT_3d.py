import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
import math

# MIN_NUM_PATCHES = 16


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
        # self.to_latent = nn.Identity()
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, 512)
        # )
        # print(dim)

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        # x = self.to_latent(x)
        # x = self.mlp_head(x)
        return x


class ViT3D(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert all([each_dimension % patch_size ==
                    0 for each_dimension in image_size])
        num_patches = (image_size[0] // patch_size) * \
            (image_size[1] // patch_size)*(image_size[2] // patch_size)
        patch_dim = channels * patch_size ** 3
        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim)
        )
        new_patch_size = (4, 4, 4)
        self.conv3d_transpose = nn.ConvTranspose3d(dim, 16, kernel_size=new_patch_size, stride=new_patch_size)
        self.conv3d_transpose_1 = nn.ConvTranspose3d(
            in_channels=16, out_channels=1, kernel_size=new_patch_size, stride=new_patch_size
        )

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(
            img, 'b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)', p1=p, p2=p, p3=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        # print("X:", x.shape)
        # print("pos_embedding:", self.pos_embedding)
        x = self.dropout(x)
        # print("X:", x.shape)
        x = self.transformer(x, mask)
        # print("X:", x.shape)
        x = self.to_latent(x)
        x = self.mlp_head(x)
        x = x.transpose(1, 2)
        cuberoot = round(math.pow(x.size()[2], 1 / 3))
        x_shape = x.size()
        x = torch.reshape(x, [x_shape[0], x_shape[1], cuberoot, cuberoot, cuberoot])
        x = self.conv3d_transpose(x)
        x = self.conv3d_transpose_1(x)
        return x