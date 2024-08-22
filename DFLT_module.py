
import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

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
    def __init__(self, dim, heads = 8, dim_head = 32, dropout = 0.):
        super().__init__()

        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class DFLT(nn.Module):
    """
        Args:
        image_size (int): Size of input images. Default: 224
        patch_size (tuple): The patch size used for the transformer (DFLT block). Default: (2,2)
        dim (int): Embedding dimenstion for each patch. Default: 256.
        depth (int): Number of transformer blocks. Default: 3.
        heads (int): Number of heads. Default: 8
        expansion (int): Expansion ratio. Default: 4
        channels (int): Input feature channel dimension. Default: 192
        use_distillation (bool): A boolean to indicate if distillation will be used or not. Default: True (for HDKD).
        dim_head (int): Dimension per head. Default: 32
        dropout (int): Dropout rate. Default 0

    """
    def __init__(self, image_size, patch_size, dim, depth, heads, expansion, channels, use_distillation=True, dim_head = 32, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.use_distillation = use_distillation
        mlp_dim = dim * expansion
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        if self.use_distillation:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        if self.use_distillation:
            self.distill_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_drop = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        if self.use_distillation:
            distill_token = repeat(self.distill_token, '1 1 d -> b 1 d', b = b)
            x = torch.cat((cls_tokens,distill_token, x), dim=1)

        else:
            x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.pos_drop(x)

        x = self.transformer(x)

        x = self.to_latent(x)
        if self.use_distillation:
            return x[:,0], x[:,1]
        else:
            return x[:,0]
