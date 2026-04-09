import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class OSTrackBackbone(nn.Module):
    """ One-Stream Vision Transformer """
    def __init__(self, template_size=128, search_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.template_size = template_size
        self.search_size = search_size
        self.patch_size = patch_size
        
        self.patch_embed_z = PatchEmbed(img_size=template_size, patch_size=patch_size, embed_dim=embed_dim)
        self.patch_embed_x = PatchEmbed(img_size=search_size, patch_size=patch_size, embed_dim=embed_dim)

        num_patches_z = self.patch_embed_z.num_patches
        num_patches_x = self.patch_embed_x.num_patches

        # Absolute positional embeddings
        self.pos_embed_z = nn.Parameter(torch.zeros(1, num_patches_z, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, num_patches_x, embed_dim))

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, qkv_bias=True)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize pos embeddings
        nn.init.trunc_normal_(self.pos_embed_z, std=.02)
        nn.init.trunc_normal_(self.pos_embed_x, std=.02)

    def forward(self, z, x):
        B = x.shape[0]
        
        # Patch embedding
        z_emb = self.patch_embed_z(z) + self.pos_embed_z
        x_emb = self.patch_embed_x(x) + self.pos_embed_x
        
        # Concatenate tokens (One-Stream Joint Processing)
        tokens = torch.cat([z_emb, x_emb], dim=1)
        
        # Transformer blocks
        for blk in self.blocks:
            tokens = blk(tokens)
            
        tokens = self.norm(tokens)
        
        # Extract search region tokens
        len_z = z_emb.shape[1]
        x_out = tokens[:, len_z:]
        
        # Reshape to 2D feature map
        H_x = self.search_size // self.patch_size
        W_x = self.search_size // self.patch_size
        x_out = x_out.transpose(1, 2).reshape(B, -1, H_x, W_x)
        
        return x_out

class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CenterPredictor(nn.Module):
    """ Prediction head for Score, Offset, and Size """
    def __init__(self, in_channels=768, hidden_channels=256):
        super().__init__()
        
        # Score head (Class)
        self.score_head = nn.Sequential(
            ConvBNRelu(in_channels, hidden_channels, 3, padding=1),
            ConvBNRelu(hidden_channels, hidden_channels, 3, padding=1),
            nn.Conv2d(hidden_channels, 1, 3, padding=1)
        )
        
        # Offset head (Sub-pixel exact center)
        self.offset_head = nn.Sequential(
            ConvBNRelu(in_channels, hidden_channels, 3, padding=1),
            ConvBNRelu(hidden_channels, hidden_channels, 3, padding=1),
            nn.Conv2d(hidden_channels, 2, 3, padding=1)
        )
        
        # Size head (Width, Height)
        self.size_head = nn.Sequential(
            ConvBNRelu(in_channels, hidden_channels, 3, padding=1),
            ConvBNRelu(hidden_channels, hidden_channels, 3, padding=1),
            nn.Conv2d(hidden_channels, 2, 3, padding=1)
        )

    def forward(self, x):
        score_map = self.score_head(x)
        offset_map = self.offset_head(x)
        size_map = self.size_head(x)
        return score_map, offset_map, size_map

class OSTrack(nn.Module):
    """ Complete OSTrack Model Architecture """
    def __init__(self, model_type='base'):
        super().__init__()
        
        # base: 12 layers, 768 dim. small: 8 layers, 384 dim.
        if model_type == 'base':
            embed_dim = 768
            depth = 12
            num_heads = 12
        elif model_type == 'small':
            embed_dim = 384
            depth = 8
            num_heads = 6
        else:
            raise ValueError("Unsupported model type")

        self.backbone = OSTrackBackbone(
            template_size=128, search_size=256, patch_size=16, 
            embed_dim=embed_dim, depth=depth, num_heads=num_heads
        )
        self.head = CenterPredictor(in_channels=embed_dim)

    def forward(self, template, search):
        # 1. Extract joint features
        features = self.backbone(template, search)
        
        # 2. Predict center score, offset, and size
        score_map, offset_map, size_map = self.head(features)
        
        # Normalize score
        score_map = torch.sigmoid(score_map)
        
        return score_map, offset_map, size_map
