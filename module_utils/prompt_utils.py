import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads=8, qkv_bias=False, scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        head_dim = embed_dim // n_heads
        self.scale = scale or head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=DEVICE)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=DEVICE)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias, device=DEVICE)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(embed_dim, embed_dim, device=DEVICE)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value):
        B_q, N_q, C = query.shape
        B_k, N_k, _ = key.shape

        q = self.q_proj(query).view(B_q, N_q, self.n_heads, C // self.n_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).view(B_k, N_k, self.n_heads, C // self.n_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).view(B_k, N_k, self.n_heads, C // self.n_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().view(B_q, N_q, C)
        return self.proj_drop(self.out_proj(x))


class PromptGeneratorLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.):
        super().__init__()
        self.cross_attn = MultiHeadAttention(d_model, nhead, proj_drop=dropout)
        self.norm1 = nn.LayerNorm(d_model, device=DEVICE)
        self.norm2 = nn.LayerNorm(d_model, device=DEVICE)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4, device=DEVICE),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model, device=DEVICE),
        )

    def forward(self, x, visual):
        q = self.norm1(x)
        x = x + self.cross_attn(q, visual, visual)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class VideoSpecificPrompt(nn.Module):
    def __init__(self, layers=2, embed_dim=512, alpha=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, device=DEVICE)
        self.decoder = nn.ModuleList(
            [PromptGeneratorLayer(embed_dim, embed_dim // 64) for _ in range(layers)]
        )
        self.alpha = nn.Parameter(torch.full((embed_dim,), alpha))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, text, visual):
        visual = self.norm(visual)
        for layer in self.decoder:
            text = layer(text, visual)
        return self.alpha * text
