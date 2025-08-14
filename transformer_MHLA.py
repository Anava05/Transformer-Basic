import torch
import torch.nn as nn
import math

class attention(nn.Module):
    def __init__(self, dropout_p=0.1):
        super(attention, self).__init__()
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        head_dim = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = self.softmax(scores)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, v)


class normalize(nn.Module):
    def __init__(self, d_model):
        super(normalize, self).__init__()
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        return self.norm(x)

class StatefulLatentCacheAttention(nn.Module):
    
    def __init__(self, d_model, heads, dropout_p=0.1):
        super(StatefulLatentCacheAttention, self).__init__()
        self.d_model = d_model
        self.heads = heads
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.head_dim = d_model // heads
        self.q_proj = nn.Linear(d_model, d_model)        
        self.latent_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn = attention(dropout_p=dropout_p)

    def forward(self, current_token_embedding, latent_cache=None):
        batch_size = current_token_embedding.size(0)
        q = self.q_proj(current_token_embedding)
        new_latent = self.latent_proj(current_token_embedding)
        if latent_cache is None:
            updated_cache = new_latent
        else:
            updated_cache = torch.cat([latent_cache, new_latent], dim=1)
        k = updated_cache
        v = updated_cache
        q = q.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)

        attn_output = self.attn(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(attn_output), updated_cache


class RecurrentLatentTransformerBlock(nn.Module):
    def __init__(self, d_model, heads, ff_dim_multiplier=4, dropout_p=0.1):
        super(RecurrentLatentTransformerBlock, self).__init__()        
        self.attention = StatefulLatentCacheAttention(d_model, heads, dropout_p)
        self.norm1 = normalize(d_model)
        self.norm2 = normalize(d_model)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)        
        self.ffnn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_dim_multiplier),
            nn.GELU(),
            nn.Linear(d_model * ff_dim_multiplier, d_model)
        )

    def forward(self, input_sequence, initial_cache=None):
        outputs = []
        current_cache = initial_cache
        normed_input = self.norm1(input_sequence)
        for t in range(input_sequence.size(1)):
            current_token = input_sequence[:, t:t+1, :]
            normed_token = normed_input[:, t:t+1, :]
            attn_output, current_cache = self.attention(normed_token, current_cache)
            x = current_token + self.dropout1(attn_output)
            ffn_output = self.ffnn(self.norm2(x))
            final_token_output = x + self.dropout2(