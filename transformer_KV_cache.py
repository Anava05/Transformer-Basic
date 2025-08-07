import torch
import torch.nn as nn
import math

class attention(nn.Module):
    def __init__(self,mask=None, dropout=None,dim=512):
        super(attention, self).__init__()
        self.mask = mask
        self.dropout = dropout
        self.softmax = nn.Softmax(dim=-1)
        self.dim = dim    

    def forward(self, q, k, v,n):
        scores = torch.matmul(q, k.transpose(-2, -1))
        if self.mask is not None:
            scores = scores.masked_fill(self.mask, -float('inf'))
        scores = scores / torch.sqrt(torch.tensor(self.dim, dtype=scores.dtype))
        #we have also extracted value of sequence length because if we want to include positional bias right here in the attention block itself then we must be able to handle it 
        p_attn = self.softmax(scores)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, v)

class normalize(nn.Module):
    def __init__(self,d_model):
        super(normalize, self).__init__()
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        return self.norm(x)


class MHA(nn.Module):
    def __init__(self,d_model,heads,Q_dim,dropout=None):
        super(MHA, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.q = nn.LazyLinear(Q_dim)
        self.k = nn.LazyLinear(Q_dim)
        self.v = nn.LazyLinear(Q_dim)
        self.dropout = nn.Dropout(dropout)
        if Q_dim % heads != 0:
            raise ValueError("Q_dim must be divisible by heads")
        self.head_dim = Q_dim // heads
        self.attn = attention(mask=None, dropout=self.dropout,dim = self.head_dim)
        self.out = nn.LazyLinear(d_model)
        self.kv_cache = None  # Initialize the KV cache

    def forward(self, embeddings, use_cache=False, past_kv=None):
        n = embeddings.size(1)
        q = self.q(embeddings)
        k = self.k(embeddings)
        v = self.v(embeddings)
        q = q.view(q.size(0), -1, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), -1, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), -1, self.heads, self.head_dim).transpose(1, 2)
        
        if use_cache:
            if past_kv is not None: #If we have past_kv , then we will add the new key,values to the cache
                k = torch.cat([past_kv[0], k], dim=2) #Concatenate along sequence length dimension
                v = torch.cat([past_kv[1], v], dim=2)
            
        attn_output = self.attn(q, k, v,n)
        attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.size(0), -1, self.heads * self.head_dim)
        attn_output = self.out(attn_output)
        
        #Update KV Cache
        if use_cache:
            self.kv_cache = (k, v) # This handles both cases when the KV_cache is None or if not None.
        return attn_output


class transformers(nn.Module):
    def __init__(self,d_model,heads,Q_dim,dropout=None):
        super(transformers, self).__init__()
        self.mha = MHA(d_model=d_model,heads=heads,Q_dim=Q_dim,dropout=dropout)
        self.norm1 = normalize(d_model)
        self.norm2 = normalize(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffnn = nn.Sequential(
            nn.LazyLinear(4*d_model),
            nn.ReLU(),
            nn.LazyLinear(d_model)
        )

    def forward(self, embeddings, use_cache=False, past_kv=None):
        x = self.mha(self.norm1(embeddings), use_cache, past_kv)
        x = x + self.dropout1(x)
        out = self.ffnn(self.norm2(x))
        x = x + self.dropout2(out)
        return x