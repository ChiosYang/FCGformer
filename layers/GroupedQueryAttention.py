import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) - 现代轻量化注意力机制
    
    将Key和Value头分组共享，Query保持原有头数
    例如：8个Query头，2个KV头，每个KV头被4个Query头共享
    参数量减少 (n_heads - n_kv_heads) / n_heads
    """
    
    def __init__(self, d_model, n_heads=8, n_kv_heads=2, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # 每个KV头重复次数
        self.d_k = d_model // n_heads
        
        # Query投影保持原有头数
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        
        # Key和Value投影减少头数
        self.w_k = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        
        # 输出投影
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # 可选：使用RoPE（旋转位置编码）
        self.use_rope = True
        if self.use_rope:
            self.rope = RotaryPositionalEncoding(self.d_k)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算Q, K, V
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = self.w_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = self.w_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        
        # 转置为 [batch, n_heads, seq_len, d_k]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 应用RoPE
        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)
        
        # 重复KV头以匹配Q头数量
        k = self.repeat_kv(k, self.n_rep)
        v = self.repeat_kv(v, self.n_rep)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        context = torch.matmul(attn_weights, v)
        
        # 重塑并应用输出投影
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(context)
        
        return output
    
    @staticmethod
    def repeat_kv(x, n_rep):
        """重复KV头以匹配Query头数量"""
        if n_rep == 1:
            return x
        batch_size, n_kv_heads, seq_len, d_k = x.shape
        x = x.unsqueeze(2).expand(batch_size, n_kv_heads, n_rep, seq_len, d_k)
        return x.reshape(batch_size, n_kv_heads * n_rep, seq_len, d_k)


class RotaryPositionalEncoding(nn.Module):
    """
    旋转位置编码 (RoPE) - 更高效的位置编码方式
    相比传统位置编码，参数量为0，计算量更小
    """
    
    def __init__(self, dim, max_seq_len=5000, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算位置编码
        self._build_cache()
    
    def _build_cache(self):
        seq_len = self.max_seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # 构建旋转矩阵
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, n_heads, seq_len, d_k]
        """
        seq_len = x.shape[2]
        
        # 应用旋转
        cos = self.cos_cached[:, :, :seq_len, :].to(x.device)
        sin = self.sin_cached[:, :, :seq_len, :].to(x.device)
        
        # 分割实部和虚部
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        
        # 应用旋转变换
        y1 = x1 * cos[..., ::2] - x2 * sin[..., ::2]
        y2 = x1 * sin[..., 1::2] + x2 * cos[..., 1::2]
        
        # 重新组合
        y = torch.stack([y1, y2], dim=-1).flatten(-2)
        
        return y


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA) - GQA的极端版本
    所有Query头共享同一组KV（n_kv_heads=1）
    参数量减少最多，但可能性能略有下降
    """
    
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        # MQA是GQA的特例，n_kv_heads=1
        self.gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads=1, dropout=dropout)
    
    def forward(self, x, mask=None):
        return self.gqa(x, mask)