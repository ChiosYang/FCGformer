import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.FourierGNN_Lite import FGN_Lite
from layers.GroupedQueryAttention import GroupedQueryAttention
from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder


class GQAEncoderLayer(nn.Module):
    """使用GQA的Encoder层"""
    
    def __init__(self, d_model, n_heads=8, n_kv_heads=2, d_ff=1024, dropout=0.1, activation='gelu'):
        super().__init__()
        
        # 使用GQA替代标准注意力
        self.attention = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 注意力块with残差连接
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # FFN块with残差连接
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        
        return x


class iTransformerGQA(nn.Module):
    """
    使用GQA的iTransformer
    优化点：
    1. 使用GQA减少KV参数60%
    2. 保持2层encoder以平衡性能
    3. d_model适度增加到384
    """
    
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        
        # 平衡的参数配置
        self.d_model = 384  # 介于256和512之间
        self.n_heads = 8    # 保持8个Query头
        self.n_kv_heads = 2 # 只用2个KV头（参数减少75%）
        self.d_ff = 1024    # 适中的FFN维度
        self.n_layers = 2   # 2层encoder平衡性能
        
        # 嵌入层
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, 
            self.d_model,
            configs.embed, 
            configs.freq,
            configs.dropout
        )
        
        # GQA Encoder层
        self.encoder_layers = nn.ModuleList([
            GQAEncoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
                d_ff=self.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for _ in range(self.n_layers)
        ])
        
        # 最终层归一化
        self.norm = nn.LayerNorm(self.d_model)
        
        # 投影层
        self.projector = nn.Linear(self.d_model, configs.pred_len, bias=True)
    
    def forward(self, x_enc, x_mark_enc):
        # 标准化
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        
        _, _, N = x_enc.shape
        
        # 嵌入
        x = self.enc_embedding(x_enc, None)
        
        # 通过GQA encoder层
        for layer in self.encoder_layers:
            x = layer(x)
        
        # 最终归一化
        x = self.norm(x)
        
        # 投影到预测维度
        dec_out = self.projector(x).permute(0, 2, 1)[:, :, :N]
        
        # 反标准化
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out[:, -self.pred_len:, :]


class Model(nn.Module):
    """
    Lite_v4_GQA - 使用现代GQA注意力机制的轻量化版本
    
    优化点：
    1. iTransformer使用GQA，参数减少60%
    2. 2层encoder平衡性能和效率
    3. 适度的d_model=384
    4. 改进的融合层with残差连接
    """
    
    def __init__(self, configs):
        super().__init__()
        
        # 轻量化FGN，但频率截断不要太激进
        self.fgnn = FGN_Lite(
            pre_length=configs.pred_len, 
            embed_size=configs.pred_len, 
            feature_size=96,
            seq_length=configs.seq_len, 
            hidden_size=64, 
            hard_thresholding_fraction=1,
            hidden_size_factor=1, 
            sparsity_threshold=0.001, 
            num_nodes=182, 
            dominance_freq=35,  # 适度的频率截断
            device=configs.gpu
        )
        
        # 使用GQA的iTransformer
        self.itransformer = iTransformerGQA(configs)
        
        # 改进的融合层with残差
        self.fusion_layer = nn.Sequential(
            nn.Linear(40, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, configs.pred_len)
        )
        
        # 可选的残差连接投影
        self.residual_proj = nn.Linear(configs.enc_in, configs.pred_len)
        
        self.dropout = nn.Dropout(p=configs.dropout)
    
    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, batch_location=None, epoch=0):
        """
        前向传播
        
        Args:
            x_enc: [batch_size, seq_len, features]
            x_mark_enc: [batch_size, seq_len, time_features]
        """
        # 保存输入用于残差连接
        residual = x_enc[:, -1, :].unsqueeze(1)  # 取最后一个时间步
        
        # 双分支处理
        fgnout, _, _, _, _, _ = self.fgnn(x_enc)
        itransformer_output = self.itransformer(x_enc, x_mark_enc)
        
        # 特征融合
        combined_output = torch.cat([fgnout, itransformer_output], dim=-1)
        combined_output = self.dropout(combined_output)
        
        # 融合层输出
        fused_output = self.fusion_layer(combined_output)
        
        # 添加残差连接
        residual_output = self.residual_proj(residual).transpose(1, 2)
        final_output = fused_output + 0.1 * residual_output  # 小权重的残差
        
        return final_output