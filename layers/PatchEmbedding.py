import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """
    PatchTST的Patch嵌入层
    将时间序列分割成不重叠的patches，并进行嵌入
    
    支持两种模式：
    1. Channel-independent: 每个变量独立处理
    2. Channel-mixed: 所有变量共享patch（不推荐，会导致维度爆炸）
    """
    
    def __init__(self, seq_len, patch_len, d_model, dropout=0.1, 
                 channel_independent=True, norm_type='batch'):
        super().__init__()
        
        assert seq_len % patch_len == 0, f"seq_len {seq_len} must be divisible by patch_len {patch_len}"
        
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.d_model = d_model
        self.channel_independent = channel_independent
        
        # Patch嵌入：将每个patch映射到d_model维
        self.patch_embedding = nn.Linear(patch_len, d_model)
        
        # 位置编码
        self.position_embedding = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        
        # 归一化层
        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(d_model)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.patch_embedding.weight)
        nn.init.zeros_(self.patch_embedding.bias)
        nn.init.normal_(self.position_embedding, std=0.02)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, n_vars] - 输入时间序列
        
        Returns:
            如果channel_independent=True:
                [batch_size, n_vars, n_patches, d_model]
            如果channel_independent=False:
                [batch_size, n_patches, d_model]
        """
        B, L, N = x.shape
        
        if self.channel_independent:
            # Channel-independent处理
            # 重塑为patches: [B, n_patches, patch_len, N]
            x = x.reshape(B, self.n_patches, self.patch_len, N)
            
            # 转置以便每个变量独立处理: [B, N, n_patches, patch_len]
            x = x.permute(0, 3, 1, 2)
            
            # 重塑为: [B*N, n_patches, patch_len]
            x = x.reshape(B * N, self.n_patches, self.patch_len)
            
            # Patch嵌入: [B*N, n_patches, d_model]
            x = self.patch_embedding(x)
            
            # 添加位置编码
            x = x + self.position_embedding
            
            # 归一化
            if isinstance(self.norm, nn.BatchNorm1d):
                # BatchNorm需要 [B*N, d_model, n_patches]
                x = x.transpose(1, 2)
                x = self.norm(x)
                x = x.transpose(1, 2)
            else:
                x = self.norm(x)
            
            # Dropout
            x = self.dropout(x)
            
            # 恢复维度: [B, N, n_patches, d_model]
            x = x.reshape(B, N, self.n_patches, self.d_model)
            
        else:
            # Channel-mixed处理（不推荐）
            # 重塑为patches: [B, n_patches, patch_len*N]
            x = x.reshape(B, self.n_patches, self.patch_len * N)
            
            # 需要调整patch_embedding的输入维度
            if self.patch_embedding.in_features != self.patch_len * N:
                self.patch_embedding = nn.Linear(self.patch_len * N, self.d_model).to(x.device)
            
            # Patch嵌入: [B, n_patches, d_model]
            x = self.patch_embedding(x)
            
            # 添加位置编码
            x = x + self.position_embedding
            
            # 归一化和Dropout
            x = self.norm(x)
            x = self.dropout(x)
        
        return x


class PatchAggregation(nn.Module):
    """
    Patch聚合层
    将patch序列聚合为固定长度的表示
    """
    
    def __init__(self, n_patches, d_model, pred_len, aggregation='mean'):
        super().__init__()
        
        self.n_patches = n_patches
        self.d_model = d_model
        self.pred_len = pred_len
        self.aggregation = aggregation
        
        if aggregation == 'linear':
            # 线性投影聚合
            self.projector = nn.Linear(n_patches * d_model, pred_len)
        elif aggregation == 'attention':
            # 注意力聚合
            self.attention_weights = nn.Parameter(torch.randn(1, n_patches))
            self.projector = nn.Linear(d_model, pred_len)
        else:
            # mean, last, max等简单聚合
            self.projector = nn.Linear(d_model, pred_len)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch_size, n_vars, n_patches, d_model] 或 [batch_size*n_vars, n_patches, d_model]
        
        Returns:
            [batch_size, pred_len, n_vars]
        """
        
        # 处理不同的输入维度
        if x.dim() == 4:
            B, N, P, D = x.shape
            # 展平处理
            x = x.reshape(B * N, P, D)
            need_reshape = True
        else:
            need_reshape = False
            BN, P, D = x.shape
        
        # 根据聚合方式处理
        if self.aggregation == 'mean':
            x = x.mean(dim=1)  # [BN, D]
        elif self.aggregation == 'last':
            x = x[:, -1, :]  # [BN, D]
        elif self.aggregation == 'max':
            x = x.max(dim=1)[0]  # [BN, D]
        elif self.aggregation == 'linear':
            x = x.reshape(x.size(0), -1)  # [BN, P*D]
        elif self.aggregation == 'attention':
            weights = F.softmax(self.attention_weights, dim=1)
            x = torch.sum(x * weights.unsqueeze(-1), dim=1)  # [BN, D]
        
        # 投影到预测长度
        output = self.projector(x)  # [BN, pred_len]
        
        # 恢复维度
        if need_reshape:
            output = output.reshape(B, N, self.pred_len)
            output = output.permute(0, 2, 1)  # [B, pred_len, N]
        
        return output


class PatchTSTBackbone(nn.Module):
    """
    PatchTST的完整backbone
    包含patch嵌入、Transformer处理和patch聚合
    """
    
    def __init__(self, configs):
        super().__init__()
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.d_model = getattr(configs, 'd_model', 512)
        self.n_heads = getattr(configs, 'n_heads', 8)
        self.e_layers = getattr(configs, 'e_layers', 3)
        self.d_ff = getattr(configs, 'd_ff', 2048)
        self.dropout = getattr(configs, 'dropout', 0.1)
        
        # Patch嵌入
        self.patch_embedding = PatchEmbedding(
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            d_model=self.d_model,
            dropout=self.dropout,
            channel_independent=True
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.e_layers)
        
        # Patch聚合
        self.patch_aggregation = PatchAggregation(
            n_patches=self.seq_len // self.patch_len,
            d_model=self.d_model,
            pred_len=self.pred_len,
            aggregation='mean'
        )
    
    def forward(self, x, x_mark=None):
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, n_vars]
            x_mark: [batch_size, seq_len, time_features] (可选)
        
        Returns:
            [batch_size, pred_len, n_vars]
        """
        B, L, N = x.shape
        
        # Patch嵌入: [B, N, n_patches, d_model]
        x = self.patch_embedding(x)
        
        # 重塑为Transformer输入: [B*N, n_patches, d_model]
        x = x.reshape(B * N, -1, self.d_model)
        
        # Transformer处理
        x = self.transformer(x)
        
        # 恢复维度: [B, N, n_patches, d_model]
        x = x.reshape(B, N, -1, self.d_model)
        
        # Patch聚合并投影: [B, pred_len, N]
        output = self.patch_aggregation(x)
        
        return output