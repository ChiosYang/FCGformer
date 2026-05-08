import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FGN_Lite(nn.Module):
    def __init__(self, pre_length, embed_size, device,
                 feature_size, seq_length, hidden_size, num_nodes, hard_thresholding_fraction=1,
                 hidden_size_factor=1, sparsity_threshold=0.01, dominance_freq=20):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = 1
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02
        
        # 更激进的频率截断 - 从50降到20
        self.dominance_freq = dominance_freq

        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        
        # 轻量化：从3层减少到2层傅里叶变换
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        # 删除了 w3, b3

        self.embeddings_10 = nn.Parameter(torch.randn(pre_length, 8))
        
        # 轻量化的fc层：大幅减少中间维度
        self.fc = nn.Sequential(
            nn.Linear(768, 32),  # 从768->64改为768->32
            nn.LeakyReLU(),
            nn.Linear(32, self.pre_length)  # 直接映射到输出，删除中间层
        )
        self.to(device)

    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y

    def fourierGC_lite(self, x, B, N, L):
        # 第一层傅里叶变换
        o1_real = F.relu(
            torch.einsum('bli,ii->bli', x.real, self.w1[0]) - \
            torch.einsum('bli,ii->bli', x.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('bli,ii->bli', x.imag, self.w1[0]) + \
            torch.einsum('bli,ii->bli', x.real, self.w1[1]) + \
            self.b1[1]
        )

        # 稀疏化
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)

        # 第二层傅里叶变换（原来的第二层，现在作为最后一层）
        o2_real = F.relu(
            torch.einsum('bli,ii->bli', o1_real, self.w2[0]) - \
            torch.einsum('bli,ii->bli', o1_imag, self.w2[1]) + \
            self.b2[0]
        )

        o2_imag = F.relu(
            torch.einsum('bli,ii->bli', o1_imag, self.w2[0]) + \
            torch.einsum('bli,ii->bli', o1_real, self.w2[1]) + \
            self.b2[1]
        )

        # 最终输出
        z = torch.stack([o2_real, o2_imag], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold)
        z = z + y  # 残差连接
        z = torch.view_as_complex(z)
        return z

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        B, N, L = x.shape
        x = x.reshape(B, -1)

        # embedding
        x = self.tokenEmb(x)

        # FFT
        x = torch.fft.rfft(x, dim=1, norm='ortho')
        
        # 保存频域输出（为了兼容性）
        x_frequency1 = x
        
        # 更激进的LPF - 只保留前20个频率分量
        x[:, self.dominance_freq:, :] = 0
        
        x_frequency2 = x
        
        x = x.reshape(B, (N * L) // 2 + 1, self.frequency_size)
        x_frequency3 = x
        
        bias = x

        # 使用轻量化的傅里叶GC
        x = self.fourierGC_lite(x, B, N, L)
        x_frequency4 = x
        
        x = x + bias
        
        x = x.reshape(B, (N * L) // 2 + 1, self.embed_size)
        x_frequency5 = x

        # ifft
        x = torch.fft.irfft(x, n=N * L, dim=1, norm="ortho")
        
        x = x.reshape(B, N, L, self.embed_size)

        # projection
        x = torch.matmul(x, self.embeddings_10)
        
        x = x.reshape(B, N, -1)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        
        return x, x_frequency1, x_frequency2, x_frequency3, x_frequency4, x_frequency5