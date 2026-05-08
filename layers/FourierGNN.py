import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.FITS import Model, Configs


class FGN(nn.Module):
    def __init__(self, pre_length, embed_size, device,
                 feature_size, seq_length, hidden_size, num_nodes, hard_thresholding_fraction=1,
                 hidden_size_factor=1, sparsity_threshold=0.01,dominance_freq=50):
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

        # FITS
        # self.fcongis = Configs()
        # self.fits = Model(self.fcongis)
        self.dominance_freq = dominance_freq


        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))

        ## 修改的
        # self.w1 = nn.Parameter(torch.eye(self.frequency_size))
        # self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        # self.w2 = nn.Parameter(torch.eye(self.frequency_size * self.hidden_size_factor))
        # self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        # self.w3 = nn.Parameter(torch.eye(self.frequency_size * self.hidden_size_factor))
        # self.b3 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))

        self.embeddings_10 = nn.Parameter(torch.randn(pre_length, 8))
        self.fc = nn.Sequential(
            nn.Linear(768, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        # self.to('cuda:0')

    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y

    # FourierGNN
    def fourierGC(self, x, B, N, L):
        # o1_real = torch.zeros([B, (N * L) // 2 + 1, self.frequency_size * self.hidden_size_factor],
        #                       device=x.device)
        # o1_imag = torch.zeros([B, (N * L) // 2 + 1, self.frequency_size * self.hidden_size_factor],
        #                       device=x.device)
        # o2_real = torch.zeros(x.shape, device=x.device)
        # o2_imag = torch.zeros(x.shape, device=x.device)
        #
        # o3_real = torch.zeros(x.shape, device=x.device)
        # o3_imag = torch.zeros(x.shape, device=x.device)

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

        # 1 layer
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)

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

        # 2 layer
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = x + y

        o3_real = F.relu(
            torch.einsum('bli,ii->bli', o2_real, self.w3[0]) - \
            torch.einsum('bli,ii->bli', o2_imag, self.w3[1]) + \
            self.b3[0]
        )

        o3_imag = F.relu(
            torch.einsum('bli,ii->bli', o2_imag, self.w3[0]) + \
            torch.einsum('bli,ii->bli', o2_real, self.w3[1]) + \
            self.b3[1]
        )

        # 3 layer
        z = torch.stack([o3_real, o3_imag], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold)
        z = z + x
        z = torch.view_as_complex(z)
        return z

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        # print(f'After permute x : {x.size()}')
        B, N, L = x.shape
        # B*N*L ==> B*NL
        # print(f'After shape x : {x.size()}')
        x = x.reshape(B, -1)

        # print(f'After reshape x : {x.size()}')
        # embedding B*NL ==> B*NL*D
        x = self.tokenEmb(x)
        # print(f'After tokenEmb x : {x.size()}')    #torch.Size([2048, 1920, 24])

        ######
        #x = self.fits(x)

        # FFT B*NL*D ==> B*NT/2*D
        x = torch.fft.rfft(x, dim=1, norm='ortho')
        # 保存频域输出第一层
        x_frequency1 = x
        #####添加的LPF
        x[:, self.dominance_freq:, :] = 0  # 应用LPF

        # 保存频域输出第2层
        x_frequency2 = x

        x = x.reshape(B, (N * L) // 2 + 1, self.frequency_size)
        # print(f'After reshape 2 x : {x.size()}')

        # 保存频域输出第2层
        x_frequency3 = x

        bias = x

        # FourierGNN
        x = self.fourierGC(x, B, N, L)
        # print(f'After fourier GC : {x.size()}')

        # 保存频域输出第4层
        x_frequency4 = x

        x = x + bias
        # print(f'x + bias : {x.size()}')

        x = x.reshape(B, (N * L) // 2 + 1, self.embed_size)
        # print(f'After reshape 3 x : {x.size()}')

        # 保存频域输出
        x_frequency5 = x

        # ifft
        x = torch.fft.irfft(x, n=N * L, dim=1, norm="ortho")
        # print(f'After fft irfft x : {x.size()}')

        x = x.reshape(B, N, L, self.embed_size)
        # print(f'增维度后的 x : {x.size()}')

        # B, N, D, L = x.shape
        # x = x.reshape(B, D, N * L)
        # x = x.permute(0, 1, 3, 2)  # B, N, D, L
        # print(f'增维度后的 x2 : {x.size()}')

        # projection
        x = torch.matmul(x, self.embeddings_10)
        # print(f'After matmul x : {x.size()}')

        x = x.reshape(B, N, -1)
        # print(f'After reshape 4 x : {x.size()}')
        x = self.fc(x)
        # print(f'最终的 x : {x.size()}')
        x = x.permute(0, 2, 1)
        # print(f'最终的permute之后 x : {x.size()}')
        return x,x_frequency1,x_frequency2,x_frequency3,x_frequency4,x_frequency5
