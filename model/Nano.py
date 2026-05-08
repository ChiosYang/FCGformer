"""
Lite_v5_nano.py - 极致微型版本
最小化模型，适用于极端资源受限环境

主要特性：
1. 极简架构（< 100K参数）
2. 纯线性层设计
3. 无注意力机制
4. 固定点数运算支持
5. 单精度优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NanoFourierBlock(nn.Module):
    """
    极简频域处理块
    仅保留最基本的频域变换
    """
    
    def __init__(self, seq_len, pred_len, n_features, freq_cut=10):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.freq_cut = freq_cut  # 极低频率截断
        
        # 单层线性变换
        self.freq_proj = nn.Linear(freq_cut * 2, pred_len)
        
    def forward(self, x):
        """
        x: [B, L, N]
        """
        B, L, N = x.shape
        
        # 简单FFT
        x_freq = torch.fft.rfft(x.transpose(1, 2), dim=-1, norm='ortho')
        
        # 极端低通滤波 - 只保留前10个频率
        x_freq = x_freq[:, :, :self.freq_cut]
        
        # 展平实部虚部
        x_freq_flat = torch.cat([x_freq.real, x_freq.imag], dim=-1)  # [B, N, freq_cut*2]
        
        # 线性投影到预测长度
        output = self.freq_proj(x_freq_flat)  # [B, N, pred_len]
        
        return output.transpose(1, 2)  # [B, pred_len, N]


class NanoLinearBlock(nn.Module):
    """
    极简线性预测块
    使用最简单的线性映射
    """
    
    def __init__(self, seq_len, pred_len):
        super().__init__()
        # 直接线性映射
        self.temporal_proj = nn.Linear(seq_len, pred_len, bias=False)
        
        # 可选的小型MLP
        self.feature_mix = nn.Sequential(
            nn.Linear(pred_len, pred_len // 2),
            nn.ReLU(inplace=True),
            nn.Linear(pred_len // 2, pred_len)
        )
        
    def forward(self, x):
        """
        x: [B, L, N]
        """
        # 时间维度投影
        x = x.transpose(1, 2)  # [B, N, L]
        x = self.temporal_proj(x)  # [B, N, pred_len]
        
        # 特征混合
        x = x + self.feature_mix(x)
        
        return x.transpose(1, 2)  # [B, pred_len, N]


class TinyAttention(nn.Module):
    """
    超轻量注意力
    使用pooling降低序列长度后计算
    """
    
    def __init__(self, d_model, pool_size=4):
        super().__init__()
        self.d_model = d_model
        self.pool_size = pool_size
        
        # 极简的QKV投影
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        
        self.scale = d_model ** -0.5
        
    def forward(self, x):
        B, L, D = x.shape
        
        # 池化降低序列长度
        if L > self.pool_size:
            x_pooled = F.adaptive_avg_pool1d(
                x.transpose(1, 2), L // self.pool_size
            ).transpose(1, 2)
        else:
            x_pooled = x
            
        # QKV计算
        qkv = self.qkv(x_pooled).reshape(B, -1, 3, D).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 简化的注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = attn @ v
        out = self.out(out)
        
        # 上采样回原始长度
        if L > self.pool_size:
            out = F.interpolate(
                out.transpose(1, 2), size=L, mode='linear', align_corners=False
            ).transpose(1, 2)
            
        return out


class Model(nn.Module):
    """
    Lite_v5_nano - 纳米级模型
    
    特点：
    1. 参数量 < 100K
    2. 推理速度极快
    3. 内存占用极小（< 1MB）
    4. 支持INT8/FP16推理
    5. 可部署到MCU级别设备
    
    适用场景：
    - 微控制器（MCU）
    - 极低功耗设备
    - 实时系统
    - IoT终端
    """
    
    def __init__(self, configs):
        super().__init__()
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # 极简配置
        self.d_model = 32  # 极小的模型维度
        self.use_fourier = True  # 是否使用频域
        self.use_attention = False  # 默认不使用注意力（太重）
        
        # 输入投影（降维）
        if configs.enc_in != self.d_model:
            self.input_proj = nn.Linear(configs.enc_in, self.d_model, bias=False)
        else:
            self.input_proj = nn.Identity()
            
        # 核心组件（二选一或组合）
        if self.use_fourier:
            self.fourier_block = NanoFourierBlock(
                seq_len=configs.seq_len,
                pred_len=configs.pred_len,
                n_features=self.d_model,
                freq_cut=min(10, configs.seq_len // 2)  # 自适应频率截断
            )
            
        self.linear_block = NanoLinearBlock(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len
        )
        
        # 可选的微型注意力
        if self.use_attention:
            self.attention = TinyAttention(self.d_model, pool_size=4)
        
        # 输出投影（升维）
        if self.d_model != configs.enc_in:
            self.output_proj = nn.Linear(self.d_model, configs.enc_in, bias=False)
        else:
            self.output_proj = nn.Identity()
            
        # 残差权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        # 初始化
        self._init_weights()
        
        # 打印模型信息
        self._print_info()
        
    def _init_weights(self):
        """极简初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用较小的初始化范围
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def _print_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        model_size_mb = total_params * 4 / 1024 / 1024  # FP32
        
        print(f"\n{'='*50}")
        print(f"Lite_v5_nano - Ultra Tiny Model")
        print(f"{'='*50}")
        print(f"Model dimension: {self.d_model}")
        print(f"Use Fourier: {self.use_fourier}")
        print(f"Use Attention: {self.use_attention}")
        print(f"Total parameters: {total_params:,}")
        print(f"Model size: {model_size_mb:.3f} MB (FP32)")
        print(f"Model size: {model_size_mb/2:.3f} MB (FP16)")
        print(f"Model size: {model_size_mb/4:.3f} MB (INT8)")
        
        # 检查是否满足纳米级要求
        if total_params < 100000:
            print(f"✓ Nano-scale model (<100K params)")
        else:
            print(f"⚠ Warning: Model exceeds nano-scale (>100K params)")
        print(f"{'='*50}\n")
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, 
                batch_location=None, epoch=0):
        """
        极简前向传播
        """
        B, L, N = x_enc.shape
        
        # 输入投影
        x = self.input_proj(x_enc)  # [B, L, d_model]
        
        # 可选注意力
        if self.use_attention:
            x = x + self.attention(x) * 0.1  # 小权重
        
        # 并行处理两个分支
        outputs = []
        
        # 线性分支 - 使用原始输入以保持维度一致
        linear_out = self.linear_block(x_enc)  # 使用原始x_enc而不是投影后的x
        outputs.append(linear_out)
        
        # 频域分支
        if self.use_fourier:
            # 需要原始维度进行FFT
            fourier_out = self.fourier_block(x_enc)
            outputs.append(fourier_out)
        
        # 简单平均融合 - 现在两个分支输出维度应该一致了
        if len(outputs) > 1:
            output = sum(outputs) / len(outputs)
        else:
            output = outputs[0]
            
        # 输出投影回原始维度
        if output.shape[-1] != N:
            # 投影
            B_out, L_out, D_out = output.shape
            output = output.reshape(B_out, L_out, -1)
            output = self.output_proj(output)
            
        # 简单残差连接（使用最后的输入值）
        last_value = x_enc[:, -1:, :].expand(-1, self.pred_len, -1)
        output = output + self.residual_weight * last_value
        
        return output
    
    def count_parameters(self):
        """统计参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_memory_usage(self, batch_size=1):
        """估算内存使用"""
        # 前向传播内存估算
        input_size = batch_size * self.seq_len * self.enc_in * 4  # FP32
        output_size = batch_size * self.pred_len * self.enc_in * 4
        
        # 模型参数内存
        param_size = self.count_parameters() * 4
        
        # 中间激活内存（粗略估算）
        activation_size = batch_size * self.seq_len * self.d_model * 4 * 2
        
        total_mb = (input_size + output_size + param_size + activation_size) / 1024 / 1024
        
        return {
            'input_mb': input_size / 1024 / 1024,
            'output_mb': output_size / 1024 / 1024,
            'param_mb': param_size / 1024 / 1024,
            'activation_mb': activation_size / 1024 / 1024,
            'total_mb': total_mb
        }
    
    @torch.jit.export
    def jit_forward(self, x_enc):
        """
        TorchScript优化的前向传播
        用于部署
        """
        return self.forward(x_enc)[:, -self.pred_len:, :]
    
    def optimize_for_deployment(self):
        """
        部署优化
        包括量化、剪枝、融合等
        """
        self.eval()
        
        # 1. 融合操作
        optimized = torch.jit.script(self)
        
        # 2. 量化准备（INT8）
        optimized = torch.quantization.quantize_dynamic(
            optimized, 
            {nn.Linear}, 
            dtype=torch.qint8
        )
        
        return optimized