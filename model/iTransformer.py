import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer, DSAttention
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  # 输入序列长度
        self.pred_len = configs.pred_len  # 预测序列长度
        self.output_attention = configs.output_attention  # 是否输出注意力权重
        self.use_norm = configs.use_norm  # 是否使用归一化
        # 使用 DataEmbedding_inverted 层将输入数据嵌入到一个高维空间中。参数包括序列长度、模型维度、嵌入方式、频率特征以及 dropout 概率
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # 从 configs 中获取分类策略，可能用于控制模型输出的处理方式
        self.class_strategy = configs.class_strategy



        # Encoder-only architecture / Encoder-only 架构
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # 注意力层，里面包含了一个全局注意力
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)  # e_layer参数作用在这里，决定这个编码器有多少层
            ],
            # 层归一化
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # 一个线性层，将编码器的输出投影到预测长度维度上
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    # 执行预测任务
    # def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    #     # 如果配置文件中指定了标准化
    #     if self.use_norm:
    #         # Normalization from Non-stationary Transformer
    #         means = x_enc.mean(1, keepdim=True).detach()
    #         x_enc = x_enc - means
    #         stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    #         # x_enc /= stdev
    #         x_enc = x_enc / stdev
    #
    #     _, _, N = x_enc.shape # B L N
    #     # B: batch_size;    E: d_model;
    #     # L: seq_len;       S: pred_len;
    #     # N: number of variate (tokens), can also includes covariates
    #
    #     # Embedding
    #     # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
    #     enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
    #
    #     # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
    #     # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
    #     enc_out, attns = self.encoder(enc_out, attn_mask=None)
    #
    #     # B N E -> B N S -> B S N
    #     dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates
    #
    #     if self.use_norm:
    #         # De-Normalization from Non-stationary Transformer
    #         dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
    #         dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
    #
    #     return dec_out
    #
    #
    # def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    #     dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
    #     return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    def forecast(self, x_enc, x_mark_enc):
        # 如果配置文件中指定了标准化,标准化输入的数据
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        # print('x_enc :', x_enc.size())

        _, _, N = x_enc.shape  # B L N
        # print('Enc out0 :', x_enc.size())  ## (Batch_Size,96,20)
        # Embedding without x_mark_enc
        enc_out = self.enc_embedding(x_enc, None)  # 仅传递 x_enc
        # print('Enc out :', enc_out.size())

        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # print('Enc out 2 :', enc_out.size())

        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc):
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
