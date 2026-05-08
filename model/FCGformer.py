import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.FourierGNN import FGN
from utils.tools import load_adj
from model.iTransformer import Model as iTransformerModel  # 确保正确导入 iTransformer 模型



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.fgnn = FGN(pre_length=configs.pred_len, embed_size=configs.pred_len, feature_size=96,
                        seq_length=configs.seq_len, hidden_size=64, hard_thresholding_fraction=1,
                        hidden_size_factor=1, sparsity_threshold=0.001, num_nodes=182, dominance_freq=configs.dominance_freq,device=configs.gpu)

        # 实例化 iTransformer
        self.itransformer = iTransformerModel(configs)
        # self.transformer = TransformerModel(configs)

        # 添加 Dropout 层
        self.dropout = nn.Dropout(p=configs.dropout)

        # 用于融合 WaveNet 和 iTransformer 的输出
        # self.fusion_layer = nn.Linear(624, 20)
        self.fusion_layer = nn.Sequential(
            nn.Linear(40, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, configs.pred_len)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_location, epoch=0):
        fgnout, x_frequency1, x_frequency2, x_frequency3, x_frequency4, x_frequency5 = self.fgnn(x_enc)
        itransformer_output = self.itransformer(x_enc, x_mark_enc)
        # transformer_output = self.transformer(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # 融合两个模块的输出（例如通过拼接）
        # combined_output = torch.cat([fgnout, itransformer_output], dim=-
        combined_output = torch.cat([fgnout, itransformer_output], dim=-1)

        # 通过融合层进一步处理
        fused_output = self.fusion_layer(combined_output)

        # self.save_frequency_output(x_frequency, epoch)
        # return fused_output, x_frequency1, x_frequency2, x_frequency3, x_frequency4, x_frequency5
        return fused_output

    # def get_weights(self):
    #     fgnn_weights = dict(self.fgnn.named_parameters())
    #     itransformer_weights = dict(self.itransformer.named_parameters())
    #     fusion_weights = dict(self.fusion_layer.named_parameters())
    #
    #     return {
    #         'FGNN_weights': fgnn_weights,
    #         'iTransformer_weights': itransformer_weights,
    #         'Fusion_weights': fusion_weights
    #     }
    def save_frequency_output(self, x_frequency, epoch):
        # 创建文件夹来存储频域输出
        output_dir = "frequency_outputs"
        os.makedirs(output_dir, exist_ok=True)

        # 将频域输出转换为 numpy 并保存
        freq_output = x_frequency.detach().cpu().numpy()
        file_path = os.path.join(output_dir, f"frequency_output_epoch_{epoch}.npy")
        np.save(file_path, freq_output)
        # print(f"Frequency output for epoch {epoch} saved to {file_path}")
