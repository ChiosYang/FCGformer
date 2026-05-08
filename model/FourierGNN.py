import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.FourierGNN import FGN
from utils.tools import load_adj

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.fgnn = FGN(pre_length=configs.pred_len, embed_size=configs.pred_len, feature_size=96,
                        seq_length=configs.seq_len, hidden_size=64, hard_thresholding_fraction=1,
                        hidden_size_factor=1, sparsity_threshold=0.001, num_nodes=182, device=configs.gpu)


    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, batch_location=None):

        fgnout, x_frequency1, x_frequency2, x_frequency3, x_frequency4, x_frequency5 = self.fgnn(x_enc)
        return fgnout
