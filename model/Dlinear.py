import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.dlinear import Model as DMdel



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.dlinear = DMdel(configs)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_location):
        dout = self.dlinear(x_enc)

        return dout

