import numpy as np
import torch.nn as nn
from lgpl.utils.torch_utils import from_numpy
import torch

class Parameter(nn.Module):
    def __init__(self, input_dim, output_dim, init):
        super(Parameter, self).__init__()
        self.output_dim = output_dim
        self.init = init
        self.param_init = from_numpy(np.zeros((1, output_dim)) + init).float()
        #TODO: fix this nn.Parameter(self.param_init) 
        self.params_var = nn.Parameter(self.param_init)

    def forward(self, x):
        batch_size = x.size()[0]
        return self.params_var.repeat(batch_size, 1) #self.output_dim)

