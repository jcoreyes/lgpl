import torch.nn as nn
import torch.nn.functional as F
from lgpl.models.weight_init import xavier_init

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(32, 32),
                 hidden_act=nn.ReLU, final_act=None, dropout=False, batchnorm=False,
                 batchnorm_input=False):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        mlp = nn.Sequential()
        prev_size = input_dim

        if batchnorm_input:
            mlp.add_module(name='batchnorm input', module=nn.BatchNorm1d(prev_size))

        for i, hidden_size in enumerate(hidden_sizes):
            mlp.add_module(name='linear %d' % i, module=nn.Linear(prev_size, hidden_size))
            if batchnorm:
                mlp.add_module(name='batchnorm %d' % i, module=nn.BatchNorm1d(hidden_size))
            mlp.add_module(name='relu %d' % i, module=hidden_act())
            if dropout:
                mlp.add_module(name='dropout %d ' %i, module=nn.Dropout())
            prev_size = hidden_size

        mlp.add_module(name='finallayer', module=nn.Linear(hidden_sizes[-1], output_dim))

        if final_act is not None:
            mlp.add_module(name='finalact', module=final_act())

        self.network = mlp

        self.apply(xavier_init)

    def forward(self, x):
        return self.network(x)