import torch.nn as nn

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = False

    def reset(self, shape):
        pass