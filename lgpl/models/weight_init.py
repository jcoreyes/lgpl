import torch.nn.init as init

def xavier_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal(m.weight.data)
        init.xavier_normal(m.bias.data)
    if classname.find('Linear') != -1:
        init.xavier_uniform_(m.weight.data)

def kaiming_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data)
        init.kaiming_normal(m.bias.data)
    if classname.find('Linear') != -1:
        init.kaiming_uniform(m.weight.data)

def orthogonal_init(m):
    classname = m.__class__.__name__
    if classname.find('LSTM') != -1:
        init.orthogonal(m.weight_hh_l0)
        init.orthogonal(m.weight_ih_l0)