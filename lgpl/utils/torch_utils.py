import torch.autograd
import torch
import numpy as np

"""
GPU wrappers
"""

_use_gpu = torch.cuda.is_available()


def set_gpu_mode(mode):
    global _use_gpu
    _use_gpu = mode


def gpu_enabled():
    return _use_gpu


# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    if _use_gpu:
        return torch.cuda.FloatTensor(*args, **kwargs)
    else:
        # noinspection PyArgumentList
        return torch.FloatTensor(*args, **kwargs)

# noinspection PyPep8Naming
def DoubleTensor(*args, **kwargs):
    if _use_gpu:
        return torch.cuda.DoubleTensor(*args, **kwargs)
    else:
        # noinspection PyArgumentList
        return torch.DoubleTensor(*args, **kwargs)

def LongTensor(*args, **kwargs):
    if _use_gpu:
        return torch.cuda.LongTensor(*args, **kwargs)
    else:
        # noinspection PyArgumentList
        return torch.LongTensor(*args, **kwargs)

def Variable(*args, **kwargs):
    var = torch.autograd.Variable(*args, **kwargs)
    if _use_gpu and not var.is_cuda:
        var = var.cuda()
    return var


def from_numpy(*args, **kwargs):
    if _use_gpu:
        return torch.from_numpy(*args, **kwargs).cuda()
    else:
        return torch.from_numpy(*args, **kwargs)


def get_numpy(tensor):
    if isinstance(tensor, torch.autograd.Variable):
        if tensor.is_cuda:
            return tensor.data.cpu().numpy()
        return tensor.data.numpy()
    if _use_gpu:
        return tensor.cpu().numpy()
    return tensor.numpy()


def np_to_var(np_array):
    return Variable(from_numpy(np_array).float())