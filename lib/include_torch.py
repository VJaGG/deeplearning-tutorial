'''
@File    : include_torch.py
@Modify Time     @Author    @Version    @Desciption
------------     -------    --------    -----------
2021/8/5 15:12   WuZhiqiang     1.0        None 
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data.sampler import *


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return seed