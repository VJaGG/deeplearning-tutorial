'''
@File    : include_torch.py
@Modify Time     @Author    @Version    @Desciption
------------     -------    --------    -----------
2021/8/5 15:17   WuZhiqiang     1.0        None 
'''
import os
import time
import yaml
from datetime import datetime
from timeit import default_timer as timer

import random
import numpy as np
import pandas as pd

import cv2
import albumentations as A
from sklearn.model_selection import StratifiedKFold

def seed_py(seed):
    random.seed(seed)
    np.random.seed(seed)
    return seed