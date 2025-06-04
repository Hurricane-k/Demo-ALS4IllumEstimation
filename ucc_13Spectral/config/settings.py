import torch
import numpy as np
import random
import os

DEVICE_TYPE = 'cuda:0'

def get_device():
    """ GPU or CPU
    
    """

    if DEVICE_TYPE == 'cpu':
        print('=============================\n')
        print('\n Running on device "cpu" \n')
        print('=============================')
        return torch.device('cpu')
    else:
        print('=============================\n')
        print(f'Running on device {DEVICE_TYPE}', '\n')
        print('=============================')
        return torch.device(DEVICE_TYPE)


DEVICE = get_device()


def reproduce(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

