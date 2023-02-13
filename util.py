import numpy as np
import torch, argparse
import matplotlib.pyplot as plt

def plot_loss(train_loss,mode):
    plt.plot(train_loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('%s Loss' %mode)
    plt.legend()
    plt.show()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2tuple(s):
    return tuple(s.split('_'))

def s2ituple(s):
    return tuple(int(_s) for _s in s.split('_'))
