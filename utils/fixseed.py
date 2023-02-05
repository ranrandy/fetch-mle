import numpy as np
# import tensorflow as tf
import torch


def fixseed(seed):
    np.random.seed(seed)
    # tf.keras.utils.set_random_seed(seed)
    torch.manual_seed(seed)
    return