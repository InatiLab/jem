# Utility functions
import numpy as np


def select(data, mask=None):

    if isinstance(data, list):
        h = []
        for dsub in data:
            h.append(select(dsub, mask))
        return np.hstack(h)
    else:
        if mask is not None:
            if len(data.shape) == 3:
                return data.reshape(-1, 1)[mask.flatten(), :]
            else:
                return data.reshape(-1, data.shape[-1])[mask.flatten(), :]
        else:
            if len(data.shape) == 3:
                return data.reshape(-1, 1)
            else:
                return data.reshape(-1, data.shape[-1])
