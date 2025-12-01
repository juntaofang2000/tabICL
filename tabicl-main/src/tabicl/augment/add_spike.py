# -*- coding: utf-8 -*-
"""
@author:
"""

import copy
import numpy as np


def add_spike(data, strength=3, seed=None):
    is_matrix = len(data.shape) == 2
    if not is_matrix:
        data = np.expand_dims(data, 0)

    n_data = data.shape[0]
    data_len = data.shape[1]
    sigma = np.std(data, axis=1, keepdims=True)

    sigma_scaling = strength * sigma
    sigma_scaling[sigma[:, 0] == 0, 0] = strength

    rng = np.random.default_rng(seed=seed)
    location = rng.integers(low=0, high=data_len, size=n_data)
    noise = rng.standard_normal(size=(n_data,)) * sigma_scaling.flatten()

    data_aug = copy.deepcopy(data)
    for i in range(n_data):
        data_aug[i, location[i]] += noise[i]

    if not is_matrix:
        data_aug = data_aug[0, :]
    return data_aug

