# -*- coding: utf-8 -*-
"""
@author:
"""

import numpy as np
from scipy import signal
from .util import cleanup


def cropping(data, min_ratio=0.1, seed=None):
    is_matrix = len(data.shape) == 2
    if not is_matrix:
        data = np.expand_dims(data, 0)

    n_data = data.shape[0]
    data_len = data.shape[1]

    rng = np.random.default_rng(seed=seed)
    segment_len = rng.random(size=n_data)
    segment_len = segment_len * (1 - min_ratio) + min_ratio
    segment_len = segment_len * data_len
    segment_start = rng.random(size=n_data)
    segment_start = segment_start * (data_len - segment_len)
    segment_end = segment_start + segment_len

    segment_start = cleanup(segment_start, 0, data_len)
    segment_end = cleanup(segment_end, 0, data_len)
    data_aug = np.zeros((n_data, data_len))
    for i in range(n_data):
        segment = data[i, segment_start[i]:segment_end[i]]
        data_aug[i, :] = signal.resample(segment, data_len)

    if not is_matrix:
        data_aug = data_aug[0, :]
    return data_aug

