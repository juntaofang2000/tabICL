# -*- coding: utf-8 -*-
"""
@author:
"""


import numpy as np


def flatten_datalist(data_list):
    for item in data_list:
        if isinstance(item, list):
            yield from flatten_datalist(item)
        else:
            yield item


def relabel(label):
    label_set = np.unique(label)
    n_class = len(label_set)

    label_re = np.zeros(label.shape[0], dtype=int)
    for i, label_i in enumerate(label_set):
        label_re[label == label_i] = i
    return label_re, n_class


def normalize_dataset(data):
    data_mu = np.mean(data, axis=2, keepdims=True)
    data_sigma = np.std(data, axis=2, keepdims=True)
    data_sigma[data_sigma <= 0] = 1
    data = (data - data_mu) / data_sigma
    return data


def reshape_data(data):
    if len(data.shape) == 2:
        return data
    n_data = data.shape[0]
    n_dim = data.shape[1]
    data_len = data.shape[2]
    data = np.reshape(data, [n_data, n_dim * data_len, ])
    return data

