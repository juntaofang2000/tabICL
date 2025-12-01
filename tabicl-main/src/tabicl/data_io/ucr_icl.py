# -*- coding: utf-8 -*-
"""
@author:
"""


import numpy as np
import torch
import sklearn.neighbors
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from .ucr_io import load_ucr_dataset
from .ucr_io import get_data_list
from .util import reshape_data


class UCRICLDatasetIter(IterableDataset):
    def __init__(self, data_dir, config_dict, partition, data_name=None):
        super(UCRICLDatasetIter).__init__()
        partition = partition.lower()
        assert partition in ['train', 'valid', 'test', ]

        data_list = get_data_list(config_dict)

        if partition == 'train':
            self.data_list = data_list['train'] + data_list['valid']
            self.seeds = np.concatenate((data_list['valid_seed'],
                                         data_list['train_seed']))
        elif partition == 'valid':
            self.data_list = data_list['valid']
            self.seeds = data_list['valid_seed']
        else:
            self.data_list = data_list['test']
            self.seeds = data_list['test_seed']

        if data_name is not None:
            self.data_list = (data_list['train'] +
                              data_list['valid'] +
                              data_list['test'])
            self.seeds = np.concatenate((data_list['valid_seed'],
                                         data_list['train_seed'],
                                         data_list['test_seed']))
            for data_name_, seed_ in zip(self.data_list, self.seeds):
                if data_name_ == data_name:
                    self.data_list = [data_name_, ]
                    self.seeds = np.array([seed_, ])
                    break

        self.data_dir = data_dir
        self.config_dict = config_dict
        self.partition = partition
        self.epoch_counter = 0

    def __len__(self):
        data_dir = self.data_dir
        config_dict = self.config_dict
        partition = self.partition
        data_list = self.data_list
        seeds = self.seeds
        n_data = len(data_list)
        n_data_total = 0
        for i in range(n_data):
            dataset = load_ucr_dataset(
                data_dir, data_list[i], seeds[i], config_dict)

            if partition == 'train':
                n_data_i = (dataset['data_train'].shape[0] +
                            dataset['data_valid'].shape[0] +
                            dataset['data_test'].shape[0])
            elif partition == 'valid':
                n_data_i = dataset['data_valid'].shape[0]
            else:
                n_data_i = dataset['data_test'].shape[0]
            n_data_total += n_data_i
        return n_data_total

    def __iter__(self):
        data_dir = self.data_dir
        config_dict = self.config_dict
        partition = self.partition
        data_list = self.data_list
        seeds = self.seeds
        n_data = len(data_list)
        n_neighbor = int(config_dict['incontext']['n_neighbor'])

        self.shuffle()
        for i in range(n_data):
            dataset = load_ucr_dataset(
                data_dir, data_list[i], seeds[i], config_dict)
            if partition == 'train':
                data = np.concatenate([dataset['data_train'],
                                       dataset['data_valid'],
                                       dataset['data_test']],
                                      axis=0)
                label = np.concatenate([dataset['label_train'],
                                        dataset['label_valid'],
                                        dataset['label_test']],
                                       axis=0)
                data_flatten = reshape_data(data)
                n_neighbor_ = min(n_neighbor, data_flatten.shape[0] - 1)
                knn_graph = sklearn.neighbors.kneighbors_graph(
                    data_flatten, n_neighbor_)

                n_sample = data_flatten.shape[0]
                order = np.random.permutation(n_sample)
                for j in order:
                    start = knn_graph.indptr[j]
                    end = knn_graph.indptr[j + 1]
                    neighbor_idx = knn_graph.indices[start:end]

                    x_train = data[neighbor_idx, ...]
                    y_train = label[neighbor_idx]
                    x_test = data[j:j+1, ...]
                    y_test = label[j:j+1]

                    mask_train = np.zeros((x_train.shape[0],), dtype=bool)
                    mask_test = np.zeros((1,), dtype=bool)

                    batch = {}
                    batch['x_train'] = torch.tensor(
                        x_train, dtype=torch.float32)
                    batch['y_train'] = torch.tensor(
                        y_train, dtype=torch.long)
                    batch['x_test'] = torch.tensor(
                        x_test, dtype=torch.float32)
                    batch['y_test'] = torch.tensor(
                        y_test, dtype=torch.long)
                    batch['mask_train'] = torch.tensor(
                        mask_train, dtype=torch.bool)
                    batch['mask_test'] = torch.tensor(
                        mask_test, dtype=torch.bool)
                    yield batch
            elif partition == 'valid':
                data_train = dataset['data_train']
                data_test = dataset['data_valid']
                label_train = dataset['label_train']
                label_test = dataset['label_valid']

                data_train_flatten = reshape_data(data_train)
                data_test_flatten = reshape_data(data_test)
                n_neighbor_ = min(n_neighbor, data_train_flatten.shape[0] - 1)

                knn_graph = sklearn.neighbors.NearestNeighbors(
                    n_neighbors=n_neighbor_)
                knn_graph.fit(data_train_flatten)
                neighbor_idx = knn_graph.kneighbors(
                    data_test_flatten, return_distance=False)

                n_sample = data_test.shape[0]
                for j in range(n_sample):
                    x_train = data_train[neighbor_idx[j], ...]
                    y_train = label_train[neighbor_idx[j]]
                    x_test = data_test[j:j+1, ...]
                    y_test = label_test[j:j+1]

                    mask_train = np.zeros((x_train.shape[0],), dtype=bool)
                    mask_test = np.zeros((1,), dtype=bool)

                    batch = {}
                    batch['x_train'] = torch.tensor(
                        x_train, dtype=torch.float32)
                    batch['y_train'] = torch.tensor(
                        y_train, dtype=torch.long)
                    batch['x_test'] = torch.tensor(
                        x_test, dtype=torch.float32)
                    batch['y_test'] = torch.tensor(
                        y_test, dtype=torch.long)
                    batch['mask_train'] = torch.tensor(
                        mask_train, dtype=torch.bool)
                    batch['mask_test'] = torch.tensor(
                        mask_test, dtype=torch.bool)
                    yield batch
            else:
                data_train = np.concatenate([dataset['data_train'],
                                             dataset['data_valid'],],
                                            axis=0)
                data_test = dataset['data_test']
                label_train = np.concatenate([dataset['label_train'],
                                              dataset['label_valid'],],
                                             axis=0)
                label_test = dataset['label_test']

                data_train_flatten = reshape_data(data_train)
                data_test_flatten = reshape_data(data_test)
                n_neighbor_ = min(n_neighbor, data_train_flatten.shape[0] - 1)

                knn_graph = sklearn.neighbors.NearestNeighbors(
                    n_neighbors=n_neighbor_)
                knn_graph.fit(data_train_flatten)
                neighbor_idx = knn_graph.kneighbors(
                    data_test_flatten, return_distance=False)

                n_sample = data_test.shape[0]
                for j in range(n_sample):
                    x_train = data_train[neighbor_idx[j], ...]
                    y_train = label_train[neighbor_idx[j]]
                    x_test = data_test[j:j+1, ...]
                    y_test = label_test[j:j+1]

                    mask_train = np.zeros((x_train.shape[0],), dtype=bool)
                    mask_test = np.zeros((1,), dtype=bool)

                    batch = {}
                    batch['x_train'] = torch.tensor(
                        x_train, dtype=torch.float32)
                    batch['y_train'] = torch.tensor(
                        y_train, dtype=torch.long)
                    batch['x_test'] = torch.tensor(
                        x_test, dtype=torch.float32)
                    batch['y_test'] = torch.tensor(
                        y_test, dtype=torch.long)
                    batch['mask_train'] = torch.tensor(
                        mask_train, dtype=torch.bool)
                    batch['mask_test'] = torch.tensor(
                        mask_test, dtype=torch.bool)
                    yield batch

    def shuffle(self):
        partition = self.partition
        data_list = self.data_list
        seeds = self.seeds
        n_data = len(data_list)

        if partition == 'train':
            order = np.random.permutation(n_data)
            data_list = [data_list[i] for i in order]
            seeds = seeds[order]
            self.data_list = data_list
            self.seeds = seeds


def _tensor_list_collator(batch):
    batch_out = {}
    for key in ['x_train', 'x_test', ]:
        batch_out[key] = [item[key] for item in batch]
        batch_out[key] = torch.nn.utils.rnn.pad_sequence(
            batch_out[key],
            batch_first=True,
            padding_value=0.0)

    for key in ['y_train', 'y_test',]:
        batch_out[key] = [item[key] for item in batch]
        batch_out[key] = torch.nn.utils.rnn.pad_sequence(
            batch_out[key],
            batch_first=True,
            padding_value=0)

    for key in ['mask_train', 'mask_test',]:
        batch_out[key] = [item[key] for item in batch]
        batch_out[key] = torch.nn.utils.rnn.pad_sequence(
            batch_out[key],
            batch_first=True,
            padding_value=True)
    return batch_out


def get_ucr_dataloader(data_dir, config_dict, partition,
                       prefetch_factor, n_job, data_name=None):
    dataset = UCRICLDatasetIter(
        data_dir, config_dict, partition, data_name=data_name)

    batch_size = int(config_dict['optim']['batch_size'])
    drop_last_flag = partition == 'train'

    if n_job == 0:
        prefetch_factor = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_job,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last_flag,
        collate_fn=_tensor_list_collator)
    return dataloader, dataset

