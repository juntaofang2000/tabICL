# -*- coding: utf-8 -*-
"""
@author:
"""


import os
import numpy as np
import scipy.signal as signal
from .util import flatten_datalist
from .util import normalize_dataset
from .util import relabel


def load_ucr_dataset(data_dir, data_name, seed, config_dict):
    max_len = int(config_dict['data']['max_len'])
    train_frac = float(config_dict['data']['dataset_train_frac'])
    test_frac = float(config_dict['data']['dataset_test_frac'])
    is_len_norm = config_dict['data']['is_len_norm']
    is_len_norm = is_len_norm.lower() == 'true'

    train_path = os.path.join(
        data_dir, 'Missing_value_and_variable_length_datasets_adjusted',
        '{0}', '{0}_TRAIN.tsv')
    test_path = os.path.join(
        data_dir, 'Missing_value_and_variable_length_datasets_adjusted',
        '{0}', '{0}_TEST.tsv')
    if not os.path.isfile(train_path.format(data_name)):
        train_path = os.path.join(data_dir, '{0}', '{0}_TRAIN.tsv')
        test_path = os.path.join(data_dir, '{0}', '{0}_TEST.tsv')

    train_path = train_path.format(data_name)
    test_path = test_path.format(data_name)

    data = np.concatenate(
        (np.loadtxt(train_path),
         np.loadtxt(test_path), ), axis=0)
    n_data = data.shape[0]
    data_len = data.shape[1] - 1

    rng = np.random.default_rng(seed)
    random_vec = rng.permutation(n_data)
    data = data[random_vec, :]

    label = data[:, 0]
    label = label.astype(int)
    data = data[:, 1:]
    data = np.expand_dims(data, 1)
    if is_len_norm and data_len != max_len:
        data = signal.resample(
            data, max_len, axis=2)

    data = normalize_dataset(data)
    label, n_class = relabel(label)

    data_train = []
    data_valid = []
    data_test = []

    label_train = []
    label_valid = []
    label_test = []
    for i in range(n_class):
        data_i = data[label == i, :, :]
        label_i = label[label == i]

        n_data_i = label_i.shape[0]

        n_train_i = np.round(train_frac * n_data_i)
        n_train_i = int(n_train_i)
        n_train_i = max(n_train_i, 1)

        n_test_i = np.round(test_frac * n_data_i)
        n_test_i = int(n_test_i)
        n_test_i = max(n_test_i, 1)
        n_valid_i = n_data_i - n_train_i - n_test_i

        train_start = 0
        train_end = n_train_i
        valid_start = train_end
        valid_end = valid_start + n_valid_i
        test_start = valid_end
        test_end = test_start + n_test_i

        data_train.append(data_i[train_start:train_end, :, :])
        data_valid.append(data_i[valid_start:valid_end, :, :])
        data_test.append(data_i[test_start:test_end, :, :])

        label_train.append(label_i[train_start:train_end])
        label_valid.append(label_i[valid_start:valid_end])
        label_test.append(label_i[test_start:test_end])

    data_train = np.concatenate(data_train, axis=0)
    data_valid = np.concatenate(data_valid, axis=0)
    data_test = np.concatenate(data_test, axis=0)

    label_train = np.concatenate(label_train, axis=0)
    label_valid = np.concatenate(label_valid, axis=0)
    label_test = np.concatenate(label_test, axis=0)

    dataset_ = {}
    dataset_['data_train'] = data_train
    dataset_['data_valid'] = data_valid
    dataset_['data_test'] = data_test
    dataset_['label_train'] = label_train
    dataset_['label_valid'] = label_valid
    dataset_['label_test'] = label_test
    dataset_['n_class'] = n_class
    dataset_['n_dim'] = data_train.shape[1]
    dataset_['data_len'] = data_train.shape[2]
    return dataset_


def get_data_list(config_dict):
    data_names = get_ucr_data_names()

    seed = int(config_dict['data']['seed'])
    archive_train_frac = float(config_dict['data']['archive_train_frac'])
    archive_valid_frac = float(config_dict['data']['archive_valid_frac'])

    rng = np.random.default_rng(seed)
    n_data = len(data_names)
    n_train = int(np.ceil(n_data * archive_train_frac))
    n_valid = int(np.ceil(n_data * archive_valid_frac))

    order = rng.permutation(n_data)
    train_list = [data_names[i] for i in order[:n_train]]
    valid_list = [data_names[i] for i in order[n_train:n_train + n_valid]]
    test_list = [data_names[i] for i in order[n_train + n_valid:]]

    data_list = {}
    data_list['train'] = list(flatten_datalist(train_list))
    data_list['valid'] = list(flatten_datalist(valid_list))
    data_list['test'] = list(flatten_datalist(test_list))

    ss = np.random.SeedSequence(seed)
    for partition in ['train', 'valid', 'test']:
        data_list[f'{partition}_seed'] = ss.generate_state(
            len(data_list[partition]))
    return data_list


def get_ucr_data_names():
    names = [
        'Adiac',
        'ArrowHead',
        'Beef',
        'BeetleFly',
        'BirdChicken',
        'Car',
        'CBF',
        'ChlorineConcentration',
        'CinCECGTorso',
        'Coffee',
        'Computers',
        ['CricketX',
         'CricketY',
         'CricketZ',],
        'DiatomSizeReduction',
        ['DistalPhalanxOutlineAgeGroup',
         'DistalPhalanxOutlineCorrect',
         'DistalPhalanxTW',],
        'Earthquakes',
        ['ECG200',
         'ECG5000',
         'ECGFiveDays',],
        'ElectricDevices',
        ['FaceAll',
         'FaceFour',
         'FacesUCR',],
        'FiftyWords',
        'Fish',
        ['FordA',
         'FordB',],
        ['GunPoint',
         'GunPointAgeSpan',
         'GunPointMaleVersusFemale',
         'GunPointOldVersusYoung',],
        'Ham',
        'HandOutlines',
        'Haptics',
        'Herring',
        'InlineSkate',
        'InsectWingbeatSound',
        'ItalyPowerDemand',
        ['LargeKitchenAppliances',
         'SmallKitchenAppliances',],
        ['Lightning2',
         'Lightning7',],
        'Mallat',
        'Meat',
        'MedicalImages',
        ['MiddlePhalanxOutlineAgeGroup',
         'MiddlePhalanxOutlineCorrect',
         'MiddlePhalanxTW',],
        'MoteStrain',
        ['NonInvasiveFetalECGThorax1',
         'NonInvasiveFetalECGThorax2',],
        'OliveOil',
        'OSULeaf',
        'PhalangesOutlinesCorrect',
        'Phoneme',
        'Plane',
        ['ProximalPhalanxOutlineAgeGroup',
         'ProximalPhalanxOutlineCorrect',
         'ProximalPhalanxTW',],
        'RefrigerationDevices',
        'ScreenType',
        'ShapeletSim',
        'ShapesAll',
        ['SonyAIBORobotSurface1',
         'SonyAIBORobotSurface2',],
        'StarLightCurves',
        'Strawberry',
        'SwedishLeaf',
        'Symbols',
        'SyntheticControl',
        ['ToeSegmentation1',
         'ToeSegmentation2',],
        'Trace',
        'TwoLeadECG',
        'TwoPatterns',
        ['UWaveGestureLibraryAll',
         'UWaveGestureLibraryX',
         'UWaveGestureLibraryY',
         'UWaveGestureLibraryZ',],
        'Wafer',
        'Wine',
        'WordSynonyms',
        ['Worms',
         'WormsTwoClass',],
        'Yoga',
        'ACSF1',
        ['AllGestureWiimoteX',
         'AllGestureWiimoteY',
         'AllGestureWiimoteZ',],
        'BME',
        'Chinatown',
        'Crop',
        ['DodgerLoopDay',
         'DodgerLoopGame',
         'DodgerLoopWeekend',],
        ['EOGHorizontalSignal',
         'EOGVerticalSignal',],
        'EthanolLevel',
        ['FreezerRegularTrain',
         'FreezerSmallTrain',],
        'Fungi',
        ['GestureMidAirD1',
         'GestureMidAirD2',
         'GestureMidAirD3',],
        ['GesturePebbleZ1',
         'GesturePebbleZ2',],
        'HouseTwenty',
        ['InsectEPGRegularTrain',
         'InsectEPGSmallTrain',],
        'MelbournePedestrian',
        ['MixedShapesRegularTrain',
         'MixedShapesSmallTrain',],
        'PickupGestureWiimoteZ',
        ['PigAirwayPressure',
         'PigArtPressure',
         'PigCVP',],
        'PLAID',
        'PowerCons',
        'Rock',
        ['SemgHandGenderCh2',
         'SemgHandMovementCh2',
         'SemgHandSubjectCh2',],
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD',
    ]
    return names

