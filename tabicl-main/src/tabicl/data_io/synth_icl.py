# -*- coding: utf-8 -*-
"""
@author:
"""

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from .kernel_synth import generate_time_series

from ..augment import jittering
from ..augment import smoothing
from ..augment import add_slope
from ..augment import add_spike
from ..augment import add_step
from ..augment import cropping
from ..augment import masking
from ..augment import shifting
from ..augment import time_warping
from ..augment import no_change



class SynthICLDatasetIter(IterableDataset):
    def __init__(self, config_dict):
        super(SynthICLDatasetIter, self).__init__()

        self.n_bit = int(config_dict['incontext']['n_bit'])
        self.n_step = int(config_dict['incontext']['n_step'])
        self.aug_bank = [
            lambda x:jittering(x, strength=0.1, seed=None),
            lambda x:smoothing(x, max_ratio=0.5, min_ratio=0.01, seed=None),
            lambda x:add_slope(x, strength=1, seed=None),
            lambda x:add_spike(x, strength=3, seed=None),
            lambda x:add_step(x, min_ratio=0.1, strength=1, seed=None),
            lambda x:cropping(x, min_ratio=0.1, seed=None),
            lambda x:masking(x, max_ratio=0.5, seed=None),
            lambda x:shifting(x, seed=None),
            lambda x:time_warping(x, min_ratio=0.5, seed=None),
            lambda x:no_change(x),
        ]
        is_aug = config_dict['incontext']['is_aug']
        is_aug = is_aug.lower() == 'true'
        self.is_aug = is_aug

        is_mix = config_dict['incontext']['is_mix']
        is_mix = is_mix.lower() == 'true'
        self.is_mix = is_mix

        self.augment_cap = int(config_dict['incontext']['augment_cap'])

        self.max_class = int(config_dict['incontext']['max_class'])
        self.config_dict = config_dict

    def __iter__(self):
        n_bit = self.n_bit
        n_step = self.n_step
        aug_bank = self.aug_bank
        is_aug = self.is_aug
        is_mix = self.is_mix
        augment_cap = self.augment_cap
        max_class = self.max_class

        while True:
            x = []
            y = []
            y_bit = []
            if is_mix:
                pattern_0 = generate_time_series()['target']
                pattern_1 = generate_time_series()['target']

                y_train_bit_0 = np.random.randint(0, 2, size=(n_bit, ))
                y_train_bit_1 = np.random.randint(0, 2, size=(n_bit, ))
                while np.all(y_train_bit_0 == y_train_bit_1):
                    y_train_bit_1 = np.random.randint(0, 2, size=(n_bit, ))

                ratios = np.random.uniform(0, 1, size=(n_step,))
                cutoff = np.random.uniform(0.1, 0.9)
                for ratio in ratios:
                    x_ = ratio * pattern_0 + (1 - ratio) * pattern_1
                    if ratio < cutoff:
                        y_ = 1
                        y_bit_ = y_train_bit_1
                    else:
                        y_ = 0
                        y_bit_ = y_train_bit_0

                    x.append(x_)
                    y.append(y_)
                    y_bit.append(y_bit_)
            else:
                n_class = np.random.randint(2, max_class + 1)
                sample_per_class = n_step // n_class
                exist_y_bit = {}
                for i in range(n_class):
                    x_ = generate_time_series()['target']
                    y_ = i
                    while True:
                        y_bit_ = np.random.randint(0, 2, size=(n_bit, ))
                        y_bit_string = ''.join(
                            [str(bit) for bit in y_bit_])
                        if y_bit_string not in exist_y_bit:
                            break

                    if i == n_class - 1:
                        n_sample = n_step - sample_per_class * i
                    else:
                        n_sample = sample_per_class

                    for _ in range(n_sample):
                        x.append(x_)
                        y.append(y_)
                        y_bit.append(y_bit_)

                    exist_y_bit[y_bit_string] = 1

            if is_aug:
                n_aug = np.random.randint(augment_cap)
                for _ in range(n_aug):
                    aug_idx = np.random.randint(0, len(aug_bank) - 1)
                    aug_func = aug_bank[aug_idx]
                    x = [aug_func(x_) for x_ in x]

            for i in range(len(x)):
                mu = np.mean(x[i])
                sigma = np.std(x[i])
                if sigma > 1e-6:
                    x[i] = (x[i] - mu) / sigma
                else:
                    x[i] = x[i] - mu

            x = np.array(x, dtype=np.float32)
            y = np.array(y, dtype=np.int64)
            y_bit = np.array(y_bit, dtype=np.float32)

            x = np.expand_dims(x, axis=1)

            n_data = x.shape[0]
            order = np.random.permutation(n_data)
            x = x[order, ...]
            y = y[order]
            y_bit = y_bit[order, ...]

            n_train = np.ceil(n_data / 2).astype(np.int64)
            x_train = x[:n_train, ...]
            y_train = y[:n_train]
            y_train_bit = y_bit[:n_train, ...]
            x_test = x[n_train:, ...]
            y_test = y[n_train:]

            mask_train = np.zeros((x_train.shape[0],), dtype=bool)
            mask_test = np.zeros((x_test.shape[0],), dtype=bool)

            batch = {}
            batch['x_train'] = torch.tensor(
                x_train, dtype=torch.float32)
            batch['y_train'] = torch.tensor(
                y_train, dtype=torch.long)
            batch['y_train_bit'] = torch.tensor(
                y_train_bit, dtype=torch.float32)
            batch['x_test'] = torch.tensor(
                x_test, dtype=torch.float32)
            batch['y_test'] = torch.tensor(
                y_test, dtype=torch.long)
            batch['mask_train'] = torch.tensor(
                mask_train, dtype=torch.bool)
            batch['mask_test'] = torch.tensor(
                mask_test, dtype=torch.bool)
            yield batch


def _tensor_list_collator(batch):
    batch_out = {}
    for key in ['x_train', 'x_test', ]:
        batch_out[key] = [item[key] for item in batch]
        batch_out[key] = torch.nn.utils.rnn.pad_sequence(
            batch_out[key],
            batch_first=True,
            padding_value=0.0)

    for key in ['y_train', 'y_test', 'y_train_bit',]:
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


def get_synth_dataloader(config_dict, prefetch_factor, n_job):
    dataset = SynthICLDatasetIter(
        config_dict)

    batch_size = int(config_dict['optim']['batch_size'])
    drop_last_flag = True

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



# juntao fang
class MultiClassMixupDataset(IterableDataset):
    def __init__(self, config_dict):
        super(MultiClassMixupDataset, self).__init__()
        
        # Configuration
        self.n_bit = int(config_dict.get('n_bit', 8))
        self.n_step = int(config_dict.get('n_step', 1000))
        self.min_n_step = config_dict.get('min_n_step', None)
        self.max_n_step = config_dict.get('max_n_step', None)
        if self.min_n_step is not None:
            self.min_n_step = int(self.min_n_step)
        if self.max_n_step is not None:
            self.max_n_step = int(self.max_n_step)

        self.max_class = int(config_dict.get('max_class', 5)) # Support > 2 classes
        self.mix_alpha = float(config_dict.get('mix_alpha', 1.0)) # Dirichlet concentration
        
        # Augmentation bank (same as original TiCT)
        self.aug_bank = [
            lambda x: jittering(x, strength=0.1, seed=None),
            # ... (Add other augmentations here) ...
            lambda x: no_change(x),
        ]
        self.augment_cap = int(config_dict.get('augment_cap', 2))
        self.min_train_size = config_dict.get('min_train_size', 0.5)
        self.max_train_size = config_dict.get('max_train_size', 0.5)
        
    def __iter__(self):
        while True:
            # Determine n_step for this iteration
            if self.min_n_step is not None and self.max_n_step is not None:
                n_step = np.random.randint(self.min_n_step, self.max_n_step + 1)
            else:
                n_step = self.n_step

            # 1. Determine number of classes for this task (e.g., random between 3 and max_class)
            # We use >=3 to differentiate from the original binary logic, but 2 works too.
            n_class = np.random.randint(3, self.max_class + 1)
            
            # 2. Generate Prototypes (Templates) for each class
            # These are the "corners" of the simplex
            prototypes = []
            for _ in range(n_class):
                ts_data = generate_time_series()['target']
                # Normalize prototypes to ensure fair mixing
                ts_data = (ts_data - np.mean(ts_data)) / (np.std(ts_data) + 1e-6)
                prototypes.append(ts_data)
            prototypes = np.array(prototypes) # Shape: (n_class, time_len)

            # 3. Generate Random Bit Labels for these classes (Symbolic Labels)
            # In TiCT, labels are arbitrary bit strings
            class_bit_labels = []
            seen_bits = set()
            for _ in range(n_class):
                while True:
                    bits = np.random.randint(0, 2, size=(self.n_bit,))
                    bit_str = ''.join(map(str, bits))
                    if bit_str not in seen_bits:
                        seen_bits.add(bit_str)
                        class_bit_labels.append(bits)
                        break
            class_bit_labels = np.array(class_bit_labels)

            # 4. Generate Samples via Dirichlet Mixup
            # We generate 'n_step' samples for this task
            x_batch = []
            y_batch = []
            y_bit_batch = []
            
            # Sample mixing weights from Dirichlet distribution
            # Shape: (n_step, n_class)
            # alpha < 1 => samples concentrate near corners (easier, distinct classes)
            # alpha > 1 => samples concentrate in center (harder, very mixed)
            alpha_vec = [self.mix_alpha] * n_class
            mix_weights = np.random.dirichlet(alpha_vec, size=n_step)

            for i in range(n_step):
                weights = mix_weights[i]
                
                # A. Mix the prototypes
                # Linear combination of all N prototypes
                # x_new = w0*P0 + w1*P1 + ... + wn*Pn
                # Reshape weights for broadcasting: (n_class, 1) * (n_class, time_len)
                weighted_protos = weights[:, np.newaxis] * prototypes
                x_mixed = np.sum(weighted_protos, axis=0)
                
                # B. Assign Label
                # The label is the class with the highest weight (Dominant Component)
                # This creates Voronoi-like decision boundaries on the simplex
                dominant_class_idx = np.argmax(weights)
                
                # Optional: Add a "margin" check.
                # If the max weight is too low (e.g., < 1/n_class + epsilon), it's an ambiguous sample.
                # TiCT handles ambiguity by forcing a choice, which is good for learning.
                
                y_label = dominant_class_idx
                y_bit = class_bit_labels[dominant_class_idx]
                
                # C. Apply Augmentations (Per sample)
                # Apply random augmentations from the bank
                n_aug = np.random.randint(self.augment_cap)
                for _ in range(n_aug):
                    aug_idx = np.random.randint(0, len(self.aug_bank))
                    x_mixed = self.aug_bank[aug_idx](x_mixed)
                
                # D. Final Normalization (Instance Norm)
                mu = np.mean(x_mixed)
                sigma = np.std(x_mixed)
                if sigma > 1e-6:
                    x_mixed = (x_mixed - mu) / sigma
                else:
                    x_mixed = x_mixed - mu

                x_batch.append(x_mixed)
                y_batch.append(y_label)
                y_bit_batch.append(y_bit)

            # 5. Format Batch (Split into Support/Query if needed, or yield full sequence)
            # Following the original code structure roughly:
            x_batch = np.array(x_batch, dtype=np.float32)
            y_batch = np.array(y_batch, dtype=np.int64)
            y_bit_batch = np.array(y_bit_batch, dtype=np.float32)
            
            # Add channel dimension
            x_batch = np.expand_dims(x_batch, axis=1) # (n_step, 1, time_len)
            
            # Shuffle
            order = np.random.permutation(n_step)
            x_batch = x_batch[order]
            y_batch = y_batch[order]
            y_bit_batch = y_bit_batch[order]

            # Split logic (Train/Test for ICL)
            min_k = self.min_train_size
            max_k = self.max_train_size
            
            if isinstance(min_k, float):
                min_k = int(min_k * n_step)
            if isinstance(max_k, float):
                max_k = int(max_k * n_step)
                
            min_k = max(1, min_k)
            max_k = min(n_step - 1, max_k)
            
            if min_k >= max_k:
                n_train = min_k
            else:
                n_train = np.random.randint(min_k, max_k + 1)
            
            batch = {
                'x_train': torch.tensor(x_batch[:n_train], dtype=torch.float32),
                'y_train': torch.tensor(y_batch[:n_train], dtype=torch.long),
                'y_train_bit': torch.tensor(y_bit_batch[:n_train], dtype=torch.float32),
                'x_test': torch.tensor(x_batch[n_train:], dtype=torch.float32),
                'y_test': torch.tensor(y_batch[n_train:], dtype=torch.long),
                # Masks would be added here as in original code
                'mask_train': torch.zeros((n_train,), dtype=torch.bool),
                'mask_test': torch.zeros((n_step - n_train,), dtype=torch.bool)
            }
            
            yield batch
