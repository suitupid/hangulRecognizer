#!/usr/bin/python3
# -*- coding: utf8 -*-

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, datasets, target_encoder=None):
        self.datasets = datasets
        self.target_encoder = target_encoder

    def __getitem__(self, index):
        path, label = self.datasets[index]
        _x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _x = _x.reshape(1, 64, 64)
        _x = torch.from_numpy(np.array(_x, dtype=np.float32))
        _y = self.target_encoder.transform([[label]])
        _y = _y.reshape(-1)
        _y = torch.from_numpy(np.array(_y, dtype=np.float32))
        return _x, _y

    def __len__(self):
        return len(self.datasets)