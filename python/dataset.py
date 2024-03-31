#!/usr/bin/python3
# -*- coding: utf8 -*-

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, index):
        fpath, label = self.datasets[index]
        x = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        x = x / 255
        x = np.float32(x)
        x = x.reshape(1, 64, 64)
        x = torch.tensor(x)
        y = torch.tensor(label)
        return x, y

    def __len__(self):
        return len(self.datasets)