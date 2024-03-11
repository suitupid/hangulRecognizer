#!/usr/bin/env python3
# -*- coding: utf8 -*-

import sys
import joblib

import cv2
import numpy as np
import torch

from model import CnnModel

class Inference():

    def __init__(self):
        super().__init__()

        self.model = CnnModel()
        self.model.load_state_dict(torch.load('model/hangulCnnClassifier.pt'))
        self.model.eval()
        self.onehot_encoder = joblib.load('data/onehotEncoder.bin')

    def preprocess(self, image_path):
        rgb = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        rgb = rgb[:,:,3]
        col_min, row_min, height, width = cv2.boundingRect(rgb)
        row_max, col_max = (row_min+width, col_min+height)
        rgb = rgb[row_min:row_max, col_min:col_max]
        rgb = cv2.resize(rgb, dsize=(54,54))
        rgb = rgb / 255
        rgb = cv2.copyMakeBorder(rgb, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0,0,0])
        rgb = rgb.reshape(1, 1, 64, 64)
        rgb = np.array(rgb, dtype=np.float32)
        return rgb

    def predict(self, rgb):
        result = self.model(torch.from_numpy(rgb)).detach().numpy()
        result = self.onehot_encoder.inverse_transform(result)[0][0]
        return result