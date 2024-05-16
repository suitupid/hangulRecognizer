#!/usr/bin/env python3
# -*- coding: utf8 -*-

import sys
import joblib

import cv2
import numpy as np
import torch

from model import CustomResNet

class Inference():

    def __init__(self):
        super().__init__()

        self.model = CustomResNet()
        self.model.load_state_dict(torch.load('model/hangulClassifier.pt'))
        self.model.eval()
        self.encoder = joblib.load('data/labelEncoder.bin')

    def preprocess(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = img[:,:,3]
        col_min, row_min, height, width = cv2.boundingRect(img)
        row_max, col_max = (row_min+width, col_min+height)
        img = img[row_min:row_max, col_min:col_max]
        img = cv2.resize(img, dsize=(54,54))
        img = img / 255
        img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0])
        img = img.reshape(1, 1, 64, 64)
        img = np.array(img, dtype=np.float32)
        return img

    def predict(self, img):
        img = torch.from_numpy(img)
        result = np.argmax(self.model(img).detach().numpy())
        result = self.encoder.inverse_transform([result])[0][0]
        return result