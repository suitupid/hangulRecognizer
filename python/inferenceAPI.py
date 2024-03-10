#!/usr/bin/env python3
# -*- coding: utf8 -*-

import sys
import joblib

import cv2
import numpy as np
import torch
from typing import Union
from fastapi import FastAPI

from model import CnnModel

app = FastAPI()

@app.get("/predict/{image_name}")
def read_predict(image_name: str):
    image_path = 'image/'+image_name
    test = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    test = test[:,:,3]
    col_min, row_min, height, width = cv2.boundingRect(test)
    row_max, col_max = (row_min+width, col_min+height)
    test = test[row_min:row_max, col_min:col_max]
    test = cv2.resize(test, dsize=(54,54))
    test = test / 255
    test = cv2.copyMakeBorder(test, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0,0,0])
    test = test.reshape(1, 1, 64, 64)
    test = np.array(test, dtype=np.float32)
    
    model = CnnModel()
    model.load_state_dict(torch.load('model/hangulCnnClassifier.pt'))
    model.eval()
    onehot_encoder = joblib.load('data/onehotEncoder.bin')
    result = model(torch.from_numpy(test)).detach().numpy()
    result = onehot_encoder.inverse_transform(result)[0][0]
    return result