#!/usr/bin/env python3
# -*- coding: utf8 -*-

import os
import json
import joblib

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from multiprocessing import Process, Manager, Pool
from functools import partial

data_info_path = 'data/handwriting/raw/handwriting_data_info_clean.json'
if not os.path.isdir('data/handwriting/final'):
    os.mkdir('data/handwriting/final')

target = open('data/targetSyllable.txt','r').read()
target = target.replace('\n', '').replace(' ', '')

data_info_raw = json.loads(open(data_info_path,'r').read())
data_info_raw = [
    item for item in data_info_raw['annotations']
    if item['attributes']['type']=='글자(음절)'
]
data_info_raw = [
    [f"data/handwriting/raw/{item['image_id']}.png", item['text']]
    for item in data_info_raw
]
data_info_raw = [
    item for item in data_info_raw
    if item[1] in target
]

def func(item):
    path, label = item
    rgb = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    row_min, col_min = \
        (round(rgb.shape[0]*0.05), round(rgb.shape[1]*0.05))
    row_max, col_max = \
        (rgb.shape[0]-row_min, rgb.shape[1]-col_min)
    rgb = rgb[row_min:row_max, col_min:col_max]
    rgb = cv2.threshold(rgb, 235, 255, cv2.THRESH_BINARY_INV)[1]
    col_min, row_min, height, width = cv2.boundingRect(rgb)
    col_max, row_max = (col_min+height, row_min+width)
    rgb = rgb[row_min:row_max, col_min:col_max]
    if len(rgb)>0:
        rgb = cv2.resize(rgb, dsize=(54,54))
        rgb = cv2.copyMakeBorder(rgb, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0])
        rgb = rgb / 255
        rgb = np.array(rgb, dtype=np.float32)
        save_path = path.replace('/raw/', '/final/')
        cv2.imwrite(save_path, rgb)
        # from matplotlib import pyplot as plt
        # plt.imshow(rgb, cmap='gray')
        return [save_path, label]
pool = Pool(30)
data_info = pool.map(func, data_info_raw)
pool.close()
pool.join()

data_info = [item for item in data_info if type(item)==list]
json.dump(
    data_info,
    open('data/dataInfo.json', 'w'),
    ensure_ascii=False
)

samples = np.unique(np.array(data_info)[:,1])
onehot_encoder = OneHotEncoder(sparse_output=False).fit(samples.reshape(-1, 1))
joblib.dump(onehot_encoder, open('data/onehotEncoder.bin','wb'))