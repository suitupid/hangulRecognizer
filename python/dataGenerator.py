#!/usr/bin/env python3
# -*- coding: utf8 -*-

import os
import json
import joblib

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Process, Manager, Pool

data_info_path = 'data/raw/handwriting_data_info_clean.json'
os.mkdir('data/final') if not os.path.isdir('data/final') else False

target = open('data/targetSyllable.txt','r').read().replace('\n', '')
label_encoder = LabelEncoder().fit(list(target))
joblib.dump(label_encoder, open('data/labelEncoder.bin','wb'))

data_info_raw = json.loads(open(data_info_path,'r').read())
data_info = [
    item for item in data_info_raw['annotations']
    if item['attributes']['type']=='글자(음절)'
]
data_info = [
    [f"data/raw/{item['image_id']}.png", item['text']]
    for item in data_info
]
data_info = [
    item for item in data_info
    if item[1] in target
]

def func(item):
    fpath, label = item
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    row_min, col_min = round(img.shape[0]*0.05), round(img.shape[1]*0.05)
    row_max, col_max = img.shape[0]-row_min, img.shape[1]-col_min
    img = img[row_min:row_max, col_min:col_max]
    _, img = cv2.threshold(img, 235, 255, cv2.THRESH_BINARY_INV)
    col_min, row_min, height, width = cv2.boundingRect(img)
    col_max, row_max = col_min+height, row_min+width
    img = img[row_min:row_max, col_min:col_max]
    if len(img)>0:
        img = cv2.resize(img, dsize=(54,54))
        img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0])
        img = cv2.cvtColor(img, cv2.IMREAD_COLOR)
        save_path = fpath.replace('/raw/', '/final/')
        # cv2.imwrite(save_path, img)
        encoder = joblib.load('data/labelEncoder.bin')
        label = encoder.transform([label]).tolist()
        return [save_path, label]
pool = Pool(5)
data_info = pool.map(func, data_info)
pool.close()
pool.join()

data_info = [item for item in data_info if item is not None]

json.dump(
    data_info,
    open('data/dataInfo.json', 'w'),
    ensure_ascii=False
)
