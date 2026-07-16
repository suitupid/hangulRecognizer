#!/usr/bin/env python3
# -*- coding: utf8 -*-

from typing import Union
from fastapi import FastAPI

from inference import Inference

tool = Inference()
app = FastAPI()

@app.get("/predict/{image_name}")
def get_result(image_name):
    image_path = 'image/'+image_name
    img = tool.preprocess(image_path)
    result = tool.predict(img)
    return result