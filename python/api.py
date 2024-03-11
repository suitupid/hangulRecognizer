#!/usr/bin/env python3
# -*- coding: utf8 -*-

from typing import Union
from fastapi import FastAPI

from inference import Inference

app = FastAPI()
tool = Inference()

@app.get("/predict/{image_name}")
def read_predict(image_name: str):
    image_path = 'image/'+image_name
    rgb = tool.preprocess(image_path)
    result = tool.predict(rgb)
    return result