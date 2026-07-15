#!/usr/bin/python3
# -*- coding: utf8 -*-

import json
import joblib

import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar

from dataset import CustomDataset
from model import CustomNetwork


data = json.loads(open('data/dataInfo.json','r').read())

dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=32, num_workers=8)

model = CustomNetwork()
trainer = Trainer(
	max_epochs=50, accelerator='gpu',
	logger=False, enable_checkpointing=False,
	callbacks=[
        EarlyStopping(monitor='train_loss', patience=10),
        RichProgressBar()
    ]
)
trainer.fit(model, dataloader)

torch.save(model.state_dict(), 'model/hangulClassifier.pt')
