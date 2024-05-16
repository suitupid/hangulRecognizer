#!/usr/bin/python3
# -*- coding: utf8 -*-

import json
import joblib

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar

from dataset import CustomDataset
from model import CustomResNet

data = json.loads(open('data/dataInfo.json','r').read())

train, valid = train_test_split(
    data,
    test_size=0.05, shuffle=True,
    stratify=[item[1] for item in data]
)

train_dataset = CustomDataset(train)
valid_dataset = CustomDataset(valid)
train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=64)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, num_workers=64)

model = CustomResNet()
trainer = Trainer(
	max_epochs=100, accelerator='gpu',
	logger=False, enable_checkpointing=False,
	callbacks=[
        EarlyStopping(monitor='val_loss', patience=10),
        RichProgressBar()
    ]
)
trainer.fit(model, train_dataloader, valid_dataloader)

torch.save(model.state_dict(), 'model/hangulClassifier.pt')