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
# from model import CnnModel

data = json.loads(open('data/dataInfo.json','r').read())

train, valid = train_test_split(
    data,
    test_size=0.05, shuffle=True,
    stratify=[item[1] for item in data]
)

train_dataset = CustomDataset(train)
valid_dataset = CustomDataset(valid)
train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=8)
valid_dataloader = DataLoader(valid_dataset, batch_size=16, num_workers=8)

for x, y in train_dataloader:
    break

from torch import nn
import lightning as L
from torch.optim import Adam

class CnnModel(L.LightningModule):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(256*32*32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2350)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        y = y.reshape(-1)
        train_loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        y = y.reshape(-1)
        val_loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

model = CnnModel()
trainer = Trainer(
	max_epochs=100, accelerator='gpu',
	logger=False, enable_checkpointing=False,
	callbacks=[
        EarlyStopping(monitor='val_loss', patience=10),
        RichProgressBar()
    ]
)
trainer.fit(model, train_dataloader, valid_dataloader)

torch.save(model.state_dict(), 'model/hangulCnnClassifier.pt')