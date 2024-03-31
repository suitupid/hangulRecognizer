#!/usr/bin/python3
# -*- coding: utf8 -*-

from torch import nn
import lightning as L
from torch.optim import Adam

class CnnModel(L.LightningModule):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(128*16*16, 512),
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