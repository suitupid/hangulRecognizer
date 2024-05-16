#!/usr/bin/python3
# -*- coding: utf8 -*-

import joblib

from torch import nn
from torch.optim import Adam
import lightning as L

class CustomResNet(L.LightningModule):

    def __init__(self):
        super().__init__()
        encoder = joblib.load('data/labelEncoder.bin')
        num_classes = len(encoder.classes_)
        
        self.init =  nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(512*5*5, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.init(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

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
