#!/usr/bin/python3
# -*- coding: utf8 -*-

import joblib

from torch import nn
from torch.optim import Adam
from torchvision.models.resnet import BasicBlock
import lightning as L


class ResNetBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        self.block = BasicBlock(inplanes, planes, stride, downsample)

    def forward(self, x):
        return self.block(x)


class CustomResNet(L.LightningModule):

    def __init__(self):
        super().__init__()

        encoder = joblib.load('data/labelEncoder.bin')
        num_classes = len(encoder.classes_)

        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.resnet_layer = nn.Sequential(
            ResNetBlock(64, 128, stride=2),
            ResNetBlock(128, 256, stride=2),
            ResNetBlock(256, 512, stride=2)
        )

        self.flatten = nn.Flatten()

        self.head = nn.Sequential(
            nn.Linear(512*4*4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x):
        out = self.stem(x)
        out = self.resnet_layer(out)
        out = self.flatten(out)
        out = self.head(out)
        return out

    def training_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        y = y.reshape(-1)
        train_loss = self.loss_fn(y_hat, y)
        self.log('train_loss', train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        y = y.reshape(-1)
        val_loss = self.loss_fn(y_hat, y)
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
