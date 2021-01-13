#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Created on 2020/11/3 13:38
# Project:
# @Author: CaoYugang

import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_class=36, num_char=4, width=100, height=35):
        super(CNN, self).__init__()
        self.num_class = num_class
        self.num_char = num_char
        self.line_size = int(512 * (width // 2 // 2 // 2 // 2) * (height // 2 // 2 // 2 // 2))
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 512, 3, padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.fc = nn.Linear(self.line_size, self.num_class * self.num_char)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, self.line_size)
        x = self.fc(x)
        return x
