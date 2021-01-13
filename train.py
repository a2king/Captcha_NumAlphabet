#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Created on 2020/11/3 13:38
# Project:
# @Author: CaoYugang
import logging

import torch
import torch.nn as nn
import yaml
from torch.autograd import Variable
from models import CNN
from datasets import CaptchaData
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
import time
import os
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s -[PID:%(process)s]-%(levelname)s-%(funcName)s-%(lineno)d: [ %(message)s ]',
                datefmt="%Y-%m-%d %H:%M:%S")

with open('./config.yaml', 'r', encoding='utf-8') as f_config:
    config_result = f_config.read()
    config = yaml.load(config_result, Loader=yaml.FullLoader)


batch_size = config["train"]["batch_size"]
base_lr = config["train"]["lr"]
max_epoch = config["train"]["epoch"]
model_path = config["train"]["out_model_path"]
train_data_path = config["train"]["train_data"]
test_data_path = config["train"]["test_data"]
num_workers = config["train"]["num_workers"]
use_gpu = config["train"]["is_gpu"]
width = config["width"]
height = config["height"]
alphabet = config["alphabet"]
numchar = config["numchar"]
# restor = False

if not os.path.exists(model_path):
    logging.info("新建训练模型保存路径：{}".format(model_path))
    os.makedirs(model_path)


def calculat_acc(output, target):
    output, target = output.view(-1, len(alphabet)), target.view(-1, len(alphabet))
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, int(numchar)), target.view(-1, int(numchar))
    correct_list = []
    for i, j in zip(target, output):
        if torch.equal(i, j):
            correct_list.append(1)
        else:
            correct_list.append(0)
    acc = sum(correct_list) / len(correct_list)
    return acc


def train():
    transforms = Compose([Resize((height, width)), ToTensor()])
    train_dataset = CaptchaData(train_data_path, num_class=len(alphabet), num_char=int(numchar), transform=transforms, alphabet=alphabet)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                   shuffle=True, drop_last=True)
    test_data = CaptchaData(test_data_path, num_class=len(alphabet), num_char=int(numchar), transform=transforms, alphabet=alphabet)
    test_data_loader = DataLoader(test_data, batch_size=batch_size,
                                  num_workers=num_workers, shuffle=True, drop_last=True)
    cnn = CNN(num_class=len(alphabet), num_char=int(numchar), width=width, height=height)
    if use_gpu:
        cnn.cuda()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=base_lr)
    criterion = nn.MultiLabelSoftMarginLoss()

    for epoch in range(max_epoch):
        start_ = time.time()

        loss_history = []
        acc_history = []
        cnn.train()
        for img, target in train_data_loader:
            img = Variable(img)
            target = Variable(target)
            if use_gpu:
                img = img.cuda()
                target = target.cuda()
            output = cnn(img)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = calculat_acc(output, target)
            acc_history.append(float(acc))
            loss_history.append(float(loss))
        print('epoch:{},train_loss: {:.4}|train_acc: {:.4}'.format(
            epoch,
            torch.mean(torch.Tensor(loss_history)),
            torch.mean(torch.Tensor(acc_history)),
        ))

        loss_history = []
        acc_history = []
        cnn.eval()
        for img, target in test_data_loader:
            img = Variable(img)
            target = Variable(target)
            if torch.cuda.is_available():
                img = img.cuda()
                target = target.cuda()
            output = cnn(img)

            acc = calculat_acc(output, target)
            acc_history.append(float(acc))
            loss_history.append(float(loss))
        print('test_loss: {:.4}|test_acc: {:.4}'.format(
            torch.mean(torch.Tensor(loss_history)),
            torch.mean(torch.Tensor(acc_history)),
        ))
        print('epoch: {}|time: {:.4f}'.format(epoch, time.time() - start_))
        torch.save(cnn.state_dict(), os.path.join(model_path, "model_{}.path".format(epoch)))


if __name__ == "__main__":
    train()
