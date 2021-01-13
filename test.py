# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 20:07:17 2019

@author: icetong
"""
import logging
from io import BytesIO

import torch
import torch.nn as nn
import yaml
from PIL import ImageSequence, Image

from models import CNN
from torchvision.transforms import Compose, ToTensor, Resize

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s -[PID:%(process)s]-%(levelname)s-%(funcName)s-%(lineno)d: [ %(message)s ]',
                    datefmt="%Y-%m-%d %H:%M:%S")

with open('./config.yaml', 'r', encoding='utf-8') as f_config:
    config_result = f_config.read()
    config = yaml.load(config_result, Loader=yaml.FullLoader)

model_path = config["test"]["model_path"]
use_gpu = config["test"]["is_gpu"]
width = config["width"]
height = config["height"]
alphabet = config["alphabet"]
numchar = config["numchar"]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_net = CNN()


def load_net():
    global model_net
    model_net = CNN(num_class=len(alphabet), num_char=int(numchar), width=width, height=height)
    if use_gpu:
        model_net = model_net.cuda()
        model_net.eval()
        model_net.load_state_dict(torch.load(model_path))
    else:
        model_net.eval()
        model_net.load_state_dict(torch.load(model_path, map_location='cpu'))


def predict_image(img):
    global model_net
    with torch.no_grad():
        img = img.convert('RGB')
        transforms = Compose([Resize((height, width)), ToTensor()])
        img = transforms(img)

        if use_gpu:
            img = img.view(1, 3, height, width).cuda()
        else:
            img = img.view(1, 3, height, width)
        output = model_net(img)

        output = output.view(-1, len(alphabet))
        output = nn.functional.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        output = output.view(-1, 4)[0]
        return ''.join([alphabet[i] for i in output.cpu().numpy()])


def gif2jpg(gif_image):
    image = gif_image
    jpg_list = []
    for index, f in enumerate(ImageSequence.Iterator(image)):
        #   获取图像序列存储
        # f.show()
        if index % 3 == 0:
            f = f.convert('RGB')
            output_buffer = BytesIO()
            f.save(output_buffer, format='JPEG')
            jpg_list.append(f)
    return jpg_list


def predict_gif(gif_image):
    result_list = {}
    for _image in gif2jpg(gif_image):
        result = predict_image(_image)
        if result in result_list:
            result_list[result] += 1
        else:
            result_list[result] = 1
    return sorted([(key, result_list[key]) for key in result_list], key=lambda x: x[1], reverse=True)[0][0]


if __name__ == "__main__":
    # predict()
    load_net()

    # gif_image = Image.open(r"F:\xftp_tmpfile\checkpoints_haiguan\1608197652.gif")
    # v_code = predict_gif(gif_image)
    # print(v_code)

    v_code = predict_image(Image.open(r"C:\Users\caoyugang\Downloads\data\test/fngg.png"))
    print(v_code)
