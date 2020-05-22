import os
from PIL import Image

import torch
import numpy as np
import torchvision
import torch.nn as nn

CLASS_NAMES = {0: 'Кошка', 1: 'Собака'}

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation=nn.ReLU(), padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.activation(x)
        return x

class resBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation=nn.ReLU()):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=int((kernel_size-1)/2))
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=int((kernel_size-1)/2))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation2 = activation
    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = input+self.bn2(x)
        x = self.activation(x)
        return x

class Predictor:
    def __init__(self, path='dogs_cats_dict'):
        self.model = nn.Sequential()
        self.model.add_module('first', Block(in_channels=3, out_channels=20, kernel_size=3))
        self.model.add_module("maxpool1", nn.MaxPool2d(kernel_size=3, stride=2))
        self.model.add_module('first2', Block(in_channels=20, out_channels=40, kernel_size=3))
        self.model.add_module("maxpool2", nn.MaxPool2d(kernel_size=2, stride=2))
        self.model.add_module('first3', Block(in_channels=40, out_channels=80, kernel_size=3))
        self.model.add_module("maxpool3", nn.MaxPool2d(kernel_size=2, stride=2))
        self.model.add_module('first21', Block(in_channels=80, out_channels=160, kernel_size=3))
        self.model.add_module("maxpool4", nn.MaxPool2d(kernel_size=2, stride=2))
        self.model.add_module('first212', Block(in_channels=160, out_channels=250, kernel_size=3))
        self.model.add_module("maxpool42", nn.MaxPool2d(kernel_size=2, stride=2))
        self.model.add_module('first213', Block(in_channels=250, out_channels=320, kernel_size=3))
        self.model.add_module("maxpool43", nn.MaxPool2d(kernel_size=2, stride=2))
        self.model.add_module('first214', Block(in_channels=320, out_channels=350, kernel_size=3))
        self.model.add_module("maxpool44", nn.MaxPool2d(kernel_size=2, stride=1))

        self.model.add_module("flatten", Flatten())
        self.model.add_module('seventh2', nn.Linear(1400, 512))
        self.model.add_module("linearr", nn.ReLU())
        self.model.add_module('seventh3', nn.Linear(512, 2))
        self.model.add_module('softmax', nn.LogSoftmax())

        self.model.load_state_dict(torch.load(path))

    def get_image_predict(self, img_path='img_path.jpg'):
        image = Image.open(img_path).convert('RGB').resize((224, 224), Image.ANTIALIAS)
        img_tensor = torch.tensor(np.transpose(np.array(image), (2, 0, 1))).unsqueeze(0)
        print('Before normalize', torch.max(img_tensor))
        img_tensor = img_tensor / 255.
        class_index = np.argmax(self.model(img_tensor).data.numpy()[0])
        result = CLASS_NAMES[class_index]
        return result
