import os
from PIL import Image

import torch
import numpy as np
import torchvision
import torch.nn as nn

CLASS_NAMES = {0: 'Бумага', 1: 'Камень', 2: 'Ножницы'}

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=nn.ReLU()):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x + inputs)
        return x
    
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
    
    

class Predictor:
    def __init__(self, path='trained_model_dict'):
        self.model = self.model = nn.Sequential(Block(in_channels=3, out_channels=48, kernel_size=3, stride=2),

                                ResidualBlock(in_channels=48, out_channels=48, kernel_size=3),
                                nn.MaxPool2d(kernel_size=2, stride=2),

                                Block(in_channels=48, out_channels=96, kernel_size=3, stride=1),
                                nn.MaxPool2d(kernel_size=2, stride=2),

                                ResidualBlock(in_channels=96, out_channels=96, kernel_size=3),
                                nn.MaxPool2d(kernel_size=2, stride=2),

                                Block(in_channels=96, out_channels=192, kernel_size=3, stride=2),
                                nn.MaxPool2d(kernel_size=2, stride=2),

                                ResidualBlock(in_channels=192, out_channels=192, kernel_size=3),
                                nn.MaxPool2d(kernel_size=2, stride=2),

                                Flatten(),
                                nn.Linear(768, 3),
                                nn.LogSoftmax())

        self.model.load_state_dict(torch.load(path))

    def get_image_predict(self, img_path='img_path.jpg'):
        image = Image.open(img_path).convert('RGB').resize((300, 300), Image.ANTIALIAS)
        img_tensor = torch.tensor(np.transpose(np.array(image), (2, 0, 1))).unsqueeze(0)
        print('Before normalize', torch.max(img_tensor))
        img_tensor = img_tensor / 255.
        class_index = np.argmax(self.model(img_tensor).data.numpy()[0])
        result = CLASS_NAMES[class_index]
        return result
