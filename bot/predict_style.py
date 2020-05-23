import os
from PIL import Image

import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        self.weight = weight

    def forward(self, input):
        self.loss = F.mse_loss(input * self.weight, self.target)
        return input.clone()

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
    
class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight

    def forward(self, input):
        self.G = gram_matrix(input)
        self.G.mul_(self.weight)
        self.loss = F.mse_loss(self.G, self.target)
        return input.clone()

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
    

class Predictor:
    def __init__(self, path='wave_model_dict'):
        self.model = nn.Sequential()
        self.model.add_module('conv_1', nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('style_loss_1', StyleLoss())
        self.model.add_module('relu_1', nn.ReLU(inplace=True))
        self.model.add_module('conv_2', nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('style_loss_2', StyleLoss())
        self.model.add_module('relu_2', nn.ReLU(inplace=True))
        self.model.add_module('pool_3', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.model.add_module('conv_3', nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('style_loss_3', StyleLoss())
        self.model.add_module('relu_3', nn.ReLU(inplace=True))
        self.model.add_module('conv_4', nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('content_loss_4', ContentLoss())
        self.model.add_module('style_loss_4', StyleLoss())
        self.model.add_module('relu_4', nn.ReLU(inplace=True))
        self.model.add_module('pool_5', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.model.add_module('conv_5', nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('style_loss_5', StyleLoss())
        self.model.add_module('relu_5', nn.ReLU(inplace=True))
        self.model.add_module('conv_6', nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('relu_6', nn.ReLU(inplace=True))
        self.model.add_module('conv_7', nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('relu_7', nn.ReLU(inplace=True))
        self.model.add_module('conv_8', nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('relu_8', nn.ReLU(inplace=True))
        self.model.add_module('pool_9', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.model.add_module('conv_9', nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('relu_9', nn.ReLU(inplace=True))
        self.model.add_module('conv_10', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('relu_10', nn.ReLU(inplace=True))
        self.model.add_module('conv_11', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('relu_11', nn.ReLU(inplace=True))
        self.model.add_module('conv_12', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('relu_12', nn.ReLU(inplace=True))
        self.model.add_module('pool_13', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.model.add_module('conv_13', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('relu_13', nn.ReLU(inplace=True))
        self.model.add_module('conv_14', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('relu_14', nn.ReLU(inplace=True))
        self.model.add_module('conv_15', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('relu_15', nn.ReLU(inplace=True))
        self.model.add_module('conv_16', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('relu_16', nn.ReLU(inplace=True))
        self.model.add_module('pool_17', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        self.model.load_state_dict(torch.load(path))

    def get_image_predict(self, img_path='img_path.jpg'):
        image = Image.open(img_path).convert('RGB').resize((256, 256), Image.ANTIALIAS)
        img_tensor = torch.tensor(np.transpose(np.array(image), (2, 0, 1))).unsqueeze(0)
        print('Before normalize', torch.max(img_tensor))
        img_tensor = img_tensor / 255.
        model(img_tensor)
        save_image(img_tensor, "res_photo.jpg")
        
