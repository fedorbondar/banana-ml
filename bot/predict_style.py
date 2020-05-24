import os
from PIL import Image

import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.autograd import Variable

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
    def __init__(self):
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

    def get_image_predict(self, img_path='img_path.jpg', option="1"):
        image = Image.open(img_path).convert('RGB').resize((256, 256), Image.ANTIALIAS)
        img_tensor = torch.tensor(np.transpose(np.array(image), (2, 0, 1))).unsqueeze(0)
        print('Before normalize', torch.max(img_tensor))
        img_tensor = img_tensor / 255.
        
        if option == "1":
            style_img = image_loader("wave.jpg").type(dtype)
        if option == "2":
            style_img = image_loader("the_scream.jpg").type(dtype)
        if option == "3":
            style_img = image_loader("starry_night.jpg").type(dtype)

        content_weight = 1            # coefficient for content loss
        style_weight = 1000           # coefficient for style loss
        content_layers = ('conv_4',)  # use these layers for content loss
        style_layers = ('conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5')
        
        content_losses = []
        style_losses = []

        i = 1
        for layer in list(self.model):
            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)

                if name in content_layers:
                    # add content loss:
                    target = self.model(img_tensor).clone()
                    content_loss = ContentLoss(target, content_weight)
                    content_losses.append(content_loss)

                if name in style_layers:
                    # add style loss:
                    target_feature = self.model(style_img).clone()
                    target_feature_gram = gram_matrix(target_feature)
                    style_loss = StyleLoss(target_feature_gram, style_weight)
                    style_losses.append(style_loss)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(i)

                if name in content_layers:
                    # add content loss:
                    target = self.model(img_tensor).clone()
                    content_loss = ContentLoss(target, content_weight)
                    content_losses.append(content_loss)

                if name in style_layers:
                    # add style loss:
                    target_feature = self.model(style_img).clone()
                    target_feature_gram = gram_matrix(target_feature)
                    style_loss = StyleLoss(target_feature_gram, style_weight)
                    style_losses.append(style_loss)

                i += 1

        input_image = Variable(img_tensor.clone().data, requires_grad=True)
        optimizer = torch.optim.LBFGS([input_image])
        
        num_steps = 300

        for i in range(num_steps):
            # correct the values of updated input image
            input_image.data.clamp_(0, 1)

            model(input_image)
            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()
                
            loss = style_score + content_score
            
            optimizer.step(lambda:loss)
            optimizer.zero_grad()
            
        input_image.data.clamp_(0, 1)
        
        save_image(input_image.cpu().data.numpy()[0].transpose(1, 2, 0), "res_photo.jpg")
        
