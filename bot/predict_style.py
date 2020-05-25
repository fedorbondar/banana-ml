import os
from PIL import Image

import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.models as models
from torch.autograd import Variable

from matplotlib.pyplot import imread
from skimage.transform import resize

def image_loader(image_name):
    image = resize(imread(image_name), [512, 512])
    image = image.transpose([2,0,1]) / image.max()
    image = Variable(torch.FloatTensor(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image

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

    def get_image_predict(self, img_path='img_path.jpg', option="1"):

        img_tensor = image_loader(img_path).type(torch.FloatTensor)
        
        if option == "1":
            style_img = image_loader("wave.jpg").type(torch.FloatTensor)
        if option == "2":
            style_img = image_loader("the_scream.jpg").type(torch.FloatTensor)
        if option == "3":
            style_img = image_loader("starry_night.jpg").type(torch.FloatTensor)
        
        content_weight = 1            # coefficient for content loss
        style_weight = 1000           # coefficient for style loss
        content_layers = ('conv_4',)  # use these layers for content loss
        style_layers = ('conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5')

        cnn = models.vgg19(pretrained=True).features

        content_losses = []
        style_losses = []

        i = 1
        for layer in list(cnn):
            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)
                self.model.add_module(name, layer)

                if name in content_layers:
                    # add content loss:
                    target = self.model(img_tensor).clone()
                    content_loss = ContentLoss(target, content_weight)
                    self.model.add_module("content_loss_" + str(i), content_loss)
                    content_losses.append(content_loss)

                if name in style_layers:
                    # add style loss:
                    target_feature = self.model(style_img).clone()
                    target_feature_gram = gram_matrix(target_feature)
                    style_loss = StyleLoss(target_feature_gram, style_weight)
                    self.model.add_module("style_loss_" + str(i), style_loss)
                    style_losses.append(style_loss)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(i)
                self.model.add_module(name, layer)

                if name in content_layers:
                    # add content loss:
                    target = self.model(img_tensor).clone()
                    content_loss = ContentLoss(target, content_weight)
                    self.model.add_module("content_loss_" + str(i), content_loss)
                    content_losses.append(content_loss)

                if name in style_layers:
                    # add style loss:
                    target_feature = self.model(style_img).clone()
                    target_feature_gram = gram_matrix(target_feature)
                    style_loss = StyleLoss(target_feature_gram, style_weight)
                    self.model.add_module("style_loss_" + str(i), style_loss)
                    style_losses.append(style_loss)

                i += 1

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(i)
                self.model.add_module(name, layer)  # ***

        input_image = Variable(img_tensor.clone().data, requires_grad=True)
        optimizer = torch.optim.LBFGS([input_image])
        
        num_steps = 300

        for i in range(num_steps):
            # correct the values of updated input image
            input_image.data.clamp_(0, 1)

            self.model(input_image)
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
        print("training finished")
        
        save_image(input_image.data, 'res_photo.jpg')
        print("saved to disc")
        
        
