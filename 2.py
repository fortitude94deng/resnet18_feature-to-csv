# -*- coding: utf-8 -*-
"""
dj
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():

    return ResNet(ResidualBlock)

import torch
import torch.nn as nn
import os
from torchvision import models, transforms
from torch.autograd import Variable   
import numpy as np
from PIL import Image  
import torchvision.models as models
import pretrainedmodels
import pandas as pd
class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class M(nn.Module):
    def __init__(self, backbone1, drop, pretrained=True):
        super(M,self).__init__()
        if pretrained:
            img_model = pretrainedmodels.__dict__[backbone1](num_classes=1000, pretrained='imagenet') 
        else:
            img_model = ResNet18()
            we='/home/cc/Desktop/dj/model1/incption--7'
            # 模型定义-ResNet
            #net = ResNet18().to(device)
            img_model.load_state_dict(torch.load(we))#diaoyong        
        self.img_encoder = list(img_model.children())[:-2]
        self.img_encoder.append(nn.AdaptiveAvgPool2d(1))
        self.img_encoder = nn.Sequential(*self.img_encoder)
        if drop > 0:
            self.img_fc = nn.Sequential(FCViewer())                                  
        else:
            self.img_fc = nn.Sequential(
                FCViewer())
    def forward(self, x_img):
        x_img = self.img_encoder(x_img)
        x_img = self.img_fc(x_img)
        return x_img 
model1=M('resnet18',0,pretrained=None)
features_dir = '/home/cc/Desktop/features' 
transform1 = transforms.Compose([
        transforms.Resize(56),
        transforms.CenterCrop(32),
        transforms.ToTensor()]) 
file_path='/home/cc/Desktop/picture'
names = os.listdir(file_path)
print(names)
for name in names:
    pic=file_path+'/'+name
    img = Image.open(pic)
    img1 = transform1(img)
    x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
    y = model1(x)
    y = y.data.numpy()
    y = y.tolist()
    #print(y)
    test=pd.DataFrame(data=y)
    #print(test)
    test.to_csv("/home/cc/Desktop/features/3.csv",mode='a+',index=None,header=None)
'''
import csv
for name in names:
    pic=file_path+'/'+name
    img = Image.open(pic)
    img1 = transform1(img)
    x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
    y = model1(x)
    y = y.data.numpy()
    y = y.tolist()
    with open("/home/cc/Desktop/features/3.csv",'a+') as f:
        csv_write =csv.writer(f)
        csv_write.writerow(y)
'''





























        
      
  


