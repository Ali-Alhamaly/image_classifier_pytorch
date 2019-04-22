# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 09:44:25 2019

@author: SMART
"""

from collections import OrderedDict
import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision import datasets, transforms, models

def import_transfrom(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    
    
    train_transforms = transforms.Compose([transforms.RandomOrder([transforms.RandomRotation(30),
                                                               transforms.RandomResizedCrop(224),
                                                               transforms.RandomHorizontalFlip(),
                                                               transforms.RandomVerticalFlip()]),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
# TODO: Load the datasets with ImageFolder

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)



# TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    return train_data, valid_data, test_data, trainloader, validloader, testloader

def my_network(input_size,num_classes,hidden_layers,p_drop=0.5):
    layers = []
    model_list = []
    layers.append(input_size)
    layers.extend(hidden_layers)
    layer_sizes = zip(layers[:-1], layers[1:])
    for h1 , h2 in layer_sizes:
        model_list.extend([nn.Linear(h1,h2),nn.ReLU(),nn.Dropout(p_drop)])    
    model_list.extend([nn.Linear(hidden_layers[-1],num_classes),nn.LogSoftmax(dim=1)])
    return model_list
    
    
def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False
        
def nn_squential(network_architecture_list):
    a = OrderedDict()
    for idx,valu in enumerate(network_architecture_list):
        name = str(idx)
        a[name] = valu
    return nn.Sequential(a)
    
def params_to_update(model):
    to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            to_update.append(param)
            print("\t",name)
    return to_update    
    
    
    
def load_checkpoint(filepath,flag):
    
    if flag ==1:
        model = models.densenet121(pretrained=True)
    else:
        model = models.resnet18(pretrained=True)
        
    set_parameter_requires_grad(model)
    
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath)
    else: 
        checkpoint = torch.load(filepath,map_location='cpu')
    
    sequential_model=nn_squential(checkpoint['network_architecture_list'])
    
    if flag ==1:
        model.classifier = sequential_model
    else:
        model.fc = sequential_model
        
    model.load_state_dict(checkpoint['state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.class_to_idx = checkpoint['class_to_idx']
    model = model.to(device)
    return model        


def process_image(image):
    
    img_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    pil_image = Image.open(image)
    pil_image = img_transforms(pil_image).float()
    pil_image.unsqueeze_(0) 
#    im = Image.open(image)
#    im
#    im = im.resize((255,255))
#    
#    left = (255 - 224)/2
#    top = (255 - 224)/2
#    right = (255 + 224)/2
#    bottom = (255 + 224)/2
#    im = im.crop((left, top, right, bottom))
#    
#    
#    np_image = np.array(im)/255.0
#    np_image[:,:,0]= (np_image[:,:,0]-.485)/0.229
#    np_image[:,:,1]= (np_image[:,:,1]-.456)/0.224
#    np_image[:,:,2]= (np_image[:,:,2]-.406)/0.225
#    np_image=np_image.transpose(2,0,1)
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    return pil_image