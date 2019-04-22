import argparse


from collections import OrderedDict
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import util_functions as fnc
import training_driver as drv
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",
                        help="data directory of images. root directory is ImageClassifier")
    
    parser.add_argument("--save_dir",
                        help ="directory to save model checkpoint")
    
    parser.add_argument("--arch",
                        help = "netwrok architecture to use in trasfer learning",
                        choices=["densenet", "resnet" ])
    
    parser.add_argument("--learning_rate",
                        help = "initial learning rate for optimizer",
                        type=float )
    
    parser.add_argument("--hidden_units",
                        help = "hidden units in the first hidden layer after frozen feature extractor net",
                        type=int )
    
    parser.add_argument("--epochs",
                        help = "number of epochs during training the network",
                        type=int )
    
    parser.add_argument("--gpu",
                        help = "flag used to specify using GPU for training the network",
                        action="store_true")
    
    args = parser.parse_args()
    train_net(args)
    

def train_net(args):
    
    data_dir = args.data_dir
    train_data, valid_data, test_data, trainloader, validloader, testloader=fnc.import_transfrom(data_dir)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    num_classes = len(cat_to_name)
    
    
    
    ##
    if args.hidden_units is None:
        first_hidden=1024
    
    else:
        first_hidden = args.hidden_units
    ##
    
    hidden_layers = [first_hidden,500]
    p_drop=0.5
    
    # load a pretrained model and add your classifier: 
    
    if args.arch is None:
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features
        fnc.set_parameter_requires_grad(model)
        network_architecture_list= fnc.my_network(input_size,num_classes,hidden_layers,p_drop)
        net_name = 'densenet121'
    
        a = OrderedDict()
        for idx,valu in enumerate(network_architecture_list):
            name = str(idx)
            a[name] = valu
        model.classifier = nn.Sequential(a)
    
    elif args.arch == 'densenet':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features
        fnc.set_parameter_requires_grad(model)
        network_architecture_list= fnc.my_network(input_size,num_classes,hidden_layers,p_drop)
        net_name = 'densenet121'
    
        a = OrderedDict()
        for idx,valu in enumerate(network_architecture_list):
            name = str(idx)
            a[name] = valu
        model.classifier = nn.Sequential(a)
    else:
        model = models.resnet18(pretrained=True)
        input_size = model.fc.in_features
        fnc.set_parameter_requires_grad(model)
        network_architecture_list= fnc.my_network(input_size,num_classes,hidden_layers,p_drop)
        net_name = 'res18'
    
        a = OrderedDict()
        for idx,valu in enumerate(network_architecture_list):
            name = str(idx)
            a[name] = valu
        model.fc = nn.Sequential(a)
    
    if args.gpu:
        device = 'cuda'
    else: 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.learning_rate is None:
        learn_rate=0.001
    else:
        learn_rate = args.learning_rate
        
    model = model.to(device)
    to_update=fnc.params_to_update(model)    
    optimizer = optim.Adam(to_update,lr=learn_rate)
    # auto scheduler for the learning rate 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=4)

    criterion = nn.NLLLoss()

    
    if args.epochs is None:
        epochs=5
    else:
        epochs = args.epochs
    
    model, val_acc_history = drv.training(model, trainloader, validloader, criterion, optimizer,scheduler,device,epochs)
    
    
    # Save the checkpoint 
    model.class_to_idx = train_data.class_to_idx
    model.optim_status = optimizer.state_dict
    model.network_architecture_list = network_architecture_list
    
    checkpoint = {'class_to_idx': train_data.class_to_idx,
                  'optim_status': optimizer.state_dict,
                  'network_architecture_list': network_architecture_list,
                  'pretrained_net': net_name,
                  'state_dict': model.state_dict()}
    
    if args.save_dir is None:
        
        checkpoint_name = 'checkpoint_' + net_name + '.pth'
    else:
        checkpoint_name = args.save_dir + '/checkpoint_' + net_name + '.pth'
        
    torch.save(checkpoint, checkpoint_name)

    
    
    

if __name__ == '__main__':
    main()
    
    