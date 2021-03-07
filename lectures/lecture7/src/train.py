import torch
from torch import optim
import torch.nn as nn
from data_loader import TorchvisionDataLoader
from torch.utils.tensorboard import SummaryWriter
from model import MLP
from tqdm import tqdm
import numpy as np

import argparse

parser = argparse.ArgumentParser()
## arguments for Model 
parser.add_argument('--hidden_units', type=list, default = [32, 64], help="number of units in hidden layers")
parser.add_argument('--input_units', type=int, default = 28*28, help="Number of units in input layer")
parser.add_argument('--output_units', type=int, default = 10, help="Number of units in output classes")
parser.add_argument('--activation', type=str, default = 'ReLU', help="Activation")
## arguments for data loaders
parser.add_argument('--dataset', type=str, default = 'MNIST', help="name of dataset class to load from Torchvision")
parser.add_argument('--batch_size', type=int, default = 16, help="Batch Size")
parser.add_argument('--data_dir', type=str, default = './../data/', help="folder location")

## arguments for training
parser.add_argument('--epochs', default=20, type=int, help="number of epochs to train")
parser.add_argument('--optimizer', default='Adam', type=str, help="optimizer")
parser.add_argument('--learning_rate', default=0.001, type=float, help="learning rate")
parser.add_argument('--device', default='cpu', type=str, help="device to train on")




args = parser.parse_args()

device = getattr(args, "device")

exp_name = f"Exp_optim_{args.optimizer}_lr_{args.learning_rate}"
writer = SummaryWriter(f'runs/{exp_name}')


### Create Model here
Model = MLP(args)
net = Model().to(device)

### create or instantiate data loader
_data = TorchvisionDataLoader(args)
_data.prepare_data()

# optimizer stuff here
Optimizer = getattr(optim, getattr(args, "optimizer"))
optimizer = Optimizer(net.parameters(), lr = getattr(args, "learning_rate"))

# loss function
loss_fn = nn.CrossEntropyLoss()

_data.setup(stage = 'fit')

for i in range(getattr(args, "epochs")):
    train_loss = []
    for x, y in tqdm(_data.train_dataloader()):
        
        
        
        x, y = x.to(device), y.to(device)
        x = x.view(x.shape[0], -1)
        out = net(x)
        loss = loss_fn(out, y)

        optimizer.zero_grad()
        
        # backward step
        loss.backward()

        optimizer.step()

        train_loss.append(loss.cpu().data)
    writer.add_scalar('training loss', np.mean(train_loss), i)

    for x, y in tqdm(_data.val_dataloader()):
        
        val_loss = []        
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            x = x.view(x.shape[0], -1)
            out = net(x)
            loss = loss_fn(out, y)
            val_loss.append(loss.cpu().data)
    writer.add_scalar('validation loss', np.mean(val_loss), i)
    print(f'At epoch {i} val loss is {np.mean(val_loss)} train loss is {np.mean(train_loss)}')
    

        
        

    


        


        


