import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from data_loader import TorchvisionDataLoader
from torch.utils.tensorboard import SummaryWriter
from model import StackNet34, StackOfConv, Net
from tqdm import tqdm
import numpy as np

import argparse

parser = argparse.ArgumentParser()
## arguments for Model 

parser.add_argument('--output_units', type=int, default = 2, help="Number of units in output classes")
parser.add_argument('--activation', type=str, default = 'ReLU', help="Activation")
## arguments for data loaders
parser.add_argument('--dataset', type=str, default = 'MNIST', help="name of dataset class to load from Torchvision")
parser.add_argument('--batch_size', type=int, default = 16, help="Batch Size")
parser.add_argument('--data_dir', type=str, default = './../datasets', help="folder location")

## arguments for training
parser.add_argument('--epochs', default=500, type=int, help="number of epochs to train")
parser.add_argument('--optimizer', default='Adam', type=str, help="optimizer")
parser.add_argument('--learning_rate', default=0.01, type=float, help="learning rate")
parser.add_argument('--device', default='cuda', type=str, help="device to train on")




args = parser.parse_args()

device = getattr(args, "device")

exp_name = f"ConvNetBigger_optim_{args.optimizer}_lr_{args.learning_rate}"
writer = SummaryWriter(f'runs/{exp_name}')

print(device)
### Create Model here
Model = StackNet34()
Model.to('cuda:0')
print(Model)

### create or instantiate data loader
_data = TorchvisionDataLoader(args)
_data.prepare_data()

# optimizer stuff here
Optimizer = getattr(optim, getattr(args, "optimizer"))
optimizer = Optimizer(Model.parameters(), lr = getattr(args, "learning_rate"))
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

# loss function
loss_fn = nn.CrossEntropyLoss()

_data.setup(stage = 'fit')

for i in range(getattr(args, "epochs")):
    train_loss = []
    for x, y in _data.train_dataloader():
        
        x, y = x.to('cuda:0'), y.to('cuda:0')
        
        optimizer.zero_grad()

        out = Model(x)
        loss = loss_fn(out, y)
        
        
        
        # backward step
        loss.backward()

        optimizer.step()

        train_loss.append(loss.cpu().data)
    writer.add_scalar('training loss', np.mean(train_loss), i)
    
    val_loss = []
    for x, y in _data.val_dataloader():
                    
        with torch.no_grad():
            x, y = x.to('cuda:0'), y.to('cuda:0')
            out = Model(x)
            loss = loss_fn(out, y)
            val_loss.append(loss.cpu().data)
    writer.add_scalar('validation loss', np.mean(val_loss), i)
    print(f'At epoch {i} train loss is {np.mean(train_loss)} and validation loss is {np.mean(val_loss)}')

    scheduler.step()
    optimizer.step()
    

        
        

    


        


        


