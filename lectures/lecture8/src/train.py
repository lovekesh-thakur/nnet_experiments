import torch
from torch import optim
import torch.nn as nn
from data_loader import TorchvisionDataLoader
from torch.utils.tensorboard import SummaryWriter
from model import RNN
from tqdm import tqdm
import numpy as np

import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
## arguments for Model 
parser.add_argument('--hidden_units', type=int, default = 64, help="number of units in hidden layers")
parser.add_argument('--input_units', type=int, default = 1, help="Number of units in input layer")
parser.add_argument('--output_units', type=int, default = 2, help="Number of units in output classes")
parser.add_argument('--activation', type=str, default = 'ReLU', help="Activation")
## arguments for data loaders
parser.add_argument('--dataset', type=str, default = 'MNIST', help="name of dataset class to load from Torchvision")
parser.add_argument('--batch_size', type=int, default = 16, help="Batch Size")
parser.add_argument('--data_dir', type=str, default = './../../lecture7/data/', help="folder location")

## arguments for training
parser.add_argument('--epochs', default=20, type=int, help="number of epochs to train")
parser.add_argument('--optimizer', default='Adam', type=str, help="optimizer")
parser.add_argument('--learning_rate', default=0.001, type=float, help="learning rate")
parser.add_argument('--device', default='cuda', type=str, help="device to train on")




args = parser.parse_args()

device = getattr(args, "device")

exp_name = f"Exp_optim_{args.optimizer}_lr_{args.learning_rate}_hu_{args.hidden_units}"
writer = SummaryWriter(f'runs/{exp_name}')

print(device)
### Create Model here
Model = RNN(args)
net = Model.to(torch.device(device))

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
    for ip, _ in tqdm(_data.train_dataloader()):
        
        x = ip[:, :-1, :]
        y = ip[:, 1:, :].reshape(-1)
        x, y = x.to(device), y.to(device)
        out = net(x)
        loss = loss_fn(out.reshape(out.shape[0]*out.shape[1], -1), y.long())

        optimizer.zero_grad()
        
        # backward step
        loss.backward()

        optimizer.step()

        train_loss.append(loss.cpu().data)
    writer.add_scalar('training loss', np.mean(train_loss), i)

    for ip, _ in tqdm(_data.val_dataloader()):
        
        val_loss = []        
        with torch.no_grad():
            x = ip[:, :-1, :]
            x = x.to(device)
            out = net(x)
            pred_img = torch.zeros([16, 784])
            pred_img[:, :-1] = torch.argmax(out, 2)
            pred_img = pred_img.int()
            pred_img = pred_img.reshape(16, 28, 28)

            fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True)
            axes = axes.ravel()
            for j in range(16):
                axes[j].imshow(pred_img[j])
            writer.add_figure(f'epoch_{i}', fig)
        break
        
    #  writer.add_scalar('validation loss', np.mean(val_loss), i)
    print(f'At epoch {i} train loss is {np.mean(train_loss)}')
    

        
        

    


        


        


