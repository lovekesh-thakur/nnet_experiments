"""Models for Training Cifar"""
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class StackNet34(nn.Module):
    """
    Simple stack of layerts
    """

    def __init__(self, n_classes = 10):
        super(StackNet34, self).__init__()
        self.n_classes = n_classes
        #self.first_layer = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding = 1)
        #self.act_layer = nn.ReLU()
        self.layers = []#self.first_layer, self.act_layer]
        
        for l in range(6):
            if l == 0:
                self.layers.append(nn.Conv2d(3, 64, kernel_size=3, stride = 2)) # downsample using stride
            else:
                self.layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            self.layers.append(nn.BatchNorm2d(64))
            self.layers.append(nn.ReLU())

        for l in range(8):
            if l == 0:
                self.layers.append(nn.Conv2d(64, 128, kernel_size=3, stride = 2))
            else:
                self.layers.append(nn.Conv2d(128, 128, kernel_size=3, padding=1))
            self.layers.append(nn.BatchNorm2d(128))
            self.layers.append(nn.ReLU())
        
        for l in range(12):
            if l == 0:
                self.layers.append(nn.Conv2d(128, 256, kernel_size=3, stride = 2))
            else:
                self.layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
            self.layers.append(nn.BatchNorm2d(256))
            self.layers.append(nn.ReLU())
        
        for l in range(6):
            if l == 0:
                self.layers.append(nn.Conv2d(256, 512, kernel_size=3, stride = 2))
            else:
                self.layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
            self.layers.append(nn.BatchNorm2d(512))
            self.layers.append(nn.ReLU())
        self.conv_net = nn.Sequential(*self.layers)
        #a, b, c = self.layers[-1].shape[1:]
        self.fc_layer = nn.Linear(512, 10)
        
        

    
    def forward(self, x):
        out = self.conv_net(x)
        
        out = out.view(out.shape[0], -1)
        out = self.fc_layer(out)
        return out 



def StackOfConv(layers, filters, ip_c, name):
    """Stack of convolution layers"""
    
    
        
        
    stack = name
    
    self_layers = []
    
    for l in range(layers):
        if l == 0:
            self_layers.append((f'stack_conv_{stack}_{l}', nn.Conv2d(ip_c, filters, kernel_size=3, stride = 2)))
        else:
            self_layers.append((f'stack_conv_{stack}_{l}', nn.Conv2d(filters, filters, kernel_size=3, padding=1)))
        self_layers.append((f'stack_act_{stack}_{l}', nn.ReLU()))
    
    net = nn.Sequential(OrderedDict(self_layers))
    return net
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 
        

if __name__ == '__main__':
    net = StackNet34()
    x = torch.rand(16, 3, 32, 32)
    out = net(x)
    print(out.shape)