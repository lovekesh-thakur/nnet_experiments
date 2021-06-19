# resnet architecture here

import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    """
    Residual block in Resnet Architecture
    """
    def __init__(self, input_channels, output_channels, downsample = False):
        """
        
        Args:
        ----- 
        input_channels : number of channels from the input
        output_channels : number of output channels
        downsample : True if we want to reduce feature size else False

        """
        super(Residual, self).__init__()
        
        self.downsample = downsample
        if self.downsample: 
            strides = 2
        else:
            strides = 1
        
        self.conv1 = nn.Conv2d(input_channels, output_channels,
                               kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

        if self.downsample:
            self.conv1x1 = nn.Conv2d(input_channels, output_channels, kernel_size=1, 
                                     stride=strides)
    
    def forward(self, x):
        """
        Forward method of Residual

        Args:
        -------
        x : tensor of shape [N, C, H, W]
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            return F.relu(self.conv1x1(x) + out)
        else:
            return F.relu(out) + x

class Resnet34(nn.Module):
    """
    Resnet 34 architecture
    """
    def __init__(self, blocks = [3, 4, 6, 3], classes = 2):
        
        super().__init__()
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride = 2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        inp_channels = 64
        resid_blks = []

        for ind, blk in enumerate(blocks):
            if ind == 0:
                out_channels = 1*inp_channels
            else:
                out_channels = 2*inp_channels

            for i in range(blk):
                if i == 0:
                    residual_block = Residual(inp_channels, out_channels, downsample=True)
                else:
                    residual_block = Residual(out_channels, out_channels)
                resid_blks.append(residual_block)
            
            inp_channels = out_channels
        self.subnet = nn.Sequential(*resid_blks)
        self.global_pooling = nn.AvgPool2d(kernel_size=5)
        self.fc1 = nn.Linear(out_channels, 1)

    
    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.global_pooling(self.subnet(out))
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        return torch.squeeze(out)


class Resnet18(nn.Module):
    """
    Resnet 34 architecture
    """
    def __init__(self, blocks = [2, 2, 2, 2], classes = 2):
        
        super().__init__()
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride = 2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        inp_channels = 64
        resid_blks = []

        for ind, blk in enumerate(blocks):
            if ind == 0:
                out_channels = 1*inp_channels
            else:
                out_channels = 2*inp_channels

            for i in range(blk):
                if i == 0:
                    residual_block = Residual(inp_channels, out_channels, downsample=True)
                else:
                    residual_block = Residual(out_channels, out_channels)
                resid_blks.append(residual_block)
            
            inp_channels = out_channels
        self.subnet = nn.Sequential(*resid_blks)
        self.global_pooling = nn.AvgPool2d(kernel_size=5)
        self.fc1 = nn.Linear(out_channels, 1)

    
    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.global_pooling(self.subnet(out))
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        return torch.squeeze(out)









if __name__ == '__main__':
    x = torch.ones((10, 3, 300, 300))
    model = Resnet18()
    print(model(x).shape)