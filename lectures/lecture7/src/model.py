import torch.nn as nn
from collections import OrderedDict

class MLP():
    
    def __init__(self, args):
        self.kwargs = args
        self.hidden_units = getattr(self.kwargs, "hidden_units")
        self.input_units = getattr(self.kwargs, "input_units")
        self.output_units = getattr(self.kwargs, "output_units")
        self.activation = getattr(nn, getattr(self.kwargs, "activation"))
        
    
    def __call__(self):
        layers = []
        prev_units = self.input_units
        for ind, units in enumerate(self.hidden_units):
            layers += [(f'layer_{ind}', nn.Linear(prev_units, units))]
            layers += [(f'act_{ind}', self.activation())]
            prev_units = units
        layers += [(f'layer_{ind+1}', nn.Linear(prev_units, self.output_units))]
        return nn.Sequential(OrderedDict(layers))



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_units', type=list, default = [32, 64], help="number of units in hidden layers")
    parser.add_argument('--input_units', type=int, default = 28*28, help="Number of units in input layer")
    parser.add_argument('--output_units', type=int, default = 10, help="Number of units in output classes")
    parser.add_argument('--activation', type=str, default = 'ReLU', help="Activation")
 
    args = parser.parse_args()

    
    Model = MLP(args)
    net = Model()



