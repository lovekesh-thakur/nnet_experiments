import torch.nn as nn
from collections import OrderedDict

class RNN(nn.Module):
    
    def __init__(self, args):
        super(RNN, self).__init__()
        self.kwargs = args
        self.hidden_units = getattr(self.kwargs, "hidden_units")
        self.input_units = getattr(self.kwargs, "input_units")
        self.output_units = getattr(self.kwargs, "output_units")
        self.activation = getattr(nn, getattr(self.kwargs, "activation"))
        self.rnn = nn.RNN(self.input_units, self.hidden_units, batch_first=True)
        self.fc = nn.Linear(self.hidden_units, self.output_units)
    
    def forward(self, x):
        out, h = self.rnn(x)
        out = out.contiguous().view(-1, out.shape[1], self.hidden_units)

        return self.fc(out)



if __name__ == '__main__':
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_units', type=list, default = 64, help="number of units in hidden layers")
    parser.add_argument('--input_units', type=int, default = 1, help="Number of units in input layer")
    parser.add_argument('--output_units', type=int, default = 1, help="Number of units in output classes")
    parser.add_argument('--activation', type=str, default = 'ReLU', help="Activation")
 
    args = parser.parse_args()

    
    Model = RNN(args)
    net = Model
    a = torch.rand(16, 783, 1)
    print(net(a).shape)


