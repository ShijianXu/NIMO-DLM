import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=1):
        super(MLP, self).__init__()
        # `layers` counts all linear layers including the final output layer.
        # e.g. layers=2: Linear(input->hidden) + ReLU + Linear(hidden->output)
        #      layers=1: Linear(input->output)  (no hidden layer)
        assert layers >= 1
        network = []
        in_dim = input_dim
        for _ in range(layers - 1):
            network.append(nn.Linear(in_dim, hidden_dim))
            network.append(nn.ReLU())
            in_dim = hidden_dim
        network.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)
