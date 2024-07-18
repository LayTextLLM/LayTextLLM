import torch.nn as nn

class LinearProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProjector, self).__init__()
        # Initialize the linear layer with the given dimensions
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Forward pass through the linear layer
        return self.linear(x)
