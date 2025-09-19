import torch
import torch.nn as nn

class Swish(nn.Module):
    '''
    Swish activation function: f(x) = x * sigmoid(x)
    '''
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)