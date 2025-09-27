import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

class protein_poisson_count(nn.Module):
    def __init__(self, num_feature, num_marker):
        super(protein_poisson_count, self).__init__()
        self.head = nn.Linear(num_feature, num_marker)
        
    def forward(self, features):
        lambda_params = nn.functional.softplus(self.head(features))
        
        return lambda_params
    