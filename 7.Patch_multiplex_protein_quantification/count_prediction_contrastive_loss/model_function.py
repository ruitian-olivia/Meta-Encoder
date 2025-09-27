import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

class protein_poisson_count(nn.Module):
    def __init__(self, common_dim, num_marker, feature_size_list):
        super(protein_poisson_count, self).__init__()
        
        self.linear1 = torch.nn.Linear(feature_size_list[0], common_dim)
        self.linear2 = torch.nn.Linear(feature_size_list[1], common_dim)
        self.linear3 = torch.nn.Linear(feature_size_list[2], common_dim)

        in_features = common_dim * 3
        self.head = nn.Linear(in_features, num_marker)
        
    def forward(self, x1, x2, x3):
        
        x1 = self.linear1(x1)
        x2 = self.linear2(x2)
        x3 = self.linear3(x3)
        
        x = torch.cat((x1, x2, x3), dim=-1)
        
        lambda_params = nn.functional.softplus(self.head(x))
        
        return lambda_params, x1, x2, x3