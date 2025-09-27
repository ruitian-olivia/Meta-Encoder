import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

from attention_utils import MultiheadAttention

class protein_poisson_count(nn.Module):
    def __init__(self, embed_dim, num_heads, num_marker, feature_size_list):
        super(protein_poisson_count, self).__init__()
        
        self.cross_attention1 = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, kdim=feature_size_list[0], vdim=feature_size_list[0], qdim=feature_size_list[1])
        self.cross_attention2 = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, kdim=feature_size_list[0], vdim=feature_size_list[0], qdim=feature_size_list[2])

        self.head = nn.Linear(embed_dim*2, num_marker)
        
    def forward(self, x1, x2, x3):
        x1, x2, x3 = x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0)
        
        x12, _ = self.cross_attention1(x2, x1, x1)
        x13, _ = self.cross_attention2(x3, x1, x1)
        x12 = x12.squeeze(0)
        x13 = x13.squeeze(0)
        x_kqv = torch.cat((x12, x13), dim=1)
        x_kqv = x_kqv.squeeze()
        
        lambda_params = nn.functional.softplus(self.head(x_kqv))
        
        return lambda_params
    
